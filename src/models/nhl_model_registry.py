"""nhl_model_registry.py — load and cache all three NHL XGBoost models."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd

log = logging.getLogger(__name__)
MODEL_DIR = Path("models")

_models: dict[str, object]  = {}
_metas:  dict[str, dict]    = {}
_NAMES = ("goals", "points", "shots")


def load() -> tuple[dict, dict]:
    """
    Load all three models. Returns ({name: model}, {name: meta}).
    Any missing model returns None for that key — pipeline degrades gracefully.
    """
    global _models, _metas
    if _models:
        return _models, _metas

    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed.")
        return {n: None for n in _NAMES}, {n: {} for n in _NAMES}

    for name in _NAMES:
        mp = MODEL_DIR / f"nhl_{name}_model.json"
        ep = MODEL_DIR / f"nhl_{name}_metadata.json"
        if not mp.exists() or not ep.exists():
            log.warning("No %s model at %s — stat baseline only.", name, mp)
            _models[name] = None
            _metas[name]  = {}
            continue
        try:
            m = xgb.XGBRegressor()
            m.load_model(str(mp))
            _models[name] = m
            _metas[name]  = json.loads(ep.read_text())
            log.info(
                "Loaded %s model — n=%d, R²=%.4f",
                name, _metas[name].get("n_training", 0), _metas[name].get("cv_r2", 0),
            )
        except Exception as exc:
            log.warning("Failed to load %s model: %s", name, exc)
            _models[name] = None
            _metas[name]  = {}

    return _models, _metas


def predict_rate(
    mp_metrics: dict,
    model_name: str,
    model,
    meta: dict,
) -> float | None:
    """Predict goals/60, points/60, or shots/60 using the appropriate model."""
    if model is None or not meta:
        return None

    features: list[str] = meta.get("features", [])
    medians:  dict      = meta.get("feature_medians", {})

    row = {}
    for feat in features:
        val = mp_metrics.get(feat, np.nan)
        try:
            f = float(val)
            row[feat] = f if np.isfinite(f) else float(medians.get(feat, np.nan))
        except (TypeError, ValueError):
            row[feat] = float(medians.get(feat, np.nan))

    X = pd.DataFrame([row])[features].fillna(pd.Series(medians))
    if X.isna().all(axis=None):
        return None

    clip = {"goals": 4.0, "points": 8.0, "shots": 20.0}.get(model_name, 10.0)
    return float(np.clip(model.predict(X)[0], 0, clip))
