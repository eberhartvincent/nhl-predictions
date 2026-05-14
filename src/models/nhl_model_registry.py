"""nhl_model_registry.py — load/cache the trained XGBoost NHL model."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd

log = logging.getLogger(__name__)
MODEL_PATH = Path("models/nhl_model.json")
META_PATH  = Path("models/nhl_feature_metadata.json")
_model = None
_meta: dict | None = None


def load() -> tuple[object | None, dict | None]:
    global _model, _meta
    if _model is not None:
        return _model, _meta
    if not MODEL_PATH.exists() or not META_PATH.exists():
        log.warning("No NHL model at %s — using statistical model only.", MODEL_PATH)
        return None, None
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(str(MODEL_PATH))
        _model = m
        _meta  = json.loads(META_PATH.read_text())
        log.info(
            "Loaded NHL XGBoost model — n=%d, CV R²=%.4f, features=%s",
            _meta.get("n_training", "?"), _meta.get("cv_r2", 0), _meta.get("features", []),
        )
        return _model, _meta
    except Exception as exc:
        log.warning("Failed to load NHL model: %s — using statistical model only.", exc)
        return None, None


def predict_goals_per_60(mp_metrics: dict, model, meta: dict) -> float | None:
    """Use XGBoost model to predict a skater's goals/60 talent rate."""
    features: list[str] = meta["features"]
    medians:  dict      = meta["feature_medians"]

    row = {}
    for feat in features:
        val = mp_metrics.get(feat, np.nan)
        try:
            f = float(val)
            row[feat] = f if np.isfinite(f) else medians.get(feat, np.nan)
        except (TypeError, ValueError):
            row[feat] = medians.get(feat, np.nan)

    X = pd.DataFrame([row])[features].fillna(pd.Series(medians))
    if X.isna().all(axis=None):
        return None
    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0.0, 4.0))
