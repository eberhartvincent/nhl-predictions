"""
nhl_train.py
============
Trains THREE separate XGBoost models from Moneypuck data:
  1. goals/60    → P(≥1 goal tonight)
  2. points/60   → P(≥1 point tonight)
  3. shots/60    → P(over X.5 shots tonight)

Each model uses year-N features → year-(N+1) target (no look-ahead bias).
All three saved to models/ and committed by the retrain workflow.
"""
from __future__ import annotations

import io, json, logging, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MIN_TOI_SECONDS = 300 * 60
HEADERS = {"User-Agent": "nhl-goal-predictor/1.0 (github-actions; open-source)"}
SKATERS_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary"
    "/{season}/regular/skaters.csv"
)

# Features shared across all three models
BASE_FEATURES = [
    "xg_per_60",
    "shots_per_60",
    "shooting_pct",
    "hd_xg_per_60",
    "hd_shooting_pct",
    "pp_toi_pct",
    "corsi_pct",
    "xg_pct",
    "toi_per_game",
    "primary_assists_per_60",   # extra signal for points model
]

# Which features each model uses
MODEL_FEATURES = {
    "goals":  ["xg_per_60", "shooting_pct", "hd_xg_per_60",
                "hd_shooting_pct", "pp_toi_pct", "corsi_pct",
                "xg_pct", "toi_per_game"],
    "points": ["xg_per_60", "shots_per_60", "primary_assists_per_60",
                "pp_toi_pct", "corsi_pct", "xg_pct", "toi_per_game"],
    "shots":  ["shots_per_60", "corsi_pct", "toi_per_game",
                "pp_toi_pct", "xg_pct"],
}

MODEL_TARGETS = {
    "goals":  "goals_per_60",
    "points": "points_per_60",
    "shots":  "shots_per_60",
}


def _col(df: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    """Return first matching column as float Series, never a scalar."""
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=df.index, dtype=float)


def fetch_season(season: int) -> pd.DataFrame | None:
    url = SKATERS_URL.format(season=season)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Moneypuck/%d: %d rows", season, len(df))
        log.info("Moneypuck/%d columns: %s", season, df.columns.tolist())
        return df
    except Exception as exc:
        log.warning("Moneypuck/%d failed: %s", season, exc)
        return None


def build_season_frame(df_raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build canonical feature frame — goals, points, AND shots targets."""
    df = df_raw[df_raw["situation"] == "all"].copy()
    if "position" in df.columns:
        df = df[df["position"].isin(["C", "L", "R"])].copy()

    toi_col = next((c for c in ["icetime", "toi", "timeOnIce"] if c in df.columns), None)
    if toi_col is None:
        log.warning("Season %d: no TOI column. Skipping.", season)
        return pd.DataFrame()

    df = df[_col(df, toi_col) >= MIN_TOI_SECONDS].copy()
    if df.empty:
        return pd.DataFrame()

    toi_sec = _col(df, "icetime", "toi", "timeOnIce")
    toi_hr  = (toi_sec / 3600).replace(0, np.nan)
    toi_min = toi_sec / 60
    gp      = _col(df, "games_played", "gamesPlayed", default=1.0).replace(0, 1)

    goals   = _col(df, "I_F_goals", "goals", "Goals")
    assists = _col(df, "I_F_primaryAssists", "primaryAssists") + \
              _col(df, "I_F_secondaryAssists", "secondaryAssists")
    points  = _col(df, "I_F_points", "points") if any(
                  c in df.columns for c in ["I_F_points", "points"]
              ) else goals + assists
    shots   = _col(df, "I_F_shotsOnGoal", "shotsOnGoal", "I_F_shots")
    xg      = _col(df, "I_F_xGoals", "xGoals", "I_F_expectedGoals")
    hd_sh   = _col(df, "I_F_highDangerShots", "highDangerShots")
    hd_xg   = _col(df, "I_F_highDangerxGoals", "highDangerxGoals",
                   "I_F_highDangerExpectedGoals")
    prim_a  = _col(df, "I_F_primaryAssists", "primaryAssists")
    corsi   = _col(df, "onIce_corsiPercentage", "corsiPercentage", default=np.nan)
    xg_pct  = _col(df, "onIce_xGoalsPercentage", "xGoalsPercentage", default=np.nan)

    sh_pct    = goals.where(shots >= 10, np.nan)    / shots.replace(0, np.nan)
    hd_sh_pct = goals.where(hd_sh >= 5, np.nan)    / hd_sh.replace(0, np.nan)

    # PP TOI fraction
    pp_raw = df_raw[
        (df_raw["situation"] == "5on4") &
        df_raw["playerId"].isin(df["playerId"].values)
    ].copy()
    pp_toi_series = pd.Series(0.0, index=df.index, dtype=float)
    if not pp_raw.empty and toi_col in pp_raw.columns:
        pp_map = (
            pd.to_numeric(pp_raw[toi_col], errors="coerce")
            .fillna(0).groupby(pp_raw["playerId"]).sum()
        )
        pp_toi_series = df["playerId"].map(pp_map).fillna(0) / toi_sec.replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["player_id"]             = df["playerId"].values
    out["season"]                = season
    out["games_played"]          = gp.values
    out["season_toi_sec"]        = toi_sec.values

    # ── Three targets ────────────────────────────────────────────────────
    out["goals_per_60"]          = (goals   / toi_hr).values
    out["points_per_60"]         = (points  / toi_hr).values
    out["shots_per_60"]          = (shots   / toi_hr).values

    # ── Shared features ──────────────────────────────────────────────────
    out["xg_per_60"]             = (xg      / toi_hr).values
    out["shooting_pct"]          = sh_pct.values
    out["hd_xg_per_60"]         = (hd_xg   / toi_hr).values
    out["hd_shooting_pct"]       = hd_sh_pct.values
    out["primary_assists_per_60"]= (prim_a  / toi_hr).values
    out["corsi_pct"]             = corsi.values
    out["xg_pct"]                = xg_pct.values
    out["toi_per_game"]          = (toi_min / gp).values
    out["pp_toi_pct"]            = pp_toi_series.values

    out = out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    log.info("Season %d: %d qualifying forwards (goals/60 avg %.3f, shots/60 avg %.3f)",
             season, len(out),
             float(out["goals_per_60"].mean(skipna=True)),
             float(out["shots_per_60"].mean(skipna=True)))
    return out


def build_training_pairs(
    stats: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Year-N features → Year-(N+1) target."""
    seasons = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n, yr_np1 = seasons[i], seasons[i + 1]
        n   = stats[stats["season"] == yr_n].set_index("player_id")
        np1 = stats[stats["season"] == yr_np1].set_index("player_id")
        common = n.index.intersection(np1.index)
        if common.empty:
            continue

        avail   = [f for f in features if f in n.columns]
        X_block = n.loc[common, avail].copy()
        y_block = np1.loc[common, target].copy()
        valid   = X_block.notna().any(axis=1) & y_block.notna() & (y_block > 0)

        all_X.append(X_block[valid])
        all_y.append(y_block[valid])
        log.info("  %s pair %d→%d: %d examples", target, yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError(f"No valid training pairs for target: {target}")
    return (
        pd.concat(all_X).reset_index(drop=True),
        pd.concat(all_y).reset_index(drop=True),
    )


def _train_one(
    stats: pd.DataFrame,
    model_name: str,
    features: list[str],
    target: str,
    clip_max: float,
) -> tuple[object, dict]:
    """Train one XGBoost model, return (model, metadata)."""
    import xgboost as xgb

    log.info("── Training %s model ──", model_name)
    available = [f for f in features if f in stats.columns
                 and stats[f].notna().sum() >= 20]
    if len(available) < 2:
        raise RuntimeError(f"{model_name}: too few features: {available}")

    X, y = build_training_pairs(stats, available, target)
    log.info("%s: %d examples × %d features", model_name, len(X), X.shape[1])

    medians = X.median()
    X = X.fillna(medians)

    n_splits = min(5, max(2, len(X) // 20))
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    maes = []

    xgb_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.5,
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, verbosity=0,
    )

    for fold, (tr, val) in enumerate(kf.split(X), 1):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = np.clip(m.predict(X.iloc[val]), 0, clip_max)
        oof[val] = p
        maes.append(mean_absolute_error(y.iloc[val], p))

    cv_mae = float(np.mean(maes))
    cv_r2  = float(r2_score(y, oof))
    log.info("%s: CV MAE=%.4f, CV R²=%.4f", model_name, cv_mae, cv_r2)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)

    importance = dict(sorted(
        zip(available, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    metadata = {
        "model_name":      model_name,
        "target":          target,
        "features":        available,
        "feature_medians": medians[available].to_dict(),
        "lg_avg":          float(y.mean()),
        "training_years":  None,   # filled in by caller
        "n_training":      len(X),
        "cv_mae":          cv_mae,
        "cv_r2":           cv_r2,
        "feature_importance": importance,
    }
    return model, metadata


def train(years: list[int] | None = None) -> dict:
    """Train all three models and save to models/."""
    if years is None:
        years = [2021, 2022, 2023, 2024]

    MODEL_DIR.mkdir(exist_ok=True)

    # Fetch data
    frames = []
    for year in years:
        df_raw = fetch_season(year)
        if df_raw is not None:
            df = build_season_frame(df_raw, year)
            if not df.empty:
                frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No data fetched from Moneypuck.")

    stats = pd.concat(frames, ignore_index=True)
    log.info("Total player-seasons: %d", len(stats))

    all_metadata = {}
    clip_maxes = {"goals": 4.0, "points": 8.0, "shots": 20.0}

    for name, features in MODEL_FEATURES.items():
        target   = MODEL_TARGETS[name]
        model, meta = _train_one(
            stats, name, features, target, clip_maxes[name]
        )
        meta["training_years"] = years

        model_path = MODEL_DIR / f"nhl_{name}_model.json"
        meta_path  = MODEL_DIR / f"nhl_{name}_metadata.json"

        model.save_model(str(model_path))
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info("Saved %s → %s", name, model_path)

        all_metadata[name] = meta

    # Summary
    log.info("=" * 55)
    log.info("NHL Training complete — all 3 models.")
    for name, meta in all_metadata.items():
        log.info("  %-7s CV R²=%.4f  MAE=%.4f  n=%d  top=%s",
                 name, meta["cv_r2"], meta["cv_mae"], meta["n_training"],
                 list(meta["feature_importance"].keys())[:2])
    log.info("=" * 55)

    return all_metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
