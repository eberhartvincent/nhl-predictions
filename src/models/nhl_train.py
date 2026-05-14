"""
nhl_train.py
============
Annual retraining for the XGBoost NHL goal-rate talent estimator.

Data source: Moneypuck free CSV downloads (moneypuck.com)
  - No API key, works from GitHub Actions
  - Confirmed columns: playerId, name, team, position, situation,
    season, games_played, icetime, goals, I_F_xGoals,
    I_F_shotsOnGoal, I_F_highDangerShots, I_F_highDangerxGoals,
    onIce_corsiPercentage, onIce_xGoalsPercentage

Training: Year-N features → Year-(N+1) goals/60 rate (no look-ahead bias)
Retrain: Once per year after season ends (April/May)
"""
from __future__ import annotations

import io
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "nhl_model.json"
META_PATH  = MODEL_DIR / "nhl_feature_metadata.json"

MIN_TOI_SECONDS = 300 * 60    # 300 minutes = qualifying threshold
HEADERS = {"User-Agent": "nhl-goal-predictor/1.0 (github-actions; open-source)"}

SKATERS_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary"
    "/{season}/regular/skaters.csv"
)

# Features to train on — all from Moneypuck all-situation rows
FEATURES = [
    "xg_per_60",         # individual xG/60 — strongest goal predictor
    "shots_per_60",      # shot volume per 60
    "shooting_pct",      # actual shooting %
    "hd_xg_per_60",      # high-danger xG/60 — shot quality
    "hd_shooting_pct",   # high-danger shooting %
    "pp_toi_pct",        # power play TOI fraction — huge for goal volume
    "corsi_pct",         # team possession share
    "xg_pct",            # team xG share when on ice
    "toi_per_game",      # average ice time per game
]


def fetch_season(season: int) -> pd.DataFrame | None:
    url = SKATERS_URL.format(season=season)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Moneypuck/%d: %d rows, cols: %s", season, len(df), df.columns.tolist()[:10])
        return df
    except Exception as exc:
        log.warning("Moneypuck/%d failed: %s", season, exc)
        return None


def build_season_frame(df_raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Build a canonical feature frame from a raw Moneypuck CSV.
    One row per qualifying forward, all-situation stats.
    """
    # All-situation rows only
    df = df_raw[df_raw["situation"] == "all"].copy()

    # Forwards only for training (defensemen have very different goal rates)
    if "position" in df.columns:
        df = df[df["position"].isin(["C", "L", "R"])].copy()

    # Filter by TOI
    toi_col = "icetime" if "icetime" in df.columns else None
    if toi_col is None:
        log.warning("No icetime column in season %d", season)
        return pd.DataFrame()

    df = df[df[toi_col] >= MIN_TOI_SECONDS].copy()

    # Compute derived features
    toi_hr = df[toi_col] / 3600
    toi_min = df[toi_col] / 60
    gp = df.get("games_played", pd.Series(1, index=df.index))

    goals   = pd.to_numeric(df.get("goals", 0), errors="coerce").fillna(0)
    xg      = pd.to_numeric(df.get("I_F_xGoals", 0), errors="coerce").fillna(0)
    shots   = pd.to_numeric(df.get("I_F_shotsOnGoal", 0), errors="coerce").fillna(0)
    hd_sh   = pd.to_numeric(df.get("I_F_highDangerShots", 0), errors="coerce").fillna(0)
    hd_xg   = pd.to_numeric(df.get("I_F_highDangerxGoals", 0), errors="coerce").fillna(0)
    corsi   = pd.to_numeric(df.get("onIce_corsiPercentage", np.nan), errors="coerce")
    xg_pct  = pd.to_numeric(df.get("onIce_xGoalsPercentage", np.nan), errors="coerce")

    # Fetch PP TOI from the same raw frame
    pp_rows = df_raw[
        (df_raw["situation"] == "5on4") &
        df_raw["playerId"].isin(df["playerId"])
    ].set_index("playerId")["icetime"].rename("pp_toi")

    out = pd.DataFrame()
    out["player_id"]       = df["playerId"].values
    out["season"]          = season
    out["games_played"]    = gp.values
    out["season_toi_sec"]  = df[toi_col].values
    out["goals"]           = goals.values
    out["goals_per_60"]    = (goals / toi_hr).values    # TARGET
    out["xg_per_60"]       = (xg / toi_hr).values
    out["shots_per_60"]    = (shots / toi_hr).values
    out["shooting_pct"]    = np.where(shots >= 10, goals / shots, np.nan)
    out["hd_xg_per_60"]    = (hd_xg / toi_hr).values
    out["hd_shooting_pct"] = np.where(hd_sh >= 5, goals / hd_sh, np.nan)
    out["corsi_pct"]       = corsi.values
    out["xg_pct"]          = xg_pct.values
    out["toi_per_game"]    = (toi_min / gp).values

    # PP TOI fraction
    pp_toi_vals = df["playerId"].map(pp_rows).fillna(0)
    out["pp_toi_pct"] = (pp_toi_vals.values / df[toi_col].values)

    log.info("Season %d: %d qualifying forwards", season, len(out))
    return out.reset_index(drop=True)


def build_training_pairs(
    stats: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Year-N features → Year-(N+1) goals/60 rate."""
    seasons = sorted(stats["season"].unique())
    all_X, all_y = [], []

    for i in range(len(seasons) - 1):
        yr_n, yr_np1 = seasons[i], seasons[i + 1]
        n   = stats[stats["season"] == yr_n].set_index("player_id")
        np1 = stats[stats["season"] == yr_np1].set_index("player_id")
        common = n.index.intersection(np1.index)

        if common.empty:
            log.warning("No common players %d→%d", yr_n, yr_np1)
            continue

        avail   = [f for f in features if f in n.columns]
        X_block = n.loc[common, avail].copy()
        y_block = np1.loc[common, "goals_per_60"].copy()
        valid   = X_block.notna().any(axis=1) & y_block.notna() & (y_block > 0)

        all_X.append(X_block[valid])
        all_y.append(y_block[valid])
        log.info("Pair %d→%d: %d examples", yr_n, yr_np1, valid.sum())

    if not all_X:
        raise RuntimeError("No valid training pairs.")

    return (
        pd.concat(all_X).reset_index(drop=True),
        pd.concat(all_y).reset_index(drop=True),
    )


def train(years: list[int] | None = None) -> dict:
    """Full training pipeline."""
    import xgboost as xgb

    if years is None:
        years = [2021, 2022, 2023, 2024]

    MODEL_DIR.mkdir(exist_ok=True)

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

    available = [f for f in FEATURES if f in stats.columns
                 and stats[f].notna().sum() >= 20]
    if len(available) < 2:
        raise RuntimeError(f"Too few features: {available}")
    log.info("Training on %d features: %s", len(available), available)

    X, y = build_training_pairs(stats, available)
    log.info("Training set: %d × %d", len(X), X.shape[1])

    medians = X.median()
    X = X.fillna(medians)

    n_splits = min(5, max(2, len(X) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
        p = np.clip(m.predict(X.iloc[val]), 0, 4.0)
        oof[val] = p
        mae = mean_absolute_error(y.iloc[val], p)
        maes.append(mae)
        log.info("  Fold %d MAE: %.4f goals/60", fold, mae)

    cv_mae = float(np.mean(maes))
    cv_r2  = float(r2_score(y, oof))
    log.info("CV MAE: %.4f | CV R²: %.4f", cv_mae, cv_r2)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)
    model.save_model(str(MODEL_PATH))

    importance = dict(sorted(
        zip(available, model.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    ))

    metadata = {
        "features":           available,
        "feature_medians":    medians[available].to_dict(),
        "lg_goals_per_60":    float(y.mean()),
        "training_years":     years,
        "n_training":         len(X),
        "cv_mae":             cv_mae,
        "cv_r2":              cv_r2,
        "feature_importance": importance,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    log.info("=" * 50)
    log.info("NHL Training complete.")
    log.info("  Samples     : %d", len(X))
    log.info("  CV MAE      : %.4f goals/60", cv_mae)
    log.info("  CV R²       : %.4f", cv_r2)
    log.info("  LG goals/60 : %.4f", float(y.mean()))
    log.info("  Top features: %s", list(importance.keys())[:4])
    log.info("=" * 50)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
