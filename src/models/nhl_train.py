"""
nhl_train.py
============
Annual retraining for the XGBoost NHL goal-rate talent estimator.

Data source: Moneypuck free CSV downloads (moneypuck.com)
Training: Year-N features → Year-(N+1) goals/60 (no look-ahead bias)
Retrain: Once per year after season ends (May 1st cron)
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

MIN_TOI_SECONDS = 300 * 60    # 300 minutes qualifying threshold
HEADERS = {"User-Agent": "nhl-goal-predictor/1.0 (github-actions; open-source)"}

SKATERS_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary"
    "/{season}/regular/skaters.csv"
)

FEATURES = [
    "xg_per_60",
    "shots_per_60",
    "shooting_pct",
    "hd_xg_per_60",
    "hd_shooting_pct",
    "pp_toi_pct",
    "corsi_pct",
    "xg_pct",
    "toi_per_game",
]


# ---------------------------------------------------------------------------
# Safe column accessor — the core fix.
# Moneypuck column names vary slightly across seasons and versions.
# e.g. goals may be "I_F_goals" or "goals"; xG may be "I_F_xGoals" or "xGoals".
# Always returns a float Series, never a scalar.
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    """Return first matching column as a numeric float Series, or a default Series."""
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
    """
    Build canonical feature frame from a raw Moneypuck CSV.
    Uses _col() for all column access so missing columns never crash.
    """
    # All-situation rows
    df = df_raw[df_raw["situation"] == "all"].copy()

    # Forwards only
    if "position" in df.columns:
        df = df[df["position"].isin(["C", "L", "R"])].copy()

    # TOI filter
    toi_col = next((c for c in ["icetime", "toi", "timeOnIce"] if c in df.columns), None)
    if toi_col is None:
        log.warning("Season %d: no TOI column found. Skipping.", season)
        return pd.DataFrame()

    df = df[pd.to_numeric(df[toi_col], errors="coerce").fillna(0) >= MIN_TOI_SECONDS].copy()
    if df.empty:
        log.warning("Season %d: no players passed TOI filter.", season)
        return pd.DataFrame()

    toi_sec = _col(df, "icetime", "toi", "timeOnIce")
    toi_hr  = toi_sec / 3600
    toi_min = toi_sec / 60
    gp      = _col(df, "games_played", "gamesPlayed", default=1.0)
    gp      = gp.replace(0, 1)   # avoid division by zero

    # Goals — try all known Moneypuck naming conventions
    goals  = _col(df, "I_F_goals", "goals", "Goals")
    xg     = _col(df, "I_F_xGoals", "xGoals", "I_F_expectedGoals")
    shots  = _col(df, "I_F_shotsOnGoal", "shotsOnGoal", "I_F_shots")
    hd_sh  = _col(df, "I_F_highDangerShots", "highDangerShots")
    hd_xg  = _col(df, "I_F_highDangerxGoals", "highDangerxGoals", "I_F_highDangerExpectedGoals")
    corsi  = _col(df, "onIce_corsiPercentage", "corsiPercentage", default=float("nan"))
    xg_pct = _col(df, "onIce_xGoalsPercentage", "xGoalsPercentage", default=float("nan"))

    # Derived per-60 rates
    toi_hr_safe = toi_hr.replace(0, np.nan)
    goals_60    = goals / toi_hr_safe
    xg_60       = xg    / toi_hr_safe
    shots_60    = shots / toi_hr_safe
    hd_xg_60    = hd_xg / toi_hr_safe
    sh_pct      = goals.where(shots >= 10, np.nan) / shots.replace(0, np.nan)
    hd_sh_pct   = goals.where(hd_sh >= 5, np.nan) / hd_sh.replace(0, np.nan)

    # PP TOI fraction from power play situation rows
    pp_raw = df_raw[
        (df_raw["situation"] == "5on4") &
        df_raw["playerId"].isin(df["playerId"].values)
    ].copy()

    pp_toi_series = pd.Series(0.0, index=df.index, dtype=float)
    if not pp_raw.empty and toi_col in pp_raw.columns:
        pp_map = (
            pd.to_numeric(pp_raw[toi_col], errors="coerce")
            .fillna(0)
            .groupby(pp_raw["playerId"])
            .sum()
        )
        pp_mapped = df["playerId"].map(pp_map).fillna(0)
        pp_toi_series = pp_mapped / toi_sec.replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["player_id"]       = df["playerId"].values
    out["season"]          = season
    out["games_played"]    = gp.values
    out["season_toi_sec"]  = toi_sec.values
    out["goals"]           = goals.values
    out["goals_per_60"]    = goals_60.values           # TARGET
    out["xg_per_60"]       = xg_60.values
    out["shots_per_60"]    = shots_60.values
    out["shooting_pct"]    = sh_pct.values
    out["hd_xg_per_60"]    = hd_xg_60.values
    out["hd_shooting_pct"] = hd_sh_pct.values
    out["corsi_pct"]       = corsi.values
    out["xg_pct"]          = xg_pct.values
    out["toi_per_game"]    = (toi_min / gp).values
    out["pp_toi_pct"]      = pp_toi_series.values

    out = out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    log.info("Season %d: %d qualifying forwards assembled.", season, len(out))
    return out


def build_training_pairs(
    stats: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Year-N features → Year-(N+1) goals/60."""
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

    available = [
        f for f in FEATURES
        if f in stats.columns and stats[f].notna().sum() >= 20
    ]
    if len(available) < 2:
        raise RuntimeError(f"Too few usable features: {available}")
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
