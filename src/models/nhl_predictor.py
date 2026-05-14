"""
nhl_predictor.py
================
Ensemble goal probability model — same two-path architecture as MLB HR predictor.

Path 1 — XGBoost talent model (when trained model exists)
  Predicts goals/60 rate from Moneypuck xG, shot quality, PP time features.

Path 2 — Statistical baseline (always runs)
  Bayesian-blended goals/60 from NHL API season + career stats.

Game-level adjustments (applied to whichever base rate is used):
  goalie quality, power play time, home ice, back-to-back fatigue, recent form.

Final: P(≥1 goal) via Poisson distribution.
"""
from __future__ import annotations
import logging, math
import numpy as np
from src.features.nhl_engineer import (
    LG_GOALS_PER_60, goals_per_60_from_stats, goalie_factor,
    recent_form_factor, pp_adjustment, home_factor,
    back_to_back_factor, poisson_goal_probability,
)

log = logging.getLogger(__name__)


def ensemble_predict(
    season_stats: dict,
    career_stats: dict,
    mp_metrics: dict,
    goalie_metrics: dict,
    recent_games: list[dict],
    is_home: bool,
    is_back_to_back: bool,
    estimated_toi_min: float,
    config: dict,
    xgb_model=None,
    xgb_meta: dict | None = None,
) -> dict:
    """
    Full ensemble prediction for one skater in tonight's game.

    Returns dict with goal_probability, goal_pct, confidence_tier, factors.
    """
    # ── Base goals/60 rate ───────────────────────────────────────────────────
    stat_rate, season_toi_min = goals_per_60_from_stats(season_stats, career_stats)

    xgb_rate = None
    if xgb_model is not None and xgb_meta is not None:
        try:
            from src.models.nhl_model_registry import predict_goals_per_60
            xgb_rate = predict_goals_per_60(mp_metrics, xgb_model, xgb_meta)
        except Exception as exc:
            log.debug("XGBoost prediction failed: %s", exc)

    if xgb_rate is not None:
        base_rate   = 0.65 * xgb_rate + 0.35 * stat_rate
        rate_source = "xgboost+statistical"
    else:
        base_rate   = stat_rate
        rate_source = "statistical"

    base_rate = float(np.clip(base_rate, 0.001, 5.0))

    # ── Game-level adjustments ───────────────────────────────────────────────
    g_factor   = goalie_factor(goalie_metrics, config)
    pp_f       = pp_adjustment(float(mp_metrics.get("pp_toi_pct") or LG_GOALS_PER_60))
    home_f     = home_factor(is_home)
    b2b_f      = back_to_back_factor(is_back_to_back)
    recent_f   = recent_form_factor(recent_games, config.get("model", {}).get("recent_form_games", 10))

    # ── Poisson probability ──────────────────────────────────────────────────
    prob, lam = poisson_goal_probability(
        goals_per_60=base_rate,
        toi_minutes=estimated_toi_min,
        goalie_f=g_factor,
        pp_f=pp_f,
        home_f=home_f,
        b2b_f=b2b_f,
        recent_f=recent_f,
    )

    # ── Confidence tier ──────────────────────────────────────────────────────
    games_played = int(season_stats.get("gamesPlayed", 0))
    has_mp       = float(mp_metrics.get("xg_per_60") or 0) > 0

    if xgb_rate is not None and games_played >= 20 and has_mp:
        tier = "High"
    elif games_played >= 10 or has_mp:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "goal_probability":  round(prob, 4),
        "goal_pct":          f"{prob * 100:.1f}%",
        "confidence_tier":   tier,
        "rate_source":       rate_source,
        "factors": {
            "base_goals_per_60": round(base_rate, 4),
            "stat_rate":         round(stat_rate, 4),
            "xgb_rate":          round(xgb_rate, 4) if xgb_rate is not None else None,
            "goalie_factor":     round(g_factor, 3),
            "pp_factor":         round(pp_f, 3),
            "home_factor":       round(home_f, 3),
            "b2b_factor":        round(b2b_f, 3),
            "recent_form":       round(recent_f, 3),
            "estimated_toi_min": round(estimated_toi_min, 1),
            "lambda":            lam,
        },
        "mp_metrics": {
            k: v for k, v in mp_metrics.items()
            if k in ("xg_per_60", "shooting_pct", "pp_toi_pct", "hd_xg_per_60", "toi_per_game")
        },
    }
