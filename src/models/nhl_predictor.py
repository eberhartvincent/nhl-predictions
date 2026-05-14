"""
nhl_predictor.py
================
Returns three predictions per player: goals, points, shots.
Each uses the corresponding XGBoost model blended with the statistical baseline.
"""
from __future__ import annotations
import logging
import numpy as np
from src.features.nhl_engineer import (
    LG_GOALS_PER_60, LG_POINTS_PER_60, LG_SHOTS_PER_60,
    goals_per_60_from_stats, points_per_60_from_stats, shots_per_60_from_stats,
    goalie_factor, recent_form_factor, pp_adjustment,
    home_factor, back_to_back_factor,
    poisson_prob_at_least_one, poisson_prob_over_line,
)

log = logging.getLogger(__name__)


def _blend(xgb_rate, stat_rate):
    if xgb_rate is not None:
        return 0.65 * xgb_rate + 0.35 * stat_rate, "xgboost+statistical"
    return stat_rate, "statistical"


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
    models: dict | None = None,
    metas:  dict | None = None,
) -> dict:
    """
    Returns a dict with three independent probability predictions:
      goals_probability, points_probability, shots_probability
    plus their display strings and all factor breakdowns.
    """
    from src.models.nhl_model_registry import predict_rate

    models = models or {}
    metas  = metas  or {}
    shots_line = float(config.get("prediction", {}).get("shots_line", 2.5))

    # ── Shared game-level factors ─────────────────────────────────────────
    g_factor = goalie_factor(goalie_metrics, config)
    home_f   = home_factor(is_home)
    b2b_f    = back_to_back_factor(is_back_to_back)
    pp_f     = pp_adjustment(float(mp_metrics.get("pp_toi_pct") or 0))

    # ── Goals ─────────────────────────────────────────────────────────────
    stat_g, season_toi  = goals_per_60_from_stats(season_stats, career_stats)
    xgb_g  = predict_rate(mp_metrics, "goals",  models.get("goals"),  metas.get("goals",  {}))
    base_g, src_g = _blend(xgb_g, stat_g)
    recent_g = recent_form_factor(recent_games, "goals", prior=LG_GOALS_PER_60)
    goal_prob, goal_lam = poisson_prob_at_least_one(
        base_g, estimated_toi_min, g_factor, home_f, b2b_f, pp_f, recent_g
    )

    # ── Points ────────────────────────────────────────────────────────────
    stat_p, _ = points_per_60_from_stats(season_stats, career_stats)
    xgb_p     = predict_rate(mp_metrics, "points", models.get("points"), metas.get("points", {}))
    base_p, src_p = _blend(xgb_p, stat_p)
    recent_p = recent_form_factor(recent_games, "points", prior=LG_POINTS_PER_60)
    # Goalie factor matters less for points (assists on goals vs any goals)
    point_prob, point_lam = poisson_prob_at_least_one(
        base_p, estimated_toi_min, g_factor * 0.5 + 0.5, home_f, b2b_f, pp_f, recent_p
    )

    # ── Shots ─────────────────────────────────────────────────────────────
    stat_s, _ = shots_per_60_from_stats(season_stats, career_stats)
    xgb_s     = predict_rate(mp_metrics, "shots", models.get("shots"), metas.get("shots", {}))
    base_s, src_s = _blend(xgb_s, stat_s)
    recent_s  = recent_form_factor(recent_games, "shots", prior=LG_SHOTS_PER_60)
    # Goalie doesn't affect shot volume, only home/b2b/pp
    shot_prob, shot_lam = poisson_prob_over_line(
        base_s, estimated_toi_min, shots_line, home_f, b2b_f, pp_f, recent_s
    )

    # ── Confidence ────────────────────────────────────────────────────────
    gp      = int(season_stats.get("gamesPlayed", 0))
    has_mp  = float(mp_metrics.get("xg_per_60") or 0) > 0
    has_xgb = any(models.get(n) is not None for n in ("goals", "points", "shots"))

    if has_xgb and gp >= 20 and has_mp:
        tier = "High"
    elif gp >= 10 or has_mp:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        # ── Goals ─────────────────────────────────────────────────────────
        "goal_probability":  round(goal_prob, 4),
        "goal_pct":          f"{goal_prob * 100:.1f}%",
        "goal_lambda":       goal_lam,
        "goal_rate_source":  src_g,

        # ── Points ────────────────────────────────────────────────────────
        "point_probability": round(point_prob, 4),
        "point_pct":         f"{point_prob * 100:.1f}%",
        "point_lambda":      point_lam,
        "point_rate_source": src_p,

        # ── Shots ─────────────────────────────────────────────────────────
        "shot_probability":  round(shot_prob, 4),
        "shot_pct":          f"{shot_prob * 100:.1f}%",
        "shot_lambda":       shot_lam,           # = expected shots tonight
        "shot_line":         shots_line,
        "shot_rate_source":  src_s,
        "expected_shots":    round(shot_lam, 1),

        # ── Shared ────────────────────────────────────────────────────────
        "confidence_tier":   tier,
        "factors": {
            "goalie_factor":     round(g_factor, 3),
            "pp_factor":         round(pp_f, 3),
            "home_factor":       round(home_f, 3),
            "b2b_factor":        round(b2b_f, 3),
            "recent_goal_form":  round(recent_g, 3),
            "estimated_toi_min": round(estimated_toi_min, 1),
            "base_goals_per_60": round(float(base_g), 4),
            "base_points_per_60":round(float(base_p), 4),
            "base_shots_per_60": round(float(base_s), 4),
        },
        "mp_metrics": {
            k: v for k, v in mp_metrics.items()
            if k in ("xg_per_60", "shots_per_60", "shooting_pct",
                     "pp_toi_pct", "toi_per_game")
        },
    }
