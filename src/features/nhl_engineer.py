"""
nhl_engineer.py
===============
Feature engineering for goals, points, and shots predictions.
"""
from __future__ import annotations
import logging, math
import numpy as np

log = logging.getLogger(__name__)

# League averages (2022-25 NHL forwards)
LG_GOALS_PER_60   = 0.65
LG_POINTS_PER_60  = 1.70
LG_SHOTS_PER_60   = 8.50
LG_SAVE_PCT       = 0.906

HOME_BOOST           = 1.05
BACK_TO_BACK_PENALTY = 0.88
LG_PP_TOI_PCT        = 0.12
MIN_TOI_SAMPLE       = 300


def _safe(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def bayesian_blend_rate(
    season_rate: float, season_toi_min: float,
    career_rate: float, career_toi_min: float,
    prior: float, prior_weight_min: float = 200.0,
) -> float:
    blended_toi    = season_toi_min + prior_weight_min
    season_blended = (season_rate * season_toi_min + prior * prior_weight_min) / blended_toi
    if career_toi_min < MIN_TOI_SAMPLE:
        return season_blended
    career_w = min(career_toi_min / (career_toi_min + 1000), 0.35)
    return (1 - career_w) * season_blended + career_w * career_rate


def _base_rate_from_stats(
    season: dict, career: dict,
    stat_key: str, prior: float,
) -> tuple[float, float]:
    """Generic Bayesian-blended per-60 rate from NHL API stats."""
    s_val   = _safe(season.get(stat_key, 0))
    s_toi_s = _safe(season.get("timeOnIce", 0))
    c_val   = _safe(career.get(stat_key, 0))
    c_gp    = _safe(career.get("gamesPlayed", 0))

    s_toi_min = s_toi_s / 60
    s_rate    = s_val / (s_toi_s / 3600) if s_toi_s > 0 else prior
    c_toi_min = c_gp * 15.5
    c_rate    = c_val / (c_toi_min / 60) if c_toi_min > 0 else prior

    blended = bayesian_blend_rate(s_rate, s_toi_min, c_rate, c_toi_min, prior)
    return float(np.clip(blended, 0.0, prior * 8)), s_toi_min


def goals_per_60_from_stats(season: dict, career: dict) -> tuple[float, float]:
    return _base_rate_from_stats(season, career, "goals", LG_GOALS_PER_60)


def points_per_60_from_stats(season: dict, career: dict) -> tuple[float, float]:
    pts = _safe(season.get("points", 0))
    if pts == 0:
        goals   = _safe(season.get("goals", 0))
        assists = _safe(season.get("assists", 0))
        pts     = goals + assists
    toi_s = _safe(season.get("timeOnIce", 0))
    s_rate = pts / (toi_s / 3600) if toi_s > 0 else LG_POINTS_PER_60
    toi_min = toi_s / 60

    c_pts = _safe(career.get("points", 0))
    c_gp  = _safe(career.get("gamesPlayed", 0))
    c_toi = c_gp * 15.5
    c_rate = c_pts / (c_toi / 60) if c_toi > 0 else LG_POINTS_PER_60

    blended = bayesian_blend_rate(s_rate, toi_min, c_rate, c_toi, LG_POINTS_PER_60)
    return float(np.clip(blended, 0.0, LG_POINTS_PER_60 * 8)), toi_min


def shots_per_60_from_stats(season: dict, career: dict) -> tuple[float, float]:
    return _base_rate_from_stats(season, career, "shots", LG_SHOTS_PER_60)


def goalie_factor(goalie_metrics: dict, config: dict) -> float:
    lg_sv  = _safe(config.get("model", {}).get("lg_avg_save_pct", LG_SAVE_PCT))
    sv_pct = _safe(goalie_metrics.get("save_pct"), lg_sv)
    gsax   = _safe(goalie_metrics.get("gsax_per_60"), 0.0)
    games  = int(goalie_metrics.get("games_started", 0))
    if games < 5:
        return 1.0
    sv_factor  = (1 - sv_pct) / (1 - lg_sv)
    gsax_adj   = 1.0 - np.clip(gsax / 4.0, -0.15, 0.15)
    w = min(games / 30, 1.0)
    return float(np.clip((1 - w) * sv_factor + w * (sv_factor * gsax_adj), 0.60, 1.60))


def recent_form_factor(recent_games: list[dict], stat_key: str,
                        window: int = 10, prior: float = LG_GOALS_PER_60) -> float:
    games = recent_games[:window]
    if len(games) < 3:
        return 1.0
    total_stat = sum(_safe(g.get(stat_key, 0)) for g in games)
    total_toi  = sum(_safe(g.get("toi", g.get("timeOnIce", 900))) for g in games)
    toi_h = total_toi / 3600
    if toi_h < 0.5:
        return 1.0
    recent_rate = total_stat / toi_h
    shrunk = 1.0 + 0.25 * (recent_rate / prior - 1.0)
    return float(np.clip(shrunk, 0.75, 1.30))


def pp_adjustment(pp_toi_pct: float) -> float:
    if math.isnan(pp_toi_pct):
        return 1.0
    return float(np.clip(1.0 + (pp_toi_pct - LG_PP_TOI_PCT) * 2.0, 0.8, 1.5))


def home_factor(is_home: bool) -> float:
    return HOME_BOOST if is_home else 1.0


def back_to_back_factor(is_b2b: bool) -> float:
    return BACK_TO_BACK_PENALTY if is_b2b else 1.0


def poisson_prob_at_least_one(rate_per_60: float, toi_min: float,
                               *adjustments: float) -> tuple[float, float]:
    """P(≥1 event) via Poisson. Returns (probability, lambda)."""
    adj_rate = rate_per_60
    for f in adjustments:
        adj_rate *= f
    adj_rate = float(np.clip(adj_rate, 0.001, 20.0))
    lam  = adj_rate * (toi_min / 60)
    prob = 1.0 - math.exp(-lam)
    return float(np.clip(prob, 0, 0.999)), round(lam, 4)


def poisson_prob_over_line(rate_per_60: float, toi_min: float,
                            line: float, *adjustments: float) -> tuple[float, float]:
    """
    P(shots > line) e.g. P(shots > 2.5) = P(shots ≥ 3).
    Returns (probability, expected_shots).
    """
    adj_rate = rate_per_60
    for f in adjustments:
        adj_rate *= f
    adj_rate = float(np.clip(adj_rate, 0.001, 20.0))
    lam  = adj_rate * (toi_min / 60)
    k    = math.ceil(line)   # first integer above the line
    # P(X >= k) = 1 - sum_{i=0}^{k-1} e^-lam * lam^i / i!
    cumulative = sum(
        math.exp(-lam) * (lam ** i) / math.factorial(i)
        for i in range(k)
    )
    prob = max(0.0, 1.0 - cumulative)
    return float(np.clip(prob, 0, 0.999)), round(lam, 4)
