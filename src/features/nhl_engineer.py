"""
nhl_engineer.py
===============
Feature engineering and scoring for NHL goal probability predictions.

Key differences from baseball
------------------------------
* Ice time replaces plate appearances — goals/60 is the base rate
* Power play time is the single biggest predictor of goal volume
* Opposing goalie quality replaces pitcher quality
* No weather/park factors — all indoor arenas
* Back-to-back games carry a meaningful fatigue penalty
* Home ice advantage is real but smaller than in baseball (~5%)
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

# League average constants (2022-25 NHL forwards)
LG_GOALS_PER_60    = 0.65     # forward average goals/60
LG_XG_PER_60       = 0.60     # forward average xG/60
LG_SAVE_PCT        = 0.906    # league average goalie save %
LG_SHOTS_PER_60    = 8.0      # forward average shots on goal/60
LG_PP_TOI_PCT      = 0.12     # average PP TOI as fraction of total TOI

# Adjustments
HOME_BOOST          = 1.05    # 5% goal rate boost at home
BACK_TO_BACK_PENALTY = 0.88   # 12% reduction on second night
MIN_TOI_SAMPLE      = 300     # minutes — minimum for reliable rates


def _safe(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def bayesian_blend_rate(
    season_rate: float,
    season_toi_min: float,
    career_rate: float,
    career_toi_min: float,
    prior: float,
    prior_weight_min: float = 200.0,
) -> float:
    """
    Bayesian shrinkage of goals/60 rate toward prior (league average).
    Uses ice time minutes as the sample size weight.
    """
    blended_toi = season_toi_min + prior_weight_min
    season_blended = (
        season_rate * season_toi_min + prior * prior_weight_min
    ) / blended_toi

    if career_toi_min < MIN_TOI_SAMPLE:
        return season_blended

    career_w = min(career_toi_min / (career_toi_min + 1000), 0.35)
    return (1 - career_w) * season_blended + career_w * career_rate


def goals_per_60_from_stats(season: dict, career: dict) -> tuple[float, float]:
    """
    Compute blended goals/60 from NHL API season + career stats.
    Returns (blended_rate, season_toi_minutes).
    """
    s_goals = _safe(season.get("goals", 0))
    s_toi_s = _safe(season.get("timeOnIce", 0))     # seconds
    c_goals = _safe(career.get("goals", 0))
    c_gp    = _safe(career.get("gamesPlayed", 0))

    s_toi_min = s_toi_s / 60
    s_toi_hr  = s_toi_s / 3600
    s_rate    = s_goals / s_toi_hr if s_toi_hr > 0 else LG_GOALS_PER_60

    # Career: estimate TOI from games played × avg TOI
    c_toi_min = c_gp * 15.5   # rough average
    c_toi_hr  = c_toi_min / 60
    c_rate    = c_goals / c_toi_hr if c_toi_hr > 0 else LG_GOALS_PER_60

    blended = bayesian_blend_rate(
        season_rate=s_rate,
        season_toi_min=s_toi_min,
        career_rate=c_rate,
        career_toi_min=c_toi_min,
        prior=LG_GOALS_PER_60,
        prior_weight_min=200,
    )
    return float(np.clip(blended, 0.0, 5.0)), s_toi_min


def goalie_factor(goalie_metrics: dict, config: dict) -> float:
    """
    Multiplicative factor for opposing goalie quality.
    1.0 = league average goalie.
    >1.0 = weak goalie (easier to score on).
    <1.0 = elite goalie (harder to score on).

    Uses save % relative to league average as primary signal,
    blended with GSAx/60 as a quality-of-competition adjustment.
    """
    lg_sv   = _safe(config.get("model", {}).get("lg_avg_save_pct", LG_SAVE_PCT))
    sv_pct  = _safe(goalie_metrics.get("save_pct"), lg_sv)
    gsax_60 = _safe(goalie_metrics.get("gsax_per_60"), 0.0)
    games   = int(goalie_metrics.get("games_started", 0))

    if games < 5:
        return 1.0   # insufficient sample

    # Save % factor: lower save % = more goals allowed = better for scorers
    sv_factor = (1 - sv_pct) / (1 - lg_sv)

    # GSAx/60 adjustment: positive GSAx = goalie is better than expected
    # Scale: ~0.5 GSAx/60 = elite, -0.5 = weak
    gsax_adj = 1.0 - np.clip(gsax_60 / 4.0, -0.15, 0.15)

    # Blend by sample size
    w = min(games / 30, 1.0)
    factor = (1 - w) * sv_factor + w * (sv_factor * gsax_adj)
    return float(np.clip(factor, 0.60, 1.60))


def recent_form_factor(recent_games: list[dict], window: int = 10) -> float:
    """
    Compare recent goals/game to expected.
    Returns a multiplicative factor: 1.0 = on pace.
    Shrunk heavily toward 1.0 — hockey is streaky but noisy.
    """
    games = recent_games[:window]
    if len(games) < 3:
        return 1.0

    total_goals = sum(_safe(g.get("goals", 0)) for g in games)
    total_toi_s = sum(_safe(g.get("toi", g.get("timeOnIce", 15 * 60))) for g in games)
    total_toi_h = total_toi_s / 3600

    if total_toi_h < 1:
        return 1.0

    recent_rate = total_goals / total_toi_h
    factor = recent_rate / LG_GOALS_PER_60
    # Shrink 75% toward 1.0 — small sample
    shrunk = 1.0 + 0.25 * (factor - 1.0)
    return float(np.clip(shrunk, 0.75, 1.30))


def pp_adjustment(pp_toi_pct: float) -> float:
    """
    Power play ice time share boosts goal probability.
    A player with 20% of TOI on the PP scores ~40% more than expected.
    Returns multiplicative factor.
    """
    if math.isnan(pp_toi_pct):
        return 1.0
    excess = pp_toi_pct - LG_PP_TOI_PCT
    # Each 1% extra PP TOI ≈ 2% goal rate boost (empirical)
    return float(np.clip(1.0 + excess * 2.0, 0.8, 1.5))


def home_factor(is_home: bool) -> float:
    return HOME_BOOST if is_home else 1.0


def back_to_back_factor(is_b2b: bool) -> float:
    return BACK_TO_BACK_PENALTY if is_b2b else 1.0


def poisson_goal_probability(
    goals_per_60: float,
    toi_minutes: float,
    goalie_f: float,
    pp_f: float,
    home_f: float,
    b2b_f: float,
    recent_f: float,
) -> tuple[float, float]:
    """
    P(≥1 goal) using Poisson distribution.
    λ = adjusted_goals_per_60 × (toi_minutes / 60)

    Returns (probability, lambda).
    """
    adj_rate = goals_per_60 * goalie_f * pp_f * home_f * b2b_f * recent_f
    adj_rate = float(np.clip(adj_rate, 0.001, 5.0))
    toi_hr   = toi_minutes / 60
    lam      = adj_rate * toi_hr
    prob     = 1.0 - math.exp(-lam)
    return float(np.clip(prob, 0, 0.999)), round(lam, 4)
