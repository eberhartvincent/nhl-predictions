"""
moneypuck_client.py
===================
Fetch advanced NHL stats from Moneypuck (moneypuck.com).
Free, no API key, works from GitHub Actions.
Moneypuck is the closest NHL equivalent to Baseball Savant —
it provides xG, Corsi, high-danger shooting %, GSAx, and more.

Season format: 2023 = 2023-24 season (year the season started).

Key columns in skaters CSV:
  playerId, name, team, position, situation, season, games_played,
  icetime (seconds), goals, I_F_xGoals (individual xG),
  I_F_shotsOnGoal, I_F_highDangerShots, I_F_highDangerxGoals,
  onIce_corsiPercentage, onIce_xGoalsPercentage

situation filter:
  "all"  — all situations
  "5on5" — even strength only
  "5on4" — power play (skater has 5, opponent has 4)
  "4on5" — penalty kill
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "nhl-goal-predictor/1.0 (github-actions; open-source)"}

SKATERS_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary"
    "/{season}/regular/skaters.csv"
)
GOALIES_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary"
    "/{season}/regular/goalies.csv"
)

# Minimum ice time seconds to include a player
MIN_TOI_SECONDS = 300 * 60   # 300 minutes


@lru_cache(maxsize=16)
def _fetch_skaters(season: int) -> pd.DataFrame | None:
    url = SKATERS_URL.format(season=season)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Moneypuck skaters/%d: %d rows", season, len(df))
        return df
    except Exception as exc:
        log.warning("Moneypuck skaters/%d failed: %s", season, exc)
        return None


@lru_cache(maxsize=16)
def _fetch_goalies(season: int) -> pd.DataFrame | None:
    url = GOALIES_URL.format(season=season)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        log.info("Moneypuck goalies/%d: %d rows", season, len(df))
        return df
    except Exception as exc:
        log.warning("Moneypuck goalies/%d failed: %s", season, exc)
        return None


def get_skater_metrics(player_id: int, season: int) -> dict:
    """
    Return advanced metrics for a skater from Moneypuck.
    Combines all-situation and power-play splits.
    """
    df = _fetch_skaters(season)
    empty = {
        "goals_per_60":      np.nan,
        "xg_per_60":         np.nan,
        "shots_per_60":      np.nan,
        "shooting_pct":      np.nan,
        "hd_shooting_pct":   np.nan,
        "hd_xg_per_60":      np.nan,
        "pp_toi_pct":        np.nan,
        "corsi_pct":         np.nan,
        "xg_pct":            np.nan,
        "toi_per_game":      np.nan,
        "games_played":      0,
        "season_toi_sec":    0,
    }

    if df is None or df.empty:
        return empty

    # All-situation row
    all_sit = df[
        (df["playerId"] == player_id) &
        (df["situation"] == "all")
    ]
    # Power play row
    pp_sit = df[
        (df["playerId"] == player_id) &
        (df["situation"] == "5on4")
    ]

    if all_sit.empty:
        return empty

    row = all_sit.iloc[0]
    toi_sec  = float(row.get("icetime", 0) or 0)
    toi_hr   = toi_sec / 3600
    games    = int(row.get("games_played", 0) or 0)

    if toi_sec < 60 or games == 0:
        return empty

    goals     = float(row.get("goals", 0) or 0)
    xg        = float(row.get("I_F_xGoals", 0) or 0)
    shots     = float(row.get("I_F_shotsOnGoal", 0) or 0)
    hd_shots  = float(row.get("I_F_highDangerShots", 0) or 0)
    hd_xg     = float(row.get("I_F_highDangerxGoals", 0) or 0)

    goals_60  = goals / toi_hr if toi_hr > 0 else np.nan
    xg_60     = xg   / toi_hr if toi_hr > 0 else np.nan
    shots_60  = shots / toi_hr if toi_hr > 0 else np.nan
    sh_pct    = goals / shots if shots >= 10 else np.nan
    hd_sh_pct = goals / hd_shots if hd_shots >= 5 else np.nan
    hd_xg_60  = hd_xg / toi_hr if toi_hr > 0 else np.nan
    corsi     = float(row.get("onIce_corsiPercentage", np.nan) or np.nan)
    xg_pct    = float(row.get("onIce_xGoalsPercentage", np.nan) or np.nan)

    # PP TOI as fraction of total TOI
    pp_toi_pct = np.nan
    if not pp_sit.empty:
        pp_toi = float(pp_sit.iloc[0].get("icetime", 0) or 0)
        pp_toi_pct = pp_toi / toi_sec if toi_sec > 0 else np.nan

    return {
        "goals_per_60":    round(float(goals_60), 4) if not np.isnan(goals_60) else np.nan,
        "xg_per_60":       round(float(xg_60), 4)   if not np.isnan(xg_60)   else np.nan,
        "shots_per_60":    round(float(shots_60), 4) if not np.isnan(shots_60) else np.nan,
        "shooting_pct":    round(float(sh_pct), 4)   if not np.isnan(sh_pct)  else np.nan,
        "hd_shooting_pct": round(float(hd_sh_pct), 4) if not np.isnan(hd_sh_pct) else np.nan,
        "hd_xg_per_60":    round(float(hd_xg_60), 4) if not np.isnan(hd_xg_60) else np.nan,
        "pp_toi_pct":      round(float(pp_toi_pct), 4) if not np.isnan(pp_toi_pct) else np.nan,
        "corsi_pct":       round(float(corsi), 4)    if not np.isnan(corsi)   else np.nan,
        "xg_pct":          round(float(xg_pct), 4)   if not np.isnan(xg_pct)  else np.nan,
        "toi_per_game":    round(toi_sec / 60 / games, 1) if games > 0 else np.nan,
        "games_played":    games,
        "season_toi_sec":  int(toi_sec),
    }


def get_goalie_metrics(player_id: int, season: int) -> dict:
    """
    Return save quality metrics for a goalie.
    GSAx (Goals Saved Above Expected) is the key signal.
    """
    df = _fetch_goalies(season)
    empty = {
        "save_pct":    np.nan,
        "gsax_per_60": np.nan,
        "xg_allowed_per_60": np.nan,
        "games_started": 0,
    }

    if df is None or df.empty:
        return empty

    # Use all-situation or 5on5 — all gives more data
    rows = df[
        (df["playerId"] == player_id) &
        (df["situation"] == "all")
    ]
    if rows.empty:
        return empty

    row      = rows.iloc[0]
    toi_sec  = float(row.get("icetime", 0) or 0)
    toi_hr   = toi_sec / 3600
    games    = int(row.get("games_played", 0) or 0)

    if toi_sec < 60:
        return empty

    xg_against  = float(row.get("xGoals", 0) or 0)          # xG against
    goals_ag    = float(row.get("goals", 0) or 0)            # actual goals against
    shots_ag    = float(row.get("shotsOnGoalAgainst", 0) or 0)
    gsax        = xg_against - goals_ag                       # positive = above average

    sv_pct      = 1 - (goals_ag / shots_ag) if shots_ag > 0 else np.nan
    gsax_60     = gsax / toi_hr if toi_hr > 0 else np.nan
    xg_ag_60    = xg_against / toi_hr if toi_hr > 0 else np.nan

    return {
        "save_pct":          round(float(sv_pct), 4)    if not np.isnan(sv_pct)   else np.nan,
        "gsax_per_60":       round(float(gsax_60), 4)   if not np.isnan(gsax_60)  else np.nan,
        "xg_allowed_per_60": round(float(xg_ag_60), 4) if not np.isnan(xg_ag_60) else np.nan,
        "games_started":     games,
    }


def get_all_skaters_season(season: int) -> pd.DataFrame:
    """Return full skaters DataFrame for a season (all situations)."""
    df = _fetch_skaters(season)
    if df is None:
        return pd.DataFrame()
    return df[df["situation"] == "all"].copy()


def get_all_goalies_season(season: int) -> pd.DataFrame:
    """Return full goalies DataFrame for a season."""
    df = _fetch_goalies(season)
    if df is None:
        return pd.DataFrame()
    return df[df["situation"] == "all"].copy()
