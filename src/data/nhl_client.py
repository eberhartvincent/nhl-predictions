"""
nhl_client.py
=============
Wrapper around the NHL Stats API v1 (api-web.nhle.com).
No API key required. Free and official.

Note: The NHL migrated from statsapi.web.nhl.com to api-web.nhle.com
around 2023. This client uses the new v1 API exclusively.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE    = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "nhl-goal-predictor/1.0 (github-actions)"}

# Maps NHL team abbreviations to full names (for display)
TEAM_NAMES = {
    "ANA":"Anaheim Ducks","BOS":"Boston Bruins","BUF":"Buffalo Sabres",
    "CGY":"Calgary Flames","CAR":"Carolina Hurricanes","CHI":"Chicago Blackhawks",
    "COL":"Colorado Avalanche","CBJ":"Columbus Blue Jackets","DAL":"Dallas Stars",
    "DET":"Detroit Red Wings","EDM":"Edmonton Oilers","FLA":"Florida Panthers",
    "LAK":"Los Angeles Kings","MIN":"Minnesota Wild","MTL":"Montreal Canadiens",
    "NSH":"Nashville Predators","NJD":"New Jersey Devils","NYI":"New York Islanders",
    "NYR":"New York Rangers","OTT":"Ottawa Senators","PHI":"Philadelphia Flyers",
    "PIT":"Pittsburgh Penguins","SEA":"Seattle Kraken","SJS":"San Jose Sharks",
    "STL":"St. Louis Blues","TBL":"Tampa Bay Lightning","TOR":"Toronto Maple Leafs",
    "UTA":"Utah Hockey Club","VAN":"Vancouver Canucks","VGK":"Vegas Golden Knights",
    "WSH":"Washington Capitals","WPG":"Winnipeg Jets",
}


def _get(path: str, params: dict | None = None) -> dict:
    url  = f"{BASE}{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(0.1)
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_schedule(game_date: date) -> list[dict]:
    """Return list of game dicts for a given date."""
    ds   = game_date.strftime("%Y-%m-%d")
    data = _get(f"/schedule/{ds}")
    games = []
    for week in data.get("gameWeek", []):
        if week.get("date") == ds:
            games = week.get("games", [])
            break
    log.info("Found %d games on %s", len(games), ds)
    return games


def extract_matchups(games: list[dict], skip_final: bool = False) -> list[dict]:
    """Convert raw game dicts to structured matchup dicts."""
    matchups = []
    skipped  = {}

    for g in games:
        state      = g.get("gameState", "")
        game_type  = g.get("gameType", 2)

        # Skip playoffs if desired, skip non-regular season exhibition
        if game_type not in (2, 3):   # 2=regular, 3=playoffs
            continue

        # Final states in the new NHL API
        is_final = state in ("OFF", "FINAL", "CRIT")

        if skip_final and is_final:
            skipped[state] = skipped.get(state, 0) + 1
            continue

        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        home_abbrev = home.get("abbrev", "")
        away_abbrev = away.get("abbrev", "")

        matchups.append({
            "gameId":          g.get("id"),
            "home_abbrev":     home_abbrev,
            "away_abbrev":     away_abbrev,
            "home_team":       TEAM_NAMES.get(home_abbrev, home_abbrev),
            "away_team":       TEAM_NAMES.get(away_abbrev, away_abbrev),
            "home_score":      home.get("score"),
            "away_score":      away.get("score"),
            "game_state":      state,
            "is_final":        is_final,
            "venue":           g.get("venue", {}).get("default", ""),
            "start_time_utc":  g.get("startTimeUTC", ""),
        })

    if skipped:
        log.info("Skipped %d game(s): %s", sum(skipped.values()),
                 ", ".join(f"{v}× {k}" for k, v in skipped.items()))
    log.info("Actionable matchups: %d", len(matchups))
    return matchups


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_roster(team_abbrev: str) -> list[dict]:
    """Return current active roster for a team."""
    try:
        data = _get(f"/roster/{team_abbrev}/current")
    except Exception as exc:
        log.warning("Roster fetch failed for %s: %s", team_abbrev, exc)
        return []

    players = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            pos = "G" if group == "goalies" else (
                "D" if group == "defensemen" else p.get("positionCode", "C")
            )
            players.append({
                "id":         p.get("id"),
                "fullName":   f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}".strip(),
                "position":   pos,
                "sweaterNo":  p.get("sweaterNumber"),
            })
    return players


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_player_info(player_id: int) -> dict:
    """Return player metadata and current season stats."""
    try:
        data = _get(f"/player/{player_id}/landing")
    except Exception as exc:
        log.warning("Player info failed %d: %s", player_id, exc)
        return {}

    bio = data.get("featuredStats", {})
    season_stats = {}
    career_stats = {}

    for block in data.get("seasonTotals", []):
        if block.get("gameTypeId") != 2:
            continue
        if block.get("leagueAbbrev") != "NHL":
            continue
        career_stats = block   # last entry = career totals available

    # Current season
    fs = data.get("featuredStats", {}).get("regularSeason", {}).get("subSeason", {})
    season_stats = fs

    return {
        "id":           player_id,
        "fullName":     data.get("firstName", {}).get("default", "") + " " +
                        data.get("lastName", {}).get("default", ""),
        "position":     data.get("position", ""),
        "shoots":       data.get("shootsCatches", "R"),
        "team":         data.get("currentTeamAbbrev", ""),
        "season_stats": season_stats,
        "career_stats": career_stats,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_player_game_log(player_id: int, season: int, last_n: int = 10) -> list[dict]:
    """
    Return last N game log entries for a player.
    season format: 2024 → "20242025"
    """
    season_str = f"{season}{season + 1}"
    try:
        data = _get(f"/player/{player_id}/game-log/{season_str}/2")
    except Exception as exc:
        log.debug("Game log failed %d: %s", player_id, exc)
        return []

    games = data.get("gameLog", [])
    return games[:last_n]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_team_schedule_recent(team_abbrev: str, n_games: int = 5) -> list[dict]:
    """
    Return the last N completed games for a team — used to identify
    the likely starting goalie (most recent starter).
    """
    try:
        data = _get(f"/club-schedule-season/{team_abbrev}/now")
    except Exception as exc:
        log.debug("Team schedule failed %s: %s", team_abbrev, exc)
        return []

    games = data.get("games", [])
    completed = [g for g in games if g.get("gameState") in ("OFF", "FINAL", "CRIT")]
    return completed[-n_games:]


def get_likely_starting_goalie(team_abbrev: str) -> dict | None:
    """
    NHL teams rarely confirm starters in advance.
    Best proxy: the goalie who started the most recent game,
    alternating if they started the previous two.
    Returns basic goalie info dict or None.
    """
    recent = get_team_schedule_recent(team_abbrev, n_games=3)
    if not recent:
        return None

    for game in reversed(recent):
        game_id = game.get("id")
        if not game_id:
            continue
        try:
            box = _get(f"/gamecenter/{game_id}/boxscore")
            home_abbrev = box.get("homeTeam", {}).get("abbrev", "")
            side = "homeTeam" if home_abbrev == team_abbrev else "awayTeam"
            goalies = box.get(side, {}).get("goalies", [])
            if goalies:
                g = goalies[0]
                return {
                    "id":       g.get("playerId"),
                    "fullName": g.get("name", {}).get("default", "Unknown"),
                    "team":     team_abbrev,
                }
        except Exception:
            continue
    return None


def is_back_to_back(team_abbrev: str, game_date: date) -> bool:
    """Return True if the team played yesterday."""
    yesterday = game_date - timedelta(days=1)
    try:
        games = get_schedule(yesterday)
        for g in games:
            home = g.get("homeTeam", {}).get("abbrev", "")
            away = g.get("awayTeam", {}).get("abbrev", "")
            if team_abbrev in (home, away):
                state = g.get("gameState", "")
                if state in ("OFF", "FINAL", "CRIT"):
                    return True
    except Exception:
        pass
    return False
