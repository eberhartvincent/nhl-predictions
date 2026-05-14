#!/usr/bin/env python3
"""
nhl_main.py — NHL Goal Scorer Prediction Pipeline

Usage:
  python nhl_main.py                        # tonight's games
  python nhl_main.py --date 2026-04-15      # specific date
  python nhl_main.py --top-n 10 --dry-run   # dry run
  python nhl_main.py --retrain              # retrain then predict
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
log = logging.getLogger("nhl_main")

NHL_API_WORKERS  = 15
MONEYPUCK_WORKERS = 6


def _import_pipeline():
    from src.data.nhl_client import (
        extract_matchups, get_roster, get_player_info, get_player_game_log,
        get_schedule, get_likely_starting_goalie, is_back_to_back,
    )
    from src.data.moneypuck_client import get_skater_metrics, get_goalie_metrics
    from src.models.nhl_predictor import ensemble_predict
    from src.models.nhl_model_registry import load as load_model
    from src.notifications.nhl_email_sender import send_email

    return {
        "get_schedule":       get_schedule,
        "extract_matchups":   extract_matchups,
        "get_roster":         get_roster,
        "get_player_info":    get_player_info,
        "get_player_game_log":get_player_game_log,
        "get_skater_metrics": get_skater_metrics,
        "get_goalie_metrics": get_goalie_metrics,
        "get_goalie":         get_likely_starting_goalie,
        "is_b2b":             is_back_to_back,
        "ensemble_predict":   ensemble_predict,
        "load_model":         load_model,
        "send_email":         send_email,
    }


def load_config(path: str = "nhl_config.yml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("TOP_N"):
        cfg.setdefault("prediction", {})["top_n"] = int(os.environ["TOP_N"])
    if os.environ.get("PREDICT_DATE"):
        cfg.setdefault("prediction", {})["date"] = os.environ["PREDICT_DATE"]
    return cfg


def resolve_date(raw: str, fn: dict) -> date:
    if raw != "today":
        from dateutil.parser import parse as dp
        return dp(raw).date()
    today    = date.today()
    tomorrow = today + timedelta(days=1)
    games    = fn["get_schedule"](today)
    matchups = fn["extract_matchups"](games, skip_final=True)
    if matchups:
        return today
    if games:
        log.warning("All games on %s are final — switching to %s.", today, tomorrow)
        return tomorrow
    log.info("No games on %s — trying %s.", today, tomorrow)
    return tomorrow


def _fetch_skater_bundle(fn, player_id: int, season: int, recent_n: int) -> dict:
    info   = fn["get_player_info"](player_id)
    recent = fn["get_player_game_log"](player_id, season, recent_n)
    return {"info": info, "recent": recent}


def _fetch_mp_metrics(fn, player_id: int, season: int) -> dict:
    try:
        return fn["get_skater_metrics"](player_id, season)
    except Exception:
        return {}


def prefetch_all(fn, matchups: list[dict], season: int, recent_n: int) -> tuple[dict, dict, dict, dict]:
    """Parallel data fetch — same pattern as MLB pipeline."""
    all_player_ids: set[int] = set()
    all_goalie_teams: set[str] = set()

    for m in matchups:
        for side in ("home_abbrev", "away_abbrev"):
            all_goalie_teams.add(m[side])
        for pid in m.get("home_roster", []) + m.get("away_roster", []):
            all_player_ids.add(pid)

    log.info("Prefetching: %d players, %d teams for goalie detection …",
             len(all_player_ids), len(all_goalie_teams))

    # ── Goalies (identify likely starters) ───────────────────────────────
    goalie_cache: dict[str, dict] = {}   # team_abbrev → goalie info
    goalie_metrics_cache: dict[int, dict] = {}

    for team in all_goalie_teams:
        goalie_info = fn["get_goalie"](team)
        goalie_cache[team] = goalie_info or {}
        if goalie_info and goalie_info.get("id"):
            gid = goalie_info["id"]
            try:
                goalie_metrics_cache[gid] = fn["get_goalie_metrics"](gid, season)
            except Exception:
                goalie_metrics_cache[gid] = {}

    log.info("Goalies identified: %d", sum(1 for g in goalie_cache.values() if g))

    # ── Player NHL API data (parallel) ────────────────────────────────────
    player_cache: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=NHL_API_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_skater_bundle, fn, pid, season, recent_n): pid
            for pid in all_player_ids
        }
        done = 0
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                player_cache[pid] = fut.result()
            except Exception as exc:
                log.debug("Player fetch failed %d: %s", pid, exc)
                player_cache[pid] = {"info": {}, "recent": []}
            done += 1
            if done % 50 == 0:
                log.info("  NHL API: %d/%d players …", done, len(all_player_ids))
    log.info("Players fetched: %d", len(player_cache))

    # ── Moneypuck metrics (parallel, fewer workers) ───────────────────────
    mp_cache: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MONEYPUCK_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_mp_metrics, fn, pid, season): pid
            for pid in all_player_ids
        }
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                mp_cache[pid] = fut.result()
            except Exception:
                mp_cache[pid] = {}
    log.info("Moneypuck metrics fetched: %d", len(mp_cache))

    return player_cache, mp_cache, goalie_cache, goalie_metrics_cache


def run(config: dict, dry_run: bool = False) -> list[dict]:
    fn       = _import_pipeline()
    pred_cfg = config.get("prediction", {})
    mdl_cfg  = config.get("model", {})

    # Load model
    xgb_model, xgb_meta = fn["load_model"]()
    if xgb_model:
        log.info("✅ NHL XGBoost model loaded (R²=%.4f, n=%d)",
                 xgb_meta.get("cv_r2", 0), xgb_meta.get("n_training", 0))
    else:
        log.info("ℹ️  No trained NHL model — using statistical model only.")

    # Date
    game_date = resolve_date(pred_cfg.get("date", "today"), fn)
    season    = game_date.year if game_date.month >= 9 else game_date.year - 1
    top_n     = int(pred_cfg.get("top_n", 20))
    min_gp    = int(pred_cfg.get("min_games_played", 10))
    min_toi   = float(pred_cfg.get("min_toi_per_game", 8))
    positions  = pred_cfg.get("positions", ["C", "L", "R"])
    recent_n  = int(mdl_cfg.get("recent_form_games", 10))
    avg_toi   = float(mdl_cfg.get("avg_toi_per_game", 15.5))

    log.info("── NHL predictions for %s (top %d, season %d-%d) ──",
             game_date, top_n, season, season + 1)

    # Schedule
    games    = fn["get_schedule"](game_date)
    matchups = fn["extract_matchups"](games)
    if not matchups:
        log.error("No actionable games on %s.", game_date)
        return []

    # Attach rosters to matchups
    for m in matchups:
        for side in ("home_abbrev", "away_abbrev"):
            abbrev  = m[side]
            roster  = fn["get_roster"](abbrev)
            b2b     = fn["is_b2b"](abbrev, game_date)
            key     = "home" if side == "home_abbrev" else "away"
            m[f"{key}_roster"] = [
                p["id"] for p in roster
                if p.get("position") in positions and p.get("id")
            ]
            m[f"{key}_b2b"]    = b2b

    # Parallel prefetch
    player_cache, mp_cache, goalie_cache, goalie_metrics_cache = prefetch_all(
        fn, matchups, season, recent_n
    )

    # Predictions
    all_predictions: list[dict] = []

    for m in matchups:
        for side in ("home", "away"):
            opp     = "away" if side == "home" else "home"
            is_home = side == "home"
            b2b     = m.get(f"{side}_b2b", False)
            team    = m[f"{side}_team"]
            abbrev  = m[f"{side}_abbrev"]
            opp_abbrev = m[f"{opp}_abbrev"]

            # Opposing goalie
            goalie_info = goalie_cache.get(opp_abbrev, {})
            goalie_id   = goalie_info.get("id")
            goalie_name = goalie_info.get("fullName", "Unknown")
            goalie_met  = goalie_metrics_cache.get(goalie_id, {}) if goalie_id else {}

            for player_id in m.get(f"{side}_roster", []):
                p_data = player_cache.get(player_id, {})
                info   = p_data.get("info", {})
                recent = p_data.get("recent", [])

                pos = info.get("position", "")
                if pos not in positions:
                    continue

                s_stats = info.get("season_stats", {})
                c_stats = info.get("career_stats", {})
                gp      = int(s_stats.get("gamesPlayed", 0))
                if gp < min_gp:
                    continue

                mp = mp_cache.get(player_id, {})
                toi_per_game = float(mp.get("toi_per_game") or avg_toi)
                if toi_per_game < min_toi:
                    continue

                try:
                    pred = fn["ensemble_predict"](
                        season_stats=s_stats,
                        career_stats=c_stats,
                        mp_metrics=mp,
                        goalie_metrics=goalie_met,
                        recent_games=recent,
                        is_home=is_home,
                        is_back_to_back=b2b,
                        estimated_toi_min=toi_per_game,
                        config=config,
                        xgb_model=xgb_model,
                        xgb_meta=xgb_meta,
                    )
                    pred.update({
                        "player_id":       player_id,
                        "player_name":     info.get("fullName", "Unknown"),
                        "team":            team,
                        "team_abbrev":     abbrev,
                        "position":        pos,
                        "shoots":          info.get("shoots", "R"),
                        "opp_goalie_name": goalie_name,
                        "opp_goalie_id":   goalie_id,
                        "goalie_metrics":  goalie_met,
                        "venue":           m.get("venue", ""),
                        "is_home":         is_home,
                        "is_b2b":          b2b,
                        "game_id":         m.get("gameId"),
                    })
                    all_predictions.append(pred)
                except Exception as exc:
                    log.debug("Prediction failed player %d: %s", player_id, exc)

    all_predictions.sort(key=lambda x: x["goal_probability"], reverse=True)
    top = all_predictions[:top_n]

    log.info("Top %d from %d candidates:", top_n, len(all_predictions))
    for i, p in enumerate(top, 1):
        log.info("  %2d. %-25s %s  conf=%s  src=%s",
                 i, p["player_name"], p["goal_pct"],
                 p["confidence_tier"], p.get("rate_source", "?"))

    if not dry_run:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        sent = fn["send_email"](
            predictions=top,
            date_str=game_date.strftime("%A, %B %-d, %Y"),
            games=len(matchups),
            ts=now,
            subject_template=config.get("email", {}).get(
                "subject", "🏒 Top {n} Goal Scorer Predictions — {date}"
            ),
        )
        if sent:
            log.info("✅ Email delivered.")
        else:
            log.error("❌ Email delivery failed.")
            sys.exit(1)
    else:
        log.info("Dry run — email not sent.")

    return top


def parse_args():
    p = argparse.ArgumentParser(description="NHL Goal Scorer Prediction Pipeline")
    p.add_argument("--config",     default="nhl_config.yml")
    p.add_argument("--date",       help="YYYY-MM-DD or 'today'")
    p.add_argument("--top-n",      type=int)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--test-email", action="store_true")
    p.add_argument("--retrain",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)
    if args.date:
        config.setdefault("prediction", {})["date"] = args.date
    if args.top_n:
        config.setdefault("prediction", {})["top_n"] = args.top_n
    if args.retrain:
        from src.models.nhl_train import train
        train()
    run(config, dry_run=args.dry_run and not args.test_email)
