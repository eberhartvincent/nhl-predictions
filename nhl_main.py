#!/usr/bin/env python3
"""nhl_main.py — NHL Predictions Pipeline (Goals · Points · Shots)"""
from __future__ import annotations

import argparse, logging, os, sys
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

NHL_API_WORKERS   = 15
MONEYPUCK_WORKERS = 6


def _import_pipeline():
    from src.data.nhl_client import (
        extract_matchups, get_roster, get_player_info, get_player_game_log,
        get_schedule, get_likely_starting_goalie, is_back_to_back,
    )
    from src.data.moneypuck_client import get_skater_metrics, get_goalie_metrics
    from src.models.nhl_predictor import ensemble_predict
    from src.models.nhl_model_registry import load as load_models
    from src.notifications.nhl_email_sender import send_email

    return {
        "get_schedule":        get_schedule,
        "extract_matchups":    extract_matchups,
        "get_roster":          get_roster,
        "get_player_info":     get_player_info,
        "get_player_game_log": get_player_game_log,
        "get_skater_metrics":  get_skater_metrics,
        "get_goalie_metrics":  get_goalie_metrics,
        "get_goalie":          get_likely_starting_goalie,
        "is_b2b":              is_back_to_back,
        "ensemble_predict":    ensemble_predict,
        "load_models":         load_models,
        "send_email":          send_email,
    }


def load_config(path: str = "nhl_config.yml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("TOP_N"):
        n = int(os.environ["TOP_N"])
        for key in ("top_n_goals", "top_n_points", "top_n_shots"):
            cfg.setdefault("prediction", {})[key] = n
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
    return tomorrow


def _fetch_player(fn, pid: int, season: int, recent_n: int) -> dict:
    info   = fn["get_player_info"](pid)
    recent = fn["get_player_game_log"](pid, season, recent_n)
    return {"info": info, "recent": recent}


def _fetch_mp(fn, pid: int, season: int) -> dict:
    try:
        return fn["get_skater_metrics"](pid, season)
    except Exception:
        return {}


def prefetch_all(fn, matchups, season, recent_n):
    all_pids: set[int] = set()
    all_teams: set[str] = set()

    for m in matchups:
        all_teams.add(m["home_abbrev"])
        all_teams.add(m["away_abbrev"])
        for pid in m.get("home_roster", []) + m.get("away_roster", []):
            all_pids.add(pid)

    log.info("Prefetching %d players, %d teams …", len(all_pids), len(all_teams))

    # Goalies
    goalie_cache: dict[str, dict] = {}
    goalie_metrics: dict[int, dict] = {}
    for team in all_teams:
        gi = fn["get_goalie"](team)
        goalie_cache[team] = gi or {}
        if gi and gi.get("id"):
            gid = gi["id"]
            try:
                goalie_metrics[gid] = fn["get_goalie_metrics"](gid, season)
            except Exception:
                goalie_metrics[gid] = {}

    # Player NHL API data
    player_cache: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=NHL_API_WORKERS) as pool:
        futures = {pool.submit(_fetch_player, fn, pid, season, recent_n): pid for pid in all_pids}
        done = 0
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                player_cache[pid] = fut.result()
            except Exception:
                player_cache[pid] = {"info": {}, "recent": []}
            done += 1
            if done % 50 == 0:
                log.info("  NHL API: %d/%d …", done, len(all_pids))
    log.info("Players fetched: %d", len(player_cache))

    # Moneypuck
    mp_cache: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MONEYPUCK_WORKERS) as pool:
        futures = {pool.submit(_fetch_mp, fn, pid, season): pid for pid in all_pids}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                mp_cache[pid] = fut.result()
            except Exception:
                mp_cache[pid] = {}
    log.info("Moneypuck fetched: %d", len(mp_cache))

    return player_cache, mp_cache, goalie_cache, goalie_metrics


def run(config: dict, dry_run: bool = False) -> dict:
    fn = _import_pipeline()
    pred_cfg = config.get("prediction", {})
    mdl_cfg  = config.get("model", {})

    # Load all three models
    models, metas = fn["load_models"]()
    loaded = [n for n, m in models.items() if m is not None]
    log.info("Models loaded: %s", loaded or ["none — statistical only"])

    # Date
    game_date  = resolve_date(pred_cfg.get("date", "today"), fn)
    season     = game_date.year if game_date.month >= 9 else game_date.year - 1
    top_goals  = int(pred_cfg.get("top_n_goals",  15))
    top_points = int(pred_cfg.get("top_n_points", 15))
    top_shots  = int(pred_cfg.get("top_n_shots",  15))
    shots_line = float(pred_cfg.get("shots_line", 2.5))
    min_gp     = int(pred_cfg.get("min_games_played", 10))
    min_toi    = float(pred_cfg.get("min_toi_per_game", 8))
    positions  = pred_cfg.get("positions", ["C", "L", "R"])
    recent_n   = int(mdl_cfg.get("recent_form_games", 10))
    avg_toi    = float(mdl_cfg.get("avg_toi_per_game", 15.5))

    log.info("── NHL predictions for %s (season %d-%d) ──", game_date, season, season + 1)

    games    = fn["get_schedule"](game_date)
    matchups = fn["extract_matchups"](games)
    if not matchups:
        log.error("No actionable games on %s.", game_date)
        return {}

    # Attach rosters
    for m in matchups:
        for side, key in (("home_abbrev", "home_roster"), ("away_abbrev", "away_roster")):
            abbrev = m[side]
            roster = fn["get_roster"](abbrev)
            m[key] = [p["id"] for p in roster
                      if p.get("position") in positions and p.get("id")]
            m[f"{'home' if side=='home_abbrev' else 'away'}_b2b"] = fn["is_b2b"](abbrev, game_date)

    player_cache, mp_cache, goalie_cache, goalie_metrics = prefetch_all(
        fn, matchups, season, recent_n
    )

    all_predictions: list[dict] = []

    for m in matchups:
        for side in ("home", "away"):
            opp     = "away" if side == "home" else "home"
            is_home = side == "home"
            b2b     = m.get(f"{side}_b2b", False)
            team    = m[f"{side}_team"]
            abbrev  = m[f"{side}_abbrev"]
            opp_ab  = m[f"{opp}_abbrev"]

            goalie_info = goalie_cache.get(opp_ab, {})
            goalie_id   = goalie_info.get("id")
            goalie_name = goalie_info.get("fullName", "Unknown")
            g_met       = goalie_metrics.get(goalie_id, {}) if goalie_id else {}

            for pid in m.get(f"{side}_roster", []):
                pd_data = player_cache.get(pid, {})
                info    = pd_data.get("info", {})
                recent  = pd_data.get("recent", [])

                pos = info.get("position", "")
                if pos not in positions:
                    continue

                s_stats = info.get("season_stats", {})
                c_stats = info.get("career_stats", {})
                if int(s_stats.get("gamesPlayed", 0)) < min_gp:
                    continue

                mp = mp_cache.get(pid, {})
                toi = float(mp.get("toi_per_game") or avg_toi)
                if toi < min_toi:
                    continue

                try:
                    pred = fn["ensemble_predict"](
                        season_stats=s_stats, career_stats=c_stats,
                        mp_metrics=mp, goalie_metrics=g_met,
                        recent_games=recent, is_home=is_home,
                        is_back_to_back=b2b, estimated_toi_min=toi,
                        config=config, models=models, metas=metas,
                    )
                    pred.update({
                        "player_id":       pid,
                        "player_name":     info.get("fullName", "Unknown"),
                        "team":            team, "team_abbrev": abbrev,
                        "position":        pos,
                        "shoots":          info.get("shoots", "R"),
                        "opp_goalie_name": goalie_name,
                        "opp_goalie_id":   goalie_id,
                        "goalie_metrics":  g_met,
                        "venue":           m.get("venue", ""),
                        "is_home":         is_home,
                        "is_b2b":          b2b,
                    })
                    all_predictions.append(pred)
                except Exception as exc:
                    log.debug("Prediction failed %d: %s", pid, exc)

    # Sort into three independent ranked lists
    goal_preds  = sorted(all_predictions, key=lambda x: x["goal_probability"],  reverse=True)[:top_goals]
    point_preds = sorted(all_predictions, key=lambda x: x["point_probability"], reverse=True)[:top_points]
    shot_preds  = sorted(all_predictions, key=lambda x: x["shot_probability"],  reverse=True)[:top_shots]

    log.info("Goals top %d | Points top %d | Shots top %d (from %d candidates)",
             len(goal_preds), len(point_preds), len(shot_preds), len(all_predictions))
    for i, p in enumerate(goal_preds[:5], 1):
        log.info("  Goals %d: %-22s %s / pt %s / sh %s exp",
                 i, p["player_name"], p["goal_pct"],
                 p["point_pct"], p.get("expected_shots","?"))

    if not dry_run:
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        sent = fn["send_email"](
            goal_preds=goal_preds, point_preds=point_preds, shot_preds=shot_preds,
            date_str=game_date.strftime("%A, %B %-d, %Y"),
            games=len(matchups), ts=now, shots_line=shots_line,
            subject_template=config.get("email", {}).get("subject", "🏒 NHL Predictions — {date}"),
        )
        if sent:
            log.info("✅ Email delivered.")
        else:
            log.error("❌ Email delivery failed.")
            sys.exit(1)
    else:
        log.info("Dry run — email not sent.")

    return {"goals": goal_preds, "points": point_preds, "shots": shot_preds}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="nhl_config.yml")
    p.add_argument("--date",       help="YYYY-MM-DD or 'today'")
    p.add_argument("--top-n",      type=int, help="Override all three top-N values")
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
        for key in ("top_n_goals", "top_n_points", "top_n_shots"):
            config.setdefault("prediction", {})[key] = args.top_n
    if args.retrain:
        from src.models.nhl_train import train
        train()
    run(config, dry_run=args.dry_run and not args.test_email)
