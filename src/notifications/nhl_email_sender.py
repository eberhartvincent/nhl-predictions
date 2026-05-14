"""
nhl_email_sender.py
===================
Three-section email: Goals · Points · Shots
Each section independently ranked by its own probability.
"""
from __future__ import annotations
import logging, os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif}
.wrapper{max-width:680px;margin:0 auto}
.header{background:linear-gradient(135deg,#0f1f2e,#0d1117);padding:32px;border-bottom:2px solid #4493f8}
.brand{font-size:11px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#4493f8}
.title{font-size:28px;font-weight:800;color:#fff;margin:4px 0}
.sub{font-size:12px;color:#7d8590;font-family:monospace}
.badge{display:inline-block;background:#4493f814;border:1px solid #4493f844;color:#4493f8;
       font-family:monospace;font-size:11px;padding:3px 10px;border-radius:20px;margin-top:8px}
.summary{background:#161b22;padding:14px 32px;display:flex;gap:28px;border-bottom:1px solid #21262d}
.si{display:flex;flex-direction:column;gap:2px}
.sl{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#7d8590}
.sv{font-size:17px;font-weight:700;color:#e6edf3}
/* Section headers — each with its own accent color */
.sec-goals{padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #4493f8}
.sec-points{padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #3fb950}
.sec-shots{padding:18px 32px 10px;border-bottom:1px solid #21262d;border-top:3px solid #e3b341}
.sec-title-goals{font-size:14px;font-weight:700;color:#4493f8;margin-bottom:2px}
.sec-title-points{font-size:14px;font-weight:700;color:#3fb950;margin-bottom:2px}
.sec-title-shots{font-size:14px;font-weight:700;color:#e3b341;margin-bottom:2px}
.sec-sub{font-size:11px;color:#7d8590;font-family:monospace}
.preds{padding:0 32px}
/* Cards */
.card{border-bottom:1px solid #1a2030;padding:16px 0}
.card:last-child{border-bottom:none}
.ct{display:flex;align-items:flex-start;gap:14px}
.rk{width:32px;height:32px;border-radius:7px;display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:800;flex-shrink:0;font-family:monospace}
.r1-g{background:#4493f822;color:#4493f8;border:1px solid #4493f844}
.r2-g{background:#4493f814;color:#79c0ff;border:1px solid #4493f830}
.r3-g{background:#1f3a5c;color:#79c0ff;border:1px solid #4493f820}
.r1-p{background:#3fb95022;color:#3fb950;border:1px solid #3fb95044}
.r2-p{background:#3fb95014;color:#56d364;border:1px solid #3fb95030}
.r3-p{background:#0d2818;color:#56d364;border:1px solid #3fb95020}
.r1-s{background:#e3b34122;color:#e3b341;border:1px solid #e3b34144}
.r2-s{background:#e3b34114;color:#d29922;border:1px solid #e3b34130}
.r3-s{background:#3d2b00;color:#d29922;border:1px solid #e3b34120}
.ro{background:#21262d;color:#7d8590;border:1px solid #30363d}
.pi{flex:1;min-width:0}
.pn{font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:1px}
.pm{font-size:11px;color:#7d8590;font-family:monospace}
.mu{font-size:11px;color:#8b949e;margin-top:4px;padding:4px 8px;background:#161b22;
    border-radius:5px;border-left:2px solid #30363d;font-family:monospace}
.mu strong{color:#e6edf3}
.bar-w{margin:7px 0 5px;height:3px;background:#21262d;border-radius:2px;overflow:hidden}
.bar-g{height:100%;border-radius:2px;background:linear-gradient(90deg,#4493f8,#79c0ff)}
.bar-p{height:100%;border-radius:2px;background:linear-gradient(90deg,#3fb950,#56d364)}
.bar-s{height:100%;border-radius:2px;background:linear-gradient(90deg,#e3b341,#f0c060)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.chip{font-family:monospace;font-size:10px;padding:2px 6px;border-radius:4px;display:inline-flex;align-items:center;gap:3px}
.c-b{background:#1d3a5c;color:#58a6ff;border:1px solid #1f6feb33}
.c-g{background:#0d2818;color:#3fb950;border:1px solid #2ea04333}
.c-r{background:#3d1515;color:#f85149;border:1px solid #da363133}
.c-a{background:#3d2b00;color:#e3b341;border:1px solid #d2941333}
.c-p{background:#2d1b4e;color:#d2a8ff;border:1px solid #8957e533}
.c-hi{background:#0d2818;color:#3fb950;border:1px solid #2ea04333}
.c-med{background:#2d2200;color:#e3b341;border:1px solid #d2941333}
.c-low{background:#3d1515;color:#f0883e;border:1px solid #e3631a33}
/* Primary stat block */
.pb{text-align:right;flex-shrink:0;min-width:60px}
.pp-g{font-size:22px;font-weight:800;line-height:1;color:#4493f8}
.pp-p{font-size:22px;font-weight:800;line-height:1;color:#3fb950}
.pp-s{font-size:22px;font-weight:800;line-height:1;color:#e3b341}
.pl{font-size:9px;color:#7d8590;letter-spacing:1px;text-transform:uppercase;margin-top:2px}
/* Footer */
.footer{padding:24px 32px;border-top:1px solid #21262d}
.fn{font-size:11px;color:#484f58;line-height:1.7;font-family:monospace}
.fd{height:1px;background:linear-gradient(90deg,transparent,#4493f844,transparent);margin:12px 0}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:3px 20px}
.mi{font-size:10px;color:#484f58;font-family:monospace}
.mi span{color:#7d8590}
.mt{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4493f8;margin-bottom:6px}
</style></head><body><div class="wrapper">
<div class="header">
  <div class="brand">NHL Analytics</div>
  <div class="title">Nightly Predictions</div>
  <div class="sub">Goals · Points · Shots · xG · Goalie GSAx · Poisson</div>
  <div class="badge">🏒 %%DATE%%</div>
</div>
<div class="summary">
  <div class="si"><span class="sl">Games</span><span class="sv">%%GAMES%%</span></div>
  <div class="si"><span class="sl">Players</span><span class="sv">%%PLAYERS%%</span></div>
  <div class="si"><span class="sl">Avg Goal Prob</span><span class="sv">%%AVGGOAL%%%</span></div>
  <div class="si"><span class="sl">Avg Point Prob</span><span class="sv">%%AVGPOINT%%%</span></div>
</div>

<div class="sec-goals">
  <div class="sec-title-goals">🎯 Top %%NG%% — Goal Probability</div>
  <div class="sec-sub">P(scores ≥1 goal) via Poisson · xG/60 + goalie GSAx</div>
</div>
<div class="preds">%%GOAL_CARDS%%</div>

<div class="sec-points">
  <div class="sec-title-points">⭐ Top %%NP%% — Point Probability</div>
  <div class="sec-sub">P(records ≥1 point) · points/60 + PP time</div>
</div>
<div class="preds">%%POINT_CARDS%%</div>

<div class="sec-shots">
  <div class="sec-title-shots">🏒 Top %%NS%% — Shots (over %%LINE%%)</div>
  <div class="sec-sub">P(shots on goal > %%LINE%%) · shots/60 + ice time</div>
</div>
<div class="preds">%%SHOT_CARDS%%</div>

<div class="footer">
  <div class="mt">Methodology</div>
  <div class="mg">
    <div class="mi">📊 <span>xG/60 (Moneypuck)</span></div>
    <div class="mi">🥅 <span>Goalie save% + GSAx</span></div>
    <div class="mi">⚡ <span>PP TOI fraction</span></div>
    <div class="mi">🏠 <span>Home ice advantage</span></div>
    <div class="mi">😴 <span>Back-to-back penalty</span></div>
    <div class="mi">📈 <span>Recent form (10G)</span></div>
    <div class="mi">🤖 <span>XGBoost (3 models)</span></div>
    <div class="mi">📐 <span>Poisson distribution</span></div>
  </div>
  <div class="fd"></div>
  <div class="fn">Stats © Moneypuck.com · NHL data © NHL Stats API<br>
  <strong>For informational purposes only.</strong> Generated %%TS%% UTC.</div>
</div></div></body></html>"""

_CARD = """<div class="card"><div class="ct">
  <div class="rk %%RC%%">%%RANK%%</div>
  <div class="pi">
    <div class="pn">%%NAME%%</div>
    <div class="pm">%%TEAM%% · %%POS%%</div>
    <div class="mu">vs <strong>%%GOALIE%%</strong> · %%VENUE%%</div>
    <div class="bar-w"><div class="%%BAR_CLS%%" style="width:%%BAR%%%%"></div></div>
    <div class="chips">%%CHIPS%%</div>
  </div>
  <div class="pb">
    <div class="%%PCT_CLS%%">%%PRIMARY%%</div>
    <div class="pl">%%LABEL%%</div>
  </div>
</div></div>"""


def _sub(t: str, d: dict) -> str:
    for k, v in d.items():
        t = t.replace(f"%%{k}%%", str(v))
    return t


def _chip(cls: str, icon: str, text: str) -> str:
    return f'<span class="chip {cls}">{icon} {text}</span>'


def _shared_chips(pred: dict, section: str) -> str:
    """Build chips — section = 'goals'|'points'|'shots'."""
    chips = []
    f    = pred.get("factors", {})
    mp   = pred.get("mp_metrics", {})
    tier = pred.get("confidence_tier", "Medium")
    cls  = {"High":"c-hi","Medium":"c-med","Low":"c-low"}.get(tier,"c-med")
    chips.append(_chip(cls, "◉", f"{tier}"))

    xg = mp.get("xg_per_60")
    if xg and xg == xg:
        chips.append(_chip("c-b", "📊", f"xG/60 {float(xg):.2f}"))

    pp = mp.get("pp_toi_pct")
    if pp and pp == pp and float(pp) > 0.05:
        chips.append(_chip("c-p", "⚡", f"PP {float(pp)*100:.0f}%"))

    gf = float(f.get("goalie_factor", 1.0))
    if gf >= 1.08:
        chips.append(_chip("c-g", "🥅", "Weak goalie"))
    elif gf <= 0.92:
        chips.append(_chip("c-r", "🥅", "Elite goalie"))

    if float(f.get("b2b_factor", 1.0)) < 1.0:
        chips.append(_chip("c-a", "😴", "B2B"))
    if float(f.get("home_factor", 1.0)) > 1.0:
        chips.append(_chip("c-g", "🏠", "Home"))

    # Section-specific secondary stat
    if section == "goals":
        sh = mp.get("shooting_pct")
        if sh and sh == sh:
            chips.append(_chip("c-b", "🎯", f"SH% {float(sh)*100:.1f}"))
    elif section == "points":
        chips.append(_chip("c-b", "⭐", f"Pt prob {pred.get('point_pct','')}"))
    elif section == "shots":
        ev = mp.get("shots_per_60")
        if ev and ev == ev:
            chips.append(_chip("c-b", "🏒", f"{float(ev):.1f} sh/60"))

    return "\n".join(chips)


def _rank_cls(i: int, section: str) -> str:
    suffix = {"goals": "g", "points": "p", "shots": "s"}.get(section, "g")
    return {1: f"r1-{suffix}", 2: f"r2-{suffix}", 3: f"r3-{suffix}"}.get(i, "ro")


def _build_section_cards(predictions: list[dict], section: str, shots_line: float) -> str:
    pct_cls = {"goals": "pp-g", "points": "pp-p", "shots": "pp-s"}[section]
    bar_cls = {"goals": "bar-g", "points": "bar-p", "shots": "bar-s"}[section]
    cards   = []

    for i, p in enumerate(predictions, 1):
        if section == "goals":
            prob    = p["goal_probability"]
            primary = p["goal_pct"]
            label   = "Goal Prob"
        elif section == "points":
            prob    = p["point_probability"]
            primary = p["point_pct"]
            label   = "Point Prob"
        else:
            prob    = p["shot_probability"]
            primary = f"{p.get('expected_shots', '?')} exp"
            label   = f"P(>{shots_line} shots)"

        bar = min(int(prob * 100 * 2.5), 100)
        cards.append(_sub(_CARD, {
            "RANK":    str(i),
            "RC":      _rank_cls(i, section),
            "NAME":    p.get("player_name", "Unknown"),
            "TEAM":    p.get("team", ""),
            "POS":     p.get("position", ""),
            "GOALIE":  p.get("opp_goalie_name", "Unknown"),
            "VENUE":   p.get("venue", ""),
            "BAR_CLS": bar_cls,
            "BAR":     str(bar),
            "CHIPS":   _shared_chips(p, section),
            "PCT_CLS": pct_cls,
            "PRIMARY": primary,
            "LABEL":   label,
        }))
    return "\n".join(cards)


def build_html(
    goal_preds:  list[dict],
    point_preds: list[dict],
    shot_preds:  list[dict],
    date_str: str,
    games: int,
    ts: str,
    shots_line: float = 2.5,
) -> str:
    total_players = len(set(
        p["player_id"] for p in goal_preds + point_preds + shot_preds
        if "player_id" in p
    ))
    avg_g = sum(p["goal_probability"]  for p in goal_preds)  / len(goal_preds)  * 100 if goal_preds  else 0
    avg_p = sum(p["point_probability"] for p in point_preds) / len(point_preds) * 100 if point_preds else 0

    return _sub(_HTML, {
        "DATE":        date_str,
        "GAMES":       str(games),
        "PLAYERS":     str(total_players),
        "AVGGOAL":     f"{avg_g:.1f}",
        "AVGPOINT":    f"{avg_p:.1f}",
        "NG":          str(len(goal_preds)),
        "NP":          str(len(point_preds)),
        "NS":          str(len(shot_preds)),
        "LINE":        str(shots_line),
        "GOAL_CARDS":  _build_section_cards(goal_preds,  "goals",  shots_line),
        "POINT_CARDS": _build_section_cards(point_preds, "points", shots_line),
        "SHOT_CARDS":  _build_section_cards(shot_preds,  "shots",  shots_line),
        "TS":          ts,
    })


def send_email(
    goal_preds:  list[dict],
    point_preds: list[dict],
    shot_preds:  list[dict],
    date_str: str,
    games: int,
    ts: str,
    shots_line: float = 2.5,
    subject_template: str = "🏒 NHL Predictions — {date}",
) -> bool:
    sender  = os.environ.get("EMAIL_SENDER", "")
    pw      = os.environ.get("EMAIL_PASSWORD", "")
    recip_r = os.environ.get("EMAIL_RECIPIENTS", "")
    host    = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
    port    = int(os.environ.get("EMAIL_PORT", "587"))

    if not sender or not pw or not recip_r:
        log.error("EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS not set.")
        return False

    recipients = [r.strip() for r in recip_r.split(",") if r.strip()]
    total      = len(set(p.get("player_id") for p in goal_preds + point_preds + shot_preds))
    subject    = subject_template.format(n=total, date=date_str)
    html_body  = build_html(goal_preds, point_preds, shot_preds,
                            date_str, games, ts, shots_line)

    # Plain text
    lines = [f"NHL Predictions — {date_str}", "=" * 55,
             f"\n🎯 TOP GOAL SCORERS"]
    for i, p in enumerate(goal_preds, 1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p['goal_pct']:>6}  vs {p.get('opp_goalie_name','?')}")
    lines.append(f"\n⭐ TOP POINT SCORERS")
    for i, p in enumerate(point_preds, 1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p['point_pct']:>6}")
    lines.append(f"\n🏒 TOP SHOT TAKERS (over {shots_line})")
    for i, p in enumerate(shot_preds, 1):
        lines.append(f"  {i:2}. {p['player_name']:25s} {p.get('expected_shots','?')} exp shots")
    text_body = "\n".join(lines)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"NHL Predictor <{sender}>"
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, port) as s:
            s.ehlo(); s.starttls(); s.login(sender, pw)
            s.sendmail(sender, recipients, msg.as_string())
        log.info("NHL email sent (%d goal, %d point, %d shot predictions).",
                 len(goal_preds), len(point_preds), len(shot_preds))
        return True
    except Exception as exc:
        log.error("Failed to send NHL email: %s", exc)
        return False
