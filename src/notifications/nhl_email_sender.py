"""
nhl_email_sender.py
===================
Sends the daily NHL goal scorer predictions via SMTP.
Same environment variables as the MLB system:
    EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS, EMAIL_HOST, EMAIL_PORT
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
.subtitle{font-size:12px;color:#7d8590;font-family:monospace}
.date-badge{display:inline-block;background:#4493f814;border:1px solid #4493f844;color:#4493f8;
            font-family:monospace;font-size:11px;padding:3px 10px;border-radius:20px;margin-top:8px}
.summary{background:#161b22;padding:14px 32px;display:flex;gap:28px;border-bottom:1px solid #21262d}
.si{display:flex;flex-direction:column;gap:2px}
.sl{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#7d8590}
.sv{font-size:17px;font-weight:700;color:#e6edf3}
.sec{padding:16px 32px 10px;border-bottom:1px solid #21262d}
.sec-t{font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#4493f8}
.preds{padding:0 32px}
.card{border-bottom:1px solid #21262d1a;padding:18px 0}
.card:last-child{border-bottom:none}
.ct{display:flex;align-items:flex-start;gap:14px}
.rk{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:800;flex-shrink:0;font-family:monospace}
.r1{background:#f1c40f22;color:#f1c40f;border:1px solid #f1c40f44}
.r2{background:#95a5a622;color:#bdc3c7;border:1px solid #95a5a644}
.r3{background:#e67e2222;color:#e67e22;border:1px solid #e67e2244}
.ro{background:#21262d;color:#7d8590;border:1px solid #30363d}
.pi{flex:1;min-width:0}
.pn{font-size:16px;font-weight:700;color:#e6edf3;margin-bottom:1px}
.pm{font-size:11px;color:#7d8590;font-family:monospace}
.mu{font-size:11px;color:#8b949e;margin-top:5px;padding:5px 8px;background:#161b22;
    border-radius:5px;border-left:2px solid #30363d;font-family:monospace}
.mu strong{color:#e6edf3}
.bar-w{margin:8px 0 6px;height:3px;background:#21262d;border-radius:2px;overflow:hidden}
.bar{height:100%;border-radius:2px;background:linear-gradient(90deg,#4493f8,#79c0ff)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.chip{font-family:monospace;font-size:10px;padding:2px 7px;border-radius:4px;display:inline-flex;align-items:center;gap:3px}
.c-blue{background:#1d3a5c;color:#58a6ff;border:1px solid #1f6feb44}
.c-green{background:#0d2818;color:#3fb950;border:1px solid #2ea04344}
.c-red{background:#3d1515;color:#f85149;border:1px solid #da363144}
.c-amber{background:#3d2b00;color:#e3b341;border:1px solid #d2941344}
.c-purple{background:#2d1b4e;color:#d2a8ff;border:1px solid #8957e544}
.c-hi{background:#0d2818;color:#3fb950;border:1px solid #2ea04344}
.c-med{background:#2d2200;color:#e3b341;border:1px solid #d2941344}
.c-low{background:#3d1515;color:#f0883e;border:1px solid #e3631a44}
.pb{text-align:right;flex-shrink:0}
.pp{font-size:24px;font-weight:800;line-height:1;color:#4493f8}
.pl{font-size:10px;color:#7d8590;letter-spacing:1px;text-transform:uppercase;margin-top:1px}
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
  <div class="title">Goal Scorer<br>Predictions</div>
  <div class="subtitle">xG · Moneypuck · Goalie GSAx · Poisson Model</div>
  <div class="date-badge">🏒 %%DATE%%</div>
</div>
<div class="summary">
  <div class="si"><span class="sl">Players</span><span class="sv">%%N%%</span></div>
  <div class="si"><span class="sl">Games</span><span class="sv">%%GAMES%%</span></div>
  <div class="si"><span class="sl">Avg Prob</span><span class="sv">%%AVG%%%</span></div>
</div>
<div class="sec"><div class="sec-t">Tonight's Top Picks</div></div>
<div class="preds">%%CARDS%%</div>
<div class="footer">
  <div class="mt">Methodology</div>
  <div class="mg">
    <div class="mi">📊 <span>xG/60 (Moneypuck)</span></div>
    <div class="mi">🥅 <span>Goalie GSAx</span></div>
    <div class="mi">⚡ <span>PP TOI share</span></div>
    <div class="mi">🏠 <span>Home ice factor</span></div>
    <div class="mi">😴 <span>Back-to-back penalty</span></div>
    <div class="mi">📈 <span>Recent form (10G)</span></div>
    <div class="mi">🤖 <span>XGBoost talent model</span></div>
    <div class="mi">📐 <span>Poisson P(≥1 goal)</span></div>
  </div>
  <div class="fd"></div>
  <div class="fn">Advanced stats © Moneypuck.com · NHL data © NHL Stats API<br>
  <strong>For informational purposes only.</strong> Generated %%TS%% UTC.</div>
</div></div></body></html>"""

_CARD = """<div class="card"><div class="ct">
  <div class="rk %%RC%%">%%RANK%%</div>
  <div class="pi">
    <div class="pn">%%NAME%%</div>
    <div class="pm">%%TEAM%% · %%POS%% · Shoots %%SHOOTS%%</div>
    <div class="mu">vs <strong>%%GOALIE%%</strong> (%%GS%%% SV%) · %%VENUE%%</div>
    <div class="bar-w"><div class="bar" style="width:%%BAR%%%%"></div></div>
    <div class="chips">%%CHIPS%%</div>
  </div>
  <div class="pb"><div class="pp">%%PCT%%</div><div class="pl">Goal Prob</div></div>
</div></div>"""


def _sub(t: str, d: dict) -> str:
    for k, v in d.items():
        t = t.replace(f"%%{k}%%", str(v))
    return t


def _chip(cls: str, icon: str, text: str) -> str:
    return f'<span class="chip {cls}">{icon} {text}</span>'


def _chips(pred: dict, goalie_sv: float | None) -> str:
    chips = []
    f    = pred.get("factors", {})
    mp   = pred.get("mp_metrics", {})
    tier = pred.get("confidence_tier", "Medium")
    cls  = {"High": "c-hi", "Medium": "c-med", "Low": "c-low"}.get(tier, "c-med")
    chips.append(_chip(cls, "◉", f"{tier} conf"))

    xg = mp.get("xg_per_60")
    if xg and xg == xg:
        chips.append(_chip("c-blue", "📊", f"xG/60 {float(xg):.2f}"))

    pp = mp.get("pp_toi_pct")
    if pp and pp == pp and float(pp) > 0.05:
        chips.append(_chip("c-purple", "⚡", f"PP {float(pp)*100:.0f}%"))

    gf = float(f.get("goalie_factor", 1.0))
    if gf >= 1.08:
        chips.append(_chip("c-green", "🥅", "Weak goalie"))
    elif gf <= 0.92:
        chips.append(_chip("c-red", "🥅", "Elite goalie"))

    if float(f.get("b2b_factor", 1.0)) < 1.0:
        chips.append(_chip("c-amber", "😴", "Back-to-back"))
    if float(f.get("home_factor", 1.0)) > 1.0:
        chips.append(_chip("c-green", "🏠", "Home ice"))

    return "\n".join(chips)


def build_html(predictions: list[dict], date_str: str, games: int, ts: str) -> str:
    cards = []
    for i, p in enumerate(predictions, 1):
        rc    = {1:"r1",2:"r2",3:"r3"}.get(i, "ro")
        prob  = p["goal_probability"]
        bar   = min(int(prob * 100 * 3), 100)
        gm    = p.get("goalie_metrics", {})
        sv    = gm.get("save_pct")
        sv_s  = f"{float(sv)*100:.1f}" if sv and sv==sv else "?"

        cards.append(_sub(_CARD, {
            "RANK":   str(i), "RC": rc,
            "NAME":   p.get("player_name", "Unknown"),
            "TEAM":   p.get("team", ""),
            "POS":    p.get("position", ""),
            "SHOOTS": p.get("shoots", "R"),
            "GOALIE": p.get("opp_goalie_name", "Unknown"),
            "GS":     sv_s,
            "VENUE":  p.get("venue", ""),
            "BAR":    str(bar),
            "CHIPS":  _chips(p, sv),
            "PCT":    p.get("goal_pct", "0.0%"),
        }))

    avg = sum(p["goal_probability"] for p in predictions) / len(predictions) * 100 if predictions else 0
    return _sub(_HTML, {
        "DATE": date_str, "N": str(len(predictions)),
        "GAMES": str(games), "AVG": f"{avg:.1f}",
        "CARDS": "\n".join(cards), "TS": ts,
    })


def send_email(
    predictions: list[dict],
    date_str: str,
    games: int,
    ts: str,
    subject_template: str = "🏒 Top {n} Goal Scorer Predictions — {date}",
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
    subject    = subject_template.format(n=len(predictions), date=date_str)
    html_body  = build_html(predictions, date_str, games, ts)

    lines = [f"NHL Goal Predictions — {date_str}", "=" * 50]
    for i, p in enumerate(predictions, 1):
        lines.append(f"{i:2}. {p['player_name']:25s} {p['goal_pct']:>6}  vs {p.get('opp_goalie_name','?')}")
    text_body = "\n".join(lines)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"NHL Goal Predictor <{sender}>"
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, port) as s:
            s.ehlo(); s.starttls(); s.login(sender, pw)
            s.sendmail(sender, recipients, msg.as_string())
        log.info("NHL email sent to %d recipient(s).", len(recipients))
        return True
    except Exception as exc:
        log.error("Failed to send NHL email: %s", exc)
        return False
