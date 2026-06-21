"""Generate static HTML dashboard for WC 2026 predictions."""
import html as _html
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

TOURNAMENT_START = pd.Timestamp("2026-06-11", tz="UTC")

FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bolivia": "🇧🇴", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Cape Verde": "🇨🇻",
    "Chile": "🇨🇱", "Colombia": "🇨🇴", "Costa Rica": "🇨🇷",
    "Croatia": "🇭🇷", "Czech Republic": "🇨🇿", "Denmark": "🇩🇰",
    "DR Congo": "🇨🇩", "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Honduras": "🇭🇳",
    "Hungary": "🇭🇺", "Indonesia": "🇮🇩", "Iran": "🇮🇷",
    "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲", "Japan": "🇯🇵",
    "Mali": "🇲🇱", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Nigeria": "🇳🇬",
    "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Peru": "🇵🇪", "Poland": "🇵🇱", "Portugal": "🇵🇹",
    "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal": "🇸🇳", "Serbia": "🇷🇸", "Slovakia": "🇸🇰",
    "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tanzania": "🇹🇿",
    "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "Ukraine": "🇺🇦",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
    "Venezuela": "🇻🇪",
}


def _flag(team: str) -> str:
    return FLAGS.get(team, "🏳")


def _load_live() -> dict:
    p = DATA_DIR / "live_events.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_params():
    cache_path = DATA_DIR / "params_cache.json"
    if not cache_path.exists():
        return None
    try:
        from model.poisson import PoissonParams
        d = json.loads(cache_path.read_text())
        return PoissonParams(
            attack=d["attack"], defense=d["defense"],
            rho=d["rho"], gamma=d["gamma"], teams=d["teams"],
        )
    except Exception:
        return None


def _retroactive_pred(home: str, away: str, params) -> dict | None:
    try:
        from model.predict import scoreline_grid
        from model.markets import derive_markets
        grid = scoreline_grid(home, away, params, is_neutral=True)
        m = derive_markets(grid)
        top3 = sorted(m.exact_scores.items(), key=lambda x: -x[1])[:3]
        return {
            "p_home": m.p_home_win, "p_draw": m.p_draw, "p_away": m.p_away_win,
            "top_scoreline": top3[0][0] if top3 else "",
            "top_2_scoreline": top3[1][0] if len(top3) > 1 else "",
            "top_3_scoreline": top3[2][0] if len(top3) > 2 else "",
            "retroactive": True,
        }
    except Exception:
        return None


def _clean_score(v) -> str:
    if v is None or v != v:
        return ""
    s = str(v)
    return "" if s == "nan" else s


def main() -> None:
    live = _load_live()
    params = _load_params()

    preds = pd.read_parquet(DATA_DIR / "predictions.parquet")
    preds["match_date"] = pd.to_datetime(preds["match_date"], utc=True)
    preds = preds.sort_values("match_date")

    results_path = DATA_DIR / "results.parquet"
    results = pd.read_parquet(results_path) if results_path.exists() else pd.DataFrame()
    if not results.empty:
        results["date"] = pd.to_datetime(results["date"], utc=True)

    with open(DATA_DIR / "tournament.json") as f:
        tourn = json.load(f)
    win_odds = sorted(tourn.get("win", {}).items(), key=lambda x: -x[1])[:16]
    tourn_updated = tourn.get("updated_at", "")[:10]

    pred_map: dict[tuple[str, str], dict] = {}
    for _, p in preds.iterrows():
        pred_map[(p["home"], p["away"])] = p.to_dict()

    result_map: dict[tuple[str, str], tuple[int, int]] = {}
    if not results.empty:
        for _, r in results.iterrows():
            result_map[(str(r["home"]), str(r["away"]))] = (int(r["home_goals"]), int(r["away_goals"]))

    all_entries = []
    seen_keys: set[tuple[str, str]] = set()

    if not results.empty:
        for _, r in results.sort_values("date").iterrows():
            dt = pd.Timestamp(r["date"])
            if dt < TOURNAMENT_START:
                continue
            home, away = str(r["home"]), str(r["away"])
            key = (home, away)
            seen_keys.add(key)
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            pred = pred_map.get(key)
            retro = False
            if pred is None and params is not None:
                pred = _retroactive_pred(home, away, params)
                retro = True
            elif pred is not None and str(pred.get("prediction_type", "")) == "retroactive":
                retro = True
            all_entries.append({
                "date": dt, "home": home, "away": away,
                "hg": hg, "ag": ag, "pred": pred, "retro": retro, "status": "completed",
            })

    for _, p in preds.sort_values("match_date").iterrows():
        key = (p["home"], p["away"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_entries.append({
            "date": p["match_date"], "home": p["home"], "away": p["away"],
            "hg": None, "ag": None, "pred": p.to_dict(), "retro": False, "status": "upcoming",
        })

    n_total = n_wdl = n_top1 = n_top3 = 0
    for e in all_entries:
        if e["status"] != "completed" or e["pred"] is None:
            continue
        hg, ag = e["hg"], e["ag"]
        actual = f"{hg}-{ag}"
        actual_wdl = "H" if hg > ag else ("D" if hg == ag else "A")
        p = e["pred"]
        pred_wdl = "H" if p["p_home"] > p["p_draw"] and p["p_home"] > p["p_away"] else (
            "D" if p["p_draw"] > p["p_away"] else "A"
        )
        s1 = _clean_score(p.get("top_scoreline"))
        s2 = _clean_score(p.get("top_2_scoreline"))
        s3 = _clean_score(p.get("top_3_scoreline"))
        n_total += 1
        if pred_wdl == actual_wdl:
            n_wdl += 1
        if s1 == actual:
            n_top1 += 1
        if actual in (s1, s2, s3):
            n_top3 += 1

    def _pct(n, d):
        return f"{n/d:.0%}" if d else "—"

    rows_html = ""
    for e in all_entries:
        home, away = e["home"], e["away"]
        date_str = e["date"].strftime("%b %d")
        status = e["status"]
        pred = e["pred"]
        hg, ag = e["hg"], e["ag"]
        actual = f"{hg}-{ag}" if hg is not None else ""

        live_key = f"{home} vs {away}"
        live_data = live.get(live_key, {})
        if live_data:
            status = "live"

        if live_data:
            ph = live_data.get("p_home", pred["p_home"] if pred else 0.33)
            pd_ = live_data.get("p_draw", pred["p_draw"] if pred else 0.33)
            pa = live_data.get("p_away", pred["p_away"] if pred else 0.33)
        elif pred:
            ph = pred["p_home"]
            pd_ = pred["p_draw"]
            pa = pred["p_away"]
        else:
            ph = pd_ = pa = 0.0

        if pred:
            s1 = _clean_score(pred.get("top_scoreline"))
            s2 = _clean_score(pred.get("top_2_scoreline"))
            s3 = _clean_score(pred.get("top_3_scoreline"))
            parts = [x for x in [s1, s2, s3] if x]
            scores_str = "  ".join(parts) if parts else "—"
        else:
            s1 = s2 = s3 = ""
            scores_str = "—"

        wdl_verdict = score_verdict = ""
        if actual and pred:
            actual_wdl = "H" if hg > ag else ("D" if hg == ag else "A")
            pred_wdl = "H" if ph > pd_ and ph > pa else ("D" if pd_ > pa else "A")
            wdl_verdict = "✓" if pred_wdl == actual_wdl else "✗"
            score_verdict = "✓" if actual == s1 else ("~" if actual in (s2, s3) else "✗")

        live_score = ""
        if live_data:
            lhg = live_data.get("home_goals", 0)
            lag = live_data.get("away_goals", 0)
            minute = live_data.get("minute", "?")
            live_score = f"🔴 {lhg}–{lag} ({minute}')"

        retro_badge = ' <sup class="retro-badge">[r]</sup>' if e.get("retro") else ""
        result_cell = live_score or actual
        no_pred_class = "" if pred else " no-pred"

        hf = _flag(home)
        af = _flag(away)
        h_esc = _html.escape(home)
        a_esc = _html.escape(away)
        scores_esc = _html.escape(scores_str)
        result_esc = _html.escape(result_cell)

        # Mini prob bars (3 segments)
        def _mini_bar(p_h, p_d, p_a):
            w_h = round(p_h * 60)
            w_d = round(p_d * 60)
            w_a = 60 - w_h - w_d
            w_a = max(0, w_a)
            return (
                f'<div class="mini-bar">'
                f'<div class="mb-h" style="width:{w_h}px"></div>'
                f'<div class="mb-d" style="width:{w_d}px"></div>'
                f'<div class="mb-a" style="width:{w_a}px"></div>'
                f'</div>'
            )

        bar_html = _mini_bar(ph, pd_, pa) if pred else '<div class="mini-bar"></div>'

        rows_html += f"""
        <tr class="{status}{no_pred_class}">
          <td class="date-col">{date_str}</td>
          <td class="team-col home-team">{hf} {h_esc}</td>
          <td class="prob home-prob">{f'{ph:.0%}' if pred else '—'}</td>
          <td class="bar-td">{bar_html}</td>
          <td class="prob away-prob">{f'{pa:.0%}' if pred else '—'}</td>
          <td class="team-col away-team">{af} {a_esc}{retro_badge}</td>
          <td class="scores-col">{scores_esc}</td>
          <td class="result-col">{result_esc}</td>
          <td class="verdict-wdl"><span class="verdict">{wdl_verdict}</span></td>
          <td class="verdict-score"><span class="verdict">{score_verdict}</span></td>
        </tr>"""

    winner_rows = ""
    max_prob = win_odds[0][1] if win_odds else 0.01
    for rank, (team, prob) in enumerate(win_odds, 1):
        implied = f"{1/prob - 1:.1f}:1" if prob > 0 else "—"
        bar_w = round((prob / max_prob) * 180)
        tf = _flag(team)
        winner_rows += f"""
        <tr class="winner-row">
          <td class="rank-col">#{rank}</td>
          <td class="wteam-col">{tf} {_html.escape(team)}</td>
          <td class="wpct">{prob:.1%}</td>
          <td class="wbar-cell">
            <div class="wbar-bg">
              <div class="wbar-fill" style="width:{bar_w}px"></div>
            </div>
          </td>
          <td class="wimplied">{implied}</td>
        </tr>"""

    updated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    n_results = len(results) if not results.empty else 0

    accuracy_html = ""
    if n_total > 0:
        accuracy_html = f"""
<div class="accuracy-box">
  <div class="acc-header">
    <span class="acc-title">Live Track Record</span>
    <span class="acc-sub">{n_total} completed predictions · locked before kickoff</span>
  </div>
  <div class="acc-grid">
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_wdl, n_total)}</div>
      <div class="acc-label">W/D/L correct</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_top3, n_total)}</div>
      <div class="acc-label">Score in top 3</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_top1, n_total)}</div>
      <div class="acc-label">Exact top-1 score</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{n_results}</div>
      <div class="acc-label">Results in</div>
    </div>
  </div>
  <div class="acc-note">
    Backtest (5,518 matches 2018–2023): log-loss 0.8961 vs 1.0986 random · 59% W/D/L accuracy
    &nbsp;|&nbsp; <span class="retro-inline">[r]</span> = retroactive prediction
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC 2026 Forecaster</title>
<meta http-equiv="refresh" content="60">
<style>
  :root {{
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #7d8590;
    --muted2: #484f58;
    --green: #3fb950;
    --green-dim: #1a4a24;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --border: #21262d;
    --border2: #30363d;
    --card: #161b22;
    --card2: #1c2128;
    --hover: #1c2128;
    --accent: #238636;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: ui-monospace, 'SF Mono', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    padding: 0;
  }}

  /* ── Header ── */
  .site-header {{
    background: var(--card);
    border-bottom: 1px solid var(--border2);
    padding: 18px 32px 16px;
    display: flex;
    align-items: flex-end;
    gap: 20px;
    flex-wrap: wrap;
  }}
  .site-title {{
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--fg);
  }}
  .site-title .ball {{ margin-right: 6px; }}
  .site-meta {{
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.3px;
    padding-bottom: 2px;
  }}
  .site-meta a {{ color: var(--blue); text-decoration: none; }}
  .site-meta a:hover {{ text-decoration: underline; }}
  .header-right {{
    margin-left: auto;
    text-align: right;
    font-size: 11px;
    color: var(--muted);
  }}
  .updated-pill {{
    display: inline-block;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid #2ea04326;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 10px;
    letter-spacing: 0.3px;
  }}

  /* ── Layout ── */
  .content {{ padding: 24px 32px; max-width: 1200px; }}

  /* ── Section headings ── */
  h2 {{
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin: 32px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  h2 .h2-badge {{
    background: var(--card2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 10px;
    color: var(--muted);
    font-weight: normal;
    letter-spacing: 0;
    text-transform: none;
  }}

  /* ── Accuracy box ── */
  .accuracy-box {{
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 8px;
  }}
  .acc-header {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }}
  .acc-title {{
    font-size: 12px;
    font-weight: 600;
    color: var(--fg);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .acc-sub {{
    font-size: 11px;
    color: var(--muted);
  }}
  .acc-grid {{
    display: flex;
    gap: 36px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .acc-val {{
    font-size: 30px;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
    letter-spacing: -0.5px;
  }}
  .acc-label {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .acc-note {{
    font-size: 10px;
    color: var(--muted2);
    border-top: 1px solid var(--border);
    padding-top: 10px;
    margin-top: 4px;
  }}
  .retro-inline {{
    color: var(--yellow);
    font-size: 10px;
  }}

  /* ── Fixtures table ── */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    text-align: left;
    color: var(--muted);
    font-weight: normal;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 5px 8px 7px;
    border-bottom: 1px solid var(--border2);
    white-space: nowrap;
  }}
  td {{
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
    white-space: nowrap;
  }}
  tr.upcoming:hover td {{ background: var(--hover); cursor: default; }}
  tr.completed td {{ color: var(--muted); }}
  tr.live td {{ background: #1a1200; }}
  tr.live td {{ color: var(--fg); }}
  tr.no-pred td {{ opacity: 0.5; }}

  .date-col {{ width: 52px; color: var(--muted); font-size: 11px; }}
  .team-col {{ max-width: 190px; overflow: hidden; text-overflow: ellipsis; }}
  .home-team {{ text-align: right; }}
  .away-team {{ text-align: left; padding-left: 4px; }}
  .prob {{ text-align: right; width: 38px; font-variant-numeric: tabular-nums; font-size: 12px; }}
  .home-prob {{ color: var(--blue); }}
  .away-prob {{ color: var(--muted); }}
  tr.completed .home-prob,
  tr.completed .away-prob {{ color: inherit; }}

  /* Mini 3-segment prob bar */
  .bar-td {{ padding: 0 6px; width: 64px; }}
  .mini-bar {{ display: flex; height: 4px; border-radius: 2px; overflow: hidden; gap: 1px; }}
  .mb-h {{ background: var(--blue); border-radius: 2px 0 0 2px; min-width: 1px; }}
  .mb-d {{ background: var(--yellow); min-width: 1px; }}
  .mb-a {{ background: var(--muted2); border-radius: 0 2px 2px 0; min-width: 1px; }}
  tr.completed .mb-h {{ background: #2a3040; }}
  tr.completed .mb-d {{ background: #2a2a1a; }}
  tr.completed .mb-a {{ background: #222; }}

  .scores-col {{ color: var(--muted); font-size: 11px; min-width: 110px; letter-spacing: 0.3px; }}
  .result-col {{ font-weight: 600; min-width: 60px; font-size: 12px; }}
  .verdict-wdl, .verdict-score {{ width: 22px; text-align: center; font-size: 13px; }}
  .verdict {{ }}
  .v-correct {{ color: var(--green); }}
  .v-partial {{ color: var(--yellow); }}
  .v-wrong {{ color: var(--red); }}
  .retro-badge {{ color: var(--yellow); font-size: 9px; vertical-align: super; }}

  /* ── Winner odds table ── */
  .winner-section {{ max-width: 560px; }}
  .winner-row:hover td {{ background: var(--hover); }}
  .rank-col {{ width: 32px; color: var(--muted2); font-size: 11px; }}
  .wteam-col {{ min-width: 180px; font-size: 13px; }}
  .wpct {{ text-align: right; width: 54px; color: var(--green); font-variant-numeric: tabular-nums; font-weight: 600; }}
  .wbar-cell {{ padding-left: 14px; width: 200px; }}
  .wbar-bg {{ background: var(--card2); border-radius: 3px; height: 8px; width: 180px; overflow: hidden; }}
  .wbar-fill {{ background: linear-gradient(90deg, #3fb950, #2ea043); border-radius: 3px; height: 100%; transition: width 0.3s; }}
  .wimplied {{ color: var(--muted); padding-left: 16px; font-size: 11px; width: 72px; }}
  .winner-note {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 10px;
    padding: 8px 0;
    border-top: 1px solid var(--border);
  }}

  /* ── Legend ── */
  .legend {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    font-size: 10px;
    color: var(--muted);
    margin-top: 10px;
    padding: 8px 0;
    border-top: 1px solid var(--border);
  }}
  .legend span {{ white-space: nowrap; }}
  .leg-h {{ color: var(--blue); }}
  .leg-d {{ color: var(--yellow); }}
  .leg-a {{ color: var(--muted); }}

  /* ── Footer ── */
  footer {{
    margin-top: 40px;
    padding: 14px 0;
    border-top: 1px solid var(--border);
    color: var(--muted2);
    font-size: 10px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: center;
  }}
  footer a {{ color: var(--muted); text-decoration: none; }}
  footer a:hover {{ color: var(--blue); }}
</style>
</head>
<body>

<header class="site-header">
  <div>
    <div class="site-title"><span class="ball">⚽</span> WC 2026 Forecaster</div>
    <div class="site-meta">
      Dixon-Coles score model &middot; 10k Monte Carlo &middot;
      <a href="https://github.com/VirajMishra1/worldcup-forecaster">github.com/VirajMishra1/worldcup-forecaster</a>
    </div>
  </div>
  <div class="header-right">
    <div class="updated-pill">⟳ updated {updated}</div>
  </div>
</header>

<div class="content">

{accuracy_html}

<h2>All Fixtures — WC 2026 <span class="h2-badge">from Jun 11</span></h2>
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th style="text-align:right">Home</th>
      <th style="text-align:right" title="Home win %">H%</th>
      <th style="width:64px"></th>
      <th title="Away win %">A%</th>
      <th>Away</th>
      <th>Top-3 Scores</th>
      <th>Result</th>
      <th title="W/D/L direction">W/D/L</th>
      <th title="Score in top-3 prediction">Score</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
<div class="legend">
  <span class="leg-h">■ H%</span>
  <span class="leg-d">■ D%</span>
  <span class="leg-a">■ A%</span>
  <span>W/D/L: ✓ correct &nbsp; ✗ wrong</span>
  <span>Score: ✓ exact &nbsp; ~ top-3 hit &nbsp; ✗ miss</span>
  <span class="retro-inline">[r]</span><span> retroactive — computed after match</span>
</div>

<h2>Tournament Winner Odds <span class="h2-badge">10k Monte Carlo · {tourn_updated}</span></h2>
<div class="winner-section">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Team</th>
      <th style="text-align:right">Win %</th>
      <th style="padding-left:14px">Probability</th>
      <th style="padding-left:16px">Implied odds</th>
    </tr>
  </thead>
  <tbody>{winner_rows}
  </tbody>
</table>
<div class="winner-note">
  Bracket-aware simulation. Refit daily with WC 2026 results at 3× weight.
  Implied odds = 1/p − 1.
</div>
</div>

<footer>
  <span>Generated {updated}</span>
  <span>·</span>
  <a href="https://github.com/VirajMishra1/worldcup-forecaster">Source on GitHub</a>
  <span>·</span>
  <span>Dixon-Coles · MLE · temperature scaling calibration · 13,779 training matches</span>
</footer>

</div>

<script>
  document.querySelectorAll('.verdict').forEach(el => {{
    const t = el.textContent.trim();
    if (t === '✓') el.classList.add('v-correct');
    else if (t === '~') el.classList.add('v-partial');
    else if (t === '✗') el.classList.add('v-wrong');
  }});
</script>
</body>
</html>"""

    DOCS_DIR = Path(__file__).parent.parent / "docs"
    DOCS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    for out in [DOCS_DIR / "index.html", REPORTS_DIR / "dashboard.html"]:
        out.write_text(html, encoding="utf-8")
    size_kb = (DOCS_DIR / "index.html").stat().st_size / 1024
    print(f"Dashboard written → docs/index.html  ({size_kb:.1f} KB)")
    print(f"Accuracy: {n_wdl}/{n_total} W/D/L ({_pct(n_wdl,n_total)})  |  "
          f"{n_top3}/{n_total} top-3 ({_pct(n_top3,n_total)})  |  "
          f"{n_top1}/{n_total} exact ({_pct(n_top1,n_total)})")


if __name__ == "__main__":
    main()
