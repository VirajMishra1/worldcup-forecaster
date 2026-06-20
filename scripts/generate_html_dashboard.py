"""Generate static HTML dashboard for WC 2026 predictions."""
import html as _html
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

TOURNAMENT_START = pd.Timestamp("2026-06-11", tz="UTC")


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


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
    """Compute prediction on-the-fly using cached params."""
    try:
        from model.predict import scoreline_grid
        from model.markets import derive_markets
        grid = scoreline_grid(home, away, params, is_neutral=True)
        m = derive_markets(grid)
        top3 = sorted(m.exact_scores.items(), key=lambda x: -x[1])[:3]
        return {
            "p_home": m.p_home_win,
            "p_draw": m.p_draw,
            "p_away": m.p_away_win,
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
    win_odds = sorted(tourn.get("win", {}).items(), key=lambda x: -x[1])[:12]

    # Build lookup: (home, away) -> prediction row
    pred_map: dict[tuple[str, str], dict] = {}
    for _, p in preds.iterrows():
        pred_map[(p["home"], p["away"])] = p.to_dict()

    # Build lookup: (home, away) -> actual score
    result_map: dict[tuple[str, str], tuple[int, int]] = {}
    if not results.empty:
        for _, r in results.iterrows():
            result_map[(str(r["home"]), str(r["away"]))] = (int(r["home_goals"]), int(r["away_goals"]))

    # Merge: all results + all upcoming preds, deduplicated and sorted
    # Start from tournament start date
    all_entries = []  # list of dicts with unified fields

    seen_keys: set[tuple[str, str]] = set()

    # First pass: results (real games played, sorted by date)
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

            all_entries.append({
                "date": dt,
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
                "pred": pred,
                "retro": retro,
                "status": "completed",
            })

    # Second pass: upcoming predictions not yet in results
    for _, p in preds.sort_values("match_date").iterrows():
        key = (p["home"], p["away"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_entries.append({
            "date": p["match_date"],
            "home": p["home"],
            "away": p["away"],
            "hg": None,
            "ag": None,
            "pred": p.to_dict(),
            "retro": False,
            "status": "upcoming",
        })

    # Compute accuracy stats (only completed games with predictions)
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

    # Build HTML rows
    rows_html = ""
    for e in all_entries:
        home, away = e["home"], e["away"]
        date_str = e["date"].strftime("%b %d")
        status = e["status"]
        pred = e["pred"]
        hg, ag = e["hg"], e["ag"]
        actual = f"{hg}-{ag}" if hg is not None else ""

        # Determine live state
        live_key = f"{home} vs {away}"
        live_data = live.get(live_key, {})
        if live_data:
            status = "live"

        # Probabilities
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

        # Scorelines
        if pred:
            s1 = _clean_score(pred.get("top_scoreline"))
            s2 = _clean_score(pred.get("top_2_scoreline"))
            s3 = _clean_score(pred.get("top_3_scoreline"))
            parts = [x for x in [s1, s2, s3] if x]
            scores_str = "  ".join(parts) if parts else "—"
        else:
            s1 = s2 = s3 = ""
            scores_str = "—"

        # Verdict
        wdl_verdict = ""
        score_verdict = ""
        if actual and pred:
            actual_wdl = "H" if hg > ag else ("D" if hg == ag else "A")
            pred_wdl = "H" if ph > pd_ and ph > pa else ("D" if pd_ > pa else "A")
            wdl_verdict = "✓" if pred_wdl == actual_wdl else "✗"
            if actual == s1:
                score_verdict = "✓"
            elif actual in (s2, s3):
                score_verdict = "~"
            else:
                score_verdict = "✗"

        # Live score override
        live_score = ""
        if live_data:
            lhg = live_data.get("home_goals", 0)
            lag = live_data.get("away_goals", 0)
            minute = live_data.get("minute", "?")
            live_score = f"🔴 {lhg}-{lag} ({minute}')"

        retro_badge = '<sup title="retroactive — computed after match">[r]</sup>' if e.get("retro") else ""

        result_cell = live_score or actual
        no_pred_class = "" if pred else " no-pred"
        h_esc = _html.escape(home)
        a_esc = _html.escape(away)
        scores_esc = _html.escape(scores_str)
        result_esc = _html.escape(result_cell)
        rows_html += f"""
        <tr class="{status}{no_pred_class}">
          <td class="date-col">{date_str}</td>
          <td class="team-col home-team">{h_esc}</td>
          <td class="prob home-prob">{f'{ph:.0%}' if pred else '—'}</td>
          <td class="prob draw-prob">{f'{pd_:.0%}' if pred else '—'}</td>
          <td class="prob away-prob">{f'{pa:.0%}' if pred else '—'}</td>
          <td class="team-col away-team">{a_esc}{retro_badge}</td>
          <td class="scores-col">{scores_esc}</td>
          <td class="result-col">{result_esc}</td>
          <td class="verdict-wdl"><span class="verdict">{wdl_verdict}</span></td>
          <td class="verdict-score"><span class="verdict">{score_verdict}</span></td>
        </tr>"""

    # Build winner odds rows
    winner_rows = ""
    for team, prob in win_odds:
        implied = f"{1/prob - 1:.1f}:1" if prob > 0 else "—"
        bar = _bar(prob)
        winner_rows += f"""
        <tr>
          <td>{_html.escape(team)}</td>
          <td class="pct">{prob:.1%}</td>
          <td class="bar-cell"><span class="bar">{bar}</span></td>
          <td class="implied">{implied}</td>
        </tr>"""

    updated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")

    accuracy_html = ""
    if n_total > 0:
        accuracy_html = f"""
<div class="accuracy-box">
  <div class="acc-title">Model Accuracy — {n_total} completed games with predictions</div>
  <div class="acc-grid">
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_wdl, n_total)}</div>
      <div class="acc-label">W/D/L direction correct</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_top3, n_total)}</div>
      <div class="acc-label">Exact score in top-3</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{_pct(n_top1, n_total)}</div>
      <div class="acc-label">Top-1 score exact</div>
    </div>
    <div class="acc-stat">
      <div class="acc-val">{n_total}</div>
      <div class="acc-label">Games evaluated</div>
    </div>
  </div>
  <div class="acc-note">
    [r] = retroactive (predicted after match played, using params trained on all data)
    &nbsp;|&nbsp; locked = pre-match predictions
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
    --muted: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --border: #30363d;
    --card: #161b22;
    --hover: #1c2128;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: 'Courier New', Courier, monospace;
    font-size: 13px;
    padding: 24px 32px;
    line-height: 1.5;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: bold;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  h2 {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 28px 0 10px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 8px;
  }}
  th {{
    text-align: left;
    color: var(--muted);
    font-weight: normal;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 4px 8px 6px;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 5px 8px;
    border-bottom: 1px solid #1c2128;
    vertical-align: middle;
    white-space: nowrap;
  }}
  tr.upcoming:hover td {{ background: var(--hover); }}
  tr.completed td {{ color: var(--muted); }}
  tr.live td {{ background: #1a1200; }}
  tr.live .home-prob, tr.live .draw-prob, tr.live .away-prob {{ font-weight: bold; }}
  .date-col {{ width: 52px; color: var(--muted); font-size: 11px; }}
  .team-col {{ max-width: 160px; overflow: hidden; text-overflow: ellipsis; }}
  .home-team {{ text-align: right; }}
  .away-team {{ text-align: left; }}
  .prob {{ text-align: right; width: 44px; font-variant-numeric: tabular-nums; }}
  .home-prob {{ color: var(--blue); }}
  .draw-prob {{ color: var(--yellow); }}
  .away-prob {{ color: var(--muted); }}
  tr.completed .home-prob,
  tr.completed .draw-prob,
  tr.completed .away-prob {{ color: inherit; }}
  .scores-col {{ color: var(--muted); font-size: 11px; min-width: 120px; }}
  .result-col {{ font-weight: bold; min-width: 70px; }}
  .verdict-wdl {{ width: 24px; text-align: center; }}
  .verdict-score {{ width: 24px; text-align: center; }}
  .verdict {{ }}
  /* winner table */
  .winner-table {{ max-width: 480px; }}
  .pct {{ text-align: right; width: 56px; color: var(--green); font-variant-numeric: tabular-nums; }}
  .bar-cell {{ padding-left: 12px; }}
  .bar {{ color: var(--green); font-size: 11px; letter-spacing: -1px; }}
  .implied {{ color: var(--muted); padding-left: 16px; }}
  .v-correct {{ color: var(--green); }}
  .v-partial {{ color: var(--yellow); }}
  .v-wrong {{ color: var(--red); }}
  /* accuracy box */
  .accuracy-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 24px;
  }}
  .acc-title {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  .acc-grid {{
    display: flex;
    gap: 32px;
    margin-bottom: 10px;
  }}
  .acc-stat {{ }}
  .acc-val {{
    font-size: 28px;
    font-weight: bold;
    color: var(--green);
    line-height: 1;
  }}
  .acc-label {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }}
  .acc-note {{
    font-size: 10px;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 8px;
    margin-top: 4px;
  }}
  footer {{
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
  }}
  .legend {{
    display: flex;
    gap: 20px;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
  }}
  .legend span {{ white-space: nowrap; }}
</style>
</head>
<body>
<header>
  <h1>&#x26BD; WC 2026 Forecaster</h1>
  <p class="subtitle">Bivariate Poisson &middot; Dixon-Coles correction &middot; 10k Monte Carlo &middot; auto-updated daily</p>
</header>

{accuracy_html}

<h2>All Fixtures — WC 2026 (from Jun 11)</h2>
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th style="text-align:right">Home</th>
      <th style="text-align:right">H%</th>
      <th style="text-align:right">D%</th>
      <th style="text-align:right">A%</th>
      <th>Away</th>
      <th>Top-3 Scores</th>
      <th>Result</th>
      <th title="Win/Draw/Loss direction">W/D/L</th>
      <th title="Score in top-3">Score</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
<div class="legend">
  <span>W/D/L: &#x2713; direction correct &nbsp; &#x2717; wrong</span>
  <span>Score: &#x2713; exact top-1 &nbsp; &#x7E; top-3 hit &nbsp; &#x2717; miss</span>
  <span style="color:var(--blue)">H%</span>
  <span style="color:var(--yellow)">D%</span>
  <span style="color:var(--muted)">A%</span>
  <span>[r] = retroactive</span>
</div>

<h2>Tournament Winner Odds (10k Monte Carlo)</h2>
<table class="winner-table">
  <thead>
    <tr>
      <th>Team</th>
      <th style="text-align:right">Win%</th>
      <th>Probability</th>
      <th>Implied odds</th>
    </tr>
  </thead>
  <tbody>{winner_rows}
  </tbody>
</table>

<footer>
  Generated {updated} &middot;
  <a href="https://github.com/VirajMishra1/worldcup-forecaster" style="color:var(--blue);text-decoration:none">github.com/VirajMishra1/worldcup-forecaster</a>
  &middot; also see <a href="https://github.com/VirajMishra1/polymath" style="color:var(--blue);text-decoration:none">Polymath</a>
</footer>

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
    print(f"Accuracy: {n_wdl}/{n_total} W/D/L ({_pct(n_wdl,n_total)})  |  {n_top3}/{n_total} top-3 score ({_pct(n_top3,n_total)})  |  {n_top1}/{n_total} exact top-1 ({_pct(n_top1,n_total)})")


if __name__ == "__main__":
    main()
