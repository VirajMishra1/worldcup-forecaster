"""Generate static HTML dashboard for WC 2026 predictions."""
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def _load_live() -> dict:
    p = DATA_DIR / "live_events.json"
    if not p.exists():
        return {}
    import json as _json
    try:
        return _json.loads(p.read_text())
    except Exception:
        return {}


def main() -> None:
    live = _load_live()  # {"{home} vs {away}": {"minute": 67, "home_goals": 1, "away_goals": 0, "red_cards": {"home":0,"away":0}, "p_home":0.72,"p_draw":0.18,"p_away":0.10}}
    preds = pd.read_parquet(DATA_DIR / "predictions.parquet")
    preds["match_date"] = pd.to_datetime(preds["match_date"], utc=True)
    preds = preds.sort_values("match_date")

    results_path = DATA_DIR / "results.parquet"
    results = pd.read_parquet(results_path) if results_path.exists() else pd.DataFrame()
    if not results.empty:
        results["date"] = pd.to_datetime(results["date"], utc=True).dt.date

    with open(DATA_DIR / "tournament.json") as f:
        tourn = json.load(f)
    win_odds = sorted(tourn.get("win", {}).items(), key=lambda x: -x[1])[:12]

    result_map: dict[tuple[str, str], str] = {}
    if not results.empty:
        for _, r in results.iterrows():
            key = (str(r["home"]), str(r["away"]))
            result_map[key] = f"{int(r['home_goals'])}-{int(r['away_goals'])}"

    # Build HTML rows
    rows_html = ""
    for _, p in preds.iterrows():
        key = (p["home"], p["away"])
        actual = result_map.get(key, "")
        status = "completed" if actual else "upcoming"
        date_str = p["match_date"].strftime("%b %d")

        s1 = p.get("top_scoreline") or ""
        s2 = p.get("top_2_scoreline") or ""
        s3 = p.get("top_3_scoreline") or ""
        # pd.isna check for NaN values stored as float
        s1 = "" if (s1 != s1 or s1 is None) else str(s1)
        s2 = "" if (s2 != s2 or s2 is None) else str(s2)
        s3 = "" if (s3 != s3 or s3 is None) else str(s3)

        parts = [x for x in [s1, s2, s3] if x]
        scores = "  ".join(parts) if parts else "—"

        correct = ""
        if actual:
            if actual == s1:
                correct = "✓"
            elif actual in [s2, s3]:
                correct = "~"
            else:
                correct = "✗"

        live_key = f"{p['home']} vs {p['away']}"
        live_data = live.get(live_key, {})
        ph = live_data.get("p_home", p["p_home"])
        pd_ = live_data.get("p_draw", p["p_draw"])
        pa = live_data.get("p_away", p["p_away"])
        live_score = ""
        if live_data:
            hg = live_data.get("home_goals", 0)
            ag = live_data.get("away_goals", 0)
            minute = live_data.get("minute", "?")
            live_score = f"🔴 {hg}-{ag} ({minute}')"
            status = "live"

        rows_html += f"""
        <tr class="{status}">
          <td class="date-col">{date_str}</td>
          <td class="team-col home-team">{p['home']}</td>
          <td class="prob home-prob">{ph:.0%}</td>
          <td class="prob draw-prob">{pd_:.0%}</td>
          <td class="prob away-prob">{pa:.0%}</td>
          <td class="team-col away-team">{p['away']}</td>
          <td class="scores-col">{scores}</td>
          <td class="result-col">{live_score or actual} <span class="verdict">{correct}</span></td>
        </tr>"""

    # Build winner odds rows
    winner_rows = ""
    for team, prob in win_odds:
        if prob > 0:
            implied = f"{1/prob - 1:.1f}:1"
        else:
            implied = "—"
        bar = _bar(prob)
        winner_rows += f"""
        <tr>
          <td>{team}</td>
          <td class="pct">{prob:.1%}</td>
          <td class="bar-cell"><span class="bar">{bar}</span></td>
          <td class="implied">{implied}</td>
        </tr>"""

    updated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")

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
  .scores-col {{ color: var(--muted); font-size: 11px; min-width: 100px; }}
  .result-col {{ font-weight: bold; min-width: 80px; }}
  .verdict {{ margin-left: 4px; }}
  tr.completed .verdict {{ }}
  /* winner table */
  .winner-table {{ max-width: 480px; }}
  .pct {{ text-align: right; width: 56px; color: var(--green); font-variant-numeric: tabular-nums; }}
  .bar-cell {{ padding-left: 12px; }}
  .bar {{ color: var(--green); font-size: 11px; letter-spacing: -1px; }}
  .implied {{ color: var(--muted); padding-left: 16px; }}
  /* verdict colors */
  .verdict {{ }}
  tr td .verdict:contains("✓") {{ color: var(--green); }}
  /* JS-applied classes are simpler */
  .v-correct {{ color: var(--green); }}
  .v-partial {{ color: var(--yellow); }}
  .v-wrong {{ color: var(--red); }}
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

<h2>All Fixtures</h2>
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
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
<div class="legend">
  <span>&#x2713; exact score hit</span>
  <span>&#x7E; top-3 hit</span>
  <span>&#x2717; miss</span>
  <span style="color:var(--blue)">H%</span>
  <span style="color:var(--yellow)">D%</span>
  <span style="color:var(--muted)">A%</span>
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
  &middot; also see <a href="https://github.com/VirajMishra1/polymath" style="color:var(--blue);text-decoration:none">Polymath</a> (Polymarket analytics terminal)
</footer>

<script>
  // Apply verdict colour classes post-render (CSS :contains not widely supported)
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
    print(f"Dashboard written to docs/index.html  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
