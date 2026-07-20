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
    "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲",
    "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Algeria": "🇩🇿", "Curaçao": "🇨🇼",
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
            if key in seen_keys:
                continue  # ponytail: guards a stale-duplicate row surviving upstream dedup
            seen_keys.add(key)
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            hp, ap = r.get("home_pens"), r.get("away_pens")
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
                "hp": int(hp) if hp is not None and hp == hp else None,
                "ap": int(ap) if ap is not None and ap == ap else None,
                "stage": str(r.get("stage", "")),
            })

    for _, p in preds.sort_values("match_date").iterrows():
        key = (p["home"], p["away"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_entries.append({
            "date": p["match_date"], "home": p["home"], "away": p["away"],
            "hg": None, "ag": None, "pred": p.to_dict(), "retro": False, "status": "upcoming",
            "stage": str(p.get("stage", "")),
        })

    n_locked = 0
    n_wdl = n_top1 = n_top3 = 0
    ko_locked = ko_wdl = ko_top1 = ko_top3 = 0
    for e in all_entries:
        if e["status"] != "completed" or e["pred"] is None or e.get("retro"):
            continue
        hg, ag = e["hg"], e["ag"]
        actual = f"{hg}-{ag}"
        p = e["pred"]
        ph, pd_s, pa = p["p_home"], p["p_draw"], p["p_away"]
        stage = e.get("stage", "")
        is_ko = stage and stage != "GROUP_STAGE"
        if is_ko:
            ph, pa = ph + pd_s / 2, pa + pd_s / 2
            pd_s = 0.0
        if hg == ag and e.get("hp") is not None:
            actual_wdl = "H" if e["hp"] > e["ap"] else "A"
        else:
            actual_wdl = "H" if hg > ag else ("D" if hg == ag else "A")
        pred_wdl = "H" if ph > pd_s and ph > pa else ("D" if pd_s > pa else "A")
        s1 = _clean_score(p.get("top_scoreline"))
        s2 = _clean_score(p.get("top_2_scoreline"))
        s3 = _clean_score(p.get("top_3_scoreline"))
        wdl_ok = pred_wdl == actual_wdl
        top1_ok = s1 == actual
        top3_ok = actual in (s1, s2, s3)
        n_locked += 1
        n_wdl += wdl_ok
        n_top1 += top1_ok
        n_top3 += top3_ok
        if is_ko:
            ko_locked += 1
            ko_wdl += wdl_ok
            ko_top1 += top1_ok
            ko_top3 += top3_ok

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

        if pred:
            ph = pred["p_home"]
            pd_ = pred["p_draw"]
            pa = pred["p_away"]
        else:
            ph = pd_ = pa = 0.0

        # ponytail: knockout matches can't actually end in a draw (extra time
        # + penalties decide it) — fold the draw probability into H/A "chance
        # to advance" for display so the bar doesn't show an impossible draw.
        stage = e.get("stage", pred.get("stage", "") if pred else "")
        if stage and stage != "GROUP_STAGE":
            ph, pa = ph + pd_ / 2, pa + pd_ / 2
            pd_ = 0.0

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
            # Knockout draws are decided by penalties, not a real "draw" outcome —
            # grade against who actually advanced, matching the H/A-only fold above.
            if hg == ag and e.get("hp") is not None:
                actual_wdl = "H" if e["hp"] > e["ap"] else "A"
            else:
                actual_wdl = "H" if hg > ag else ("D" if hg == ag else "A")
            pred_wdl = "H" if ph > pd_ and ph > pa else ("D" if pd_ > pa else "A")
            wdl_verdict = "✓" if pred_wdl == actual_wdl else "✗"
            score_verdict = "✓" if actual == s1 else ("~" if actual in (s2, s3) else "✗")

        retro_badge = ' <sup class="retro-badge">[r]</sup>' if e.get("retro") else ""
        pens_suffix = f" ({e['hp']}-{e['ap']} pens)" if e.get("hp") is not None else ""
        result_cell = actual + pens_suffix
        no_pred_class = "" if pred else " no-pred"

        hf = _flag(home)
        af = _flag(away)
        h_esc = _html.escape(home)
        a_esc = _html.escape(away)
        scores_esc = _html.escape(scores_str)
        result_esc = _html.escape(result_cell)

        # Mini prob bars (3 segments)
        def _mini_bar(p_h, p_d, p_a):
            w_h = round(p_h * 68)
            w_d = round(p_d * 68)
            w_a = 68 - w_h - w_d
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

    FINAL_STANDINGS = [
        ("Spain", "Winner"),
        ("Argentina", "Runner-up"),
        ("England", "Third place"),
        ("France", "Fourth place"),
    ]

    winner_rows = ""
    for rank, (team, result) in enumerate(FINAL_STANDINGS, 1):
        tf = _flag(team)
        model_prob = tourn.get("win", {}).get(team, 0)
        model_pct = f"{model_prob:.1%}"
        winner_rows += f"""
        <tr class="winner-row">
          <td class="rank-col">#{rank}</td>
          <td class="wteam-col">{tf} {_html.escape(team)}</td>
          <td class="wpct">{_html.escape(result)}</td>
          <td class="wbar-cell" style="font-size:12px;color:var(--muted)">{model_pct} final model odds</td>
        </tr>"""

    updated = "Tournament complete · Jul 19, 2026"
    n_results = len(results) if not results.empty else 0

    accuracy_html = ""
    if n_locked > 0:
        accuracy_html = f"""
<div class="section-heading" style="margin-bottom:14px">
  <span class="section-title">All Pre-Kickoff Predictions</span>
  <span class="section-badge">{n_locked} matches · locked before kickoff</span>
</div>
<div class="stat-cards">
  <div class="stat-card">
    <div class="stat-val">{_pct(n_wdl, n_locked)}</div>
    <div class="stat-frac">{n_wdl}/{n_locked}</div>
    <div class="stat-label">Win/Draw/Loss correct</div>
    <div class="stat-baseline">Random baseline: 33%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{_pct(n_top3, n_locked)}</div>
    <div class="stat-frac">{n_top3}/{n_locked}</div>
    <div class="stat-label">Score in top-3 predicted</div>
    <div class="stat-baseline">Random: ~5-8%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{_pct(n_top1, n_locked)}</div>
    <div class="stat-frac">{n_top1}/{n_locked}</div>
    <div class="stat-label">Top-1 exact score hit</div>
    <div class="stat-baseline">Random: ~2-3%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{n_locked}</div>
    <div class="stat-frac">{n_results} total WC results</div>
    <div class="stat-label">Pre-kickoff predictions</div>
    <div class="stat-baseline">All predictions locked before kickoff</div>
  </div>
</div>
<div class="section-heading" style="margin-bottom:14px; margin-top:24px">
  <span class="section-title">Knockout Stage Performance</span>
  <span class="section-badge">{ko_locked} matches · Round of 32 to Final</span>
</div>
<div class="stat-cards" style="margin-bottom:16px">
  <div class="stat-card">
    <div class="stat-val">{_pct(ko_wdl, ko_locked)}</div>
    <div class="stat-frac">{ko_wdl}/{ko_locked}</div>
    <div class="stat-label">Winner predicted correctly</div>
    <div class="stat-baseline">Random baseline: 50%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{_pct(ko_top3, ko_locked)}</div>
    <div class="stat-frac">{ko_top3}/{ko_locked}</div>
    <div class="stat-label">Score in top-3 predicted</div>
    <div class="stat-baseline">Random: ~5-8%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{_pct(ko_top1, ko_locked)}</div>
    <div class="stat-frac">{ko_top1}/{ko_locked}</div>
    <div class="stat-label">Top-1 exact score hit</div>
    <div class="stat-baseline">Random: ~2-3%</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{ko_locked}</div>
    <div class="stat-frac">R32 to Final</div>
    <div class="stat-label">Knockout matches predicted</div>
    <div class="stat-baseline">Draw prob folded into H/A</div>
  </div>
</div>
<div class="backtest-note">
  <strong>Backtest (5,518 matches, 2018-2023):</strong>
  log-loss 0.8961 vs 1.0986 random &middot; Brier 0.5265 vs 0.6667 random &middot; 59% W/D/L accuracy.
  Matches marked <span class="retro-inline">[r]</span> were computed after kickoff and excluded from stats above.
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC 2026 Forecaster</title>
<meta name="description" content="Probabilistic match predictions for FIFA World Cup 2026. Dixon-Coles model, locked before kickoff.">
<style>
  :root {{
    --bg: #0d1117;
    --fg: #e6edf3;
    --fg2: #c9d1d9;
    --muted: #8b949e;
    --muted2: #484f58;
    --green: #3fb950;
    --green-dim: #0f2d14;
    --red: #f85149;
    --yellow: #e3b341;
    --blue: #58a6ff;
    --border: #21262d;
    --border2: #30363d;
    --card: #161b22;
    --card2: #1c2128;
    --hover: #1f2937;
    --radius: 8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  a {{ color: var(--blue); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ── Centering wrapper ── */
  .page-wrap {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 28px;
  }}
  @media (max-width: 640px) {{
    .page-wrap {{ padding: 0 10px; }}
  }}

  /* ── Header ── */
  .site-header {{
    border-bottom: 1px solid var(--border2);
    padding: 20px 0 18px;
    margin-bottom: 32px;
  }}
  .header-inner {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .site-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--fg);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .site-subtitle {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 3px;
  }}
  .updated-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid #3fb95030;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
  }}

  /* ── Stat cards ── */
  .stat-cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }}
  @media (max-width: 640px) {{
    .stat-cards {{ grid-template-columns: repeat(2, 1fr); }}
    .team-col {{ min-width: 90px; max-width: 130px; font-size: 12px; }}
    td {{ padding: 7px 8px; }}
    th {{ padding: 8px 8px 7px; }}
    .bar-td,
    th:nth-child(4) {{ display: none; }}
    .site-title {{ font-size: 18px; }}
    .updated-pill {{ font-size: 11px; padding: 3px 9px; }}
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 16px 18px;
  }}
  .stat-val {{
    font-size: 32px;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -1px;
  }}
  .stat-frac {{
    font-size: 13px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
    line-height: 1;
  }}
  .stat-label {{
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }}
  .stat-baseline {{
    font-size: 11px;
    color: var(--muted2);
    margin-top: 2px;
  }}
  .backtest-note {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 16px;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 32px;
    display: flex;
    gap: 8px;
    align-items: flex-start;
    flex-wrap: wrap;
  }}
  .backtest-note strong {{ color: var(--fg2); font-weight: 600; }}

  /* ── Section heading ── */
  .section-heading {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .section-title {{
    font-size: 16px;
    font-weight: 600;
    color: var(--fg);
  }}
  .section-badge {{
    font-size: 11px;
    color: var(--muted);
    background: var(--card2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2px 10px;
  }}

  /* ── Fixtures table ── */
  .table-wrap {{
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    overflow-x: auto;
    margin-bottom: 12px;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    text-align: left;
    color: var(--muted);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 14px 9px;
    background: var(--card2);
    border-bottom: 1px solid var(--border2);
    white-space: nowrap;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
    white-space: nowrap;
    font-size: 13px;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tr.upcoming:hover td {{ background: var(--hover); }}
  tr.completed td {{ color: var(--muted); }}
  tr.no-pred td {{ opacity: 0.5; }}

  .date-col {{ width: 58px; color: var(--muted); font-size: 12px; }}
  .team-col {{ min-width: 140px; max-width: 200px; }}
  .home-team {{ text-align: right; }}
  .away-team {{ text-align: left; }}
  .prob {{
    text-align: right;
    width: 42px;
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    font-weight: 500;
  }}
  .home-prob {{ color: var(--blue); }}
  .away-prob {{ color: var(--muted); }}
  tr.completed .home-prob,
  tr.completed .away-prob {{ color: inherit; font-weight: normal; }}

  /* Mini 3-segment prob bar */
  .bar-td {{ padding: 0 8px; width: 84px; }}
  .mini-bar {{ display: flex; height: 6px; border-radius: 3px; overflow: hidden; gap: 1px; }}
  .mb-h {{ background: var(--blue); min-width: 2px; }}
  .mb-d {{ background: var(--yellow); min-width: 2px; }}
  .mb-a {{ background: var(--muted2); min-width: 2px; }}
  tr.completed .mb-h {{ background: #1e2d40; }}
  tr.completed .mb-d {{ background: #28240f; }}
  tr.completed .mb-a {{ background: #222; }}

  .scores-col {{ color: var(--muted); font-size: 12px; min-width: 120px; letter-spacing: 0.2px; font-variant-numeric: tabular-nums; }}
  .result-col {{ font-weight: 600; min-width: 54px; font-variant-numeric: tabular-nums; }}
  .verdict-wdl, .verdict-score {{ width: 26px; text-align: center; font-size: 14px; }}
  .v-correct {{ color: var(--green); }}
  .v-partial {{ color: var(--yellow); }}
  .v-wrong {{ color: var(--red); }}
  .retro-badge {{ color: var(--yellow); font-size: 9px; vertical-align: super; }}

  .legend {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--muted);
    padding: 10px 0 4px;
  }}
  .legend span {{ white-space: nowrap; }}
  .leg-h {{ color: var(--blue); }}
  .leg-d {{ color: var(--yellow); }}
  .retro-inline {{ color: var(--yellow); }}

  /* ── Winner odds ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
    margin-bottom: 40px;
  }}
  @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  .winner-table-wrap {{
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .winner-row:hover td {{ background: var(--hover); }}
  .rank-col {{ width: 34px; color: var(--muted2); font-size: 12px; padding-right: 4px; }}
  .wteam-col {{ font-size: 13px; min-width: 150px; }}
  .wpct {{
    text-align: right;
    width: 52px;
    color: var(--green);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 13px;
  }}
  .wbar-cell {{ padding-left: 12px; width: 120px; }}
  .wbar-bg {{ background: var(--card2); border-radius: 4px; height: 6px; width: 100px; overflow: hidden; }}
  .wbar-fill {{ background: linear-gradient(90deg, #3fb950, #2ea043); border-radius: 4px; height: 100%; }}
  .wimplied {{ color: var(--muted); padding-left: 12px; font-size: 12px; white-space: nowrap; }}

  .winner-explainer {{
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 18px 20px;
    font-size: 13px;
    color: var(--fg2);
    line-height: 1.7;
  }}
  .winner-explainer h3 {{
    font-size: 13px;
    font-weight: 600;
    color: var(--fg);
    margin-bottom: 10px;
  }}
  .winner-explainer p {{ margin-bottom: 10px; color: var(--muted); font-size: 12px; }}
  .winner-explainer p:last-child {{ margin-bottom: 0; }}
  .winner-explainer strong {{ color: var(--fg2); font-weight: 600; }}

  /* ── Footer ── */
  footer {{
    border-top: 1px solid var(--border);
    padding: 20px 0 32px;
    color: var(--muted2);
    font-size: 12px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: center;
  }}
  footer a {{ color: var(--muted); }}
  footer a:hover {{ color: var(--blue); }}
  .dot {{ color: var(--muted2); }}
</style>
</head>
<body>

<div class="page-wrap">

<header class="site-header">
  <div class="header-inner">
    <div>
      <div class="site-title">⚽ WC 2026 Forecaster</div>
      <div class="site-subtitle">
        Dixon-Coles score model &middot; predictions locked before kickoff &middot;
        <a href="https://github.com/VirajMishra1/worldcup-forecaster">GitHub</a>
      </div>
    </div>
    <div class="updated-pill">
      <span>⟳</span> {updated}
    </div>
  </div>
</header>

{accuracy_html}

<div class="section-heading">
  <span class="section-title">Match Predictions</span>
  <span class="section-badge">WC 2026 · from Jun 11</span>
</div>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th style="text-align:right">Home</th>
      <th style="text-align:right" title="Home win probability">H%</th>
      <th style="width:68px"></th>
      <th title="Away win probability">A%</th>
      <th>Away</th>
      <th>Top 3 scorelines</th>
      <th>Result</th>
      <th title="Win/draw/loss direction correct?">W/D/L</th>
      <th title="Exact score in top 3?">Score</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
</div>
<div class="legend">
  <span><span class="leg-h">▮</span> Home win</span>
  <span><span class="leg-d">▮</span> Draw</span>
  <span>▮ Away win</span>
  <span>&nbsp;·&nbsp; W/D/L: ✓ correct &nbsp; ✗ wrong</span>
  <span>&nbsp;·&nbsp; Score: ✓ exact top pick &nbsp; ~ in top 3 &nbsp; ✗ miss</span>
  <span>&nbsp;·&nbsp; <span class="retro-inline">[r]</span> computed after kickoff</span>
</div>

<div style="height:36px"></div>

<div class="section-heading" style="margin-top:4px">
  <span class="section-title">Final Standings</span>
  <span class="section-badge">FIFA World Cup 2026</span>
</div>
<div class="two-col">
  <div class="winner-table-wrap">
  <table>
    <thead>
      <tr>
        <th></th>
        <th>Team</th>
        <th>Result</th>
        <th style="padding-left:12px">Model prediction</th>
      </tr>
    </thead>
    <tbody>{winner_rows}
    </tbody>
  </table>
  </div>
  <div class="winner-explainer">
    <h3>How the model worked</h3>
    <p>
      The model ran <strong>10,000 bracket simulations</strong> of the full tournament
      using match probabilities from the Dixon-Coles model — the same model that generated
      the per-match predictions above. After each completed result, the model was refit and
      the simulations rerun.
    </p>
    <p>
      <strong>Model prediction</strong> shows each team's win probability from the final
      simulation run before the tournament ended.
    </p>
    <p>
      All predictions were <em>locked before kickoff</em> — no hindsight adjustments.
    </p>
  </div>
</div>

<footer>
  <span>Built by <a href="https://github.com/VirajMishra1">Viraj Mishra</a></span>
  <span class="dot">·</span>
  <a href="https://github.com/VirajMishra1/worldcup-forecaster">Source on GitHub</a>
  <span class="dot">·</span>
  <span>Dixon-Coles (1997) · MLE · temperature scaling · 13,779 training matches</span>
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
    print(f"Pre-kickoff accuracy: {n_wdl}/{n_locked} W/D/L ({_pct(n_wdl,n_locked)})  |  "
          f"{n_top3}/{n_locked} top-3 ({_pct(n_top3,n_locked)})  |  "
          f"{n_top1}/{n_locked} exact ({_pct(n_top1,n_locked)})")


if __name__ == "__main__":
    main()
