"""Regenerate reports/track_record.md and splice it into README.md."""
import json
import logging
from pathlib import Path

import pandas as pd

from backtest.metrics import log_loss_match, brier_score, aggregate_metrics

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
README = Path(__file__).parent.parent / "README.md"

MARKER_START = "<!-- TRACK_RECORD_START -->"
MARKER_END = "<!-- TRACK_RECORD_END -->"

WINNER_MARKER_START = "<!-- WINNER_ODDS_START -->"
WINNER_MARKER_END = "<!-- WINNER_ODDS_END -->"

_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bolivia": "🇧🇴", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Cape Verde": "🇨🇻",
    "Chile": "🇨🇱", "Colombia": "🇨🇴", "Costa Rica": "🇨🇷",
    "Croatia": "🇭🇷", "Czech Republic": "🇨🇿", "Denmark": "🇩🇰",
    "DR Congo": "🇨🇩", "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪",
    "Algeria": "🇩🇿", "Curaçao": "🇨🇼",
    "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Honduras": "🇭🇳",
    "Indonesia": "🇮🇩", "Iran": "🇮🇷", "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Mali": "🇲🇱",
    "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Nigeria": "🇳🇬", "Norway": "🇳🇴",
    "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Peru": "🇵🇪",
    "Poland": "🇵🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
    "Serbia": "🇷🇸", "Slovakia": "🇸🇰", "South Africa": "🇿🇦",
    "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭", "Tanzania": "🇹🇿", "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "United States": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿", "Venezuela": "🇻🇪",
}


def _outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def build_track_record(include_table: bool = True) -> str:
    preds_path = DATA_DIR / "predictions.parquet"
    results_path = DATA_DIR / "results.parquet"

    if not preds_path.exists() or not results_path.exists():
        return "_No predictions yet._\n"

    preds = pd.read_parquet(preds_path)
    retroactive = (
        preds[preds["prediction_type"] == "retroactive"]
        if "prediction_type" in preds.columns else pd.DataFrame()
    )
    if "prediction_type" in preds.columns:
        preds = preds[preds["prediction_type"] == "locked"]
    results = pd.read_parquet(results_path)
    results["outcome"] = results.apply(
        lambda r: _outcome(int(r["home_goals"]), int(r["away_goals"])), axis=1
    )
    preds["match_date"] = pd.to_datetime(preds["match_date"]).dt.date
    results["date"] = pd.to_datetime(results["date"]).dt.date

    joined = preds.merge(
        results[["date", "home", "away", "home_goals", "away_goals", "outcome"]],
        left_on=["match_date", "home", "away"],
        right_on=["date", "home", "away"],
        how="inner",
    )

    if len(joined) == 0:
        return "_No completed matches yet._\n"

    joined["log_loss"] = joined.apply(
        lambda r: log_loss_match(r["p_home"], r["p_draw"], r["p_away"], r["outcome"]), axis=1
    )
    joined["brier"] = joined.apply(
        lambda r: brier_score(r["p_home"], r["p_draw"], r["p_away"], r["outcome"]), axis=1
    )
    joined["pred"] = joined.apply(
        lambda r: "H" if r["p_home"] >= r["p_draw"] and r["p_home"] >= r["p_away"]
        else ("D" if r["p_draw"] >= r["p_away"] else "A"),
        axis=1,
    )
    joined["correct"] = joined["pred"] == joined["outcome"]
    metrics = aggregate_metrics(joined)

    lines = [
        f"## Live Track Record ({len(joined)} matches)\n",
        "| Metric | Value | Random baseline |",
        "|--------|-------|-----------------|",
        f"| W/D/L accuracy | {metrics['accuracy']:.1%} | 33.3% |",
        f"| Log-loss | {metrics['log_loss']:.4f} | 1.0986 |",
        f"| Brier score | {metrics['brier']:.4f} | 0.6667 |",
    ]

    if len(retroactive) > 0:
        teams = retroactive.apply(
            lambda r: f"{r['home']} vs {r['away']}", axis=1
        ).tolist()
        team_list = ", ".join(teams)
        lines.append(
            f"\n_{len(retroactive)} predictions generated after kickoff "
            f"({team_list}) are excluded from this table. "
            f"Visible with an [r] badge on the "
            f"[live dashboard](https://virajmishra1.github.io/worldcup-forecaster/)._"
        )

    if not include_table:
        lines.append(
            "\n_Per-match breakdown on the "
            "[live dashboard](https://virajmishra1.github.io/worldcup-forecaster/)._"
        )
        return "\n".join(lines) + "\n"

    lines += [
        "",
        "### Per-match predictions\n",
        "| Date | Match | H% / D% / A% | Result | LL | ✓ |",
        "|------|-------|--------------|--------|----|---|",
    ]

    for _, r in joined.sort_values("match_date").iterrows():
        result_str = {"H": r["home"], "D": "Draw", "A": r["away"]}[r["outcome"]]
        score_str = f"{int(r['home_goals'])}-{int(r['away_goals'])}"
        lines.append(
            f"| {r['match_date']} | {r['home']} vs {r['away']} "
            f"| {r['p_home']:.0%}/{r['p_draw']:.0%}/{r['p_away']:.0%} "
            f"| {result_str} ({score_str}) | {r['log_loss']:.3f} | {'✓' if r['correct'] else '✗'} |"
        )

    return "\n".join(lines) + "\n"


def build_winner_odds() -> str:
    tournament_path = DATA_DIR / "tournament.json"
    if not tournament_path.exists():
        return "## WC 2026 Winner Odds\n\n_Not yet generated._\n"

    with open(tournament_path) as f:
        t = json.load(f)

    win: dict = t.get("win", {})
    updated_at: str = t.get("updated_at", "")
    top = sorted(win.items(), key=lambda x: -x[1])[:12]

    lines = [
        "## WC 2026 Winner Odds\n",
        "10,000 Monte Carlo bracket simulations, updated after every result.\n",
        "Implied odds = 1/p − 1. At 20% win probability, fair implied odds are 4.0:1"
        " (a £10 bet at fair value returns £50 total).\n",
        "| Team | Win % | Implied odds |",
        "|------|-------|--------------|",
    ]
    for team, prob in top:
        flag = _FLAGS.get(team, "🏳")
        implied = f"{1/prob - 1:.1f}:1" if prob > 0 else "—"
        lines.append(f"| {flag} {team} | {prob:.1%} | {implied} |")

    if updated_at:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(updated_at)
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = updated_at[:10]
        n_results = (
            len(pd.read_parquet(DATA_DIR / "results.parquet"))
            if (DATA_DIR / "results.parquet").exists() else 0
        )
        lines.append(f"\n_{n_results} completed WC 2026 results included. Updated {date_str}._")

    return "\n".join(lines) + "\n"


def _splice_section(content: str, start_marker: str, end_marker: str, body: str) -> str:
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start == -1 or end == -1:
        content += f"\n{start_marker}\n{body}\n{end_marker}\n"
    else:
        content = (
            content[: start + len(start_marker)]
            + "\n"
            + body
            + "\n"
            + content[end:]
        )
    return content


def update_readme(readme_track_record: str) -> None:
    content = README.read_text()
    content = _splice_section(content, WINNER_MARKER_START, WINNER_MARKER_END, build_winner_odds())
    content = _splice_section(content, MARKER_START, MARKER_END, readme_track_record)
    README.write_text(content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)
    full_report = build_track_record(include_table=True)
    readme_section = build_track_record(include_table=False)
    (REPORTS / "track_record.md").write_text(full_report)
    update_readme(readme_section)
    logging.info("Track record regenerated.")


if __name__ == "__main__":
    main()
