"""Regenerate reports/track_record.md and splice it into README.md."""
import logging
from pathlib import Path

import pandas as pd

from backtest.metrics import log_loss_match, brier_score, aggregate_metrics

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
README = Path(__file__).parent.parent / "README.md"

MARKER_START = "<!-- TRACK_RECORD_START -->"
MARKER_END = "<!-- TRACK_RECORD_END -->"


def _outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def build_track_record() -> str:
    preds_path = DATA_DIR / "predictions.parquet"
    results_path = DATA_DIR / "results.parquet"

    if not preds_path.exists() or not results_path.exists():
        return "_No predictions yet._\n"

    preds = pd.read_parquet(preds_path)
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


def update_readme(track_record: str) -> None:
    content = README.read_text()
    start = content.find(MARKER_START)
    end = content.find(MARKER_END)
    if start == -1 or end == -1:
        content += f"\n{MARKER_START}\n{track_record}\n{MARKER_END}\n"
    else:
        content = (
            content[: start + len(MARKER_START)]
            + "\n"
            + track_record
            + "\n"
            + content[end:]
        )
    README.write_text(content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)
    track_record = build_track_record()
    (REPORTS / "track_record.md").write_text(track_record)
    update_readme(track_record)
    logging.info("Track record regenerated.")


if __name__ == "__main__":
    main()
