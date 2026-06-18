"""Print all locked predictions as a sorted table.

Usage:
  python -m cli.schedule
  python -m cli.schedule --stage GROUP_STAGE
"""
import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show all locked WC 2026 predictions.")
    parser.add_argument("--stage", default=None, help="Filter by stage (e.g. GROUP_STAGE)")
    args = parser.parse_args()

    preds = pd.read_parquet(DATA_DIR / "predictions.parquet")
    preds["match_date"] = pd.to_datetime(preds["match_date"])
    preds = preds.sort_values("match_date").reset_index(drop=True)

    if args.stage:
        preds = preds[preds["stage"].str.upper() == args.stage.upper()]

    results_path = DATA_DIR / "results.parquet"
    results = None
    if results_path.exists():
        results = pd.read_parquet(results_path)
        results["date"] = pd.to_datetime(results["date"]).dt.date
        preds["_date"] = preds["match_date"].dt.date

    hdr = f"{'Date':<8}  {'Team 1':<26}  {'Team 2':<26}  {'T1%':>5}  {'D%':>5}  {'T2%':>5}  {'xG':>7}  {'Top-3 scores':<22}  Result"
    print(hdr)
    print("-" * len(hdr))

    for _, r in preds.iterrows():
        result_str = ""
        if results is not None:
            match = results[
                (results["date"] == r["match_date"].date())
                & (results["home"] == r["home"])
                & (results["away"] == r["away"])
            ]
            if not match.empty:
                m = match.iloc[0]
                result_str = f"{int(m['home_goals'])}-{int(m['away_goals'])}"

        s1 = r.get("top_scoreline") or ""
        s2 = r.get("top_2_scoreline") or ""
        s3 = r.get("top_3_scoreline") or ""
        scores_str = f"{s1}  {s2}  {s3}".strip() if (s2 or s3) else s1

        print(
            f"{r['match_date'].strftime('%b %d'):<8}  "
            f"{r['home']:<26}  {r['away']:<26}  "
            f"{r['p_home']:>4.0%}  {r['p_draw']:>4.0%}  {r['p_away']:>4.0%}  "
            f"{r['xg_home']:.1f}-{r['xg_away']:.1f}  "
            f"{scores_str:<22}  {result_str}"
        )


if __name__ == "__main__":
    main()
