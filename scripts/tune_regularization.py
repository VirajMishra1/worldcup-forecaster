"""Cross-validate L2 regularization base strength."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model.poisson import fit
from model.predict import scoreline_grid
from model.markets import derive_markets

DATA_DIR = Path(__file__).parent.parent / "data"
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_loss_match(p_home, p_draw, p_away, outcome):
    p = {"H": p_home, "D": p_draw, "A": p_away}[outcome]
    return -np.log(max(p, 1e-9))


def _outcome(hg, ag):
    return "H" if hg > ag else ("D" if hg == ag else "A")


def evaluate(train_df, test_df, base_reg):
    test_sample = test_df.sample(min(200, len(test_df)), random_state=42)

    params = fit(train_df, neutral=True, base_reg=base_reg)

    losses = []
    for _, row in test_sample.iterrows():
        h, a = str(row["home"]), str(row["away"])
        if h not in params.teams or a not in params.teams:
            continue
        grid = scoreline_grid(h, a, params, is_neutral=True)
        mkts = derive_markets(grid)
        o = _outcome(int(row["home_goals"]), int(row["away_goals"]))
        losses.append(log_loss_match(mkts.p_home_win, mkts.p_draw, mkts.p_away_win, o))
    return float(np.mean(losses)) if losses else float("inf")


def main():
    df = pd.read_parquet(DATA_DIR / "historical_matches.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # Simple time split: train on pre-2021, test on 2021-2023
    train = df[df["date"] < "2021-01-01"]
    test = df[(df["date"] >= "2021-01-01") & (df["date"] < "2024-01-01")]

    logging.info(
        "Train: %d matches (pre-2021), Test: %d matches (2021-2023)",
        len(train),
        len(test),
    )

    grid = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]
    results = []
    for base in grid:
        ll = evaluate(train, test, base)
        logging.info("base_reg=%.3f  log_loss=%.4f", base, ll)
        results.append((base, ll))

    best_base, best_ll = min(results, key=lambda x: x[1])
    logging.info("\nBest base_reg=%.3f  log_loss=%.4f", best_base, best_ll)
    logging.info("Current default=0.010")

    out = {
        "grid": [{"base_reg": b, "log_loss": ll} for b, ll in results],
        "best_base_reg": best_base,
        "best_log_loss": best_ll,
        "current_default": 0.01,
    }
    out_path = Path(__file__).parent.parent / "reports" / "reg_tuning.json"
    out_path.write_text(json.dumps(out, indent=2))
    logging.info("Saved to reports/reg_tuning.json")


if __name__ == "__main__":
    main()
