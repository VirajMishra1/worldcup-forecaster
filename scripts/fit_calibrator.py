"""Fit temperature scaling calibration from walk-forward backtest and save to data/calibrator.joblib.

Uses the last 20% of backtest rows (by date) as a held-out evaluation set.
The first 80% is used for fitting so the reported improvement is not in-sample.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model.calibrate import calibrate, fit_calibrator, save_calibrator

REPORTS = Path(__file__).parent.parent / "reports"
BACKTEST_PATH = REPORTS / "backtest_results.parquet"
HOLDOUT_FRAC = 0.20


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not BACKTEST_PATH.exists():
        logging.error("No backtest results. Run: uv run python -m cli.backtest first.")
        return

    df = pd.read_parquet(BACKTEST_PATH).sort_values("date").reset_index(drop=True)
    logging.info("Backtest rows: %d", len(df))

    cutoff = int(len(df) * (1 - HOLDOUT_FRAC))
    train_df = df.iloc[:cutoff]
    eval_df = df.iloc[cutoff:]
    logging.info("Fitting on %d rows, evaluating on %d held-out rows", len(train_df), len(eval_df))

    cal = fit_calibrator(train_df)
    if cal is None:
        logging.error("Too few training samples to fit (need >= 200).")
        return

    logging.info("Temperature T = %.4f", cal.temperature)
    save_calibrator(cal)
    logging.info("Calibrator saved -> data/calibrator.joblib")

    eps = 1e-7
    p = eval_df[["p_home", "p_draw", "p_away"]].values
    y = np.zeros_like(p)
    outcomes = eval_df["outcome"].values
    y[outcomes == "H", 0] = 1.0
    y[outcomes == "D", 1] = 1.0
    y[outcomes == "A", 2] = 1.0

    raw_ll = -float(np.mean(np.sum(y * np.log(np.clip(p, eps, 1)), axis=1)))

    cal_rows = [calibrate(cal, r["p_home"], r["p_draw"], r["p_away"])
                for _, r in eval_df.iterrows()]
    cal_p = np.array(cal_rows)
    cal_ll = -float(np.mean(np.sum(y * np.log(np.clip(cal_p, eps, 1)), axis=1)))

    logging.info(
        "Held-out log-loss: raw=%.4f  calibrated=%.4f  (delta=%.4f)",
        raw_ll, cal_ll, raw_ll - cal_ll,
    )


if __name__ == "__main__":
    main()
