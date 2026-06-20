"""Fit isotonic calibration from walk-forward backtest results and save to data/calibrator.joblib.

Run after walk-forward backtest has been executed (reports/backtest_results.parquet must exist).
"""
import logging
from pathlib import Path

import pandas as pd

from model.calibrate import fit_calibrator, save_calibrator

REPORTS = Path(__file__).parent.parent / "reports"
BACKTEST_PATH = REPORTS / "backtest_results.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not BACKTEST_PATH.exists():
        logging.error("No backtest results. Run: uv run python -m cli.backtest first.")
        return

    df = pd.read_parquet(BACKTEST_PATH)
    logging.info("Fitting calibrator on %d backtest rows", len(df))

    cal = fit_calibrator(df)
    if cal is None:
        logging.error("Too few samples to fit calibrator (need ≥200).")
        return

    save_calibrator(cal)
    logging.info("Calibrator saved → data/calibrator.joblib")

    # Sanity check: log improvement in calibration error
    from model.calibrate import calibrate
    import numpy as np

    raw_err = []
    cal_err = []
    for _, r in df.iterrows():
        ph_c, pd_c, pa_c = calibrate(cal, r["p_home"], r["p_draw"], r["p_away"])
        actual_h = float(r["outcome"] == "H")
        actual_d = float(r["outcome"] == "D")
        actual_a = float(r["outcome"] == "A")
        raw_err.append(abs(r["p_home"] - actual_h) + abs(r["p_draw"] - actual_d) + abs(r["p_away"] - actual_a))
        cal_err.append(abs(ph_c - actual_h) + abs(pd_c - actual_d) + abs(pa_c - actual_a))

    logging.info("Mean absolute calibration error: raw=%.4f  calibrated=%.4f",
                 np.mean(raw_err), np.mean(cal_err))


if __name__ == "__main__":
    main()
