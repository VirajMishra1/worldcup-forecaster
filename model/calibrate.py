"""Isotonic regression calibration for 3-way (H/D/A) probabilities.

Fits one IsotonicRegression per outcome class on out-of-sample backtest predictions.
Normalises the three calibrated outputs so they sum to 1.

Usage:
    cal = fit_calibrator(backtest_df)   # df must have p_home/p_draw/p_away/outcome
    save_calibrator(cal)
    cal = load_calibrator()
    ph, pd, pa = calibrate(cal, p_home, p_draw, p_away)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path(__file__).parent.parent / "data"
CAL_PATH = DATA_DIR / "calibrator.joblib"

MIN_SAMPLES = 200


@dataclass
class Calibrator:
    home: IsotonicRegression
    draw: IsotonicRegression
    away: IsotonicRegression


def fit_calibrator(df: pd.DataFrame) -> Optional["Calibrator"]:
    """Fit isotonic calibration from walk-forward backtest results.

    df columns required: p_home, p_draw, p_away, outcome ('H'/'D'/'A').
    Returns None if fewer than MIN_SAMPLES rows.
    """
    if len(df) < MIN_SAMPLES:
        return None

    y_h = (df["outcome"] == "H").astype(float).values
    y_d = (df["outcome"] == "D").astype(float).values
    y_a = (df["outcome"] == "A").astype(float).values

    cal_h = IsotonicRegression(out_of_bounds="clip").fit(df["p_home"].values, y_h)
    cal_d = IsotonicRegression(out_of_bounds="clip").fit(df["p_draw"].values, y_d)
    cal_a = IsotonicRegression(out_of_bounds="clip").fit(df["p_away"].values, y_a)

    return Calibrator(home=cal_h, draw=cal_d, away=cal_a)


def calibrate(cal: "Calibrator", p_home: float, p_draw: float, p_away: float
              ) -> tuple[float, float, float]:
    """Apply calibration and renormalise to sum to 1."""
    ph = float(cal.home.predict([p_home])[0])
    pd_ = float(cal.draw.predict([p_draw])[0])
    pa = float(cal.away.predict([p_away])[0])
    total = ph + pd_ + pa
    if total <= 0:
        return p_home, p_draw, p_away
    return round(ph / total, 4), round(pd_ / total, 4), round(pa / total, 4)


def save_calibrator(cal: "Calibrator") -> None:
    dump(cal, CAL_PATH)


def load_calibrator() -> Optional["Calibrator"]:
    if not CAL_PATH.exists():
        return None
    return load(CAL_PATH)
