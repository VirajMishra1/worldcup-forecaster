"""Temperature scaling calibration for 3-way (H/D/A) match probabilities.

Learns a single temperature T by minimizing NLL on backtest predictions.
T > 1 softens overconfident outputs; T < 1 sharpens them.
Outputs always sum to 1 (softmax), unlike 3 independent isotonic regressors.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump, load
from scipy.optimize import minimize_scalar

DATA_DIR = Path(__file__).parent.parent / "data"
CAL_PATH = DATA_DIR / "calibrator.joblib"
MIN_SAMPLES = 200


@dataclass
class Calibrator:
    temperature: float


def _softmax_temp(p_arr: np.ndarray, T: float) -> np.ndarray:
    log_p = np.log(np.clip(p_arr, 1e-7, 1.0)) / T
    log_p -= log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def _nll(T: float, p_arr: np.ndarray, y_arr: np.ndarray) -> float:
    cal = _softmax_temp(p_arr, T)
    return -float(np.mean(np.sum(y_arr * np.log(np.clip(cal, 1e-7, 1.0)), axis=1)))


def fit_calibrator(df) -> Optional["Calibrator"]:
    if len(df) < MIN_SAMPLES:
        return None
    p = np.column_stack([df["p_home"].values, df["p_draw"].values, df["p_away"].values])
    y = np.zeros_like(p)
    outcomes = df["outcome"].values
    y[outcomes == "H", 0] = 1.0
    y[outcomes == "D", 1] = 1.0
    y[outcomes == "A", 2] = 1.0
    result = minimize_scalar(_nll, args=(p, y), bounds=(0.5, 3.0), method="bounded")
    return Calibrator(temperature=float(result.x))


def calibrate(cal: "Calibrator", p_home: float, p_draw: float, p_away: float
              ) -> tuple[float, float, float]:
    p = np.array([[p_home, p_draw, p_away]])
    out = _softmax_temp(p, cal.temperature)[0]
    return round(float(out[0]), 4), round(float(out[1]), 4), round(float(out[2]), 4)


def save_calibrator(cal: "Calibrator") -> None:
    dump(cal, CAL_PATH)


def load_calibrator() -> Optional["Calibrator"]:
    if not CAL_PATH.exists():
        return None
    return load(CAL_PATH)
