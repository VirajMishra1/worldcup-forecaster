"""Log-loss, Brier score, and calibration bin computation."""
import math
from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-9


def log_loss_match(p_home: float, p_draw: float, p_away: float, outcome: str) -> float:
    """outcome: 'H', 'D', 'A'"""
    p = {"H": p_home, "D": p_draw, "A": p_away}[outcome]
    return -math.log(max(p, EPS))


def brier_score(p_home: float, p_draw: float, p_away: float, outcome: str) -> float:
    targets = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[outcome]
    probs = (p_home, p_draw, p_away)
    return sum((p - t) ** 2 for p, t in zip(probs, targets))


def aggregate_metrics(predictions: pd.DataFrame) -> dict:
    """
    predictions: DataFrame with p_home, p_draw, p_away, outcome (H/D/A).
    Returns dict: log_loss, brier, accuracy, n.
    """
    ll, bs, acc = [], [], []
    for _, row in predictions.iterrows():
        o = row["outcome"]
        ph, pd_, pa = float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])
        ll.append(log_loss_match(ph, pd_, pa, o))
        bs.append(brier_score(ph, pd_, pa, o))
        pred = "H" if ph >= pd_ and ph >= pa else ("D" if pd_ >= pa else "A")
        acc.append(1 if pred == o else 0)
    return {
        "log_loss": round(float(np.mean(ll)), 4),
        "brier": round(float(np.mean(bs)), 4),
        "accuracy": round(float(np.mean(acc)), 4),
        "n": len(ll),
    }


def calibration_bins(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: mean_predicted vs mean_actual per probability bin."""
    rows = []
    for col, val in [("p_home", "H"), ("p_draw", "D"), ("p_away", "A")]:
        df = predictions.copy()
        df["pred"] = df[col].astype(float)
        df["actual"] = (df["outcome"] == val).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        df["bin"] = pd.cut(df["pred"], bins=bins, labels=False, include_lowest=True)
        for b, grp in df.groupby("bin", observed=True):
            if len(grp) >= 3:
                rows.append({
                    "market": val,
                    "bin": b,
                    "mean_predicted": float(grp["pred"].mean()),
                    "mean_actual": float(grp["actual"].mean()),
                    "count": len(grp),
                })
    return pd.DataFrame(rows)
