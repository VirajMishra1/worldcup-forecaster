"""Rest-days adjustment for tournament scheduling."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def rest_factor(team: str, match_date: str, df: pd.DataFrame, alpha: float = 0.03) -> float:
    """
    Returns multiplicative lambda adjustment based on rest days since last match.
    Baseline = 4 days. Each day above/below baseline adds/subtracts alpha.
    Max effect ±6% (2 days difference × 3%).
    """
    date = pd.Timestamp(match_date)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    prev = df[((df["home"] == team) | (df["away"] == team)) & (df["date"] < date)]
    if prev.empty:
        return 1.0
    last_match = prev["date"].max()
    rest_days = (date - last_match).days
    BASELINE = 4
    delta = np.clip(rest_days - BASELINE, -2, 2)
    return float(1.0 + alpha * delta)
