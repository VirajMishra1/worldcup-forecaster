"""Recent-form adjustment: last-5 competitive match goal ratio."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
_FRIENDLIES = {"Friendly", "friendly"}


def form_factors(home: str, away: str, match_date: str, df: pd.DataFrame,
                 n: int = 5, alpha: float = 0.15) -> tuple[float, float]:
    """
    Returns (home_factor, away_factor) — multiplicative adjustments to lambda.
    alpha=0.15 means at most ±15% from form alone.
    """
    date = pd.Timestamp(match_date)
    df = df[~df["tournament"].isin(_FRIENDLIES)].copy()
    df["date"] = pd.to_datetime(df["date"])

    def _team_form(team: str) -> float:
        rows = df[((df["home"] == team) | (df["away"] == team)) & (df["date"] < date)]
        rows = rows.sort_values("date").tail(n)
        if len(rows) < 2:
            return 1.0
        gf, ga = [], []
        for _, r in rows.iterrows():
            if r["home"] == team:
                gf.append(r["home_goals"]); ga.append(r["away_goals"])
            else:
                gf.append(r["away_goals"]); ga.append(r["home_goals"])
        # ratio vs global mean (historical average: ~1.2 goals/team/match)
        GLOBAL_MEAN = 1.2
        ratio = np.mean(gf) / GLOBAL_MEAN
        return float(np.clip(1.0 + alpha * (ratio - 1.0), 1 - alpha, 1 + alpha))

    return _team_form(home), _team_form(away)
