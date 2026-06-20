"""Recent-form adjustment: last-5 competitive match goal-difference, opponent-strength weighted."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
_FRIENDLIES = {"Friendly", "friendly"}
_SQUAD_RATIOS: dict | None = None


def _squad_ratios() -> dict:
    """Load opponent quality weights from squad_values.json (ratio vs WC mean)."""
    global _SQUAD_RATIOS
    if _SQUAD_RATIOS is None:
        path = DATA_DIR / "squad_values.json"
        if path.exists():
            data = json.loads(path.read_text())
            mean_val = float(data.get("_mean", 282))
            _SQUAD_RATIOS = {k: v / mean_val for k, v in data.items()
                             if not k.startswith("_") and isinstance(v, (int, float))}
        else:
            _SQUAD_RATIOS = {}
    return _SQUAD_RATIOS


def form_factors(home: str, away: str, match_date: str, df: pd.DataFrame,
                 n: int = 5, alpha: float = 0.15) -> tuple[float, float]:
    """
    Returns (home_factor, away_factor) — multiplicative adjustments to lambda.
    alpha=0.15 means at most ±15% from form alone.
    Each match is weighted by opponent squad strength (sqrt of ratio vs WC mean).
    Beating Argentina counts more than beating Haiti.
    """
    date = pd.Timestamp(match_date)
    if date.tzinfo is not None:
        date = date.tz_convert(None)
    df = df[~df["tournament"].isin(_FRIENDLIES)].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    ratios = _squad_ratios()

    def _team_form(team: str) -> float:
        rows = df[((df["home"] == team) | (df["away"] == team)) & (df["date"] < date)]
        rows = rows.sort_values("date").tail(n)
        if len(rows) < 2:
            return 1.0
        gf, ga, weights = [], [], []
        for _, r in rows.iterrows():
            if r["home"] == team:
                opp = r["away"]
                gf.append(r["home_goals"])
                ga.append(r["away_goals"])
            else:
                opp = r["home"]
                gf.append(r["away_goals"])
                ga.append(r["home_goals"])
            # sqrt dampens extremes: elite opp (ratio 4) → weight 2, minnow (ratio 0.06) → 0.25
            opp_ratio = ratios.get(opp, 1.0)
            weights.append(max(0.25, min(2.5, opp_ratio ** 0.5)))

        w = np.array(weights)
        w /= w.sum()
        gd_weighted = float(np.dot(w, np.array(gf) - np.array(ga)))
        GLOBAL_MEAN = 1.2
        return float(np.clip(1.0 + alpha * gd_weighted / GLOBAL_MEAN, 1 - alpha, 1 + alpha))

    return _team_form(home), _team_form(away)
