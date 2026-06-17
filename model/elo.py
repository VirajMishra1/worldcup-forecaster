"""Time-decay Elo ratings for national teams. Half-life = 2 years."""
import math
from datetime import date
from typing import Dict

import pandas as pd

INITIAL_ELO = 1500.0
HALF_LIFE_DAYS = 730
HOME_ADVANTAGE = 100


def _tournament_k(tournament: str) -> float:
    t = str(tournament)
    if "World Cup" in t and "qualif" not in t.lower():
        return 60.0
    if any(x in t for x in ["Euro", "Copa América", "Nations", "Gold Cup", "Asian Cup"]):
        return 50.0
    if "qualif" in t.lower() or "Qualifier" in t:
        return 40.0
    return 20.0


def _expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _goal_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _time_weight(match_date: date, reference_date: date) -> float:
    days = (reference_date - match_date).days
    return math.exp(-math.log(2) * days / HALF_LIFE_DAYS)


def compute_elo(df: pd.DataFrame, reference_date: date | None = None) -> Dict[str, float]:
    """Return {team: elo} computed from match history with time-decay weights."""
    if reference_date is None:
        reference_date = date.today()

    ratings: Dict[str, float] = {}

    for _, row in df.iterrows():
        home, away = row["home"], row["away"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        tournament = str(row.get("tournament", "Friendly"))
        match_date = pd.Timestamp(row["date"]).date()

        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        is_neutral = bool(row.get("neutral", True))
        ha = 0.0 if is_neutral else HOME_ADVANTAGE

        exp_home = _expected(r_home + ha, r_away)
        score_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        k = _tournament_k(tournament)
        gm = _goal_multiplier(hg - ag)
        tw = _time_weight(match_date, reference_date)

        delta = k * gm * tw * (score_home - exp_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return ratings


def elo_win_probability(elo_home: float, elo_away: float) -> tuple[float, float, float]:
    """(p_home_win, p_draw, p_away_win) from Elo difference."""
    exp_home = _expected(elo_home, elo_away)
    p_draw = max(0.10, min(0.35, 0.25 * (1.0 - abs(exp_home - 0.5) * 2)))
    p_home = exp_home * (1.0 - p_draw)
    p_away = (1.0 - exp_home) * (1.0 - p_draw)
    return p_home, p_draw, p_away
