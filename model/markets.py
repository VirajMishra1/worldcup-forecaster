"""Derive betting markets from a scoreline probability grid."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Markets:
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_25: float
    p_under_25: float
    p_btts: float
    p_btts_no: float
    exact_scores: dict
    expected_home_goals: float
    expected_away_goals: float


def derive_markets(grid: np.ndarray) -> Markets:
    n = grid.shape[0]
    p_home = p_draw = p_away = p_over = p_btts = exp_h = exp_a = 0.0
    exact: dict[str, float] = {}

    for i in range(n):
        for j in range(n):
            p = float(grid[i, j])
            if p == 0.0:
                continue
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j > 2:
                p_over += p
            if i > 0 and j > 0:
                p_btts += p
            exp_h += i * p
            exp_a += j * p
            exact[f"{i}-{j}"] = round(p, 5)

    top_exact = dict(sorted(exact.items(), key=lambda x: -x[1])[:10])

    return Markets(
        p_home_win=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away_win=round(p_away, 4),
        p_over_25=round(p_over, 4),
        p_under_25=round(1.0 - p_over, 4),
        p_btts=round(p_btts, 4),
        p_btts_no=round(1.0 - p_btts, 4),
        exact_scores=top_exact,
        expected_home_goals=round(exp_h, 3),
        expected_away_goals=round(exp_a, 3),
    )
