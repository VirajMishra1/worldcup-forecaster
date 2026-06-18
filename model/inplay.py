"""In-play win probability adjustment.

Given current score, minute, and red cards, integrates remaining-time
Poisson distributions to compute updated p_home / p_draw / p_away.
"""
import math
import numpy as np

from model.poisson import PoissonParams
from model.predict import scoreline_grid
from model.markets import derive_markets
from model.lineup import squad_ratio

MAX_GOALS = 8  # per team, for remaining-time grid


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam + k * math.log(lam + 1e-12) - math.lgamma(k + 1))


def inplay_probs(
    home: str,
    away: str,
    params: PoissonParams,
    home_goals: int,
    away_goals: int,
    minute: int,
    red_home: int = 0,
    red_away: int = 0,
) -> tuple[float, float, float]:
    """
    Returns (p_home_win, p_draw, p_away_win) given current in-play state.

    Approach: compute expected full-match lambdas from model, scale by
    remaining time fraction, adjust for red cards, then sum over all
    possible additional-goal combinations.
    """
    minute = max(1, min(minute, 90))
    remaining = (90 - minute) / 90.0

    # Full-match lambdas from pre-match model
    grid_full = scoreline_grid(
        home, away, params, is_neutral=True,
        home_lineup_ratio=squad_ratio(home),
        away_lineup_ratio=squad_ratio(away),
    )
    mkts_full = derive_markets(grid_full)
    lam_h_full = mkts_full.expected_home_goals
    lam_a_full = mkts_full.expected_away_goals

    # Remaining-time expected goals (scale by time fraction)
    lam_h = lam_h_full * remaining
    lam_a = lam_a_full * remaining

    # Red card penalty: each red card reduces that team's attack ~30%
    # and boosts opponent's ~20% for remaining time
    if red_home > 0:
        lam_h *= (0.7 ** red_home)
        lam_a *= (1.2 ** red_home)
    if red_away > 0:
        lam_a *= (0.7 ** red_away)
        lam_h *= (1.2 ** red_away)

    # Sum over additional goals
    p_home_win = p_draw = p_away_win = 0.0
    for dh in range(MAX_GOALS + 1):
        ph = _poisson_pmf(lam_h, dh)
        for da in range(MAX_GOALS + 1):
            pa = _poisson_pmf(lam_a, da)
            p = ph * pa
            total_h = home_goals + dh
            total_a = away_goals + da
            if total_h > total_a:
                p_home_win += p
            elif total_h == total_a:
                p_draw += p
            else:
                p_away_win += p

    # Normalise (floating point)
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

    return p_home_win, p_draw, p_away_win
