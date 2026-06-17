"""Generate full scoreline probability distribution for a match."""
import math

import numpy as np

from model.poisson import PoissonParams, tau, MAX_GOALS


def scoreline_grid(
    home: str,
    away: str,
    params: PoissonParams,
    is_neutral: bool = True,
    home_lineup_ratio: float = 1.0,
    away_lineup_ratio: float = 1.0,
) -> np.ndarray:
    """
    Returns (MAX_GOALS+1) x (MAX_GOALS+1) array: grid[i,j] = P(home=i, away=j).
    home_lineup_ratio = sum_mv(XI) / mean_mv(team_12mo); 1.0 = no adjustment.
    """
    a_home = params.attack.get(home, 0.0)
    d_home = params.defense.get(home, 0.0)
    a_away = params.attack.get(away, 0.0)
    d_away = params.defense.get(away, 0.0)
    gamma = 0.0 if is_neutral else params.gamma

    eps_home = 0.4 * math.log(max(0.5, min(1.5, home_lineup_ratio)))
    eps_away = 0.4 * math.log(max(0.5, min(1.5, away_lineup_ratio)))

    lam_h = math.exp(a_home - d_away + gamma + eps_home)
    lam_a = math.exp(a_away - d_home + eps_away)

    size = MAX_GOALS + 1
    grid = np.zeros((size, size))
    factorials = [math.factorial(k) for k in range(size)]

    for i in range(size):
        p_h = math.exp(-lam_h) * (lam_h**i) / factorials[i]
        for j in range(size):
            p_a = math.exp(-lam_a) * (lam_a**j) / factorials[j]
            t = tau(i, j, lam_h, lam_a, params.rho)
            grid[i, j] = max(0.0, t * p_h * p_a)

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid
