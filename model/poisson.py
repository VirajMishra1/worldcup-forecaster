"""Bivariate Poisson + Dixon-Coles correction, fit by MLE with time-decay weights."""
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

HALF_LIFE_YEARS = 1.5
MAX_GOALS = 10


@dataclass
class PoissonParams:
    attack: Dict[str, float] = field(default_factory=dict)
    defense: Dict[str, float] = field(default_factory=dict)
    rho: float = -0.13
    gamma: float = 0.0
    teams: List[str] = field(default_factory=list)


def _time_weight(date_series: pd.Series) -> np.ndarray:
    ref = pd.Timestamp.now()
    days = (ref - pd.to_datetime(date_series)).dt.days.values.astype(float)
    return np.exp(-np.log(2) * days / (HALF_LIFE_YEARS * 365.25))


def tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _match_ll(hg: int, ag: int, lam_h: float, lam_a: float, rho: float, w: float) -> float:
    t = tau(hg, ag, lam_h, lam_a, rho)
    if t <= 0:
        return -1e9
    return w * (
        math.log(t)
        + hg * math.log(lam_h) - lam_h - math.lgamma(hg + 1)
        + ag * math.log(lam_a) - lam_a - math.lgamma(ag + 1)
    )


def fit(df: pd.DataFrame, neutral: bool = True) -> PoissonParams:
    """
    Fit team attack/defense via MLE with time-decay + tournament weighting.
    neutral=True disables home advantage (correct for WC venues).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    tw = _time_weight(df["date"])
    tourney_w = df.get("tournament_weight", pd.Series(1.0, index=df.index))
    df["w"] = tw * tourney_w.values

    teams = sorted(set(df["home"]) | set(df["away"]))
    n = len(teams)
    tidx = {t: i for i, t in enumerate(teams)}

    home_idx = df["home"].map(tidx).values
    away_idx = df["away"].map(tidx).values
    hg_arr = df["home_goals"].values.astype(int)
    ag_arr = df["away_goals"].values.astype(int)
    w_arr = df["w"].values

    # Per-team regularization: teams with few matches pulled harder toward 0.
    # Shrinkage factor = 1 + 150/max(appearances, 50). Thin-data nations
    # (Curaçao ~96, New Zealand ~91) get ~3× stronger prior than top nations.
    appearances = np.zeros(n)
    for idx in home_idx:
        appearances[idx] += 1
    for idx in away_idx:
        appearances[idx] += 1
    reg_strength = 0.01 * (1.0 + 150.0 / np.maximum(appearances, 50.0))

    n_params = 2 * n + 1 + (0 if neutral else 1)

    def neg_ll(x: np.ndarray) -> float:
        atk = x[:n]
        dfc = x[n : 2 * n]
        rho_ = float(np.clip(x[2 * n], -0.99, 0.5))
        gam = float(x[2 * n + 1]) if not neutral else 0.0

        lam_h = np.exp(atk[home_idx] - dfc[away_idx] + gam)
        lam_a = np.exp(atk[away_idx] - dfc[home_idx])

        # Vectorised Poisson log-likelihood
        log_p = (
            hg_arr * np.log(lam_h) - lam_h - gammaln(hg_arr + 1)
            + ag_arr * np.log(lam_a) - lam_a - gammaln(ag_arr + 1)
        )

        # Vectorised Dixon-Coles tau correction
        tau_v = np.ones(len(hg_arr))
        m00 = (hg_arr == 0) & (ag_arr == 0)
        m10 = (hg_arr == 1) & (ag_arr == 0)
        m01 = (hg_arr == 0) & (ag_arr == 1)
        m11 = (hg_arr == 1) & (ag_arr == 1)
        tau_v[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho_
        tau_v[m10] = 1.0 + lam_a[m10] * rho_
        tau_v[m01] = 1.0 + lam_h[m01] * rho_
        tau_v[m11] = 1.0 - rho_
        tau_v = np.clip(tau_v, 1e-9, np.inf)

        total = float(np.sum(w_arr * (np.log(tau_v) + log_p)))
        reg = float(np.sum(reg_strength * (atk**2 + dfc**2)))
        return -(total - reg)

    x0 = np.zeros(n_params)
    x0[2 * n] = -0.1
    bounds = [(None, None)] * n_params
    bounds[0] = (0.0, 0.0)  # anchor first team attack for identifiability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            neg_ll, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7},
        )

    atk = result.x[:n]
    dfc = result.x[n : 2 * n]
    rho_fit = float(np.clip(result.x[2 * n], -0.99, 0.5))
    gam_fit = float(result.x[2 * n + 1]) if not neutral else 0.0

    return PoissonParams(
        attack={t: float(atk[i]) for t, i in tidx.items()},
        defense={t: float(dfc[i]) for t, i in tidx.items()},
        rho=rho_fit,
        gamma=gam_fit,
        teams=teams,
    )
