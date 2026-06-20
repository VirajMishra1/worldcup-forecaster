"""Minimal correctness tests: MLE convergence, scoreline grid, market derivation."""
import numpy as np
import pandas as pd
import pytest

from model.poisson import fit, tau
from model.predict import scoreline_grid
from model.markets import derive_markets


@pytest.fixture(scope="module")
def synthetic_df():
    rng = np.random.default_rng(42)
    teams = ["Brazil", "France", "Germany", "Argentina", "Spain", "England"]
    rows = []
    for _ in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.poisson(1.4))
        ag = int(rng.poisson(1.1))
        rows.append({"date": "2022-06-01", "home": h, "away": a,
                     "home_goals": hg, "away_goals": ag})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_params(synthetic_df):
    return fit(synthetic_df, neutral=True)


def test_tau_special_cases():
    lh, la, rho = 1.0, 1.0, -0.1
    assert tau(0, 0, lh, la, rho) == pytest.approx(1.0 - lh * la * rho)
    assert tau(1, 0, lh, la, rho) == pytest.approx(1.0 + la * rho)
    assert tau(0, 1, lh, la, rho) == pytest.approx(1.0 + lh * rho)
    assert tau(1, 1, lh, la, rho) == pytest.approx(1.0 - rho)
    assert tau(2, 2, lh, la, rho) == 1.0


def test_fit_returns_all_teams(fitted_params, synthetic_df):
    teams = set(synthetic_df["home"]) | set(synthetic_df["away"])
    assert teams == set(fitted_params.teams)
    assert all(t in fitted_params.attack for t in teams)
    assert all(t in fitted_params.defense for t in teams)


def test_rho_in_bounds(fitted_params):
    assert -0.99 <= fitted_params.rho <= 0.5, "rho must stay within clipped bounds"


def test_scoreline_grid_sums_to_one(fitted_params):
    grid = scoreline_grid("Brazil", "France", fitted_params, is_neutral=True)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)
    assert grid.min() >= 0.0


def test_markets_sum_to_one(fitted_params):
    grid = scoreline_grid("Germany", "Argentina", fitted_params, is_neutral=True)
    mkts = derive_markets(grid)
    total = mkts.p_home_win + mkts.p_draw + mkts.p_away_win
    assert total == pytest.approx(1.0, abs=1e-6)


def test_markets_btts_and_ou_bounds(fitted_params):
    grid = scoreline_grid("Spain", "England", fitted_params, is_neutral=True)
    mkts = derive_markets(grid)
    assert 0.0 <= mkts.p_btts <= 1.0
    assert 0.0 <= mkts.p_over_25 <= 1.0
    assert mkts.p_over_25 + mkts.p_under_25 == pytest.approx(1.0, abs=1e-6)


def test_as_of_date_reproducibility(synthetic_df):
    as_of = pd.Timestamp("2024-01-01")
    p1 = fit(synthetic_df, neutral=True, as_of=as_of)
    p2 = fit(synthetic_df, neutral=True, as_of=as_of)
    assert p1.attack == p2.attack
    assert p1.rho == pytest.approx(p2.rho)
