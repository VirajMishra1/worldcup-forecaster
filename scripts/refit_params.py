"""Refit model params and save to data/params_cache.json.

Includes completed WC 2026 results so knockout predictions update daily.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from model.poisson import fit, PoissonParams

DATA_DIR = Path(__file__).parent.parent / "data"
HIST_PATH = DATA_DIR / "historical_matches.parquet"
RESULTS_PATH = DATA_DIR / "results.parquet"
CACHE_PATH = DATA_DIR / "params_cache.json"

TEAM_NAME_MAP = {
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    hist = pd.read_parquet(HIST_PATH)
    logging.info("Historical matches: %d", len(hist))

    frames = [hist]
    if RESULTS_PATH.exists():
        live = pd.read_parquet(RESULTS_PATH)
        live = live.rename(columns={"home_goals": "home_goals", "away_goals": "away_goals"})
        live["home"] = live["home"].replace(TEAM_NAME_MAP)
        live["away"] = live["away"].replace(TEAM_NAME_MAP)
        live["tournament_weight"] = 3.0  # WC 2026 results weighted 3× — same tournament, highest signal
        live["date"] = pd.to_datetime(live["date"]).dt.tz_localize(None)
        live = live[["date", "home", "away", "home_goals", "away_goals", "tournament_weight"]]
        frames.append(live)
        logging.info("Adding %d completed WC 2026 results", len(live))

    df = pd.concat(frames, ignore_index=True)
    logging.info("Fitting on %d total matches...", len(df))
    params = fit(df, neutral=True)

    CACHE_PATH.write_text(json.dumps({
        "attack": params.attack,
        "defense": params.defense,
        "rho": params.rho,
        "gamma": params.gamma,
        "teams": params.teams,
    }))
    logging.info("Saved params → %s  (%d teams)", CACHE_PATH, len(params.teams))


if __name__ == "__main__":
    main()
