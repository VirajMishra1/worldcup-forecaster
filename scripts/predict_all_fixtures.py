"""Lock predictions for all upcoming WC fixtures. predictions.parquet is append-only."""
import json
import logging
from pathlib import Path

import pandas as pd

from model.poisson import PoissonParams
from model.predict import scoreline_grid
from model.markets import derive_markets

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "params_cache.json"
FIXTURES_PATH = DATA_DIR / "fixtures_2026.parquet"
PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"


def _load_params() -> PoissonParams:
    with open(CACHE_PATH) as f:
        d = json.load(f)
    return PoissonParams(attack=d["attack"], defense=d["defense"],
                         rho=d["rho"], gamma=d["gamma"], teams=d["teams"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not CACHE_PATH.exists():
        logging.error("No params cache. Run scripts/refit_params.py first.")
        return
    params = _load_params()
    logging.info("Loaded cached params (%d teams)", len(params.teams))

    if not FIXTURES_PATH.exists():
        logging.error("No fixtures file. Run scripts/fetch_fixtures.py first.")
        return

    fixtures = pd.read_parquet(FIXTURES_PATH)

    existing_keys: set[tuple] = set()
    if PREDICTIONS_PATH.exists():
        existing = pd.read_parquet(PREDICTIONS_PATH)
        existing_keys = set(
            zip(existing["home"], existing["away"], existing["match_date"].astype(str))
        )

    rows = []
    for _, fix in fixtures.iterrows():
        home, away = fix["home"], fix["away"]
        key = (home, away, str(fix["date"]))
        if key in existing_keys:
            continue
        if home not in params.teams or away not in params.teams:
            logging.warning("Unknown team(s): %s vs %s — skipping", home, away)
            continue

        grid = scoreline_grid(home, away, params, is_neutral=True)
        mkts = derive_markets(grid)

        rows.append({
            "locked_at": pd.Timestamp.now(),
            "match_date": fix["date"],
            "home": home,
            "away": away,
            "stage": fix.get("stage", ""),
            "p_home": mkts.p_home_win,
            "p_draw": mkts.p_draw,
            "p_away": mkts.p_away_win,
            "p_over_25": mkts.p_over_25,
            "p_btts": mkts.p_btts,
            "xg_home": mkts.expected_home_goals,
            "xg_away": mkts.expected_away_goals,
            "top_scoreline": max(mkts.exact_scores, key=mkts.exact_scores.get),
        })
        logging.info(
            "%s vs %s → H%.0f%% D%.0f%% A%.0f%%",
            home, away,
            mkts.p_home_win * 100, mkts.p_draw * 100, mkts.p_away_win * 100,
        )

    new_preds = pd.DataFrame(rows)
    if PREDICTIONS_PATH.exists() and len(rows) > 0:
        combined = pd.concat([existing, new_preds], ignore_index=True)
    elif len(rows) > 0:
        combined = new_preds
    else:
        logging.info("No new fixtures to predict.")
        return

    combined.to_parquet(PREDICTIONS_PATH, index=False)
    logging.info("Saved %d total predictions", len(combined))


if __name__ == "__main__":
    main()
