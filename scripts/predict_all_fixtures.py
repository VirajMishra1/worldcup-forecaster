"""Lock predictions for all upcoming WC fixtures. predictions.parquet is append-only."""
import json
import logging
from pathlib import Path

import pandas as pd

from model.calibrate import calibrate, load_calibrator
from model.form import form_factors
from model.lineup import squad_ratio
from model.poisson import PoissonParams
from model.predict import scoreline_grid
from model.markets import derive_markets
from model.rest import rest_factor

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

    cal = load_calibrator()
    if cal:
        logging.info("Isotonic calibrator loaded")
    else:
        logging.info("No calibrator found — predictions will use raw model output")

    if not FIXTURES_PATH.exists():
        logging.error("No fixtures file. Run scripts/fetch_fixtures.py first.")
        return

    fixtures = pd.read_parquet(FIXTURES_PATH)

    hist = pd.read_parquet(DATA_DIR / "historical_matches.parquet")
    results_path = DATA_DIR / "results.parquet"
    if results_path.exists():
        results = pd.read_parquet(results_path)
        all_matches = pd.concat([hist, results], ignore_index=True)
    else:
        all_matches = hist

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

        hf, af = form_factors(home, away, str(fix["date"]), all_matches)
        hr = rest_factor(home, str(fix["date"]), all_matches)
        ar = rest_factor(away, str(fix["date"]), all_matches)

        grid = scoreline_grid(
            home, away, params, is_neutral=True,
            home_lineup_ratio=squad_ratio(home) * hf * hr,
            away_lineup_ratio=squad_ratio(away) * af * ar,
        )
        mkts = derive_markets(grid)

        ph = mkts.p_home_win
        pd_ = mkts.p_draw
        pa = mkts.p_away_win
        if cal:
            ph, pd_, pa = calibrate(cal, ph, pd_, pa)

        top3 = sorted(mkts.exact_scores.items(), key=lambda x: -x[1])[:3]
        rows.append({
            "locked_at": pd.Timestamp.now(),
            "match_date": fix["date"],
            "home": home,
            "away": away,
            "stage": fix.get("stage", ""),
            "p_home": ph,
            "p_draw": pd_,
            "p_away": pa,
            "p_over_25": mkts.p_over_25,
            "p_btts": mkts.p_btts,
            "xg_home": mkts.expected_home_goals,
            "xg_away": mkts.expected_away_goals,
            "top_scoreline": top3[0][0],
            "top_scoreline_p": round(top3[0][1], 4),
            "top_2_scoreline": top3[1][0] if len(top3) > 1 else "",
            "top_3_scoreline": top3[2][0] if len(top3) > 2 else "",
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
