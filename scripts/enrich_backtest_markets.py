"""
Enrich reports/backtest_results.parquet with p_over_25, p_btts,
actual_over_25, actual_btts by running current model params_cache.json
over each saved row.  Much faster than re-running walk_forward.

Run once:
  uv run python3 -m scripts.enrich_backtest_markets
"""
import json
import logging
from pathlib import Path

import pandas as pd

from model.poisson import PoissonParams
from model.predict import scoreline_grid
from model.markets import derive_markets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PARQUET = ROOT / "reports" / "backtest_results.parquet"
PARAMS_PATH = ROOT / "data" / "params_cache.json"


def load_params(path: Path) -> PoissonParams:
    with open(path) as f:
        d = json.load(f)
    return PoissonParams(
        attack=d["attack"],
        defense=d["defense"],
        rho=float(d["rho"]),
        gamma=float(d["gamma"]),
        teams=d["teams"],
    )


def main() -> None:
    df = pd.read_parquet(PARQUET)
    log.info("Loaded %d rows from %s", len(df), PARQUET)

    params = load_params(PARAMS_PATH)
    log.info("Loaded params for %d teams", len(params.teams))

    p_over_list, p_btts_list, top_scoreline_list = [], [], []
    skipped = 0

    for i, row in df.iterrows():
        home, away = str(row["home"]), str(row["away"])
        if home not in params.attack or away not in params.attack:
            p_over_list.append(float("nan"))
            p_btts_list.append(float("nan"))
            top_scoreline_list.append("")
            skipped += 1
            continue
        grid = scoreline_grid(home, away, params, is_neutral=True)
        mkts = derive_markets(grid)
        p_over_list.append(mkts.p_over_25)
        p_btts_list.append(mkts.p_btts)
        top_scoreline_list.append(
            sorted(mkts.exact_scores.items(), key=lambda x: -x[1])[0][0]
        )

        if (i + 1) % 500 == 0:
            log.info("  %d / %d rows processed", i + 1, len(df))

    df["p_over_25"] = p_over_list
    df["p_btts"] = p_btts_list
    df["top_scoreline"] = top_scoreline_list
    df["actual_over_25"] = (df["home_goals"] + df["away_goals"] > 2).astype(int)
    df["actual_btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    df.to_parquet(PARQUET, index=False)
    log.info(
        "Saved enriched parquet (%d rows, %d skipped) to %s",
        len(df), skipped, PARQUET,
    )


if __name__ == "__main__":
    main()
