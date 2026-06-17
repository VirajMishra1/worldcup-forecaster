"""Walk-forward backtest: re-fit weekly, predict next window. No lookahead."""
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from model.poisson import fit
from model.predict import scoreline_grid
from model.markets import derive_markets
from backtest.metrics import aggregate_metrics, log_loss_match, brier_score, calibration_bins
from backtest.plots import reliability_diagram, log_loss_curve

DATA_DIR = Path(__file__).parent.parent / "data"


def _outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def run(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    refit_freq_days: int = 30,
    min_train_years: int = 5,
) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "historical_matches.parquet")
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    test_df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    records = []
    next_refit = start_dt
    params = None

    for _, row in test_df.iterrows():
        match_date = pd.Timestamp(row["date"])

        if match_date >= next_refit or params is None:
            train = df[df["date"] < match_date]
            if len(train) < 500:
                next_refit = match_date + timedelta(days=refit_freq_days)
                continue
            logging.info("Refitting at %s on %d matches", match_date.date(), len(train))
            params = fit(train, neutral=True)
            next_refit = match_date + timedelta(days=refit_freq_days)

        home, away = str(row["home"]), str(row["away"])
        if home not in params.teams or away not in params.teams:
            continue

        grid = scoreline_grid(home, away, params, is_neutral=True)
        mkts = derive_markets(grid)
        outcome = _outcome(int(row["home_goals"]), int(row["away_goals"]))

        records.append({
            "date": match_date,
            "home": home,
            "away": away,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "outcome": outcome,
            "p_home": mkts.p_home_win,
            "p_draw": mkts.p_draw,
            "p_away": mkts.p_away_win,
            "log_loss": log_loss_match(mkts.p_home_win, mkts.p_draw, mkts.p_away_win, outcome),
            "brier": brier_score(mkts.p_home_win, mkts.p_draw, mkts.p_away_win, outcome),
        })

    results = pd.DataFrame(records)
    if results.empty:
        logging.warning("No predictions generated — check date range and team coverage.")
        return results

    metrics = aggregate_metrics(results)
    logging.info(
        "Backtest %s–%s: log_loss=%.4f brier=%.4f accuracy=%.1f%% n=%d",
        start, end,
        metrics["log_loss"], metrics["brier"], metrics["accuracy"] * 100, metrics["n"],
    )

    cal = calibration_bins(results)
    reliability_diagram(cal)
    log_loss_curve(results)
    return results
