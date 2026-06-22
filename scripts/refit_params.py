"""Refit model params and save to data/params_cache.json.

Includes completed WC 2026 results so knockout predictions update daily.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from model.poisson import fit

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

    # Boost past WC tournament matches — same competition, strongest signal after WC 2026 itself.
    # WC 2022 (most recent, 64-team transition era) > 2018 > 2014 > 2010.
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["tournament_weight"] = 1.0  # reset so stale values from parquet don't survive
    wc_mask = hist["tournament"] == "FIFA World Cup"
    yr = hist["date"].dt.year
    hist.loc[wc_mask & (yr >= 2022), "tournament_weight"] = 3.0
    hist.loc[wc_mask & (yr == 2018), "tournament_weight"] = 2.0
    hist.loc[wc_mask & (yr == 2014), "tournament_weight"] = 1.5
    hist.loc[wc_mask & (yr == 2010), "tournament_weight"] = 1.2
    logging.info(
        "WC boost applied: 2022×3.0, 2018×2.0, 2014×1.5, 2010×1.2 (%d matches)",
        wc_mask.sum(),
    )

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

    # Substitute WC 2022 xG for actual goals where available.
    # xG is a better signal of true quality than luck-adjusted scorelines.
    xg_path = DATA_DIR / "wc2022_xg.json"
    if xg_path.exists():
        xg_data = json.loads(xg_path.read_text())
        xg_by_date_teams: dict = {}
        for entry in xg_data.values():
            k = (entry["date"], entry["home"], entry["away"])
            xg_by_date_teams[k] = (entry["home_xg"], entry["away_xg"])
        # Cast to float so continuous xG values can be assigned (goals column is int64)
        df["home_goals"] = df["home_goals"].astype(float)
        df["away_goals"] = df["away_goals"].astype(float)
        substituted = 0
        for idx, row in df.iterrows():
            date_str = str(pd.to_datetime(row["date"]).date())
            k = (date_str, str(row["home"]), str(row["away"]))
            if k in xg_by_date_teams:
                h_xg, a_xg = xg_by_date_teams[k]
                df.at[idx, "home_goals"] = h_xg
                df.at[idx, "away_goals"] = a_xg
                substituted += 1
        if substituted:
            logging.info("Substituted xG for actual goals in %d WC 2022 matches", substituted)

    # Substitute WC 2026 xG for actual goals where available.
    # Keyed by "Home vs Away" (no date) — Kaggle dataset lags by ~1 day so date matching is unreliable.
    xg2026_path = DATA_DIR / "wc2026_xg.json"
    if xg2026_path.exists():
        xg2026 = json.loads(xg2026_path.read_text())
        xg_by_teams_2026: dict = {
            (v["home"], v["away"]): (v["home_xg"], v["away_xg"])
            for v in xg2026.values()
        }
        wc2026_start = pd.Timestamp("2026-06-11")
        df["home_goals"] = df["home_goals"].astype(float)
        df["away_goals"] = df["away_goals"].astype(float)
        sub2026 = 0
        for idx, row in df.iterrows():
            if pd.to_datetime(row["date"]) < wc2026_start:
                continue
            k = (str(row["home"]), str(row["away"]))
            if k in xg_by_teams_2026:
                h_xg, a_xg = xg_by_teams_2026[k]
                df.at[idx, "home_goals"] = h_xg
                df.at[idx, "away_goals"] = a_xg
                sub2026 += 1
        if sub2026:
            logging.info("Substituted xG for actual goals in %d WC 2026 matches", sub2026)

    as_of = pd.Timestamp.today().normalize()
    logging.info("Fitting on %d total matches (as_of %s)...", len(df), as_of.date())
    params = fit(df, neutral=True, as_of=as_of)

    CACHE_PATH.write_text(json.dumps({
        "attack": params.attack,
        "defense": params.defense,
        "rho": params.rho,
        "gamma": params.gamma,
        "teams": params.teams,
        "fit_at": pd.Timestamp.utcnow().isoformat(),
        "as_of": str(as_of.date()),
        "n_matches": len(df),
    }))
    logging.info("Saved params → %s  (%d teams)", CACHE_PATH, len(params.teams))


if __name__ == "__main__":
    main()
