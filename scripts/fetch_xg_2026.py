"""Fetch WC 2026 match-level xG from mominullptr/FIFA-World-Cup-2026-Dataset.

Saves data/wc2026_xg.json keyed by "Home vs Away" — used by refit_params.py
to substitute xG for actual goals, same as WC 2022 StatsBomb xG.

Source: https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset (public, daily updates)
"""
import json
import logging
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "wc2026_xg.json"

URL = (
    "https://raw.githubusercontent.com/mominullptr/"
    "FIFA-World-Cup-2026-Dataset/main/matches_detailed.csv"
)

# Kaggle dataset uses different team names than football-data.org API
TEAM_NAME_MAP = {
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
    "USA": "United States",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        resp = httpx.get(URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logging.warning("Failed to fetch WC 2026 xG data: %s — skipping", e)
        return

    df = pd.read_csv(StringIO(resp.text))
    completed = df[(df["status"] == "Completed")].dropna(subset=["home_xg", "away_xg"]).copy()

    completed["home_team_name"] = completed["home_team_name"].replace(TEAM_NAME_MAP)
    completed["away_team_name"] = completed["away_team_name"].replace(TEAM_NAME_MAP)

    xg_data: dict = {}
    for _, row in completed.iterrows():
        key = f"{row['home_team_name']} vs {row['away_team_name']}"
        xg_data[key] = {
            "home": row["home_team_name"],
            "away": row["away_team_name"],
            "home_xg": round(float(row["home_xg"]), 3),
            "away_xg": round(float(row["away_xg"]), 3),
        }

    OUT_PATH.write_text(json.dumps(xg_data, indent=2))
    logging.info("Saved %d WC 2026 xG records → %s", len(xg_data), OUT_PATH)


if __name__ == "__main__":
    main()
