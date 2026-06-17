"""Fetch completed WC 2026 results from football-data.org. Append-only."""
import os
import logging
from pathlib import Path

import httpx
import pandas as pd

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results.parquet"

TEAM_NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
}


def _norm(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_results() -> pd.DataFrame:
    headers = {"X-Auth-Token": API_KEY} if API_KEY else {}
    url = f"{BASE_URL}/competitions/WC/matches?status=FINISHED"
    r = httpx.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("matches", []):
        rows.append({
            "date": pd.Timestamp(m["utcDate"]),
            "home": _norm(m["homeTeam"]["name"]),
            "away": _norm(m["awayTeam"]["name"]),
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "stage": m["stage"],
            "matchday": m.get("matchday"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    new = fetch_results()
    if RESULTS_PATH.exists():
        existing = pd.read_parquet(RESULTS_PATH)
        combined = pd.concat([existing, new]).drop_duplicates(subset=["date", "home", "away"])
    else:
        combined = new
    combined.to_parquet(RESULTS_PATH, index=False)
    logging.info("Saved %d results → %s", len(combined), RESULTS_PATH)


if __name__ == "__main__":
    main()
