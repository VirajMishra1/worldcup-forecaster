"""Fetch upcoming WC 2026 fixtures from football-data.org."""
import os
import logging
from pathlib import Path

import httpx
import pandas as pd

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
BASE_URL = "https://api.football-data.org/v4"
DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURES_PATH = DATA_DIR / "fixtures_2026.parquet"

# football-data.org name → historical dataset name
TEAM_NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
}


def _norm(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_fixtures() -> pd.DataFrame:
    headers = {"X-Auth-Token": API_KEY} if API_KEY else {}
    # Fetch all matches, filter client-side — API comma-separated status filter is unreliable
    url = f"{BASE_URL}/competitions/WC/matches"
    r = httpx.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("matches", []):
        if m["status"] not in ("SCHEDULED", "TIMED"):
            continue
        hn = m["homeTeam"].get("name")
        an = m["awayTeam"].get("name")
        if not hn or not an:
            continue
        rows.append({
            "date": pd.Timestamp(m["utcDate"]),
            "home": _norm(hn),
            "away": _norm(an),
            "stage": m["stage"],
            "matchday": m.get("matchday"),
            "venue": m.get("venue", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_fixtures()
    df.to_parquet(FIXTURES_PATH, index=False)
    logging.info("Saved %d fixtures → %s", len(df), FIXTURES_PATH)


if __name__ == "__main__":
    main()
