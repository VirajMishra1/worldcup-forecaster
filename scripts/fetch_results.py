"""Fetch completed WC 2026 results from football-data.org. Append-only."""
import os
import logging
from pathlib import Path

import httpx
import pandas as pd

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
if not API_KEY:
    _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            if _line.startswith("FOOTBALL_DATA_API_KEY="):
                API_KEY = _line.split("=", 1)[1].strip()
                break
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
        score = m["score"]
        # ponytail: football-data.org's fullTime = regularTime + penalty count for
        # PENALTY_SHOOTOUT matches (not goals). regularTime is the 90-min score and
        # extraTime is goals scored *during* ET only (not cumulative) — sum them to
        # get the actual AET score before penalties.
        on_pens = score.get("duration") == "PENALTY_SHOOTOUT"
        if on_pens:
            et = score.get("extraTime") or {"home": 0, "away": 0}
            home_goals = score["regularTime"]["home"] + (et.get("home") or 0)
            away_goals = score["regularTime"]["away"] + (et.get("away") or 0)
        else:
            home_goals = score["fullTime"]["home"]
            away_goals = score["fullTime"]["away"]
        pens = score.get("penalties") or {}
        rows.append({
            "date": pd.Timestamp(m["utcDate"]),
            "home": _norm(m["homeTeam"]["name"]),
            "away": _norm(m["awayTeam"]["name"]),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_pens": pens.get("home") if on_pens else None,
            "away_pens": pens.get("away") if on_pens else None,
            "stage": m["stage"],
            "matchday": m.get("matchday"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    new = fetch_results()
    if RESULTS_PATH.exists():
        existing = pd.read_parquet(RESULTS_PATH)
        # ponytail: dedupe on (home, away) only, not date — two teams play at
        # most once in this tournament, but the API can correct a fixture's
        # kickoff time between fetches, which would otherwise look like a
        # second, separate match and duplicate the row.
        combined = pd.concat([existing, new]).drop_duplicates(subset=["home", "away"], keep="last")
    else:
        combined = new
    combined.to_parquet(RESULTS_PATH, index=False)
    logging.info("Saved %d results → %s", len(combined), RESULTS_PATH)


if __name__ == "__main__":
    main()
