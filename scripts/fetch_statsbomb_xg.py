"""Fetch WC 2022 match-level xG from StatsBomb open data and save to data/wc2022_xg.json.

StatsBomb open data: https://github.com/statsbomb/open-data (MIT licence)
WC 2022 = competition_id=43, season_id=106

Each match's events are fetched, shot xG values summed per team.
Stored as {match_key: {home_xg: float, away_xg: float}} where
match_key = "HomeTeam vs AwayTeam YYYY-MM-DD".
"""
import json
import logging
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "wc2022_xg.json"

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43
SEASON_ID = 106   # WC 2022


def _get(client: httpx.Client, url: str) -> dict | list:
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with httpx.Client() as client:
        matches_url = f"{BASE}/matches/{COMPETITION_ID}/{SEASON_ID}.json"
        logging.info("Fetching match list from %s", matches_url)
        matches = _get(client, matches_url)
        logging.info("Found %d matches", len(matches))

        xg_data: dict = {}
        for i, m in enumerate(matches, 1):
            match_id = m["match_id"]
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            date_str = m["match_date"]

            events_url = f"{BASE}/events/{match_id}.json"
            try:
                events = _get(client, events_url)
            except Exception as e:
                logging.warning("Failed to fetch events for match %d: %s", match_id, e)
                continue

            home_xg = sum(
                ev.get("shot", {}).get("statsbomb_xg", 0.0)
                for ev in events
                if ev.get("type", {}).get("name") == "Shot"
                and ev.get("team", {}).get("name") == home
                and ev.get("shot", {}).get("outcome", {}).get("name") != "Blocked"
            )
            away_xg = sum(
                ev.get("shot", {}).get("statsbomb_xg", 0.0)
                for ev in events
                if ev.get("type", {}).get("name") == "Shot"
                and ev.get("team", {}).get("name") == away
                and ev.get("shot", {}).get("outcome", {}).get("name") != "Blocked"
            )

            key = f"{home} vs {away} {date_str}"
            xg_data[key] = {
                "home": home, "away": away, "date": date_str,
                "home_xg": round(home_xg, 3), "away_xg": round(away_xg, 3),
                "home_goals": m["home_score"], "away_goals": m["away_score"],
            }
            logging.info(
                "[%d/%d] %s vs %s: xG %.2f-%.2f (actual %d-%d)",
                i, len(matches), home, away,
                home_xg, away_xg, m["home_score"], m["away_score"],
            )
            # Be a good citizen — don't hammer GitHub CDN
            time.sleep(0.1)

    OUT_PATH.write_text(json.dumps(xg_data, indent=2))
    logging.info("Saved %d matches → %s", len(xg_data), OUT_PATH)


if __name__ == "__main__":
    main()
