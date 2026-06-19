"""Fetch live WC 2026 match events from ESPN public API.

No API key required. Works from GitHub Actions (no Cloudflare block).
Saves data/live_events.json. Exits 0 on any network failure.
"""
import json
import logging
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
OUT = DATA_DIR / "live_events.json"

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"

TEAM_MAP = {
    "United States": "United States",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cape Verde",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
}


def _canonical(name: str) -> str:
    return TEAM_MAP.get(name, name)


def _red_cards(event_id: str) -> tuple[int, int]:
    """Return (home_reds, away_reds) from ESPN summary endpoint."""
    try:
        r = httpx.get(ESPN_SUMMARY, params={"event": event_id}, timeout=8)
        if r.status_code != 200:
            return 0, 0
        data = r.json()
        home_reds = away_reds = 0
        for play in data.get("plays", []):
            ptype = play.get("type", {}).get("text", "")
            if "Red Card" in ptype or "red card" in ptype.lower():
                team_id = play.get("team", {}).get("id", "")
                comp = data.get("header", {}).get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                home_id = competitors[0].get("team", {}).get("id", "") if competitors else ""
                if team_id == home_id:
                    home_reds += 1
                else:
                    away_reds += 1
    except Exception:
        pass
    return home_reds, away_reds


def _fetch_live() -> dict:
    try:
        r = httpx.get(ESPN_SCOREBOARD, timeout=10)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        logging.warning("ESPN fetch failed: %s", e)
        return {}

    result = {}
    for ev in events:
        comp = ev["competitions"][0]
        status_type = comp["status"]["type"]["name"]

        # Only process live or recently finished matches
        if status_type not in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME",
                                "STATUS_END_PERIOD", "STATUS_FINAL"):
            continue

        competitors = comp["competitors"]
        home_c = next((c for c in competitors if c["homeAway"] == "home"), competitors[0])
        away_c = next((c for c in competitors if c["homeAway"] == "away"), competitors[1])

        home = _canonical(home_c["team"]["displayName"])
        away = _canonical(away_c["team"]["displayName"])
        hg = int(home_c.get("score", 0) or 0)
        ag = int(away_c.get("score", 0) or 0)
        clock = comp["status"].get("displayClock", "?").rstrip("'")
        try:
            minute = int(clock.split(":")[0]) if ":" in clock else int(clock)
        except (ValueError, IndexError):
            minute = 45

        red_home, red_away = _red_cards(str(ev["id"]))

        key = f"{home} vs {away}"
        entry = {
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "minute": minute,
            "red_cards": {"home": red_home, "away": red_away},
            "status": status_type,
        }

        # Enrich with in-play probabilities
        try:
            from model.inplay import inplay_probs
            from model.poisson import PoissonParams
            import json as _j
            with open(DATA_DIR / "params_cache.json") as f:
                cache = _j.load(f)
            params = PoissonParams(
                attack=cache["attack"], defense=cache["defense"],
                rho=cache["rho"], gamma=cache["gamma"], teams=cache["teams"],
            )
            ph, pd_, pa = inplay_probs(
                home, away, params,
                home_goals=hg, away_goals=ag, minute=minute,
                red_home=red_home, red_away=red_away,
            )
            entry["p_home"] = round(ph, 4)
            entry["p_draw"] = round(pd_, 4)
            entry["p_away"] = round(pa, 4)
        except Exception as e:
            logging.debug("inplay_probs failed for %s: %s", key, e)

        result[key] = entry
        logging.info("Live: %s %d-%d (%s')", key, hg, ag, minute)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    live = _fetch_live()
    OUT.write_text(json.dumps(live, indent=2))
    if live:
        logging.info("Saved %d live matches to %s", len(live), OUT)
    else:
        logging.info("No live WC matches right now — saved empty file.")


if __name__ == "__main__":
    main()
