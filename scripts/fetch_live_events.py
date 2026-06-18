"""Fetch live WC 2026 match events from SofaScore public API.

No API key required. Saves data/live_events.json.
Exits 0 on any network failure (CI must not break if SofaScore is unreachable).
"""
import json
import logging
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
OUT = DATA_DIR / "live_events.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; worldcup-forecaster/1.0)",
    "Accept": "application/json",
}

# SofaScore → our canonical names
TEAM_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curaçao",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cape Verde",
}


def _canonical(name: str) -> str:
    return TEAM_MAP.get(name, name)


def _fetch_live() -> dict:
    url = "https://api.sofascore.com/api/v1/sport/football/events/live"
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as e:
        logging.warning("SofaScore live fetch failed: %s", e)
        return {}

    result = {}
    for ev in events:
        # Filter to WC 2026 only
        cat = ev.get("tournament", {}).get("category", {}).get("name", "")
        tourn = ev.get("tournament", {}).get("name", "")
        if "World Cup" not in tourn and "World Cup" not in cat:
            continue

        home = _canonical(ev.get("homeTeam", {}).get("name", ""))
        away = _canonical(ev.get("awayTeam", {}).get("name", ""))
        status = ev.get("status", {})
        score = ev.get("homeScore", {}), ev.get("awayScore", {})
        hg = score[0].get("current", 0)
        ag = score[1].get("current", 0)
        minute = status.get("description", "?")

        # Fetch incidents for red cards
        eid = ev.get("id")
        red_home, red_away = 0, 0
        try:
            inc_resp = httpx.get(
                f"https://api.sofascore.com/api/v1/event/{eid}/incidents",
                headers=HEADERS, timeout=8,
            )
            if inc_resp.status_code == 200:
                for inc in inc_resp.json().get("incidents", []):
                    if inc.get("incidentType") == "card" and inc.get("cardType") in ("red", "yellowRed"):
                        if inc.get("isHome"):
                            red_home += 1
                        else:
                            red_away += 1
        except Exception:
            pass

        key = f"{home} vs {away}"
        result[key] = {
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "minute": minute,
            "red_cards": {"home": red_home, "away": red_away},
            "status": status.get("type", "inprogress"),
        }
        logging.info("Live: %s %d-%d (%s')", key, hg, ag, minute)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    live = _fetch_live()

    # Enrich with in-play win probabilities
    if live:
        try:
            from model.inplay import inplay_probs
            import json as _j
            with open(DATA_DIR / "params_cache.json") as f:
                cache = _j.load(f)
            from model.poisson import PoissonParams
            params = PoissonParams(
                attack=cache["attack"], defense=cache["defense"],
                rho=cache["rho"], gamma=cache["gamma"], teams=cache["teams"],
            )
            for key, ev in live.items():
                try:
                    ph, pd, pa = inplay_probs(
                        ev["home"], ev["away"], params,
                        home_goals=ev["home_goals"], away_goals=ev["away_goals"],
                        minute=int(ev["minute"]) if str(ev["minute"]).isdigit() else 45,
                        red_home=ev["red_cards"]["home"],
                        red_away=ev["red_cards"]["away"],
                    )
                    ev["p_home"] = round(ph, 4)
                    ev["p_draw"] = round(pd, 4)
                    ev["p_away"] = round(pa, 4)
                except Exception as e:
                    logging.warning("inplay_probs failed for %s: %s", key, e)
        except ImportError:
            pass

    OUT.write_text(json.dumps(live, indent=2))
    logging.info("Saved %d live matches to %s", len(live), OUT)


if __name__ == "__main__":
    main()
