"""
Fetch Polymarket WC 2026 winner odds via the Gamma API.

Saves data/polymarket_odds.json with structure:
  {"updated_at": "...", "win": {"Spain": 0.18, ...}}

On any fetch failure, writes an empty structure and exits 0 so the
rest of the pipeline continues.

Usage:
    python -m scripts.fetch_odds
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "polymarket_odds.json"

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Map Polymarket outcome labels → our canonical team names.
# Polymarket typically uses full English names; adjust as markets are observed.
TEAM_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "United States of America": "United States",
    "US": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
}


def _norm(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _empty_result(note: str) -> dict:
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "win": {}, "note": note}


def _find_wc_winner_market(client: httpx.Client) -> dict | None:
    """Return the first market that looks like the WC 2026 outright winner."""
    # Strategy 1: search via /markets
    try:
        r = client.get(
            f"{GAMMA_BASE}/markets",
            params={"search": "2026 FIFA World Cup", "closed": "false"},
            timeout=20,
        )
        if r.status_code == 200:
            markets = r.json()
            if isinstance(markets, list):
                for m in markets:
                    q = (m.get("question") or m.get("title") or "").lower()
                    if "win" in q and ("world cup" in q or "wc 2026" in q or "fifa" in q):
                        if m.get("outcomes") and m.get("outcomePrices"):
                            return m
        elif r.status_code == 429:
            log.warning("Polymarket rate-limited (markets search): 429")
            return None
    except Exception as exc:
        log.warning("markets search failed: %s", exc)

    # Strategy 2: search via /events
    try:
        r = client.get(
            f"{GAMMA_BASE}/events",
            params={"search": "2026 FIFA World Cup"},
            timeout=20,
        )
        if r.status_code == 200:
            events = r.json()
            items = events if isinstance(events, list) else events.get("data", [])
            for ev in items:
                for m in ev.get("markets", []):
                    q = (m.get("question") or m.get("title") or "").lower()
                    if "win" in q or "winner" in q:
                        if m.get("outcomes") and m.get("outcomePrices"):
                            return m
        elif r.status_code == 429:
            log.warning("Polymarket rate-limited (events search): 429")
            return None
    except Exception as exc:
        log.warning("events search failed: %s", exc)

    return None


def _parse_market(market: dict) -> dict[str, float]:
    """Extract {team: probability} from a Polymarket market dict."""
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")

    # Both fields are JSON-encoded strings in the Gamma API
    if isinstance(outcomes_raw, str):
        outcomes = json.loads(outcomes_raw)
    else:
        outcomes = outcomes_raw

    if isinstance(prices_raw, str):
        prices = json.loads(prices_raw)
    else:
        prices = prices_raw

    result: dict[str, float] = {}
    for team, price in zip(outcomes, prices):
        try:
            p = float(price)
        except (ValueError, TypeError):
            continue
        canonical = _norm(team)
        result[canonical] = round(p, 6)

    return result


def fetch_odds() -> dict:
    headers = {
        "User-Agent": "worldcup-forecaster/1.0 (github.com/VirajMishra1/worldcup-forecaster)",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(headers=headers, follow_redirects=True) as client:
            market = _find_wc_winner_market(client)
    except httpx.ConnectError as exc:
        log.warning("Cannot reach Polymarket (ConnectError): %s", exc)
        return _empty_result("ConnectError: Polymarket unreachable")
    except httpx.TimeoutException as exc:
        log.warning("Polymarket request timed out: %s", exc)
        return _empty_result("TimeoutException")
    except Exception as exc:
        log.warning("Unexpected error fetching Polymarket data: %s", exc)
        return _empty_result(f"Error: {exc}")

    if market is None:
        log.warning("WC 2026 winner market not found on Polymarket")
        return _empty_result("Market not found")

    win = _parse_market(market)
    if not win:
        log.warning("Market found but no parseable outcome prices")
        return _empty_result("No parseable outcome prices")

    log.info("Fetched Polymarket odds for %d teams", len(win))
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "win": win,
        "market_id": market.get("id", ""),
        "market_question": market.get("question", market.get("title", "")),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = fetch_odds()
    DATA_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    if result.get("win"):
        log.info("Saved Polymarket odds (%d teams) → %s", len(result["win"]), OUT_PATH)
    else:
        log.warning("Saved empty Polymarket odds → %s  (%s)", OUT_PATH, result.get("note", ""))


if __name__ == "__main__":
    main()
