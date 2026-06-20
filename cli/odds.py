"""
CLI: compare model win probabilities vs Polymarket market prices.

Usage:
    python -m cli.odds
"""
import json
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TOURNAMENT_PATH = DATA_DIR / "tournament.json"
POLYMARKET_PATH = DATA_DIR / "polymarket_odds.json"


def main() -> None:
    # Load model
    try:
        with open(TOURNAMENT_PATH) as f:
            tournament = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TOURNAMENT_PATH} not found. Run: python -m scripts.simulate_tournament")
        sys.exit(1)

    model_win: dict[str, float] = tournament.get("win", {})

    # Load Polymarket
    market_win: dict[str, float] = {}
    market_note = ""
    market_updated = ""
    if POLYMARKET_PATH.exists():
        with open(POLYMARKET_PATH) as f:
            poly = json.load(f)
        market_win = poly.get("win", {})
        market_note = poly.get("note", "")
        market_updated = poly.get("updated_at", "")
    else:
        market_note = f"{POLYMARKET_PATH.name} not found — run: python -m scripts.fetch_odds"

    if not market_win:
        print(f"\nPolymarket data unavailable: {market_note or 'empty'}")
        print("Run `python -m scripts.fetch_odds` to refresh.\n")

    # Format updated timestamp
    updated_str = ""
    if market_updated:
        try:
            dt = datetime.fromisoformat(market_updated)
            updated_str = dt.strftime("%b %d %H:%M UTC")
        except ValueError:
            updated_str = market_updated

    # Build rows: all teams with a model win% >= 0.1%
    teams = [t for t, p in model_win.items() if p >= 0.001]
    teams.sort(key=lambda t: model_win[t], reverse=True)

    team_w = 24
    header = (
        f"{'Team':<{team_w}} "
        f"{'Model':>7} "
        f"{'Market':>8} "
        f"{'Edge':>7}"
    )
    sep = "─" * len(header)

    n_sims = tournament.get("simulations", 0)
    print(f"\nWC 2026 — Model vs Polymarket  ({n_sims:,} simulations)")
    if updated_str:
        print(f"Polymarket updated: {updated_str}")
    print()
    print(header)
    print(sep)

    for team in teams:
        p_model = model_win[team]
        p_market = market_win.get(team)

        model_str = f"{p_model * 100:.1f}%"

        if p_market is not None:
            market_str = f"{p_market * 100:.1f}%"
            edge = (p_model - p_market) * 100
            edge_str = f"{edge:+.1f}%"
        else:
            market_str = "—"
            edge_str = "—"

        print(
            f"{team:<{team_w}} "
            f"{model_str:>7} "
            f"{market_str:>8} "
            f"{edge_str:>7}"
        )

    print(sep)

    if not market_win:
        print()
        return

    # Summary: top positive edges (model > market)
    edges = [
        (t, model_win[t], market_win[t], model_win[t] - market_win[t])
        for t in teams
        if t in market_win
    ]
    pos = sorted([e for e in edges if e[3] > 0], key=lambda x: x[3], reverse=True)
    if pos:
        print("\nTop positive edges (model > market):")
        for team, pm, mkt, edge in pos[:5]:
            print(f"  {team}: model {pm*100:.1f}%  market {mkt*100:.1f}%  edge {edge*100:+.1f}%")

    print()


if __name__ == "__main__":
    main()
