"""CLI: predict a match outcome.

Usage:
  python -m cli.predict --home France --away Argentina
  python -m cli.predict --list-teams
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from model.poisson import fit, PoissonParams
from model.predict import scoreline_grid
from model.markets import derive_markets

DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET = DATA_DIR / "historical_matches.parquet"
CACHE = DATA_DIR / "params_cache.json"


def _save_params(params: PoissonParams) -> None:
    CACHE.write_text(json.dumps({
        "attack": params.attack,
        "defense": params.defense,
        "rho": params.rho,
        "gamma": params.gamma,
        "teams": params.teams,
    }))


def _load_cached() -> PoissonParams:
    d = json.loads(CACHE.read_text())
    return PoissonParams(
        attack=d["attack"],
        defense=d["defense"],
        rho=d["rho"],
        gamma=d["gamma"],
        teams=d["teams"],
    )


def load_params(force_refit: bool = False) -> PoissonParams:
    if not PARQUET.exists():
        print("Run:  python scripts/fetch_historical.py", file=sys.stderr)
        sys.exit(1)
    parquet_mtime = PARQUET.stat().st_mtime
    if not force_refit and CACHE.exists() and CACHE.stat().st_mtime >= parquet_mtime:
        print("Loading cached model params...", file=sys.stderr)
        return _load_cached()
    df = pd.read_parquet(PARQUET)
    print(f"Fitting model on {len(df):,} historical matches...", file=sys.stderr)
    params = fit(df, neutral=True)
    _save_params(params)
    return params


def print_prediction(home: str, away: str, params) -> None:
    if home not in params.teams:
        print(f"Unknown team: '{home}'  (use --list-teams to see all)", file=sys.stderr)
        sys.exit(1)
    if away not in params.teams:
        print(f"Unknown team: '{away}'  (use --list-teams to see all)", file=sys.stderr)
        sys.exit(1)

    grid = scoreline_grid(home, away, params, is_neutral=True)
    m = derive_markets(grid)

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {home}  vs  {away}")
    print(f"{sep}")

    print("\nOUTCOME PROBABILITIES  (neutral venue — no home advantage)")
    print(f"  {home:<25} {m.p_home_win:.1%}")
    print(f"  {'Draw':<25} {m.p_draw:.1%}")
    print(f"  {away:<25} {m.p_away_win:.1%}")

    print("\nEXPECTED GOALS")
    print(f"  {home}: {m.expected_home_goals:.2f}   {away}: {m.expected_away_goals:.2f}")

    print("\nGOALS MARKETS")
    print(f"  Over  2.5  {m.p_over_25:.1%}    Under 2.5  {m.p_under_25:.1%}")
    print(f"  BTTS yes   {m.p_btts:.1%}    BTTS no    {m.p_btts_no:.1%}")

    print("\nTOP SCORELINES")
    for score, prob in sorted(m.exact_scores.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(prob * 200)
        print(f"  {home} {score} {away}  {prob:.1%}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="World Cup match predictor.",
        usage="%(prog)s [HOME] [AWAY]  or  %(prog)s --home HOME --away AWAY",
    )
    parser.add_argument("teams", nargs="*", help="Home and away team (positional)")
    parser.add_argument("--home", "--team1", dest="home", help="Home team")
    parser.add_argument("--away", "--team2", dest="away", help="Away team")
    parser.add_argument("--list-teams", action="store_true", help="Print all known teams")
    parser.add_argument("--refit", action="store_true", help="Force model re-fit (ignore cache)")
    args = parser.parse_args()

    params = load_params(force_refit=args.refit)

    if args.list_teams:
        for t in sorted(params.teams):
            print(t)
        return

    # Accept positional or named args
    home = args.home or (args.teams[0] if len(args.teams) > 0 else None)
    away = args.away or (args.teams[1] if len(args.teams) > 1 else None)

    if not home or not away:
        parser.error("Provide two teams: predict France Iraq  or  predict --home France --away Iraq")

    print_prediction(home, away, params)


if __name__ == "__main__":
    main()
