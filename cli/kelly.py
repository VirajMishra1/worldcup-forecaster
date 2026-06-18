"""CLI: Kelly-criterion bet sizing from model probabilities and Polymarket odds.

Usage:
  python -m cli.kelly
  python -m cli.kelly --bankroll 5000
  python -m cli.kelly --full-kelly
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
TOURNAMENT_JSON = DATA_DIR / "tournament.json"
POLYMARKET_JSON = DATA_DIR / "polymarket_odds.json"
PREDICTIONS_PARQUET = DATA_DIR / "predictions.parquet"


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full Kelly fraction. Returns 0 if no edge."""
    if decimal_odds <= 1.0:
        return 0.0
    f = (p * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return max(f, 0.0)


def half_kelly(p: float, decimal_odds: float) -> float:
    return kelly_fraction(p, decimal_odds) / 2.0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_tournament() -> dict[str, float]:
    """Return {team: win_probability}."""
    data = json.loads(TOURNAMENT_JSON.read_text())
    return data.get("win", {})


def load_polymarket() -> tuple[dict[str, float] | None, str | None]:
    """
    Returns (market_probs, note).
    market_probs is None when the market is unavailable.
    """
    data = json.loads(POLYMARKET_JSON.read_text())
    win = data.get("win", {})
    note = data.get("note")
    if win:
        return win, None
    return None, note or "unavailable"


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_PARQUET)


# ---------------------------------------------------------------------------
# Tournament winner Kelly table
# ---------------------------------------------------------------------------

def tournament_kelly_table(
    model_probs: dict[str, float],
    market_probs: dict[str, float] | None,
    bankroll: float,
    kelly_scale: float,
) -> list[dict]:
    """
    Build rows for tournament winner Kelly table.
    If market_probs is None, returns model-only rows (no edge/stake computed).
    """
    rows = []
    if market_probs is None:
        # model-only — sorted by win probability descending
        for team, p_model in sorted(model_probs.items(), key=lambda x: -x[1]):
            if p_model < 0.001:
                continue
            rows.append({
                "team": team,
                "model_pct": p_model * 100,
                "market_pct": None,
                "edge_pct": None,
                "f_half": None,
                "stake": None,
            })
        return rows

    for team, p_model in sorted(model_probs.items(), key=lambda x: -x[1]):
        if p_model < 0.001:
            continue
        p_market = market_probs.get(team)
        if p_market is None or p_market <= 0:
            continue
        edge = p_model - p_market
        if edge <= 0:
            continue  # only positive edge
        d = 1.0 / p_market  # decimal odds implied by market
        f = half_kelly(p_model, d) * kelly_scale
        stake = f * bankroll
        rows.append({
            "team": team,
            "model_pct": p_model * 100,
            "market_pct": p_market * 100,
            "edge_pct": edge * 100,
            "f_half": f * 100,
            "stake": stake,
        })

    return rows


# ---------------------------------------------------------------------------
# Match-level Kelly table
# ---------------------------------------------------------------------------

def match_kelly_table(
    df: pd.DataFrame,
    market_probs: dict[str, float] | None,
    bankroll: float,
    kelly_scale: float,
) -> list[dict]:
    """
    Polymarket does not expose per-match odds in the current feed (only
    tournament winner market). Return top-5 by p_home confidence instead.
    """
    if market_probs is not None:
        # If per-match data were available we'd compute Kelly here.
        # The current polymarket_odds.json only carries tournament winner
        # probs, so this branch is currently unreachable.
        pass

    # Fallback: top-5 by home-win confidence, upcoming matches only
    now = pd.Timestamp.now("UTC")
    upcoming = df[df["match_date"] >= now].copy()
    if upcoming.empty:
        upcoming = df.copy()

    top = (
        upcoming
        .sort_values("p_home", ascending=False)
        .head(5)
    )

    rows = []
    for _, row in top.iterrows():
        rows.append({
            "home": row["home"],
            "away": row["away"],
            "match_date": row["match_date"],
            "p_home": row["p_home"],
            "p_draw": row["p_draw"],
            "p_away": row["p_away"],
        })
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


def _money(v: float | None) -> str:
    return f"${v:.0f}" if v is not None else "N/A"


def print_tournament_table(rows: list[dict], bankroll: float, market_available: bool) -> None:
    if not rows:
        print("  (no positive-edge opportunities found)\n")
        return

    if market_available:
        header = f"{'Team':<22} {'Model%':>7} {'Market%':>8} {'Edge':>7} {'f*(half)':>9} {'Stake/$' + str(int(bankroll)):>12}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in rows:
            line = (
                f"{r['team']:<22}"
                f" {_pct(r['model_pct']):>7}"
                f" {_pct(r['market_pct']):>8}"
                f" {'+' + _pct(r['edge_pct']):>7}"
                f" {_pct(r['f_half']):>9}"
                f" {_money(r['stake']):>12}"
            )
            print(line)
    else:
        header = f"{'Team':<22} {'Model Win%':>10}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in rows:
            print(f"{r['team']:<22} {_pct(r['model_pct']):>10}")
    print()


def print_match_table(rows: list[dict], market_note: str | None) -> None:
    if not rows:
        print("  (no upcoming match predictions found)\n")
        return

    header = f"{'Home':<18} {'Away':<18} {'p_home':>7} {'p_draw':>7} {'p_away':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['home']:<18}"
            f" {r['away']:<18}"
            f" {r['p_home']:>7.1%}"
            f" {r['p_draw']:>7.1%}"
            f" {r['p_away']:>7.1%}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(bankroll: float = 1000.0, full_kelly: bool = False) -> None:
    kelly_scale = 1.0 if full_kelly else 0.5
    kelly_label = "full Kelly" if full_kelly else "half Kelly"

    model_probs = load_tournament()
    market_probs, market_note = load_polymarket()
    predictions = load_predictions()

    market_available = market_probs is not None

    # --- Tournament winner table ---
    print("=" * 60)
    print("TOURNAMENT WINNER — Kelly Sizing")
    print(f"  Bankroll: ${bankroll:,.0f}  |  Method: {kelly_label}")
    if not market_available:
        print(f"  NOTE: Polymarket market data unavailable ({market_note}).")
        print("        Showing model probabilities only; no stake computed.")
    print("=" * 60)

    t_rows = tournament_kelly_table(model_probs, market_probs, bankroll, kelly_scale)
    print_tournament_table(t_rows, bankroll, market_available)

    # --- Match-level table ---
    print("=" * 60)
    print("NEXT MATCHES — Top-5 by Home-Win Confidence")
    if not market_available:
        print(f"  NOTE: Per-match Polymarket odds unavailable ({market_note}).")
        print("        Showing model predictions only; Kelly stake not computed.")
    print("=" * 60)

    m_rows = match_kelly_table(predictions, market_probs, bankroll, kelly_scale)
    print_match_table(m_rows, market_note)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kelly-criterion bet sizing from model probabilities and Polymarket odds."
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Total bankroll in dollars (default: 1000)"
    )
    parser.add_argument(
        "--full-kelly", action="store_true",
        help="Use full Kelly instead of half Kelly (riskier)"
    )
    args = parser.parse_args()
    run(bankroll=args.bankroll, full_kelly=args.full_kelly)


if __name__ == "__main__":
    main()
