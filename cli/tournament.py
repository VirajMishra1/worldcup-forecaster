"""
CLI display for WC 2026 tournament win probabilities.

Usage:
    python -m cli.tournament
"""
import json
import sys
from datetime import datetime, timezone


def main() -> None:
    path = "data/tournament.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {path} not found. Run: python -m scripts.simulate_tournament")
        sys.exit(1)

    n_sims = data.get("simulations", 0)
    updated_raw = data.get("updated_at", "")
    try:
        updated_dt = datetime.fromisoformat(updated_raw)
        updated_str = updated_dt.strftime("%b %d %H:%M UTC")
    except ValueError:
        updated_str = updated_raw

    qualify = data.get("qualify", {})
    r16 = data.get("r16", {})
    qf = data.get("qf", {})
    sf = data.get("sf", {})
    final = data.get("final", {})
    win = data.get("win", {})

    # All teams that appeared in any stage
    all_teams = set(qualify) | set(r16) | set(qf) | set(sf) | set(final) | set(win)

    # Filter: win% > 0.1%
    display_teams = [t for t in all_teams if win.get(t, 0) >= 0.001]
    display_teams.sort(key=lambda t: win.get(t, 0), reverse=True)

    # Column widths
    team_w = 30
    header = (
        f"{'Team':<{team_w}} "
        f"{'Qualify':>7} "
        f"{'R16':>6} "
        f"{'QF':>6} "
        f"{'SF':>6} "
        f"{'Final':>7} "
        f"{'Win':>7} "
        f"{'Odds':>7}"
    )
    sep = "─" * len(header)

    print(f"\nWC 2026 — Tournament Win Probabilities  ({n_sims:,} simulations)")
    print(f"Updated: {updated_str}\n")
    print(header)
    print(sep)

    for team in display_teams:
        p_win = win.get(team, 0.0)
        p_qualify = qualify.get(team, 0.0)
        p_r16 = r16.get(team, 0.0)
        p_qf = qf.get(team, 0.0)
        p_sf = sf.get(team, 0.0)
        p_final = final.get(team, 0.0)

        implied_odds = f"{1.0/p_win:.1f}x" if p_win > 0 else "—"

        q_str = f"{p_qualify*100:.0f}%"
        r16_str = f"{p_r16*100:.0f}%"
        qf_str = f"{p_qf*100:.0f}%"
        sf_str = f"{p_sf*100:.0f}%"
        fin_str = f"{p_final*100:.0f}%"
        win_str = f"{p_win*100:.1f}%"

        print(
            f"{team:<{team_w}} "
            f"{q_str:>7} "
            f"{r16_str:>6} "
            f"{qf_str:>6} "
            f"{sf_str:>6} "
            f"{fin_str:>7} "
            f"{win_str:>7} "
            f"{implied_odds:>7}"
        )

    print(sep)
    print()


if __name__ == "__main__":
    main()
