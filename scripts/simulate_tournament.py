"""
Monte Carlo WC 2026 tournament simulator.

Usage:
    python -m scripts.simulate_tournament
    python -m scripts.simulate_tournament --quick   # 1,000 sims
"""
import argparse
import json
import logging
import random
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from model.form import form_factors
from model.lineup import squad_ratio
from model.markets import derive_markets
from model.poisson import PoissonParams
from model.predict import scoreline_grid
from model.rest import rest_factor

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WC 2026 group definitions
# ---------------------------------------------------------------------------
GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Switzerland", "Canada", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Scotland", "Morocco", "Brazil", "Haiti"],
    "D": ["United States", "Australia", "Turkey", "Paraguay"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Sweden", "Japan", "Netherlands", "Tunisia"],
    "G": ["New Zealand", "Iran", "Belgium", "Egypt"],
    "H": ["Uruguay", "Saudi Arabia", "Spain", "Cape Verde"],
    "I": ["Norway", "France", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Jordan", "Algeria"],
    "K": ["DR Congo", "Colombia", "Portugal", "Uzbekistan"],
    "L": ["England", "Ghana", "Croatia", "Panama"],
}

# Normalise result team names → canonical WC names
TEAM_NAME_MAP: Dict[str, str] = {
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
}

ALL_TEAMS: List[str] = [t for teams in GROUPS.values() for t in teams]
TEAM_TO_GROUP: Dict[str, str] = {
    t: g for g, teams in GROUPS.items() for t in teams
}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_params(path: str = "data/params_cache.json") -> PoissonParams:
    with open(path) as f:
        d = json.load(f)
    return PoissonParams(
        attack=d["attack"],
        defense=d["defense"],
        rho=d["rho"],
        gamma=d["gamma"],
        teams=d["teams"],
    )


def load_results(path: str = "data/results.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["home"] = df["home"].replace(TEAM_NAME_MAP)
    df["away"] = df["away"].replace(TEAM_NAME_MAP)
    return df


# ---------------------------------------------------------------------------
# Pre-compute match probability matrices
# ---------------------------------------------------------------------------

# For each ordered pair (i, j): probability team i advances against team j
# at neutral venue. p_advance = p_home_win + 0.5 * p_draw
# (used in knockout rounds where we pick a winner — draws resolved by
# coin-flip equivalent).

def build_advance_matrix(
    params: PoissonParams,
    all_matches: pd.DataFrame,
    ko_date: str = "2026-07-01",
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Returns (team_index, p_advance) where p_advance[i, j] = P(team i beats team j).
    Draws give 0.5 to each side (penalty shoot-out).
    Form and rest adjustments use ko_date as the reference knockout date.
    """
    n = len(ALL_TEAMS)
    tidx = {t: i for i, t in enumerate(ALL_TEAMS)}
    p_advance = np.zeros((n, n))

    log.info(f"Pre-computing advance matrix ({n}×{n} = {n*n} grids)...")
    computed = 0
    for i, home in enumerate(ALL_TEAMS):
        for j, away in enumerate(ALL_TEAMS):
            if i == j:
                continue
            hf, af = form_factors(home, away, ko_date, all_matches)
            hr = rest_factor(home, ko_date, all_matches)
            ar = rest_factor(away, ko_date, all_matches)
            grid = scoreline_grid(
                home, away, params, is_neutral=True,
                home_lineup_ratio=squad_ratio(home) * hf * hr,
                away_lineup_ratio=squad_ratio(away) * af * ar,
            )
            m = derive_markets(grid)
            p_advance[i, j] = m.p_home_win + 0.5 * m.p_draw
            computed += 1
    log.info(f"  {computed} grids computed.")
    return tidx, p_advance


# ---------------------------------------------------------------------------
# Pre-compute group stage match info (for remaining/simulated matches)
# ---------------------------------------------------------------------------

MatchInfo = Tuple[str, str, float, float, float, float, float]
# (home, away, p_home_win, p_draw, p_away_win, xg_home, xg_away)


def build_group_match_info(
    params: PoissonParams,
    all_matches: pd.DataFrame,
    group_stage_date: str = "2026-06-11",
) -> Dict[Tuple[str, str], MatchInfo]:
    """For every C(4,2)=6 matchup in each group, store market probabilities."""
    info: Dict[Tuple[str, str], MatchInfo] = {}
    for group, teams in GROUPS.items():
        for home, away in combinations(teams, 2):
            hf, af = form_factors(home, away, group_stage_date, all_matches)
            hr = rest_factor(home, group_stage_date, all_matches)
            ar = rest_factor(away, group_stage_date, all_matches)
            grid = scoreline_grid(
                home, away, params, is_neutral=True,
                home_lineup_ratio=squad_ratio(home) * hf * hr,
                away_lineup_ratio=squad_ratio(away) * af * ar,
            )
            m = derive_markets(grid)
            info[(home, away)] = (
                home, away,
                m.p_home_win, m.p_draw, m.p_away_win,
                m.expected_home_goals, m.expected_away_goals,
            )
    log.info(f"Group match info: {len(info)} matchups pre-computed.")
    return info


# ---------------------------------------------------------------------------
# Simulate a single group stage match
# ---------------------------------------------------------------------------

def sample_match(
    home: str,
    away: str,
    match_info: Dict[Tuple[str, str], MatchInfo],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Sample (home_goals, away_goals) from the Poisson model for a group stage match.
    Outcome is implicitly encoded in the goals.
    """
    key = (home, away)
    if key in match_info:
        _, _, ph, pd_, pa, xg_h, xg_a = match_info[key]
    else:
        # Reversed fixture — swap and invert
        _, _, ph, pd_, pa, xg_h, xg_a = match_info[(away, home)]
        ph, pa, xg_h, xg_a = pa, ph, xg_a, xg_h

    # Sample outcome, then sample goals consistent with that outcome
    r = rng.random()
    if r < ph:
        # Home win: hg > ag
        for _ in range(20):
            hg = int(rng.poisson(xg_h))
            ag = int(rng.poisson(xg_a))
            if hg > ag:
                return hg, ag
        # Fallback if Poisson keeps producing draws/away wins
        return max(1, int(rng.poisson(xg_h))), 0
    elif r < ph + pd_:
        # Draw: hg == ag
        for _ in range(20):
            hg = int(rng.poisson(max(xg_h, xg_a) * 0.8))
            if rng.random() < 0.5:
                return hg, hg
        return 1, 1
    else:
        # Away win: ag > hg
        for _ in range(20):
            hg = int(rng.poisson(xg_h))
            ag = int(rng.poisson(xg_a))
            if ag > hg:
                return hg, ag
        return 0, max(1, int(rng.poisson(xg_a)))


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def simulate_group(
    group: str,
    teams: List[str],
    completed: Dict[Tuple[str, str], Tuple[int, int]],
    match_info: Dict[Tuple[str, str], MatchInfo],
    rng: np.random.Generator,
) -> List[Tuple[str, int, int, int]]:
    """
    Returns list of (team, points, goal_diff, goals_for) sorted by standing.
    """
    pts: Dict[str, int] = {t: 0 for t in teams}
    gd: Dict[str, int] = {t: 0 for t in teams}
    gf: Dict[str, int] = {t: 0 for t in teams}

    for home, away in combinations(teams, 2):
        key = (home, away)
        rev_key = (away, home)
        if key in completed:
            hg, ag = completed[key]
        elif rev_key in completed:
            ag, hg = completed[rev_key]
        else:
            hg, ag = sample_match(home, away, match_info, rng)

        gf[home] += hg
        gf[away] += ag
        gd[home] += hg - ag
        gd[away] += ag - hg
        if hg > ag:
            pts[home] += 3
        elif hg == ag:
            pts[home] += 1
            pts[away] += 1
        else:
            pts[away] += 3

    standing = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t]),
        reverse=True,
    )
    return [(t, pts[t], gd[t], gf[t]) for t in standing]


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    tidx: Dict[str, int],
    p_advance: np.ndarray,
    rng: np.random.Generator,
) -> str:
    """Returns winner. Uses pre-computed advance probability."""
    ia, ib = tidx[team_a], tidx[team_b]
    p = p_advance[ia, ib]
    return team_a if rng.random() < p else team_b


def run_simulation(
    completed: Dict[Tuple[str, str], Tuple[int, int]],
    match_info: Dict[Tuple[str, str], MatchInfo],
    tidx: Dict[str, int],
    p_advance: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, int]]:
    """
    Returns dict of stage -> {team: 1 if reached}
    Stages: qualify, r16, qf, sf, final, win
    """
    reached: Dict[str, Dict[str, int]] = {
        "qualify": {}, "r16": {}, "qf": {}, "sf": {}, "final": {}, "win": {},
    }

    # --- Group stage ---
    group_standings: Dict[str, List[Tuple[str, int, int, int]]] = {}
    third_place_teams: List[Tuple[int, int, int, str]] = []  # (pts, gd, gf, team)

    for group, teams in GROUPS.items():
        standing = simulate_group(group, teams, completed, match_info, rng)
        group_standings[group] = standing
        # 1st and 2nd qualify directly
        for pos, (team, pts, gd, gf_val) in enumerate(standing):
            if pos < 2:
                reached["qualify"][team] = 1
            elif pos == 2:
                third_place_teams.append((pts, gd, gf_val, team))

    # Best 8 third-place teams qualify
    third_place_teams.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    for pts, gd, gf_val, team in third_place_teams[:8]:
        reached["qualify"][team] = 1

    # Categorise the 32 qualifiers by group position
    group_winners = [standing[0][0] for standing in group_standings.values()]   # 12 teams
    group_runners_up = [standing[1][0] for standing in group_standings.values()]  # 12 teams
    group_thirds_q = [t for _, _, _, t in third_place_teams[:8]]                 # 8 teams
    assert len(group_winners) == 12
    assert len(group_runners_up) == 12
    assert len(group_thirds_q) == 8

    # Shuffle within each tier (preserves seeding constraint across rounds)
    rng.shuffle(group_winners)
    rng.shuffle(group_runners_up)
    rng.shuffle(group_thirds_q)

    # --- Constrained R32 bracket ---
    # Rule: no group winner faces another group winner in R32.
    # 12 group winners each play one of the 20 non-winners.
    # Remaining 8 non-winners play each other (4 matches).
    # Total: 12 + 4 = 16 R32 matches.
    non_winners = group_runners_up + group_thirds_q  # 20 teams
    rng.shuffle(non_winners)

    r32_pairs: List[Tuple[str, str]] = []
    for i, w in enumerate(group_winners):
        r32_pairs.append((w, non_winners[i]))         # winner vs non-winner
    for i in range(0, 8, 2):                          # remaining 8 non-winners
        r32_pairs.append((non_winners[12 + i], non_winners[12 + i + 1]))
    assert len(r32_pairs) == 16

    # R32 → 16 survivors
    winners_r32 = []
    for team_a, team_b in r32_pairs:
        w = simulate_knockout_match(team_a, team_b, tidx, p_advance, rng)
        winners_r32.append(w)
        reached["r16"][w] = 1

    # R16 → QF
    winners_r16 = []
    for i in range(0, 16, 2):
        w = simulate_knockout_match(winners_r32[i], winners_r32[i + 1], tidx, p_advance, rng)
        winners_r16.append(w)
        reached["qf"][w] = 1

    # QF → SF
    winners_qf = []
    for i in range(0, 8, 2):
        w = simulate_knockout_match(winners_r16[i], winners_r16[i + 1], tidx, p_advance, rng)
        winners_qf.append(w)
        reached["sf"][w] = 1

    # SF → Final
    winners_sf = []
    for i in range(0, 4, 2):
        w = simulate_knockout_match(winners_qf[i], winners_qf[i + 1], tidx, p_advance, rng)
        winners_sf.append(w)
        reached["final"][w] = 1

    # Final
    champion = simulate_knockout_match(winners_sf[0], winners_sf[1], tidx, p_advance, rng)
    reached["win"][champion] = 1

    return reached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 Monte Carlo simulator")
    parser.add_argument("--quick", action="store_true", help="Run 1,000 sims instead of 10,000")
    args = parser.parse_args()

    N_SIMS = 1_000 if args.quick else 10_000
    np.random.seed(42)
    rng = np.random.default_rng(42)

    log.info("Loading params...")
    params = load_params()

    log.info("Loading completed results...")
    results_df = load_results()

    # Build completed match dict keyed by (home, away) as they appear in the data
    completed: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for _, row in results_df.iterrows():
        completed[(row["home"], row["away"])] = (
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
    log.info(f"  {len(completed)} completed matches loaded.")

    hist = pd.read_parquet("data/historical_matches.parquet")
    all_matches = pd.concat([hist, results_df], ignore_index=True)

    log.info("Pre-computing group match info...")
    match_info = build_group_match_info(params, all_matches)

    tidx, p_advance = build_advance_matrix(params, all_matches)

    # Accumulators: stage → team → count
    counts: Dict[str, Dict[str, int]] = {
        stage: {t: 0 for t in ALL_TEAMS}
        for stage in ["qualify", "r16", "qf", "sf", "final", "win"]
    }

    log.info(f"Running {N_SIMS:,} simulations...")
    for sim_i in range(N_SIMS):
        if (sim_i + 1) % 1000 == 0:
            log.info(f"  {sim_i + 1:,} / {N_SIMS:,}")
        result = run_simulation(completed, match_info, tidx, p_advance, rng)
        for stage, teams_reached in result.items():
            for team in teams_reached:
                counts[stage][team] += 1

    # Normalise to probabilities
    probs: Dict[str, Dict[str, float]] = {
        stage: {t: round(counts[stage][t] / N_SIMS, 4) for t in ALL_TEAMS}
        for stage in counts
    }

    # Build output
    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "simulations": N_SIMS,
        **probs,
    }

    out_path = "data/tournament.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved → {out_path}")

    # Print sorted table
    sorted_teams = sorted(ALL_TEAMS, key=lambda t: probs["win"][t], reverse=True)
    header = f"{'Team':<30} {'Qualify':>7} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>7} {'Win':>7}"
    sep = "─" * len(header)
    print(f"\nWC 2026 — Tournament Win Probabilities  ({N_SIMS:,} simulations)")
    print(f"Updated: {datetime.now(timezone.utc).strftime('%b %d %H:%M UTC')}\n")
    print(header)
    print(sep)
    for team in sorted_teams:
        if probs["win"][team] < 0.001:
            continue
        q = f"{probs['qualify'][team]*100:.0f}%"
        r16 = f"{probs['r16'][team]*100:.0f}%"
        qf = f"{probs['qf'][team]*100:.0f}%"
        sf = f"{probs['sf'][team]*100:.0f}%"
        fin = f"{probs['final'][team]*100:.0f}%"
        win = f"{probs['win'][team]*100:.1f}%"
        print(f"{team:<30} {q:>7} {r16:>6} {qf:>6} {sf:>6} {fin:>7} {win:>7}")


if __name__ == "__main__":
    main()
