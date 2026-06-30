"""
Monte Carlo WC 2026 tournament simulator.

Usage:
    python -m scripts.simulate_tournament
    python -m scripts.simulate_tournament --quick   # 1,000 sims
"""
import argparse
import json
import logging
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
# Actual WC 2026 R32 bracket (FIFA official)
# Source: FIFA competition regulations + ESPN/Wikipedia knockout schedule
# ---------------------------------------------------------------------------

# Each entry: (slot_id, side_a, side_b)
# side spec: ("W", group) = group winner, ("R", group) = runner-up,
#            ("3", slot_id) = third-place team assigned to this slot
R32_MATCHES: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = [
    ("M73", ("R", "A"), ("R", "B")),
    ("M74", ("W", "E"), ("3", "M74")),
    ("M75", ("W", "F"), ("R", "C")),
    ("M76", ("W", "C"), ("R", "F")),
    ("M77", ("W", "I"), ("3", "M77")),
    ("M78", ("R", "E"), ("R", "I")),
    ("M79", ("W", "A"), ("3", "M79")),
    ("M80", ("W", "L"), ("3", "M80")),
    ("M81", ("W", "D"), ("3", "M81")),
    ("M82", ("W", "G"), ("3", "M82")),
    ("M83", ("R", "K"), ("R", "L")),
    ("M84", ("W", "H"), ("R", "J")),
    ("M85", ("W", "B"), ("3", "M85")),
    ("M86", ("W", "J"), ("R", "H")),
    ("M87", ("W", "K"), ("3", "M87")),
    ("M88", ("R", "D"), ("R", "G")),
]

# Which groups' third-place teams are eligible for each slot
THIRD_SLOT_ELIGIBLE: Dict[str, set] = {
    "M74": {"A", "B", "C", "D", "F"},
    "M77": {"C", "D", "F", "G", "H"},
    "M79": {"C", "E", "F", "H", "I"},
    "M80": {"E", "H", "I", "J", "K"},
    "M81": {"B", "E", "F", "I", "J"},
    "M82": {"A", "E", "H", "I", "J"},
    "M85": {"E", "F", "G", "I", "J"},
    "M87": {"D", "E", "I", "J", "L"},
}

# R16: which two R32 match winners play each other
R16_MATCHES: List[Tuple[str, str, str]] = [
    ("M89", "M74", "M77"),
    ("M90", "M73", "M75"),
    ("M91", "M76", "M78"),
    ("M92", "M79", "M80"),
    ("M93", "M83", "M84"),
    ("M94", "M81", "M82"),
    ("M95", "M86", "M88"),
    ("M96", "M85", "M87"),
]

# QF: R16 winners bracket
QF_MATCHES: List[Tuple[str, str, str]] = [
    ("QF1", "M89", "M90"),
    ("QF2", "M91", "M92"),
    ("QF3", "M93", "M94"),
    ("QF4", "M95", "M96"),
]

# SF
SF_MATCHES: List[Tuple[str, str, str]] = [
    ("SF1", "QF1", "QF2"),
    ("SF2", "QF3", "QF4"),
]


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
    group_stage_date: str | None = None,
) -> Dict[Tuple[str, str], MatchInfo]:
    """For every C(4,2)=6 matchup in each group, store market probabilities."""
    if group_stage_date is None:
        group_stage_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
    Tiebreaker: pts → H2H pts → H2H GD → overall GD → overall GF (FIFA rules).
    """
    pts: Dict[str, int] = {t: 0 for t in teams}
    gd: Dict[str, int] = {t: 0 for t in teams}
    gf: Dict[str, int] = {t: 0 for t in teams}
    results: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for home, away in combinations(teams, 2):
        key = (home, away)
        rev_key = (away, home)
        if key in completed:
            hg, ag = completed[key]
        elif rev_key in completed:
            ag, hg = completed[rev_key]
        else:
            hg, ag = sample_match(home, away, match_info, rng)

        results[(home, away)] = (hg, ag)
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

    def _h2h_pts(team: str, rivals: List[str]) -> int:
        p = 0
        for opp in rivals:
            if (team, opp) in results:
                hg, ag = results[(team, opp)]
                p += 3 if hg > ag else (1 if hg == ag else 0)
            elif (opp, team) in results:
                ag, hg = results[(opp, team)]
                p += 3 if hg > ag else (1 if hg == ag else 0)
        return p

    def _h2h_gd(team: str, rivals: List[str]) -> int:
        d = 0
        for opp in rivals:
            if (team, opp) in results:
                hg, ag = results[(team, opp)]
                d += hg - ag
            elif (opp, team) in results:
                ag, hg = results[(opp, team)]
                d += hg - ag
        return d

    def _sort_key(team: str, tied_with: List[str]) -> tuple:
        return (pts[team], _h2h_pts(team, tied_with), _h2h_gd(team, tied_with), gd[team], gf[team])

    # Sort with FIFA tiebreaker: group teams by pts, apply H2H within tied sets
    from itertools import groupby
    sorted_by_pts = sorted(teams, key=lambda t: pts[t], reverse=True)
    final_order: List[str] = []
    for _, group_iter in groupby(sorted_by_pts, key=lambda t: pts[t]):
        tied = list(group_iter)
        if len(tied) == 1:
            final_order.extend(tied)
        else:
            tied.sort(key=lambda t: (_h2h_pts(t, [x for x in tied if x != t]),
                                     _h2h_gd(t, [x for x in tied if x != t]),
                                     gd[t], gf[t]), reverse=True)
            final_order.extend(tied)

    return [(t, pts[t], gd[t], gf[t]) for t in final_order]


def _assign_thirds_to_slots(
    qualifying_groups: List[str],
    third_by_group: Dict[str, str],
    rng: np.random.Generator,
) -> Dict[str, str]:
    """
    Assign 8 qualifying third-place teams to their R32 slots via randomised
    augmenting-path bipartite matching.  FIFA guarantees a perfect matching
    exists for every valid set of 8 qualifying groups.

    Returns {slot_id: team_name}.
    """
    eligible: Dict[str, List[str]] = {
        slot: [g for g in groups if g in qualifying_groups]
        for slot, groups in THIRD_SLOT_ELIGIBLE.items()
    }

    match_slot: Dict[str, str] = {}   # slot -> group
    match_group: Dict[str, str] = {}  # group -> slot

    def _augment(slot: str, visited: set) -> bool:
        candidates = eligible[slot].copy()
        rng.shuffle(candidates)
        for grp in candidates:
            if grp not in visited:
                visited.add(grp)
                if grp not in match_group or _augment(match_group[grp], visited):
                    match_slot[slot] = grp
                    match_group[grp] = slot
                    return True
        return False

    slots = list(THIRD_SLOT_ELIGIBLE.keys())
    rng.shuffle(slots)
    for slot in slots:
        _augment(slot, set())

    return {slot: third_by_group[grp] for slot, grp in match_slot.items()}


def _compute_real_third_slots(
    group_standings: Dict[str, List[Tuple[str, int, int, int]]],
    results_df: pd.DataFrame,
) -> Dict[str, str]:
    """Recover the real 3rd-place opponent for any R32 slot already played.

    Group winner/runner-up identity is deterministic once groups are
    complete, so a played LAST_32 fixture against a known W/R team reveals
    the real 3rd-place team for that slot — this must override the random
    bipartite slot assignment, which only applies to *unplayed* R32 ties.
    """
    gw = {g: s[0][0] for g, s in group_standings.items()}
    gr = {g: s[1][0] for g, s in group_standings.items()}
    last32 = results_df[results_df["stage"] == "LAST_32"]
    opponent: Dict[str, str] = {}
    for _, row in last32.iterrows():
        opponent[row["home"]] = row["away"]
        opponent[row["away"]] = row["home"]

    real_slots: Dict[str, str] = {}
    for slot, side_a, side_b in R32_MATCHES:
        for spec, other in ((side_a, side_b), (side_b, side_a)):
            kind, ref = spec
            if kind not in ("W", "R") or other[0] != "3":
                continue
            known_team = gw.get(ref) if kind == "W" else gr.get(ref)
            if known_team in opponent:
                real_slots[other[1]] = opponent[known_team]
    return real_slots


def _actual_winner(
    team_a: str,
    team_b: str,
    results_lookup: Dict[Tuple[str, str], Tuple[int, int, float, float]],
) -> str | None:
    """If this knockout fixture has already been played, return the real winner."""
    for home, away in ((team_a, team_b), (team_b, team_a)):
        row = results_lookup.get((home, away))
        if row is None:
            continue
        hg, ag, hp, ap = row
        if hg != ag:
            return home if hg > ag else away
        if hp == hp and ap == ap:  # not NaN — decided on penalties
            return home if hp > ap else away
    return None


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    tidx: Dict[str, int],
    p_advance: np.ndarray,
    rng: np.random.Generator,
    results_lookup: Dict[Tuple[str, str], Tuple[int, int, float, float]] | None = None,
) -> str:
    """Returns winner. Uses the real result if already played, else simulates."""
    if results_lookup is not None:
        actual = _actual_winner(team_a, team_b, results_lookup)
        if actual is not None:
            return actual
    ia, ib = tidx[team_a], tidx[team_b]
    p = p_advance[ia, ib]
    return team_a if rng.random() < p else team_b


def run_simulation(
    completed: Dict[Tuple[str, str], Tuple[int, int]],
    match_info: Dict[Tuple[str, str], MatchInfo],
    tidx: Dict[str, int],
    p_advance: np.ndarray,
    rng: np.random.Generator,
    results_lookup: Dict[Tuple[str, str], Tuple[int, int, float, float]] | None = None,
    real_third_slot: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, int]]:
    """
    Simulate one full WC 2026 tournament using the actual FIFA bracket.
    Returns {stage: {team: 1}} for stages: qualify, r16, qf, sf, final, win.
    """
    reached: Dict[str, Dict[str, int]] = {
        "qualify": {}, "r16": {}, "qf": {}, "sf": {}, "final": {}, "win": {},
    }

    # --- Group stage ---
    group_standings: Dict[str, List[Tuple[str, int, int, int]]] = {}
    third_place_all: List[Tuple[int, int, int, str]] = []

    for group, teams in GROUPS.items():
        standing = simulate_group(group, teams, completed, match_info, rng)
        group_standings[group] = standing
        for pos, (team, pts, gd, gf_val) in enumerate(standing):
            if pos < 2:
                reached["qualify"][team] = 1
            elif pos == 2:
                third_place_all.append((pts, gd, gf_val, team))

    # Best 8 third-place teams qualify
    third_place_all.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    qualifying_thirds = third_place_all[:8]
    for _, _, _, team in qualifying_thirds:
        reached["qualify"][team] = 1

    # Group position lookups
    gw: Dict[str, str] = {g: s[0][0] for g, s in group_standings.items()}   # winner
    gr: Dict[str, str] = {g: s[1][0] for g, s in group_standings.items()}   # runner-up
    third_by_group: Dict[str, str] = {
        TEAM_TO_GROUP[t]: t for _, _, _, t in qualifying_thirds
    }
    qualifying_groups = list(third_by_group.keys())

    # Assign 3rd-place teams to their eligible R32 slots via bipartite matching
    third_slot: Dict[str, str] = _assign_thirds_to_slots(
        qualifying_groups, third_by_group, rng
    )

    def _resolve(spec: Tuple[str, str]) -> str:
        kind, ref = spec
        if kind == "W":
            return gw[ref]
        if kind == "R":
            return gr[ref]
        # ref = slot_id for 3rd-place. Once this slot's fixture has actually
        # been played, its real opponent is fixed — don't let the random
        # bipartite matching substitute a different (fictional) team.
        if real_third_slot and ref in real_third_slot:
            return real_third_slot[ref]
        return third_slot.get(ref, "")

    # --- R32 ---
    r32_winners: Dict[str, str] = {}
    for slot, side_a, side_b in R32_MATCHES:
        team_a, team_b = _resolve(side_a), _resolve(side_b)
        w = simulate_knockout_match(team_a, team_b, tidx, p_advance, rng, results_lookup)
        r32_winners[slot] = w
        reached["r16"][w] = 1

    # --- R16 ---
    r16_winners: Dict[str, str] = {}
    for slot, src_a, src_b in R16_MATCHES:
        team_a, team_b = r32_winners[src_a], r32_winners[src_b]
        w = simulate_knockout_match(team_a, team_b, tidx, p_advance, rng, results_lookup)
        r16_winners[slot] = w
        reached["qf"][w] = 1

    # --- QF ---
    qf_winners: Dict[str, str] = {}
    for slot, src_a, src_b in QF_MATCHES:
        team_a, team_b = r16_winners[src_a], r16_winners[src_b]
        w = simulate_knockout_match(team_a, team_b, tidx, p_advance, rng, results_lookup)
        qf_winners[slot] = w
        reached["sf"][w] = 1

    # --- SF ---
    sf_winners: Dict[str, str] = {}
    for slot, src_a, src_b in SF_MATCHES:
        team_a, team_b = qf_winners[src_a], qf_winners[src_b]
        w = simulate_knockout_match(team_a, team_b, tidx, p_advance, rng, results_lookup)
        sf_winners[slot] = w
        reached["final"][w] = 1

    # --- Final ---
    champion = simulate_knockout_match(sf_winners["SF1"], sf_winners["SF2"], tidx, p_advance, rng, results_lookup)
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
    results_lookup: Dict[Tuple[str, str], Tuple[int, int, float, float]] = {}
    for _, row in results_df.iterrows():
        key = (row["home"], row["away"])
        completed[key] = (int(row["home_goals"]), int(row["away_goals"]))
        results_lookup[key] = (
            int(row["home_goals"]), int(row["away_goals"]),
            row.get("home_pens", float("nan")), row.get("away_pens", float("nan")),
        )
    log.info(f"  {len(completed)} completed matches loaded.")

    hist = pd.read_parquet("data/historical_matches.parquet")
    all_matches = pd.concat([hist, results_df], ignore_index=True)

    log.info("Pre-computing group match info...")
    match_info = build_group_match_info(params, all_matches)

    tidx, p_advance = build_advance_matrix(params, all_matches)

    # Group standings are deterministic once every group match is in `completed`
    # — compute once to derive any already-played 3rd-place R32 matchups.
    group_standings_real = {
        group: simulate_group(group, teams, completed, match_info, rng)
        for group, teams in GROUPS.items()
    }
    real_third_slot = _compute_real_third_slots(group_standings_real, results_df)
    if real_third_slot:
        log.info(f"  {len(real_third_slot)} R32 3rd-place slots pinned to real results.")

    # Accumulators: stage → team → count
    counts: Dict[str, Dict[str, int]] = {
        stage: {t: 0 for t in ALL_TEAMS}
        for stage in ["qualify", "r16", "qf", "sf", "final", "win"]
    }

    log.info(f"Running {N_SIMS:,} simulations...")
    for sim_i in range(N_SIMS):
        if (sim_i + 1) % 1000 == 0:
            log.info(f"  {sim_i + 1:,} / {N_SIMS:,}")
        result = run_simulation(completed, match_info, tidx, p_advance, rng, results_lookup, real_third_slot)
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
