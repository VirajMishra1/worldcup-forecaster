"""Squad-strength ratio for lineup adjustment.

scoreline_grid accepts home_lineup_ratio / away_lineup_ratio where
  eps = 0.4 * log(clip(ratio, 0.5, 1.5))

We pass ratio^0.375 so the effective coefficient is 0.4 * 0.375 = 0.15,
which avoids over-correcting teams the model already rates via match history.
"""
import json
import math
from pathlib import Path

_PATH = Path(__file__).parent.parent / "data" / "squad_values.json"
_cache: dict[str, float] | None = None


def _load() -> dict[str, float]:
    global _cache
    if _cache is None:
        d = json.loads(_PATH.read_text())
        _cache = {k: float(v) for k, v in d.items() if k != "_note" and k != "_mean"}
    return _cache


def squad_ratio(team: str) -> float:
    """Return lineup_ratio for team (pass directly to scoreline_grid)."""
    vals = _load()
    if team not in vals:
        return 1.0
    mean = sum(vals.values()) / len(vals)
    raw = vals[team] / mean
    return raw ** 0.375
