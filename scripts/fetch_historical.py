"""Download and clean 30+ years of international football results."""
import io
import logging
from pathlib import Path

import pandas as pd
import requests

_URLS = [
    "https://raw.githubusercontent.com/JamshedAli18/International-football-results-from-1872-to-2024/main/results.csv",
    "https://raw.githubusercontent.com/JamshedAli18/International-football-results-from-1872-to-2024/master/results.csv",
]
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "historical_matches.parquet"
RAW_PATH = DATA_DIR / "raw" / "results.csv"

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.9,
    "Copa América": 0.9,
    "Africa Cup of Nations": 0.9,
    "AFC Asian Cup": 0.85,
    "CONCACAF Gold Cup": 0.85,
    "FIFA World Cup qualification": 0.85,
    "UEFA Euro qualification": 0.8,
    "UEFA Nations League": 0.8,
    "CONCACAF Nations League": 0.75,
    "Copa América qualification": 0.65,
    "Friendly": 0.15,
}


def _tournament_weight(t: str) -> float:
    for k, v in TOURNAMENT_WEIGHTS.items():
        if k in str(t):
            return v
    return 0.65


def fetch() -> pd.DataFrame:
    for url in _URLS:
        logging.info("Downloading %s", url)
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_PATH.write_bytes(r.content)
            return pd.read_csv(io.BytesIO(r.content), parse_dates=["date"])
        logging.warning("Got %d — trying next URL", r.status_code)
    raise RuntimeError("All data sources failed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "home_team": "home",
        "away_team": "away",
        "home_score": "home_goals",
        "away_score": "away_goals",
    })
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False, errors="coerce")
    df = df[df["date"] >= pd.Timestamp("2010-01-01")].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["tournament_weight"] = df["tournament"].map(_tournament_weight)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df = clean(df)
    df.to_parquet(OUT_PATH, index=False)
    logging.info("Saved %d matches → %s", len(df), OUT_PATH)


if __name__ == "__main__":
    main()
