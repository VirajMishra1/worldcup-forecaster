"""CLI: per-market backtest statistics.

Usage:
  uv run python3 -m cli.market_stats
"""
from pathlib import Path

import pandas as pd

from backtest.metrics import aggregate_metrics, binary_market_metrics
from backtest.plots import market_calibration_plot

REPORTS = Path(__file__).parent.parent / "reports"
PARQUET = REPORTS / "backtest_results.parquet"


def main() -> None:
    df = pd.read_parquet(PARQUET)

    wdl = aggregate_metrics(df)
    ou = binary_market_metrics(df.dropna(subset=["p_over_25"]), "p_over_25", "actual_over_25")
    btts = binary_market_metrics(df.dropna(subset=["p_btts"]), "p_btts", "actual_btts")

    header = f"{'Market':<12} {'Accuracy':>10} {'LogLoss':>10} {'N':>6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'W/D/L':<12} {wdl['accuracy']*100:>9.1f}% {wdl['log_loss']:>10.4f} {wdl['n']:>6}")
    print(f"{'O/U 2.5':<12} {ou['accuracy']*100:>9.1f}% {ou['log_loss']:>10.4f} {ou['n']:>6}")
    print(f"{'BTTS':<12} {btts['accuracy']*100:>9.1f}% {btts['log_loss']:>10.4f} {btts['n']:>6}")
    print(sep)

    sub = df.dropna(subset=["p_over_25", "p_btts"])
    market_calibration_plot(sub)
    print(f"\nCalibration chart saved to {REPORTS / 'market_calibration.png'}")


if __name__ == "__main__":
    main()
