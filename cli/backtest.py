"""CLI: walk-forward backtest.

Usage:
  python -m cli.backtest --start 2018-01-01 --end 2024-12-31
  python -m cli.backtest --start 2018-01-01 --end 2024-12-31 --refit-days 30
"""
import argparse
import logging
from pathlib import Path

from backtest.walk_forward import run

REPORTS = Path(__file__).parent.parent / "reports"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward backtest.")
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--refit-days", type=int, default=30,
                        help="Days between model refits (default: 30)")
    args = parser.parse_args()

    results = run(start=args.start, end=args.end, refit_freq_days=args.refit_days)

    if not results.empty:
        out = REPORTS / "backtest_results.parquet"
        REPORTS.mkdir(exist_ok=True)
        results.to_parquet(out, index=False)
        print(f"\nResults:           {out}")
        print(f"Calibration plot:  {REPORTS / 'calibration.png'}")
        print(f"Log-loss curve:    {REPORTS / 'log_loss_curve.png'}")


if __name__ == "__main__":
    main()
