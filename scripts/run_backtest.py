"""Re-run walk-forward backtest and save results + charts."""
import logging
from pathlib import Path

from backtest.walk_forward import run

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

results = run()
if not results.empty:
    out = REPORTS / "backtest_results.parquet"
    results.to_parquet(out, index=False)
    print(f"Saved {len(results)} rows to {out}")
