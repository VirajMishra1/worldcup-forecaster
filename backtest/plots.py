"""Reliability diagram and log-loss curve plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def reliability_diagram(
    cal_df: pd.DataFrame,
    out: Path = REPORTS / "calibration.png",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    colors = {"H": "#2196F3", "D": "#9E9E9E", "A": "#F44336"}
    labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
    for market, grp in cal_df.groupby("market"):
        ax.scatter(
            grp["mean_predicted"],
            grp["mean_actual"],
            s=grp["count"] * 3,
            color=colors.get(str(market), "black"),
            label=labels.get(str(market), str(market)),
            alpha=0.8,
        )
    ax.set(
        xlabel="Predicted probability",
        ylabel="Actual frequency",
        title="Reliability diagram — bivariate Poisson + Dixon-Coles",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def log_loss_curve(
    results: pd.DataFrame,
    out: Path = REPORTS / "log_loss_curve.png",
) -> None:
    """results: DataFrame with date, log_loss, optionally pinnacle_log_loss."""
    fig, ax = plt.subplots(figsize=(10, 4))
    results = results.sort_values("date").reset_index(drop=True)
    results["cum_ll"] = results["log_loss"].expanding().mean()
    ax.plot(results["date"], results["cum_ll"], label="Model", color="#2196F3", lw=2)
    ax.axhline(1.099, color="red", linestyle="--", lw=1, label="Random (1.099)")
    if "pinnacle_log_loss" in results.columns:
        results["cum_pin"] = results["pinnacle_log_loss"].expanding().mean()
        ax.plot(
            results["date"], results["cum_pin"],
            label="Pinnacle", color="#4CAF50", lw=1.5, linestyle=":",
        )
    ax.set(
        xlabel="Date",
        ylabel="Cumulative mean log-loss (lower = better)",
        title="Model log-loss vs random baseline",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
