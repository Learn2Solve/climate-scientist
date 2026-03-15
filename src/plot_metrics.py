#!/usr/bin/env python3
"""Plot track/wind MAE vs lead hours from metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("lead_hours")
        ax.plot(sub["lead_hours"], sub[metric], marker="o", label=model)
    ax.set_xlabel("Lead hours")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs Lead")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAE curves from metrics CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        df,
        metric="track_mae_km_mean",
        ylabel="Track MAE (km, mean)",
        out_path=args.out_dir / "track_mae_mean.png",
    )
    plot_metric(
        df,
        metric="wind_mae_mean",
        ylabel="Wind MAE (kt, mean)",
        out_path=args.out_dir / "wind_mae_mean.png",
    )


if __name__ == "__main__":
    main()
