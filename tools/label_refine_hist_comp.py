
"""
Label distribution (original vs refined) histogram utility.

- Reads a point CSV containing columns: label, label_refined
- Computes per-class counts and saves an intermediate stats CSV
- Saves a side-by-side histogram figure

Usage
-----
python label_refine_hist_comp.py --input your_file.csv \
    --stats_out label_histogram_stats.csv \
    --fig_out label_histogram.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_INPUT_CSV = "/home/fzhcis/Downloads/pcd_ALRSET12_7489_refined.csv"
CLASS_NAMES = {
                0: "Void",
                1: "Ground & Water",
                2: "Stem",
                3: "Canopy",
                4: "Roots",
                5: "Objects"
            }

@dataclass
class LabelHistConfig:
    input_csv: Optional[Path] = None
    stats_out: Path = Path("label_histogram_stats.csv")
    fig_out: Path = Path("label_histogram.png")
    label_col: str = "label"
    refined_col: str = "label_refined"
    dpi: int = 300
    figsize: Tuple[float, float] = (6.0, 5.0)
    bar_width: float = 0.35
    annotate: bool = True
    annotate_fontsize: int = 12
    logy: bool = False


# -----------------------------
# Main utility class
# -----------------------------
class LabelHistogram:
    def __init__(self, cfg: LabelHistConfig):
        self.cfg = cfg

    def compute_stats_from_points(self) -> pd.DataFrame:
        """Read the raw point CSV and compute per-class counts for both columns."""
        if self.cfg.input_csv is None:
            raise ValueError("input_csv is required to compute stats from point CSV.")

        df = pd.read_csv(self.cfg.input_csv)

        for col in (self.cfg.label_col, self.cfg.refined_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in {self.cfg.input_csv}")

        # Ensure integer labels (safe even if already int)
        df[self.cfg.label_col] = df[self.cfg.label_col].astype(int)
        df[self.cfg.refined_col] = df[self.cfg.refined_col].astype(int)

        all_labels = np.sort(
            np.unique(
                np.concatenate([df[self.cfg.label_col].values, df[self.cfg.refined_col].values])
            )
        )

        original_counts = df[self.cfg.label_col].value_counts().reindex(all_labels, fill_value=0)
        refined_counts = df[self.cfg.refined_col].value_counts().reindex(all_labels, fill_value=0)

        stats = pd.DataFrame(
            {
                "class_label": all_labels,
                "original_count": original_counts.values,
                "refined_count": refined_counts.values,
            }
        )
        return stats

    def save_stats(self, stats: pd.DataFrame) -> None:
        """Save stats to CSV for easy replotting later."""
        self.cfg.stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(self.cfg.stats_out, index=False)

    def _annotate_bars(self, ax: plt.Axes, bars, labels, offset: float = 0, color: str = 'black') -> None:
        """Add labels on top of bars (strings provided by caller)."""
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=self.cfg.annotate_fontsize,
                rotation=0,
                color=color,
            )

    def plot_from_stats(self, stats: pd.DataFrame) -> None:
        """Plot side-by-side bar chart from stats and save figure."""
        stats_15 = stats[stats["class_label"].between(1, 5)]
        labels = stats_15["class_label"].values
        orig = stats_15["original_count"].values
        ref = stats_15["refined_count"].values

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=self.cfg.figsize)

        bars1 = ax.bar(
            x - self.cfg.bar_width / 2,
            orig,
            width=self.cfg.bar_width,
            label="Original label",
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + self.cfg.bar_width / 2,
            ref,
            width=self.cfg.bar_width,
            label="Refined label",
            edgecolor="black",
        )

        # ax.set_xlabel("Class label", fontsize=11)
        ax.set_ylabel("Number of points", fontsize=11)
        ax.set_xticks(x)
        # Create tick labels with class names
        tick_labels = [f"{CLASS_NAMES.get(int(lbl), 'Unknown')}" for lbl in labels]
        ax.set_xticklabels(tick_labels)

        if self.cfg.logy:
            ax.set_yscale("log")

        # ax.legend(frameon=False, loc='upper right')
        ax.legend()
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        if self.cfg.annotate:
            orig_labels = [f"{int(v):,}" for v in orig]
            ref_labels = []
            for o, r in zip(orig, ref):
                if o == 0:
                    pct = "NA"
                else:
                    pct = f"{(r - o) / o * 100:+.1f}%"
                ref_labels.append(f"{int(r):,}\n({pct})")

            # Calculate offset to prevent overlap (use 4% of y-axis range for more spacing)
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = y_range * 0.04
            
            # Use bar colors for annotations
            self._annotate_bars(ax, bars1, orig_labels, offset=0, color='tab:blue')
            self._annotate_bars(ax, bars2, ref_labels, offset=offset, color='tab:orange')

        # Add some headroom at top for annotations
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=y_max * 1.08)
        
        fig.tight_layout()
        self.cfg.fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.cfg.fig_out, dpi=self.cfg.dpi)
        plt.close(fig)

    def run(self) -> None:
        """Compute stats from point CSV, save stats, and save histogram figure."""
        stats_df = self.compute_stats_from_points()
        self.save_stats(stats_df)
        print(f"[OK] Saved stats CSV: {self.cfg.stats_out}")
        self.plot_from_stats(stats_df)
        print(f"[OK] Saved figure: {self.cfg.fig_out}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot label vs label_refined histograms with saved intermediate stats.")
    p.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Path to raw point CSV (must include label columns).",
    )
    p.add_argument("--stats_out", type=str, default=None, help="Output stats CSV path.")
    p.add_argument("--fig_out", type=str, default=None, help="Output figure path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input)

    stats_out = (
        Path(args.stats_out)
        if args.stats_out is not None
        else input_csv.parent / "label_histogram_stats.csv"
    )
    fig_out = (
        Path(args.fig_out) if args.fig_out is not None else input_csv.parent / "label_refinement_histogram.png"
    )

    cfg = LabelHistConfig(
        input_csv=input_csv,
        stats_out=stats_out,
        fig_out=fig_out,
    )

    LabelHistogram(cfg).run()


if __name__ == "__main__":
    main()
