"""
Plotting functions for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_generalization_gap(
    summary: pd.DataFrame,
    output_path: Optional[str] = None,
):
    """
    Plot generalization gap vs alpha for each layer and intervention type.

    Gap = refusal_rate(harm_train) - refusal_rate(harm_test)

    Args:
        summary: Summary DataFrame
        output_path: Optional path to save figure
    """
    # Filter to harm splits
    harm_data = summary[summary["split"].isin(["harm_train", "harm_test"])]

    # Pivot to compute gap
    pivot = harm_data.pivot_table(
        index=["layer", "alpha", "intervention_type"],
        columns="split",
        values="is_refusal_mean"
    ).reset_index()

    if "harm_train" not in pivot.columns or "harm_test" not in pivot.columns:
        print("Warning: Cannot compute generalization gap (missing data)")
        return

    pivot["gap"] = pivot["harm_train"] - pivot["harm_test"]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, intervention in enumerate(["add", "ablate"]):
        ax = axes[idx]
        data = pivot[pivot["intervention_type"] == intervention]

        for layer in data["layer"].unique():
            layer_data = data[data["layer"] == layer]
            ax.plot(
                layer_data["alpha"],
                layer_data["gap"],
                marker='o',
                label=f"Layer {layer}"
            )

        ax.set_xlabel("Alpha (Steering Strength)")
        ax.set_ylabel("Generalization Gap")
        ax.set_title(f"Intervention: {intervention}")
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Generalization Gap: harm_train vs harm_test", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved generalization gap plot to {output_path}")

    plt.close()


def plot_tradeoff_curve(
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    output_path: Optional[str] = None,
):
    """
    Plot tradeoff curve: refusal_rate(harm_test) vs helpfulness_drop(benign).

    Args:
        summary: Summary DataFrame with all conditions
        baseline_summary: Baseline summary
        output_path: Optional path to save figure
    """
    # Get baseline benign helpfulness
    baseline_benign = baseline_summary[
        baseline_summary["split"] == "benign"
    ]

    if len(baseline_benign) == 0:
        print("Warning: No baseline benign data found")
        return

    baseline_helpful = baseline_benign["is_helpful_mean"].values[0]

    # Get harm_test refusal rates
    harm_test = summary[
        (summary["split"] == "harm_test") &
        (summary["intervention_type"] != "none")
    ][["layer", "alpha", "intervention_type", "is_refusal_mean"]].copy()

    # Get benign helpfulness
    benign = summary[
        (summary["split"] == "benign") &
        (summary["intervention_type"] != "none")
    ][["layer", "alpha", "intervention_type", "is_helpful_mean"]].copy()

    # Merge
    merged = harm_test.merge(
        benign,
        on=["layer", "alpha", "intervention_type"],
        suffixes=("_harm", "_benign")
    )

    # Compute helpfulness drop
    merged["helpfulness_drop"] = baseline_helpful - merged["is_helpful_mean"]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, intervention in enumerate(["add", "ablate"]):
        ax = axes[idx]
        data = merged[merged["intervention_type"] == intervention]

        for layer in data["layer"].unique():
            layer_data = data[data["layer"] == layer]
            ax.scatter(
                layer_data["is_refusal_mean"],
                layer_data["helpfulness_drop"],
                label=f"Layer {layer}",
                s=100,
                alpha=0.7
            )

        ax.set_xlabel("Refusal Rate on harm_test")
        ax.set_ylabel("Helpfulness Drop on benign")
        ax.set_title(f"Intervention: {intervention}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Tradeoff: Refusal vs Side Effects", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved tradeoff curve to {output_path}")

    plt.close()


def plot_layer_alpha_heatmap(
    summary: pd.DataFrame,
    split: str = "harm_test",
    metric: str = "is_refusal_mean",
    output_path: Optional[str] = None,
):
    """
    Plot heatmap of metric across layer × alpha grid.

    Args:
        summary: Summary DataFrame
        split: Dataset split to plot
        metric: Metric column name
        output_path: Optional path to save figure
    """
    data = summary[
        (summary["split"] == split) &
        (summary["intervention_type"] != "none")
    ]

    if len(data) == 0:
        print(f"Warning: No data for split={split}")
        return

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, intervention in enumerate(["add", "ablate"]):
        ax = axes[idx]
        subset = data[data["intervention_type"] == intervention]

        # Pivot for heatmap
        pivot = subset.pivot_table(
            index="layer",
            columns="alpha",
            values=metric
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r" if "refusal" in metric else "RdYlGn",
            ax=ax,
            cbar_kws={"label": metric}
        )

        ax.set_title(f"Intervention: {intervention}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Layer")

    plt.suptitle(f"Heatmap: {metric} on {split}", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {output_path}")

    plt.close()


def generate_all_plots(
    results_dir: str,
    output_dir: str,
):
    """
    Generate all required plots.

    Args:
        results_dir: Directory containing results parquet files
        output_dir: Directory to save plots
    """
    from .aggregate import combine_baseline_and_sweep, compute_summary_stats

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = combine_baseline_and_sweep(
        results_dir / "baseline_results.parquet",
        results_dir / "sweep_results.parquet",
    )

    print("Computing summary statistics...")
    summary = compute_summary_stats(df)
    baseline_summary = summary[summary["intervention_type"] == "none"]

    print("\nGenerating plots...")
    print("1. Generalization gap...")
    plot_generalization_gap(
        summary,
        output_dir / "generalization_gap.png"
    )

    print("2. Tradeoff curve...")
    plot_tradeoff_curve(
        summary,
        baseline_summary,
        output_dir / "tradeoff_curve.png"
    )

    print("3. Refusal rate heatmap...")
    plot_layer_alpha_heatmap(
        summary,
        split="harm_test",
        metric="is_refusal_mean",
        output_path=output_dir / "heatmap_refusal_harm_test.png"
    )

    print("4. Helpfulness rate heatmap...")
    plot_layer_alpha_heatmap(
        summary,
        split="benign",
        metric="is_helpful_mean",
        output_path=output_dir / "heatmap_helpfulness_benign.png"
    )

    print(f"\n✓ All plots saved to {output_dir}")
