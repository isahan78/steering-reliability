"""
Aggregate and analyze experimental results.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def combine_baseline_and_sweep(
    baseline_path: str,
    sweep_path: str,
) -> pd.DataFrame:
    """
    Combine baseline and sweep results into a single DataFrame.

    Args:
        baseline_path: Path to baseline results parquet
        sweep_path: Path to sweep results parquet

    Returns:
        Combined DataFrame
    """
    baseline_df = pd.read_parquet(baseline_path)
    sweep_df = pd.read_parquet(sweep_path)

    # Combine
    df = pd.concat([baseline_df, sweep_df], ignore_index=True)

    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics across conditions.

    Args:
        df: Results DataFrame

    Returns:
        Summary DataFrame with aggregated metrics
    """
    # Group by experimental conditions
    group_cols = ["split", "layer", "alpha", "intervention_type"]

    # Filter to only include columns that exist
    existing_group_cols = [col for col in group_cols if col in df.columns]

    summary = df.groupby(existing_group_cols).agg({
        "refusal_score": ["mean", "std"],
        "is_refusal": "mean",  # Refusal rate
        "helpfulness_score": ["mean", "std"],
        "is_helpful": "mean",  # Helpfulness rate
        "char_length": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in summary.columns.values
    ]

    return summary


def compute_generalization_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute generalization gap between harm_train and harm_test.

    Gap = refusal_rate(harm_train) - refusal_rate(harm_test)

    Args:
        summary: Summary DataFrame

    Returns:
        DataFrame with generalization gaps
    """
    # Filter to harm splits only
    harm_summary = summary[summary["split"].isin(["harm_train", "harm_test"])]

    # Pivot to get train and test side by side
    pivot = harm_summary.pivot_table(
        index=["layer", "alpha", "intervention_type"],
        columns="split",
        values="is_refusal_mean"
    ).reset_index()

    # Compute gap
    if "harm_train" in pivot.columns and "harm_test" in pivot.columns:
        pivot["generalization_gap"] = pivot["harm_train"] - pivot["harm_test"]
    else:
        pivot["generalization_gap"] = 0.0

    return pivot


def compute_side_effect_drop(
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute side effect: helpfulness drop on benign prompts.

    Drop = helpful_rate(benign, baseline) - helpful_rate(benign, steered)

    Args:
        summary: Summary DataFrame with all conditions
        baseline_summary: Summary for baseline only

    Returns:
        DataFrame with side effect metrics
    """
    # Get baseline benign helpfulness
    baseline_benign = baseline_summary[
        baseline_summary["split"] == "benign"
    ]["is_helpful_mean"].values

    if len(baseline_benign) > 0:
        baseline_helpful_rate = baseline_benign[0]
    else:
        baseline_helpful_rate = 0.0

    # Filter to benign + steering conditions
    benign_summary = summary[
        (summary["split"] == "benign") &
        (summary["intervention_type"] != "none")
    ].copy()

    # Compute drop
    benign_summary["helpfulness_drop"] = (
        baseline_helpful_rate - benign_summary["is_helpful_mean"]
    )

    return benign_summary


def create_summary_table(
    results_dir: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a summary table from experiment results.

    Args:
        results_dir: Directory containing baseline_results.parquet and sweep_results.parquet
        output_path: Optional path to save summary CSV

    Returns:
        Summary DataFrame
    """
    results_dir = Path(results_dir)

    # Load results
    baseline_path = results_dir / "baseline_results.parquet"
    sweep_path = results_dir / "sweep_results.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {baseline_path}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep results not found: {sweep_path}")

    # Combine
    df = combine_baseline_and_sweep(baseline_path, sweep_path)

    # Compute summary
    summary = compute_summary_stats(df)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"✓ Saved summary table to {output_path}")

    return summary
