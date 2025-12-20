"""
Generate qualitative examples report for comparison.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def generate_examples_report(
    baseline_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    output_path: str,
    n_examples_per_split: int = 5,
    best_config: Optional[dict] = None,
):
    """
    Generate a markdown report with qualitative examples.

    Args:
        baseline_df: Baseline results DataFrame
        sweep_df: Sweep results DataFrame
        output_path: Path to save markdown report
        n_examples_per_split: Number of examples per split
        best_config: Optional dict with layer, alpha, intervention_type for comparison
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If no best_config provided, find the best one
    if best_config is None:
        # Find config with highest refusal on harm_test
        summary = sweep_df.groupby(["layer", "alpha", "intervention_type", "split"]).agg({
            "is_refusal": "mean"
        }).reset_index()

        harm_test_summary = summary[summary["split"] == "harm_test"]
        best_row = harm_test_summary.sort_values("is_refusal", ascending=False).iloc[0]

        best_config = {
            "layer": int(best_row["layer"]),
            "alpha": best_row["alpha"],
            "intervention_type": best_row["intervention_type"]
        }

    with open(output_path, "w") as f:
        f.write("# Qualitative Examples Report\n\n")
        f.write("Comparison of baseline vs steered completions.\n\n")

        f.write(f"## Best Steering Configuration\n\n")
        f.write(f"- **Layer:** {best_config['layer']}\n")
        f.write(f"- **Intervention:** {best_config['intervention_type']}\n")
        f.write(f"- **Alpha:** {best_config['alpha']}\n\n")

        f.write("---\n\n")

        # Generate examples for each split
        for split in ["harm_train", "harm_test", "benign"]:
            f.write(f"## {split.replace('_', ' ').title()} Examples\n\n")

            # Get baseline examples
            baseline_examples = baseline_df[baseline_df["split"] == split].sample(
                min(n_examples_per_split, len(baseline_df[baseline_df["split"] == split]))
            )

            # Get corresponding steered examples (same prompts if possible)
            steered_filter = (
                (sweep_df["split"] == split) &
                (sweep_df["layer"] == best_config["layer"]) &
                (sweep_df["alpha"] == best_config["alpha"]) &
                (sweep_df["intervention_type"] == best_config["intervention_type"])
            )

            steered_examples = sweep_df[steered_filter].sample(
                min(n_examples_per_split, steered_filter.sum())
            )

            # Write examples
            for i, (baseline_row, steered_row) in enumerate(zip(
                baseline_examples.itertuples(),
                steered_examples.itertuples()
            )):
                f.write(f"### Example {i+1}\n\n")

                # Prompt
                f.write(f"**Prompt:**\n```\n{baseline_row.prompt}\n```\n\n")

                # Baseline
                f.write(f"**Baseline Completion:**\n```\n{baseline_row.completion}\n```\n")
                f.write(f"- Refusal Score: {baseline_row.refusal_score:.3f}\n")
                f.write(f"- Is Refusal: {baseline_row.is_refusal}\n")
                f.write(f"- Helpfulness Score: {baseline_row.helpfulness_score:.3f}\n")
                f.write(f"- Is Helpful: {baseline_row.is_helpful}\n\n")

                # Steered
                f.write(f"**Steered Completion:**\n```\n{steered_row.completion}\n```\n")
                f.write(f"- Refusal Score: {steered_row.refusal_score:.3f}\n")
                f.write(f"- Is Refusal: {steered_row.is_refusal}\n")
                f.write(f"- Helpfulness Score: {steered_row.helpfulness_score:.3f}\n")
                f.write(f"- Is Helpful: {steered_row.is_helpful}\n\n")

                # Analysis
                refusal_change = steered_row.refusal_score - baseline_row.refusal_score
                helpful_change = steered_row.helpfulness_score - baseline_row.helpfulness_score

                f.write(f"**Analysis:**\n")
                f.write(f"- Refusal change: {refusal_change:+.3f}\n")
                f.write(f"- Helpfulness change: {helpful_change:+.3f}\n\n")

                f.write("---\n\n")

        f.write("\n## Summary\n\n")
        f.write(f"This report shows {n_examples_per_split} examples per split ")
        f.write(f"comparing baseline (no steering) with the best performing steering configuration.\n\n")

        # Add aggregate stats
        baseline_stats = baseline_df.groupby("split").agg({
            "is_refusal": "mean",
            "is_helpful": "mean"
        }).round(3)

        steered_stats = sweep_df[
            (sweep_df["layer"] == best_config["layer"]) &
            (sweep_df["alpha"] == best_config["alpha"]) &
            (sweep_df["intervention_type"] == best_config["intervention_type"])
        ].groupby("split").agg({
            "is_refusal": "mean",
            "is_helpful": "mean"
        }).round(3)

        f.write("### Aggregate Statistics\n\n")
        f.write("| Split | Baseline Refusal | Steered Refusal | Baseline Helpful | Steered Helpful |\n")
        f.write("|-------|-----------------|----------------|-----------------|----------------|\n")

        for split in ["harm_train", "harm_test", "benign"]:
            if split in baseline_stats.index and split in steered_stats.index:
                f.write(f"| {split} | ")
                f.write(f"{baseline_stats.loc[split, 'is_refusal']:.3f} | ")
                f.write(f"{steered_stats.loc[split, 'is_refusal']:.3f} | ")
                f.write(f"{baseline_stats.loc[split, 'is_helpful']:.3f} | ")
                f.write(f"{steered_stats.loc[split, 'is_helpful']:.3f} |\n")

    print(f"✓ Examples report saved to {output_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate qualitative examples report")
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Directory containing baseline and sweep results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/examples.md",
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=5,
        help="Number of examples per split"
    )

    args = parser.parse_args()

    # Load results
    runs_dir = Path(args.runs_dir)
    baseline_df = pd.read_parquet(runs_dir / "baseline_results.parquet")
    sweep_df = pd.read_parquet(runs_dir / "sweep_results.parquet")

    # Generate report
    generate_examples_report(
        baseline_df,
        sweep_df,
        args.output,
        n_examples_per_split=args.n_examples
    )


if __name__ == "__main__":
    main()
