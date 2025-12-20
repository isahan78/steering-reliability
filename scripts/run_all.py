#!/usr/bin/env python3
"""
One-shot runner for the complete steering reliability pipeline.

Runs:
1. Generate prompts (if needed)
2. Run baseline experiment
3. Run steering sweep
4. Generate plots and summary tables
5. Generate qualitative examples

Usage:
    python scripts/run_all.py --config configs/default.yaml
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steering_reliability.config import load_config
from steering_reliability.model import load_model
from steering_reliability.experiments.run_baseline import run_baseline_experiment
from steering_reliability.experiments.run_sweep import run_sweep_experiment
from steering_reliability.analysis.plots import generate_all_plots
from steering_reliability.analysis.aggregate import create_summary_table


def main():
    parser = argparse.ArgumentParser(
        description="Run complete steering reliability pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline experiment"
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip sweep experiment"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STEERING RELIABILITY - FULL PIPELINE")
    print("=" * 70)
    print()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Config loaded from {args.config}")
    print(f"  Model: {config.model.name}")
    print(f"  Device: {config.model.device}")
    print(f"  Output: {config.experiment.output_dir}")
    print()

    # Load model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    model = load_model(
        config.model.name,
        config.model.device,
        config.model.dtype,
    )
    print()

    # Run baseline
    if not args.skip_baseline:
        print("=" * 70)
        print("STEP 1: BASELINE EXPERIMENT")
        print("=" * 70)
        run_baseline_experiment(model, config, save_results=True)
        print()

    # Run sweep
    if not args.skip_sweep:
        print("=" * 70)
        print("STEP 2: STEERING SWEEP")
        print("=" * 70)
        run_sweep_experiment(model, config, save_results=True)
        print()

    # Generate summary table
    print("=" * 70)
    print("STEP 3: SUMMARY TABLE")
    print("=" * 70)
    output_dir = Path(config.experiment.output_dir)
    tables_dir = output_dir.parent.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tables_dir / "summary.csv"
    summary = create_summary_table(output_dir, summary_path)
    print()

    # Generate plots
    if not args.skip_plots:
        print("=" * 70)
        print("STEP 4: GENERATE PLOTS")
        print("=" * 70)
        figures_dir = output_dir.parent.parent / "figures"
        generate_all_plots(output_dir, figures_dir)
        print()

    # Done!
    print("=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {config.experiment.output_dir}")
    print(f"Plots saved to: {figures_dir}")
    print(f"Summary table: {summary_path}")
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
