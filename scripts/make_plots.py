#!/usr/bin/env python3
"""
Generate all analysis plots from experiment results.

Usage:
    python scripts/make_plots.py --runs_dir artifacts/runs/run_001 --out_dir artifacts/figures
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steering_reliability.analysis.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from experiment results"
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Directory containing baseline_results.parquet and sweep_results.parquet"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for figures"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING ANALYSIS PLOTS")
    print("=" * 60)
    print(f"Results directory: {args.runs_dir}")
    print(f"Output directory: {args.out_dir}")
    print()

    generate_all_plots(args.runs_dir, args.out_dir)

    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
