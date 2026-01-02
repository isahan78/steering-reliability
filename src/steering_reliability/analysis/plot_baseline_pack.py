"""
Generate figures and tables for baseline pack results.

Creates:
1. Figure A: Direction specificity barplot (refusal on harm_test)
2. Figure B: Benign preservation barplot (helpfulness on benign)
3. Summary table
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_direction_specificity(df, alphas, output_path):
    """
    Plot refusal rates on harm_test for each direction type.

    Shows that learned direction achieves higher refusal than baselines.
    """
    # Filter to harm_test only
    harm_test = df[df['split'] == 'harm_test'].copy()

    # For random directions, compute mean ± std
    random_stats = harm_test[harm_test['direction_type'] == 'random'].groupby('alpha').agg({
        'is_refusal': ['mean', 'std']
    }).reset_index()
    random_stats.columns = ['alpha', 'refusal_mean', 'refusal_std']

    # For non-random directions, just get mean
    non_random = harm_test[harm_test['direction_type'] != 'random'].groupby(
        ['direction_type', 'alpha']
    ).agg({
        'is_refusal': 'mean'
    }).reset_index()

    # Create figure with subplots for each alpha
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 5))

    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        # Get data for this alpha
        alpha_data = non_random[non_random['alpha'] == alpha]
        random_data = random_stats[random_stats['alpha'] == alpha]

        # Prepare data for plotting
        direction_types = alpha_data['direction_type'].tolist()
        refusal_rates = alpha_data['is_refusal'].tolist()

        # Add random mean
        if len(random_data) > 0:
            direction_types.append('random')
            refusal_rates.append(random_data.iloc[0]['refusal_mean'])

        # Create bar plot
        bars = ax.bar(range(len(direction_types)), refusal_rates, alpha=0.7)

        # Color learned direction differently
        for i, dir_type in enumerate(direction_types):
            if dir_type == 'learned':
                bars[i].set_color('#2ecc71')  # Green
            elif dir_type == 'random':
                bars[i].set_color('#95a5a6')  # Gray
            else:
                bars[i].set_color('#3498db')  # Blue

        # Add error bars for random
        if len(random_data) > 0:
            random_idx = direction_types.index('random')
            ax.errorbar(
                random_idx,
                random_data.iloc[0]['refusal_mean'],
                yerr=random_data.iloc[0]['refusal_std'],
                fmt='none',
                ecolor='black',
                capsize=5,
                capthick=2
            )

        ax.set_xticks(range(len(direction_types)))
        ax.set_xticklabels(direction_types, rotation=45, ha='right')
        ax.set_ylabel('Refusal Rate (harm_test)')
        ax.set_title(f'α={alpha}')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        # Add horizontal line at learned performance for reference
        learned_val = alpha_data[alpha_data['direction_type'] == 'learned']['is_refusal'].values
        if len(learned_val) > 0:
            ax.axhline(learned_val[0], color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved direction specificity plot to {output_path}")


def plot_benign_preservation(df, alphas, output_path):
    """
    Plot helpfulness on benign queries for each direction type.

    Shows that interventions preserve benign helpfulness.
    """
    # Filter to benign only
    benign = df[df['split'] == 'benign'].copy()

    # For random directions, compute mean ± std
    random_stats = benign[benign['direction_type'] == 'random'].groupby('alpha').agg({
        'is_helpful': ['mean', 'std']
    }).reset_index()
    random_stats.columns = ['alpha', 'helpful_mean', 'helpful_std']

    # For non-random directions, just get mean
    non_random = benign[benign['direction_type'] != 'random'].groupby(
        ['direction_type', 'alpha']
    ).agg({
        'is_helpful': 'mean'
    }).reset_index()

    # Create figure
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 5))

    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        # Get data for this alpha
        alpha_data = non_random[non_random['alpha'] == alpha]
        random_data = random_stats[random_stats['alpha'] == alpha]

        # Prepare data for plotting
        direction_types = alpha_data['direction_type'].tolist()
        helpful_rates = alpha_data['is_helpful'].tolist()

        # Add random mean
        if len(random_data) > 0:
            direction_types.append('random')
            helpful_rates.append(random_data.iloc[0]['helpful_mean'])

        # Create bar plot
        bars = ax.bar(range(len(direction_types)), helpful_rates, alpha=0.7)

        # Color learned direction differently
        for i, dir_type in enumerate(direction_types):
            if dir_type == 'learned':
                bars[i].set_color('#2ecc71')  # Green
            elif dir_type == 'random':
                bars[i].set_color('#95a5a6')  # Gray
            else:
                bars[i].set_color('#3498db')  # Blue

        # Add error bars for random
        if len(random_data) > 0:
            random_idx = direction_types.index('random')
            ax.errorbar(
                random_idx,
                random_data.iloc[0]['helpful_mean'],
                yerr=random_data.iloc[0]['helpful_std'],
                fmt='none',
                ecolor='black',
                capsize=5,
                capthick=2
            )

        ax.set_xticks(range(len(direction_types)))
        ax.set_xticklabels(direction_types, rotation=45, ha='right')
        ax.set_ylabel('Helpfulness Rate (benign)')
        ax.set_title(f'α={alpha}')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved benign preservation plot to {output_path}")


def create_summary_table(df, output_path):
    """
    Create summary table with key metrics for each direction type and alpha.
    """
    # For random directions, compute mean ± std
    random_summary = df[df['direction_type'] == 'random'].groupby(['alpha', 'split']).agg({
        'is_refusal': ['mean', 'std'],
        'is_helpful': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    random_summary.columns = ['alpha', 'split', 'refusal_mean', 'refusal_std', 'helpful_mean', 'helpful_std']
    random_summary['direction_type'] = 'random'

    # For non-random, just get mean
    non_random_summary = df[df['direction_type'] != 'random'].groupby(['direction_type', 'alpha', 'split']).agg({
        'is_refusal': 'mean',
        'is_helpful': 'mean'
    }).reset_index()
    non_random_summary.columns = ['direction_type', 'alpha', 'split', 'refusal_mean', 'helpful_mean']

    # Pivot to wide format
    table_rows = []

    direction_types = non_random_summary['direction_type'].unique()
    direction_types = list(direction_types) + ['random']

    for dir_type in direction_types:
        if dir_type == 'random':
            data = random_summary
        else:
            data = non_random_summary[non_random_summary['direction_type'] == dir_type]

        row = {'direction_type': dir_type}

        for alpha in sorted(df['alpha'].unique()):
            # Harm_test refusal
            harm_data = data[(data['alpha'] == alpha) & (data['split'] == 'harm_test')]
            if len(harm_data) > 0:
                if dir_type == 'random':
                    row[f'harm_refusal_α{alpha}'] = f"{harm_data.iloc[0]['refusal_mean']:.2f}±{harm_data.iloc[0]['refusal_std']:.2f}"
                else:
                    row[f'harm_refusal_α{alpha}'] = f"{harm_data.iloc[0]['refusal_mean']:.2f}"

            # Benign helpfulness
            benign_data = data[(data['alpha'] == alpha) & (data['split'] == 'benign')]
            if len(benign_data) > 0:
                if dir_type == 'random':
                    row[f'benign_helpful_α{alpha}'] = f"{benign_data.iloc[0]['helpful_mean']:.2f}±{benign_data.iloc[0]['helpful_std']:.2f}"
                else:
                    row[f'benign_helpful_α{alpha}'] = f"{benign_data.iloc[0]['helpful_mean']:.2f}"

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)

    # Save as CSV
    table_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table to {output_path}")

    # Also print to console
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(table_df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description='Plot baseline pack results')
    parser.add_argument('--in_parquet', type=str, required=True,
                        help='Input parquet file with results')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    args = parser.parse_args()

    # Read data
    df = pd.read_parquet(args.in_parquet)

    # Determine output directory
    if args.out_dir is None:
        out_dir = Path(args.in_parquet).parent
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BASELINE PACK ANALYSIS")
    print("="*80)
    print(f"Input: {args.in_parquet}")
    print(f"Output: {out_dir}")
    print()

    # Get alphas (excluding 0)
    alphas = sorted([a for a in df['alpha'].unique() if a > 0])
    print(f"Alphas: {alphas}")
    print()

    # Generate plots
    print("Generating plots...")

    plot_direction_specificity(
        df[df['alpha'] > 0],  # Exclude alpha=0
        alphas,
        out_dir / 'direction_specificity.png'
    )

    plot_benign_preservation(
        df[df['alpha'] > 0],
        alphas,
        out_dir / 'benign_preservation.png'
    )

    # Create summary table
    create_summary_table(df, out_dir / 'baseline_pack_table.csv')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {out_dir}:")
    print("  - direction_specificity.png")
    print("  - benign_preservation.png")
    print("  - baseline_pack_table.csv")


if __name__ == '__main__':
    main()
