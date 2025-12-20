"""
Analysis Notebook - Generate plots and tables from results

To use in Jupyter/Colab:
1. Copy cells below into notebook cells
2. Ensure you have run 02_full_sweep.ipynb first
3. Run top to bottom
"""

# %% [markdown]
# # Steering Reliability - Analysis & Visualization
#
# Generate publication-ready plots and tables from experiment results.
#
# **Expected runtime:** < 5 minutes
#
# **Requirements:**
# - Must have run `02_full_sweep.ipynb` first
# - Or have `baseline_results.parquet` and `sweep_results.parquet`

# %% [markdown]
# ## Setup

# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd() / "src"))

from steering_reliability.analysis.aggregate import (
    combine_baseline_and_sweep,
    compute_summary_stats,
    compute_generalization_gap,
    compute_side_effect_drop
)
from steering_reliability.analysis.plots import (
    plot_generalization_gap,
    plot_tradeoff_curve,
    plot_layer_alpha_heatmap
)
from steering_reliability.notebook_utils import *

print("✅ Imports successful!")

# %% [markdown]
# ## Load Results

# %%
print_header("Loading Experiment Results", level=2)

# Specify results directory
RESULTS_DIR = "artifacts/runs/notebook_run"  # Change if needed

# Load
baseline_path = Path(RESULTS_DIR) / "baseline_results.parquet"
sweep_path = Path(RESULTS_DIR) / "sweep_results.parquet"

if not baseline_path.exists() or not sweep_path.exists():
    print_warning(f"Results not found in {RESULTS_DIR}")
    print_info("Make sure you've run 02_full_sweep.ipynb first")
    print_info(f"Looking for:")
    print_info(f"  - {baseline_path}")
    print_info(f"  - {sweep_path}")
else:
    # Combine results
    df = combine_baseline_and_sweep(baseline_path, sweep_path)
    print_success(f"Loaded {len(df)} completions total")

    # Show breakdown
    print("\nBreakdown by split:")
    print(df.groupby("split").size())

# %% [markdown]
# ## Compute Summary Statistics

# %%
print_header("Computing Summary Statistics", level=2)

summary = compute_summary_stats(df)

print(f"Summary table shape: {summary.shape}")
display_dataframe(summary, title="Summary Statistics (first 10 rows)", max_rows=10)

# Save summary
summary_path = Path(RESULTS_DIR).parent.parent / "tables" / "summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(summary_path, index=False)
print_success(f"Summary table saved to {summary_path}")

# %% [markdown]
# ## Baseline Performance

# %%
print_header("Baseline Performance", level=2)

baseline_summary = summary[summary["intervention_type"] == "none"]

print("Refusal rates (baseline, no intervention):")
display(baseline_summary[["split", "is_refusal_mean"]].round(3))

print("\nHelpfulness rates (baseline):")
display(baseline_summary[["split", "is_helpful_mean"]].round(3))

# %% [markdown]
# ## Plot 1: Generalization Gap

# %%
print_header("Generalization Gap Analysis", level=2)

figures_dir = Path(RESULTS_DIR).parent.parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plot_generalization_gap(
    summary,
    output_path=figures_dir / "generalization_gap.png"
)

# Display in notebook
from IPython.display import Image
display(Image(filename=figures_dir / "generalization_gap.png"))

# %%
# Compute and display gap table
gap_df = compute_generalization_gap(summary)

print_header("Generalization Gap Table", level=3)
print("Gap = refusal_rate(harm_train) - refusal_rate(harm_test)")
print("Positive gap = better generalization")
print()

display(gap_df[["layer", "alpha", "intervention_type", "generalization_gap"]].round(3))

# %% [markdown]
# ## Plot 2: Tradeoff Curve

# %%
print_header("Refusal vs Side Effects Tradeoff", level=2)

baseline_summary = summary[summary["intervention_type"] == "none"]

plot_tradeoff_curve(
    summary,
    baseline_summary,
    output_path=figures_dir / "tradeoff_curve.png"
)

display(Image(filename=figures_dir / "tradeoff_curve.png"))

# %%
# Display side effect table
side_effects = compute_side_effect_drop(summary, baseline_summary)

print_header("Side Effects on Benign Prompts", level=3)
display(side_effects[
    ["layer", "alpha", "intervention_type", "is_helpful_mean", "helpfulness_drop"]
].round(3).sort_values("helpfulness_drop"))

# %% [markdown]
# ## Plot 3: Refusal Rate Heatmap (harm_test)

# %%
print_header("Refusal Rate Heatmap - harm_test", level=2)

plot_layer_alpha_heatmap(
    summary,
    split="harm_test",
    metric="is_refusal_mean",
    output_path=figures_dir / "heatmap_refusal_harm_test.png"
)

display(Image(filename=figures_dir / "heatmap_refusal_harm_test.png"))

# %% [markdown]
# ## Plot 4: Helpfulness Rate Heatmap (benign)

# %%
print_header("Helpfulness Rate Heatmap - benign", level=2)

plot_layer_alpha_heatmap(
    summary,
    split="benign",
    metric="is_helpful_mean",
    output_path=figures_dir / "heatmap_helpfulness_benign.png"
)

display(Image(filename=figures_dir / "heatmap_helpfulness_benign.png"))

# %% [markdown]
# ## Qualitative Examples

# %%
print_header("Qualitative Examples", level=2)

# Show baseline vs best steering example
print_header("Baseline vs Best Steering (harm_test)", level=3)

# Find best steering config (highest refusal on harm_test)
best_config = summary[
    (summary["split"] == "harm_test") &
    (summary["intervention_type"] != "none")
].sort_values("is_refusal_mean", ascending=False).iloc[0]

print(f"Best config: Layer {int(best_config['layer'])}, "
      f"{best_config['intervention_type']}, alpha={best_config['alpha']}")
print(f"Refusal rate: {best_config['is_refusal_mean']:.3f}")
print()

# Get example completions
baseline_examples = df[
    (df["split"] == "harm_test") &
    (df["intervention_type"] == "none")
].sample(3)

steered_examples = df[
    (df["split"] == "harm_test") &
    (df["layer"] == best_config["layer"]) &
    (df["alpha"] == best_config["alpha"]) &
    (df["intervention_type"] == best_config["intervention_type"])
].sample(3)

for i, (baseline_row, steered_row) in enumerate(zip(
    baseline_examples.itertuples(),
    steered_examples.itertuples()
)):
    display(Markdown(f"""
---
### Example {i+1}

**Prompt:** {baseline_row.prompt[:100]}...

**Baseline:**
{baseline_row.completion[:150]}...
- Refusal: {baseline_row.is_refusal} (score: {baseline_row.refusal_score:.2f})

**Steered (L{int(best_config['layer'])}, {best_config['intervention_type']}, α={best_config['alpha']}):**
{steered_row.completion[:150]}...
- Refusal: {steered_row.is_refusal} (score: {steered_row.refusal_score:.2f})
"""))

# %% [markdown]
# ## Side Effects on Benign

# %%
print_header("Side Effects Examples (Benign)", level=3)

# Get benign examples
baseline_benign = df[
    (df["split"] == "benign") &
    (df["intervention_type"] == "none")
].sample(2)

# Use same best config
steered_benign = df[
    (df["split"] == "benign") &
    (df["layer"] == best_config["layer"]) &
    (df["alpha"] == best_config["alpha"]) &
    (df["intervention_type"] == best_config["intervention_type"])
].sample(2)

for i, (baseline_row, steered_row) in enumerate(zip(
    baseline_benign.itertuples(),
    steered_benign.itertuples()
)):
    display(Markdown(f"""
---
### Benign Example {i+1}

**Prompt:** {baseline_row.prompt}

**Baseline:**
{baseline_row.completion[:150]}...
- Helpful: {baseline_row.is_helpful} (score: {baseline_row.helpfulness_score:.2f})

**Steered:**
{steered_row.completion[:150]}...
- Helpful: {steered_row.is_helpful} (score: {steered_row.helpfulness_score:.2f})
"""))

# %% [markdown]
# ## Key Findings Summary

# %%
print_header("Key Findings Summary", level=2)

# 1. Best performing layer
best_layer = summary[
    (summary["split"] == "harm_test") &
    (summary["intervention_type"] == "add")
].groupby("layer")["is_refusal_mean"].max().idxmax()

print_info(f"**Best performing layer:** {int(best_layer)}")

# 2. Generalization gap
max_gap = gap_df["generalization_gap"].max()
min_gap = gap_df["generalization_gap"].min()

print_info(f"**Generalization gap range:** {min_gap:.3f} to {max_gap:.3f}")

if max_gap > 0.2:
    print_warning("⚠️ Significant generalization failure detected!")
else:
    print_success("✅ Good generalization across distributions")

# 3. Side effects
max_drop = side_effects["helpfulness_drop"].max()
print_info(f"**Maximum helpfulness drop on benign:** {max_drop:.3f}")

if max_drop > 0.3:
    print_warning("⚠️ Significant side effects on benign prompts!")
else:
    print_success("✅ Minimal side effects on benign prompts")

# 4. Best tradeoff
# Find config with high refusal on harm_test and low drop on benign
tradeoff_score = summary[
    (summary["split"] == "harm_test") &
    (summary["intervention_type"] != "none")
].copy()

# Get corresponding benign drops
benign_drops = side_effects[["layer", "alpha", "intervention_type", "helpfulness_drop"]]
tradeoff_score = tradeoff_score.merge(benign_drops, on=["layer", "alpha", "intervention_type"])

# Score = refusal - 2*drop (penalize side effects more)
tradeoff_score["tradeoff_score"] = (
    tradeoff_score["is_refusal_mean"] - 2 * tradeoff_score["helpfulness_drop"]
)

best_tradeoff = tradeoff_score.sort_values("tradeoff_score", ascending=False).iloc[0]

print_info(f"**Best tradeoff config:** Layer {int(best_tradeoff['layer'])}, "
          f"{best_tradeoff['intervention_type']}, alpha={best_tradeoff['alpha']}")
print_info(f"  - Refusal on harm_test: {best_tradeoff['is_refusal_mean']:.3f}")
print_info(f"  - Helpfulness drop on benign: {best_tradeoff['helpfulness_drop']:.3f}")

# %% [markdown]
# ## Export Summary Report

# %%
print_header("Exporting Summary Report", level=2)

report_path = Path(RESULTS_DIR).parent.parent / "reports" / "summary_report.md"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w") as f:
    f.write(f"""# Steering Reliability - Experiment Report

## Configuration
- Model: EleutherAI/pythia-160m
- Layers tested: {[4, 8, 12]}
- Interventions: add, ablate
- Alphas: [0, 0.5, 1, 2, 4]

## Key Findings

### 1. Best Performing Layer
Layer **{int(best_layer)}** showed the highest refusal rates on harm_test prompts.

### 2. Generalization
Generalization gap (harm_train - harm_test) ranged from **{min_gap:.3f}** to **{max_gap:.3f}**.

### 3. Side Effects
Maximum helpfulness drop on benign prompts: **{max_drop:.3f}**.

### 4. Recommended Configuration
For best tradeoff between refusal and side effects:
- **Layer:** {int(best_tradeoff['layer'])}
- **Intervention:** {best_tradeoff['intervention_type']}
- **Alpha:** {best_tradeoff['alpha']}
- **Performance:**
  - Refusal rate on harm_test: {best_tradeoff['is_refusal_mean']:.3f}
  - Helpfulness drop on benign: {best_tradeoff['helpfulness_drop']:.3f}

## Plots
All plots saved to `artifacts/figures/`:
- generalization_gap.png
- tradeoff_curve.png
- heatmap_refusal_harm_test.png
- heatmap_helpfulness_benign.png

## Data
- Full results: `{RESULTS_DIR}/`
- Summary table: `artifacts/tables/summary.csv`
""")

print_success(f"Report saved to {report_path}")

# %% [markdown]
# ## Analysis Complete!

# %%
print_header("Analysis Complete!", level=2)
print_success("✅ All plots and tables generated!")
print_info(f"📊 Figures: {figures_dir}")
print_info(f"📋 Tables: {summary_path}")
print_info(f"📝 Report: {report_path}")
