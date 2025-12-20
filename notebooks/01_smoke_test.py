"""
Smoke Test Notebook - Quick validation (10 minutes, 20 prompts)

To use in Jupyter/Colab:
1. Copy cells below into notebook cells
2. Run top to bottom
"""

# %% [markdown]
# # Steering Reliability - Smoke Test
#
# Quick validation of the pipeline on a small dataset.
#
# **Expected runtime:** 5-10 minutes on GPU, 15-20 minutes on CPU
#
# **What this does:**
# - Loads pythia-160m
# - Runs baseline on 20 prompts per split
# - Builds one steering direction (layer 8)
# - Tests one steering config (add, alpha=2)
# - Compares baseline vs steered

# %% [markdown]
# ## Setup

# %%
# Install dependencies (uncomment if running in Colab)
# !pip install -q torch transformer-lens transformers datasets pandas numpy matplotlib seaborn pyyaml tqdm pyarrow

# %%
# Clone repo (uncomment if running in Colab)
# !git clone https://github.com/yourusername/steering-reliability.git
# %cd steering-reliability

# %%
# Imports
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

import torch
import pandas as pd
import matplotlib.pyplot as plt

from steering_reliability.config import create_config
from steering_reliability.model import load_model
from steering_reliability.data import load_all_splits
from steering_reliability.generation import generate_completions
from steering_reliability.directions.build_direction import build_contrastive_direction
from steering_reliability.interventions.steer import create_steering_hook_fn
from steering_reliability.metrics.refusal import detect_refusal
from steering_reliability.metrics.helpfulness import detect_helpfulness
from steering_reliability.notebook_utils import *

print("✅ Imports successful!")

# %% [markdown]
# ## Check GPU

# %%
display_gpu_info()

# %% [markdown]
# ## Configuration

# %%
# Create config for smoke test
config = create_config(
    model={
        "name": "EleutherAI/pythia-160m",
        "device": "auto",
        "dtype": "float32"
    },
    data={
        "harm_train_path": "data/prompts/harm_train.jsonl",
        "harm_test_path": "data/prompts/harm_test.jsonl",
        "benign_path": "data/prompts/benign.jsonl",
        "max_prompts_per_split": 20  # Small dataset for quick test
    },
    generation={
        "max_new_tokens": 60,
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 4,
        "seed": 42
    },
    direction={
        "refusal_prefix": " I can't help with that.",
        "compliance_prefix": " Sure, here's how:",
        "hook_point": "resid_post",
        "token_position": "last_prompt_token",
        "normalize": True
    }
)

display_config_summary(config)

# %% [markdown]
# ## Load Model

# %%
model = load_model(
    config.model.name,
    config.model.device,
    config.model.dtype
)

print_success("Model loaded!")

# %% [markdown]
# ## Load Data

# %%
# Generate prompts if needed
from pathlib import Path
if not Path("data/prompts/harm_train.jsonl").exists():
    print_info("Generating prompts...")
    !python scripts/make_prompts.py --out_dir data/prompts --n_train 150 --n_test 150 --n_benign 200 --seed 0

# Load datasets
datasets = load_all_splits(
    config.data.harm_train_path,
    config.data.harm_test_path,
    config.data.benign_path,
    config.data.max_prompts_per_split
)

print_success(f"Loaded {sum(len(v) for v in datasets.values())} prompts total")
for split, prompts in datasets.items():
    print(f"  - {split}: {len(prompts)} prompts")

# %% [markdown]
# ## Step 1: Baseline (No Steering)

# %%
print_header("Running Baseline", level=2)

baseline_results = []

for split_name, prompts_data in datasets.items():
    print(f"\nProcessing {split_name}...")
    prompts = [p["prompt"] for p in prompts_data]

    # Generate
    completions = generate_completions(
        model=model,
        prompts=prompts,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        batch_size=config.generation.batch_size,
        seed=config.generation.seed,
        show_progress=True
    )

    # Compute metrics
    for prompt_data, completion in zip(prompts_data, completions):
        refusal = detect_refusal(completion)
        helpful = detect_helpfulness(completion)

        baseline_results.append({
            "split": split_name,
            "prompt": prompt_data["prompt"],
            "completion": completion,
            "refusal_score": refusal["refusal_score"],
            "is_refusal": refusal["is_refusal"],
            "helpfulness_score": helpful["helpfulness_score"],
            "is_helpful": helpful["is_helpful"],
            "char_length": len(completion)
        })

baseline_df = pd.DataFrame(baseline_results)
print_success("Baseline complete!")

# %%
# Display baseline summary
display_metric_summary(baseline_df)

# %%
# Show some examples
display_examples(baseline_df, n_examples=2, split="harm_test")

# %% [markdown]
# ## Step 2: Build Steering Direction

# %%
print_header("Building Steering Direction (Layer 8)", level=2)

harm_train_prompts = [p["prompt"] for p in datasets["harm_train"]]

direction, metadata = build_contrastive_direction(
    model=model,
    prompts=harm_train_prompts,
    refusal_prefix=config.direction.refusal_prefix,
    compliance_prefix=config.direction.compliance_prefix,
    layer=8,  # Middle layer
    hook_point=config.direction.hook_point,
    token_position=config.direction.token_position,
    normalize=config.direction.normalize,
    batch_size=config.generation.batch_size,
    show_progress=True
)

print_success(f"Direction built! Norm: {direction.norm():.4f}")
print_info(f"Mean diff norm: {metadata['mean_diff_norm']:.4f}")

# %% [markdown]
# ## Step 3: Steering Test (Add, Alpha=2)

# %%
print_header("Testing Steering (add, alpha=2)", level=2)

alpha = 2.0
layer = 8

# Create steering hooks
hooks = create_steering_hook_fn(
    direction=direction,
    alpha=alpha,
    layer=layer,
    intervention_type="add",
    hook_point=config.direction.hook_point
)

steered_results = []

for split_name, prompts_data in datasets.items():
    print(f"\nProcessing {split_name}...")
    prompts = [p["prompt"] for p in prompts_data]

    # Generate with steering
    completions = generate_completions(
        model=model,
        prompts=prompts,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        batch_size=config.generation.batch_size,
        seed=config.generation.seed,
        show_progress=True,
        hook_fn=hooks
    )

    # Compute metrics
    for prompt_data, completion in zip(prompts_data, completions):
        refusal = detect_refusal(completion)
        helpful = detect_helpfulness(completion)

        steered_results.append({
            "split": split_name,
            "prompt": prompt_data["prompt"],
            "completion": completion,
            "refusal_score": refusal["refusal_score"],
            "is_refusal": refusal["is_refusal"],
            "helpfulness_score": helpful["helpfulness_score"],
            "is_helpful": helpful["is_helpful"],
            "char_length": len(completion)
        })

steered_df = pd.DataFrame(steered_results)
print_success("Steering test complete!")

# %%
# Display steered summary
display_metric_summary(steered_df)

# %%
# Show steered examples
display_examples(steered_df, n_examples=2, split="harm_test")

# %% [markdown]
# ## Step 4: Compare Results

# %%
print_header("Baseline vs Steered Comparison", level=2)

# Compare refusal rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, split in enumerate(["harm_train", "harm_test", "benign"]):
    ax = axes[idx]

    baseline_val = baseline_df[baseline_df["split"] == split]["is_refusal"].mean()
    steered_val = steered_df[steered_df["split"] == split]["is_refusal"].mean()

    ax.bar(["Baseline", "Steered"], [baseline_val, steered_val],
           color=["#3498db", "#e74c3c"], alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Refusal Rate")
    ax.set_title(f"{split}")
    ax.grid(True, alpha=0.3)

    # Add value labels
    ax.text(0, baseline_val + 0.03, f"{baseline_val:.2f}", ha='center', fontweight='bold')
    ax.text(1, steered_val + 0.03, f"{steered_val:.2f}", ha='center', fontweight='bold')

plt.suptitle("Refusal Rates: Baseline vs Steered (Layer 8, Alpha=2)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Helpfulness comparison
fig, ax = plt.subplots(figsize=(8, 5))

baseline_help = baseline_df[baseline_df["split"] == "benign"]["is_helpful"].mean()
steered_help = steered_df[steered_df["split"] == "benign"]["is_helpful"].mean()

ax.bar(["Baseline", "Steered"], [baseline_help, steered_help],
       color=["#2ecc71", "#f39c12"], alpha=0.7)
ax.set_ylim([0, 1])
ax.set_ylabel("Helpfulness Rate")
ax.set_title("Helpfulness on Benign Prompts")
ax.grid(True, alpha=0.3)

ax.text(0, baseline_help + 0.03, f"{baseline_help:.2f}", ha='center', fontweight='bold')
ax.text(1, steered_help + 0.03, f"{steered_help:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %%
print_header("Smoke Test Summary", level=2)

# Create summary table
summary_data = {
    "Split": ["harm_train", "harm_test", "benign"],
    "Baseline Refusal": [
        baseline_df[baseline_df["split"] == "harm_train"]["is_refusal"].mean(),
        baseline_df[baseline_df["split"] == "harm_test"]["is_refusal"].mean(),
        baseline_df[baseline_df["split"] == "benign"]["is_refusal"].mean()
    ],
    "Steered Refusal": [
        steered_df[steered_df["split"] == "harm_train"]["is_refusal"].mean(),
        steered_df[steered_df["split"] == "harm_test"]["is_refusal"].mean(),
        steered_df[steered_df["split"] == "benign"]["is_refusal"].mean()
    ],
    "Baseline Helpful": [
        baseline_df[baseline_df["split"] == "harm_train"]["is_helpful"].mean(),
        baseline_df[baseline_df["split"] == "harm_test"]["is_helpful"].mean(),
        baseline_df[baseline_df["split"] == "benign"]["is_helpful"].mean()
    ],
    "Steered Helpful": [
        steered_df[steered_df["split"] == "harm_train"]["is_helpful"].mean(),
        steered_df[steered_df["split"] == "harm_test"]["is_helpful"].mean(),
        steered_df[steered_df["split"] == "benign"]["is_helpful"].mean()
    ]
}

summary_df = pd.DataFrame(summary_data).round(3)
display(summary_df)

print_success("✅ Smoke test complete! Pipeline is working correctly.")
print_info("Next: Run `02_full_sweep.ipynb` for the complete experiment")
