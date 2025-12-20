"""
Full Sweep Notebook - Complete experiment (1-2 hours on GPU)

To use in Jupyter/Colab:
1. Copy cells below into notebook cells
2. Run top to bottom
3. Save results to Drive (optional)
"""

# %% [markdown]
# # Steering Reliability - Full Sweep
#
# Complete layer × alpha × intervention grid experiment.
#
# **Expected runtime:** 1-2 hours on GPU, 4-6 hours on CPU
#
# **What this does:**
# - Runs baseline on full datasets
# - Builds directions for layers [4, 8, 12]
# - Tests all intervention types [add, ablate]
# - Sweeps alphas [0, 0.5, 1, 2, 4]
# - Saves results for analysis

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
# Mount Google Drive (uncomment if using Colab and want to save results)
# from google.colab import drive
# drive.mount('/content/drive')
# SAVE_TO_DRIVE = True
# DRIVE_PATH = "/content/drive/MyDrive/steering_reliability_results"

SAVE_TO_DRIVE = False  # Set to True if using Drive

# %%
# Imports
import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd() / "src"))

from steering_reliability.config import create_config
from steering_reliability.model import load_model
from steering_reliability.experiments.run_baseline import run_baseline_experiment
from steering_reliability.experiments.run_sweep import run_sweep_experiment
from steering_reliability.notebook_utils import *

print("✅ Imports successful!")

# %% [markdown]
# ## Check GPU

# %%
display_gpu_info()
monitor_gpu_memory()

# %% [markdown]
# ## Configuration

# %%
# Full experiment configuration
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
        "max_prompts_per_split": None  # Use all prompts
    },
    generation={
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 8,  # Adjust based on GPU memory
        "seed": 0
    },
    direction={
        "refusal_prefix": " I can't help with that.",
        "compliance_prefix": " Sure, here's how:",
        "hook_point": "resid_post",
        "token_position": "last_prompt_token",
        "normalize": True
    },
    experiment={
        "layers": [4, 8, 12],  # Test 3 layers
        "interventions": ["add", "ablate"],
        "alphas": [0, 0.5, 1, 2, 4],
        "output_dir": "artifacts/runs/notebook_run",
        "save_completions": True
    }
)

display_config_summary(config)

# %%
# Calculate expected runtime
n_prompts = 150 + 150 + 200  # harm_train + harm_test + benign
n_layers = len(config.experiment.layers)
n_interventions = len(config.experiment.interventions)
n_alphas = len([a for a in config.experiment.alphas if a > 0])  # Exclude baseline

total_completions = n_prompts * (1 + n_layers * n_interventions * n_alphas)  # baseline + sweep

print_info(f"Total completions to generate: ~{total_completions:,}")
print_info(f"Estimated time on GPU: ~{total_completions * 0.1 / 60:.0f} minutes")

# %% [markdown]
# ## Generate Prompts (if needed)

# %%
from pathlib import Path
if not Path("data/prompts/harm_train.jsonl").exists():
    print_header("Generating Prompts", level=2)
    !python scripts/make_prompts.py --out_dir data/prompts --n_train 150 --n_test 150 --n_benign 200 --seed 0
else:
    print_success("Prompts already generated!")

# %% [markdown]
# ## Load Model

# %%
print_header("Loading Model", level=2)

model = load_model(
    config.model.name,
    config.model.device,
    config.model.dtype
)

print_success("Model loaded!")
monitor_gpu_memory()

# %% [markdown]
# ## Run Baseline Experiment

# %%
print_header("Running Baseline Experiment", level=2)
print_info("This will generate completions for all prompts without any steering...")

baseline_df = run_baseline_experiment(
    model=model,
    config=config,
    save_results=True
)

print_success(f"Baseline complete! Generated {len(baseline_df)} completions.")
monitor_gpu_memory()

# %%
# Quick baseline summary
display_metric_summary(baseline_df)

# %% [markdown]
# ## Run Steering Sweep

# %%
print_header("Running Steering Sweep", level=2)
print_warning("This will take 1-2 hours on GPU. You can monitor progress below.")

sweep_df = run_sweep_experiment(
    model=model,
    config=config,
    save_results=True
)

print_success(f"Sweep complete! Generated {len(sweep_df)} completions.")
monitor_gpu_memory()

# %%
# Quick sweep summary
print_header("Sweep Summary by Intervention", level=3)

summary = sweep_df.groupby(["intervention_type", "layer", "alpha", "split"]).agg({
    "is_refusal": "mean",
    "is_helpful": "mean"
}).round(3)

display(summary.head(20))

# %% [markdown]
# ## Save Results (Optional: to Google Drive)

# %%
if SAVE_TO_DRIVE:
    print_header("Saving to Google Drive", level=2)

    # Create Drive directory
    drive_dir = Path(DRIVE_PATH)
    drive_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    baseline_df.to_parquet(drive_dir / "baseline_results.parquet", index=False)
    sweep_df.to_parquet(drive_dir / "sweep_results.parquet", index=False)

    # Save config
    import yaml
    with open(drive_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

    print_success(f"Results saved to {DRIVE_PATH}")
else:
    print_info("Results saved to local artifacts/ directory")
    print_info("To save to Drive, set SAVE_TO_DRIVE=True and configure DRIVE_PATH")

# %% [markdown]
# ## Quick Analysis Preview

# %%
print_header("Quick Analysis Preview", level=2)

# Combine results
combined_df = pd.concat([baseline_df, sweep_df], ignore_index=True)

# %%
# Generalization gap for layer 8, add intervention
print_header("Generalization Gap (Layer 8, Add)", level=3)

layer_8_add = combined_df[
    (combined_df["layer"].isin([None, 8])) &
    (combined_df["intervention_type"].isin(["none", "add"])) &
    (combined_df["split"].isin(["harm_train", "harm_test"]))
]

gap_data = layer_8_add.groupby(["alpha", "split"])["is_refusal"].mean().unstack()

if "harm_train" in gap_data.columns and "harm_test" in gap_data.columns:
    gap_data["gap"] = gap_data["harm_train"] - gap_data["harm_test"]
    display(gap_data.round(3))

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(gap_data.index, gap_data["gap"], marker='o', linewidth=2, markersize=8)
    plt.xlabel("Alpha")
    plt.ylabel("Generalization Gap")
    plt.title("Generalization Gap: harm_train - harm_test (Layer 8, Add)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.show()

# %%
# Side effects on benign
print_header("Side Effects on Benign (Layer 8)", level=3)

baseline_benign_helpful = baseline_df[baseline_df["split"] == "benign"]["is_helpful"].mean()

benign_data = combined_df[
    (combined_df["split"] == "benign") &
    (combined_df["layer"].isin([None, 8])) &
    (combined_df["intervention_type"] != "none")
]

benign_summary = benign_data.groupby(["intervention_type", "alpha"])["is_helpful"].mean()
benign_summary = benign_summary.to_frame()
benign_summary["drop"] = baseline_benign_helpful - benign_summary["is_helpful"]

display(benign_summary.round(3))

# %% [markdown]
# ## Next Steps

# %%
print_header("Experiment Complete!", level=2)
print_success("✅ Baseline and sweep experiments finished!")
print_info("Results saved to artifacts/runs/notebook_run/")
print_info("")
print_info("Next steps:")
print_info("1. Run `03_analysis.ipynb` to generate full plots and analysis")
print_info("2. Or use scripts/make_plots.py from the command line")
print_info("3. Review qualitative examples in the results files")

# %%
# Clean up GPU memory
import gc
gc.collect()
torch.cuda.empty_cache()
print_success("GPU memory cleared")
