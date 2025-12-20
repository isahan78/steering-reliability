"""
Utilities for Jupyter notebooks: inline display, progress tracking, GPU monitoring.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown
import torch
from typing import Dict, Any, List, Optional


def print_header(text: str, level: int = 1):
    """Print a formatted header in notebooks."""
    if level == 1:
        display(Markdown(f"# {text}"))
    elif level == 2:
        display(Markdown(f"## {text}"))
    elif level == 3:
        display(Markdown(f"### {text}"))
    else:
        display(Markdown(f"**{text}**"))


def print_info(text: str):
    """Print info message with styling."""
    display(Markdown(f"ℹ️ {text}"))


def print_success(text: str):
    """Print success message with styling."""
    display(Markdown(f"✅ {text}"))


def print_warning(text: str):
    """Print warning message with styling."""
    display(Markdown(f"⚠️ {text}"))


def display_gpu_info():
    """Display GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        display(Markdown(f"""
### 🖥️ GPU Information
- **Device**: {gpu_name}
- **Memory**: {gpu_memory:.1f} GB
- **CUDA Version**: {torch.version.cuda}
"""))
    elif torch.backends.mps.is_available():
        display(Markdown(f"""
### 🖥️ Apple Silicon GPU
- **Device**: MPS (Metal Performance Shaders)
- **Available**: ✅
"""))
    else:
        display(Markdown(f"""
### 🖥️ Hardware
- **Device**: CPU only
- **Note**: Consider using Google Colab for GPU acceleration
"""))


def display_config_summary(config):
    """Display configuration summary in a nice format."""
    display(Markdown(f"""
### ⚙️ Configuration Summary

**Model:**
- Name: `{config.model.name}`
- Device: `{config.model.device}`
- Dtype: `{config.model.dtype}`

**Experiment:**
- Layers: {config.experiment.layers}
- Interventions: {config.experiment.interventions}
- Alphas: {config.experiment.alphas}

**Generation:**
- Max tokens: {config.generation.max_new_tokens}
- Temperature: {config.generation.temperature}
- Batch size: {config.generation.batch_size}

**Direction:**
- Refusal prefix: "{config.direction.refusal_prefix}"
- Compliance prefix: "{config.direction.compliance_prefix}"
- Hook point: `{config.direction.hook_point}`
- Normalize: {config.direction.normalize}
"""))


def display_dataframe(df: pd.DataFrame, title: str = "", max_rows: int = 10):
    """Display a DataFrame with nice formatting."""
    if title:
        print_header(title, level=3)

    # Style the dataframe
    styled = df.head(max_rows).style.set_properties(**{
        'text-align': 'left',
        'font-size': '11pt',
    })

    display(styled)

    if len(df) > max_rows:
        display(Markdown(f"*Showing {max_rows} of {len(df)} rows*"))


def display_metric_summary(df: pd.DataFrame, group_by: str = "split"):
    """Display summary statistics grouped by a column."""
    print_header("Metric Summary", level=3)

    summary = df.groupby(group_by).agg({
        "refusal_score": "mean",
        "is_refusal": "mean",
        "helpfulness_score": "mean",
        "is_helpful": "mean",
        "char_length": "mean",
    }).round(3)

    summary.columns = ["Refusal Score", "Refusal Rate", "Helpful Score", "Helpful Rate", "Avg Length"]

    display(summary)


def display_examples(
    df: pd.DataFrame,
    n_examples: int = 3,
    split: str = "harm_test",
    conditions: Optional[Dict[str, Any]] = None
):
    """Display example completions with metrics."""
    print_header(f"Example Completions: {split}", level=3)

    # Filter data
    filtered = df[df["split"] == split]

    if conditions:
        for key, value in conditions.items():
            filtered = filtered[filtered[key] == value]

    # Sample examples
    examples = filtered.sample(min(n_examples, len(filtered)))

    for idx, row in examples.iterrows():
        display(Markdown(f"""
---
**Prompt:** {row['prompt'][:100]}...

**Completion:** {row['completion'][:200]}...

**Metrics:**
- Refusal Score: {row['refusal_score']:.2f} | Is Refusal: {row['is_refusal']}
- Helpful Score: {row['helpfulness_score']:.2f} | Is Helpful: {row['is_helpful']}
- Length: {row['char_length']} chars
"""))


def plot_inline_comparison(
    baseline_df: pd.DataFrame,
    steered_df: pd.DataFrame,
    metric: str = "is_refusal",
    split: str = "harm_test"
):
    """Plot inline comparison between baseline and steered results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_val = baseline_df[baseline_df["split"] == split][metric].mean()
    steered_val = steered_df[steered_df["split"] == split][metric].mean()

    categories = ["Baseline", "Steered"]
    values = [baseline_val, steered_val]

    ax.bar(categories, values, color=["#3498db", "#e74c3c"], alpha=0.7)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} on {split}")
    ax.set_ylim([0, 1])

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def create_quick_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a quick summary table from experiment results."""
    df = pd.DataFrame(results)

    summary = df.groupby("split").agg({
        "is_refusal": ["mean", "std"],
        "is_helpful": ["mean", "std"],
        "char_length": ["mean", "std"],
    }).round(3)

    return summary


def monitor_gpu_memory():
    """Display current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9

        display(Markdown(f"""
**GPU Memory:**
- Allocated: {allocated:.2f} GB
- Reserved: {reserved:.2f} GB
"""))
    else:
        display(Markdown("*GPU not available*"))


def save_notebook_results(results_df: pd.DataFrame, output_path: str):
    """Save results from notebook with user feedback."""
    results_df.to_parquet(output_path, index=False)
    print_success(f"Results saved to: `{output_path}`")
    print_info(f"Total rows: {len(results_df)}")


def display_progress_summary(total: int, completed: int):
    """Display progress as a visual bar."""
    pct = (completed / total) * 100
    bar_length = 40
    filled = int(bar_length * completed / total)
    bar = "█" * filled + "░" * (bar_length - filled)

    display(Markdown(f"""
**Progress:** {completed}/{total} ({pct:.1f}%)
`{bar}`
"""))
