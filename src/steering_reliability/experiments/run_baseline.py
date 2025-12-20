"""
Run baseline experiment (no steering intervention).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from transformer_lens import HookedTransformer

from ..config import Config
from ..data import load_all_splits
from ..generation import generate_completions
from ..metrics.refusal import detect_refusal
from ..metrics.helpfulness import detect_helpfulness
from ..metrics.confounds import compute_confounds


def run_baseline_experiment(
    model: HookedTransformer,
    config: Config,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run baseline experiment (no intervention).

    Generates completions for all three splits and computes metrics.

    Args:
        model: HookedTransformer model
        config: Config object
        save_results: Whether to save results to disk

    Returns:
        DataFrame with results (one row per prompt)
    """
    print("=" * 60)
    print("BASELINE EXPERIMENT (No Intervention)")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_splits(
        config.data.harm_train_path,
        config.data.harm_test_path,
        config.data.benign_path,
        config.data.max_prompts_per_split,
    )

    results = []

    # Process each split
    for split_name, prompts_data in datasets.items():
        print(f"\n--- Processing {split_name} ({len(prompts_data)} prompts) ---")

        # Extract prompt texts
        prompts = [p["prompt"] for p in prompts_data]

        # Generate completions
        completions = generate_completions(
            model=model,
            prompts=prompts,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            batch_size=config.generation.batch_size,
            seed=config.generation.seed,
            show_progress=True,
        )

        # Compute metrics for each completion
        for i, (prompt_data, completion) in enumerate(zip(prompts_data, completions)):
            # Refusal metrics
            refusal_result = detect_refusal(
                completion,
                threshold=config.metrics.refusal["threshold"]
            )

            # Helpfulness metrics
            helpfulness_result = detect_helpfulness(
                completion,
                threshold=config.metrics.helpfulness["threshold"]
            )

            # Confound metrics
            confound_result = compute_confounds(completion)

            # Combine all metrics
            result = {
                "id": prompt_data["id"],
                "split": split_name,
                "prompt": prompt_data["prompt"],
                "completion": completion,
                "intervention": "baseline",
                "layer": None,
                "alpha": 0.0,
                "intervention_type": "none",
                # Refusal metrics
                "refusal_score": refusal_result["refusal_score"],
                "is_refusal": refusal_result["is_refusal"],
                # Helpfulness metrics
                "helpfulness_score": helpfulness_result["helpfulness_score"],
                "is_helpful": helpfulness_result["is_helpful"],
                "has_structure": helpfulness_result["has_structure"],
                # Confound metrics
                "char_length": confound_result["char_length"],
            }

            # Add metadata if present
            if "meta" in prompt_data:
                for key, value in prompt_data["meta"].items():
                    result[f"meta_{key}"] = value

            results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal completions: {len(df)}")
    print("\nRefusal rates by split:")
    print(df.groupby("split")["is_refusal"].mean())
    print("\nHelpfulness rates by split:")
    print(df.groupby("split")["is_helpful"].mean())
    print("\nAverage completion length by split:")
    print(df.groupby("split")["char_length"].mean())

    # Save results
    if save_results:
        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_path = output_dir / "baseline_results.parquet"
        df.to_parquet(baseline_path, index=False)
        print(f"\n✓ Saved results to {baseline_path}")

        # Also save as CSV for easy inspection
        csv_path = output_dir / "baseline_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to {csv_path}")

    return df


def main():
    """CLI entry point."""
    import argparse
    from ..config import load_config
    from ..model import load_model

    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    model = load_model(
        config.model.name,
        config.model.device,
        config.model.dtype,
    )

    # Run experiment
    run_baseline_experiment(model, config, save_results=True)


if __name__ == "__main__":
    main()
