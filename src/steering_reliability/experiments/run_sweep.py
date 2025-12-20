"""
Run steering sweep experiment across layers, alphas, and intervention types.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from ..config import Config
from ..data import load_all_splits
from ..generation import generate_completions
from ..directions.build_direction import build_contrastive_direction, save_direction
from ..interventions.steer import create_steering_hook_fn
from ..metrics.refusal import detect_refusal
from ..metrics.helpfulness import detect_helpfulness
from ..metrics.confounds import compute_confounds


def run_sweep_experiment(
    model: HookedTransformer,
    config: Config,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run full steering sweep experiment.

    For each (layer, intervention_type, alpha):
    - Build or load steering direction
    - Generate steered completions
    - Compute metrics

    Args:
        model: HookedTransformer model
        config: Config object
        save_results: Whether to save results

    Returns:
        DataFrame with all results
    """
    print("=" * 60)
    print("STEERING SWEEP EXPERIMENT")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_splits(
        config.data.harm_train_path,
        config.data.harm_test_path,
        config.data.benign_path,
        config.data.max_prompts_per_split,
    )

    # Build directions for each layer using harm_train
    print("\n" + "=" * 60)
    print("BUILDING STEERING DIRECTIONS")
    print("=" * 60)

    directions = {}
    harm_train_prompts = [p["prompt"] for p in datasets["harm_train"]]

    for layer in config.experiment.layers:
        print(f"\nBuilding direction for layer {layer}...")

        direction, metadata = build_contrastive_direction(
            model=model,
            prompts=harm_train_prompts,
            refusal_prefix=config.direction.refusal_prefix,
            compliance_prefix=config.direction.compliance_prefix,
            layer=layer,
            hook_point=config.direction.hook_point,
            token_position=config.direction.token_position,
            normalize=config.direction.normalize,
            batch_size=config.generation.batch_size,
            show_progress=True,
        )

        directions[layer] = direction

        # Save direction
        if save_results:
            output_dir = Path(config.experiment.output_dir)
            dir_path = output_dir / f"directions/layer_{layer}/v.pt"
            save_direction(direction, metadata, dir_path)

        print(f"  Direction norm: {direction.norm():.4f}")

    # Run sweep
    print("\n" + "=" * 60)
    print("RUNNING STEERING SWEEP")
    print("=" * 60)

    results = []
    total_configs = (
        len(config.experiment.layers) *
        len(config.experiment.interventions) *
        len(config.experiment.alphas)
    )

    pbar = tqdm(total=total_configs, desc="Sweep progress")

    for layer in config.experiment.layers:
        direction = directions[layer]

        for intervention_type in config.experiment.interventions:
            for alpha in config.experiment.alphas:
                pbar.set_description(
                    f"L{layer} {intervention_type} α={alpha}"
                )

                # Skip baseline (alpha=0)
                if alpha == 0:
                    pbar.update(1)
                    continue

                # Create steering hooks
                hooks = create_steering_hook_fn(
                    direction=direction,
                    alpha=alpha,
                    layer=layer,
                    intervention_type=intervention_type,
                    hook_point=config.direction.hook_point,
                )

                # Process each split
                for split_name, prompts_data in datasets.items():
                    prompts = [p["prompt"] for p in prompts_data]

                    # Generate with steering
                    completions = generate_completions(
                        model=model,
                        prompts=prompts,
                        max_new_tokens=config.generation.max_new_tokens,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        batch_size=config.generation.batch_size,
                        seed=config.generation.seed,
                        show_progress=False,
                        hook_fn=hooks,
                    )

                    # Compute metrics
                    for i, (prompt_data, completion) in enumerate(zip(prompts_data, completions)):
                        refusal_result = detect_refusal(
                            completion,
                            threshold=config.metrics.refusal["threshold"]
                        )
                        helpfulness_result = detect_helpfulness(
                            completion,
                            threshold=config.metrics.helpfulness["threshold"]
                        )
                        confound_result = compute_confounds(completion)

                        result = {
                            "id": prompt_data["id"],
                            "split": split_name,
                            "prompt": prompt_data["prompt"],
                            "completion": completion if config.experiment.save_completions else "",
                            "intervention": "steering",
                            "layer": layer,
                            "alpha": alpha,
                            "intervention_type": intervention_type,
                            "refusal_score": refusal_result["refusal_score"],
                            "is_refusal": refusal_result["is_refusal"],
                            "helpfulness_score": helpfulness_result["helpfulness_score"],
                            "is_helpful": helpfulness_result["is_helpful"],
                            "has_structure": helpfulness_result["has_structure"],
                            "char_length": confound_result["char_length"],
                        }

                        if "meta" in prompt_data:
                            for key, value in prompt_data["meta"].items():
                                result[f"meta_{key}"] = value

                        results.append(result)

                pbar.update(1)

    pbar.close()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    if save_results:
        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "sweep_results.parquet"
        df.to_parquet(results_path, index=False)
        print(f"\n✓ Saved results to {results_path}")

    print(f"\n✓ Sweep complete! Generated {len(df)} completions.")

    return df


def main():
    """CLI entry point."""
    import argparse
    from ..config import load_config
    from ..model import load_model

    parser = argparse.ArgumentParser(description="Run steering sweep experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(config.model.name, config.model.device, config.model.dtype)

    run_sweep_experiment(model, config, save_results=True)


if __name__ == "__main__":
    main()
