"""
Sanity checks for steering reliability experiments.

Implements:
1. Template robustness: test with alternative prefixes
2. Entropy/length confounds: verify effects aren't solely due to text degradation
3. Seed robustness: test with different random seeds
"""

import pandas as pd
from typing import Dict, List
import torch
from transformer_lens import HookedTransformer

from ..config import Config
from ..data import load_all_splits
from ..generation import generate_completions
from ..directions.build_direction import build_contrastive_direction
from ..interventions.steer import create_steering_hook_fn
from ..metrics.refusal import detect_refusal
from ..metrics.helpfulness import detect_helpfulness
from ..metrics.confounds import compute_confounds


def sanity_check_template_robustness(
    model: HookedTransformer,
    config: Config,
    layer: int,
    alpha: float,
    intervention_type: str = "add",
) -> Dict[str, pd.DataFrame]:
    """
    Test if direction generalizes to alternative prefix templates.

    Returns:
        Dict with 'original' and 'alternative' DataFrames
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 1: Template Robustness")
    print("=" * 60)

    # Load data (subset)
    datasets = load_all_splits(
        config.data.harm_train_path,
        config.data.harm_test_path,
        config.data.benign_path,
        max_prompts_per_split=config.sanity_checks.rerun_subset_n
    )

    results = {}

    # Test 1: Original prefixes
    print("\n--- Testing with ORIGINAL prefixes ---")
    print(f"Refusal: '{config.direction.refusal_prefix}'")
    print(f"Compliance: '{config.direction.compliance_prefix}'")

    direction_orig, _ = build_contrastive_direction(
        model=model,
        prompts=[p["prompt"] for p in datasets["harm_train"]],
        refusal_prefix=config.direction.refusal_prefix,
        compliance_prefix=config.direction.compliance_prefix,
        layer=layer,
        hook_point=config.direction.hook_point,
        token_position=config.direction.token_position,
        normalize=config.direction.normalize,
        batch_size=config.generation.batch_size,
        show_progress=False
    )

    results["original"] = run_test_with_direction(
        model, config, datasets, direction_orig, layer, alpha, intervention_type
    )

    # Test 2: Alternative prefixes
    print("\n--- Testing with ALTERNATIVE prefixes ---")
    alt_refusal = config.sanity_checks.alt_prefixes["refusal_prefix"]
    alt_compliance = config.sanity_checks.alt_prefixes["compliance_prefix"]
    print(f"Refusal: '{alt_refusal}'")
    print(f"Compliance: '{alt_compliance}'")

    direction_alt, _ = build_contrastive_direction(
        model=model,
        prompts=[p["prompt"] for p in datasets["harm_train"]],
        refusal_prefix=alt_refusal,
        compliance_prefix=alt_compliance,
        layer=layer,
        hook_point=config.direction.hook_point,
        token_position=config.direction.token_position,
        normalize=config.direction.normalize,
        batch_size=config.generation.batch_size,
        show_progress=False
    )

    results["alternative"] = run_test_with_direction(
        model, config, datasets, direction_alt, layer, alpha, intervention_type
    )

    # Compare
    print("\n--- Comparison ---")
    for split in ["harm_train", "harm_test", "benign"]:
        orig_refusal = results["original"][results["original"]["split"] == split]["is_refusal"].mean()
        alt_refusal = results["alternative"][results["alternative"]["split"] == split]["is_refusal"].mean()
        diff = abs(orig_refusal - alt_refusal)

        print(f"{split}: Original={orig_refusal:.3f}, Alt={alt_refusal:.3f}, Diff={diff:.3f}")

    return results


def sanity_check_confounds(
    baseline_df: pd.DataFrame,
    steered_df: pd.DataFrame,
) -> Dict[str, any]:
    """
    Check if steering effects are confounded with length/entropy changes.

    Returns:
        Dict with confound analysis
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 2: Entropy/Length Confounds")
    print("=" * 60)

    results = {}

    for split in ["harm_train", "harm_test", "benign"]:
        baseline_split = baseline_df[baseline_df["split"] == split]
        steered_split = steered_df[steered_df["split"] == split]

        # Compare lengths
        baseline_len = baseline_split["char_length"].mean()
        steered_len = steered_split["char_length"].mean()
        len_change = (steered_len - baseline_len) / baseline_len * 100

        # Check for collapse (length < 20 chars typically indicates degenerate output)
        collapsed_pct = (steered_split["char_length"] < 20).mean() * 100

        results[split] = {
            "baseline_length": baseline_len,
            "steered_length": steered_len,
            "length_change_pct": len_change,
            "collapsed_pct": collapsed_pct
        }

        print(f"\n{split}:")
        print(f"  Baseline length: {baseline_len:.1f}")
        print(f"  Steered length: {steered_len:.1f}")
        print(f"  Change: {len_change:+.1f}%")
        print(f"  Collapsed outputs: {collapsed_pct:.1f}%")

        if collapsed_pct > 10:
            print(f"  ⚠️  WARNING: High collapse rate!")

    return results


def sanity_check_seed_robustness(
    model: HookedTransformer,
    config: Config,
    layer: int,
    alpha: float,
    intervention_type: str = "add",
) -> Dict[str, pd.DataFrame]:
    """
    Test robustness to random seed changes.

    Returns:
        Dict with 'seed_0' and 'seed_alt' DataFrames
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 3: Seed Robustness")
    print("=" * 60)

    datasets = load_all_splits(
        config.data.harm_train_path,
        config.data.harm_test_path,
        config.data.benign_path,
        max_prompts_per_split=config.sanity_checks.rerun_subset_n
    )

    results = {}

    # Build direction (same for both)
    direction, _ = build_contrastive_direction(
        model=model,
        prompts=[p["prompt"] for p in datasets["harm_train"]],
        refusal_prefix=config.direction.refusal_prefix,
        compliance_prefix=config.direction.compliance_prefix,
        layer=layer,
        hook_point=config.direction.hook_point,
        token_position=config.direction.token_position,
        normalize=config.direction.normalize,
        batch_size=config.generation.batch_size,
        show_progress=False
    )

    # Test with original seed
    print(f"\n--- Testing with seed {config.generation.seed} ---")
    results["seed_0"] = run_test_with_direction(
        model, config, datasets, direction, layer, alpha, intervention_type,
        seed=config.generation.seed
    )

    # Test with alternative seed
    alt_seed = config.sanity_checks.alt_seed
    print(f"\n--- Testing with seed {alt_seed} ---")
    results["seed_alt"] = run_test_with_direction(
        model, config, datasets, direction, layer, alpha, intervention_type,
        seed=alt_seed
    )

    # Compare
    print("\n--- Comparison ---")
    for split in ["harm_train", "harm_test", "benign"]:
        seed0_refusal = results["seed_0"][results["seed_0"]["split"] == split]["is_refusal"].mean()
        seed_alt_refusal = results["seed_alt"][results["seed_alt"]["split"] == split]["is_refusal"].mean()
        diff = abs(seed0_refusal - seed_alt_refusal)

        print(f"{split}: Seed0={seed0_refusal:.3f}, SeedAlt={seed_alt_refusal:.3f}, Diff={diff:.3f}")

    return results


def run_test_with_direction(
    model: HookedTransformer,
    config: Config,
    datasets: Dict,
    direction: torch.Tensor,
    layer: int,
    alpha: float,
    intervention_type: str,
    seed: int = None,
) -> pd.DataFrame:
    """Helper to run a test with a given direction."""
    if seed is None:
        seed = config.generation.seed

    hooks = create_steering_hook_fn(
        direction=direction,
        alpha=alpha,
        layer=layer,
        intervention_type=intervention_type,
        hook_point=config.direction.hook_point
    )

    results = []

    for split_name, prompts_data in datasets.items():
        prompts = [p["prompt"] for p in prompts_data]

        completions = generate_completions(
            model=model,
            prompts=prompts,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            batch_size=config.generation.batch_size,
            seed=seed,
            show_progress=False,
            hook_fn=hooks
        )

        for prompt_data, completion in zip(prompts_data, completions):
            refusal = detect_refusal(completion)
            helpful = detect_helpfulness(completion)

            results.append({
                "split": split_name,
                "refusal_score": refusal["refusal_score"],
                "is_refusal": refusal["is_refusal"],
                "helpfulness_score": helpful["helpfulness_score"],
                "is_helpful": helpful["is_helpful"],
                "char_length": len(completion)
            })

    return pd.DataFrame(results)


def run_all_sanity_checks(
    model: HookedTransformer,
    config: Config,
    baseline_df: pd.DataFrame,
    steered_df: pd.DataFrame,
    test_layer: int = 8,
    test_alpha: float = 2.0,
    test_intervention: str = "add",
):
    """
    Run all sanity checks.

    Args:
        model: Model instance
        config: Config object
        baseline_df: Baseline results
        steered_df: Steered results (for confound check)
        test_layer: Layer to test
        test_alpha: Alpha to test
        test_intervention: Intervention type to test
    """
    if not config.sanity_checks.enabled:
        print("Sanity checks disabled in config.")
        return

    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECKS")
    print("=" * 60)

    # 1. Template robustness
    template_results = sanity_check_template_robustness(
        model, config, test_layer, test_alpha, test_intervention
    )

    # 2. Confounds
    confound_results = sanity_check_confounds(baseline_df, steered_df)

    # 3. Seed robustness
    seed_results = sanity_check_seed_robustness(
        model, config, test_layer, test_alpha, test_intervention
    )

    print("\n" + "=" * 60)
    print("SANITY CHECKS COMPLETE")
    print("=" * 60)

    return {
        "template": template_results,
        "confounds": confound_results,
        "seed": seed_results
    }
