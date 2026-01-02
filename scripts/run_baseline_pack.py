"""
Run baseline pack experiments to test direction specificity.

This script tests whether the strong ablation effects are specific
to the learned direction or artifacts of ablating any direction.

Baselines:
1. Random direction ablation (n=10 trials)
2. Shuffled contrast direction ablation
3. Benign contrast direction ablation (optional)

Compared against:
- Learned direction ablation (the "real" condition)
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from steering_reliability.model import load_model
from steering_reliability.data import load_prompts
from steering_reliability.config import load_config
from steering_reliability.directions.build_direction import (
    build_contrastive_direction,
    build_shuffled_contrast_direction,
    save_direction,
    load_direction
)
from steering_reliability.directions.random_direction import (
    generate_random_direction_ensemble
)
from steering_reliability.directions.benign_contrast import (
    build_benign_contrast_direction
)
from steering_reliability.generation import generate_completions
from steering_reliability.interventions.steer import create_steering_hook_fn
from steering_reliability.metrics.refusal import compute_refusal_score
from steering_reliability.metrics.helpfulness import compute_helpfulness_score


def run_baseline_condition(
    model,
    prompts_data,
    direction,
    direction_type,
    layer,
    alpha,
    split_name,
    config,
    random_trial=None,
):
    """Run one baseline condition and return results."""
    results = []

    # Create steering hooks for ablation (skip if alpha=0)
    if alpha == 0:
        hook_fn = None  # No intervention
    else:
        hook_fn = create_steering_hook_fn(
            direction=direction,
            alpha=alpha,
            layer=layer,
            intervention_type='ablate',
            hook_point=config.direction.hook_point
        )

    # Generate completions with ablation
    completions = generate_completions(
        model=model,
        prompts=prompts_data['prompts'],
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        seed=config.generation.seed,
        batch_size=config.generation.batch_size,
        show_progress=False,
        hook_fn=hook_fn
    )

    # Score each completion
    for prompt, completion in zip(prompts_data['prompts'], completions):
        # Refusal score
        refusal_score, _ = compute_refusal_score(completion)
        is_refusal = refusal_score >= 0.5

        # Helpfulness score (for benign only)
        if split_name == 'benign':
            helpfulness_score = compute_helpfulness_score(completion)
            is_helpful = helpfulness_score >= 0.5
        else:
            helpfulness_score = None
            is_helpful = None

        result = {
            'direction_type': direction_type,
            'random_trial': random_trial,
            'layer': layer,
            'alpha': alpha,
            'intervention': 'ablate',
            'split': split_name,
            'prompt': prompt,
            'completion': completion,
            'refusal_score': refusal_score,
            'is_refusal': is_refusal,
            'helpfulness_score': helpfulness_score,
            'is_helpful': is_helpful,
            'char_length': len(completion),
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run baseline pack experiments')
    parser.add_argument('--layer', type=int, default=10, help='Layer to test (default: 10)')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0, 1, 4, 8],
                        help='Alpha values to test')
    parser.add_argument('--n_harm_test', type=int, default=50, help='Number of harm_test prompts')
    parser.add_argument('--n_benign', type=int, default=50, help='Number of benign prompts')
    parser.add_argument('--n_random', type=int, default=10, help='Number of random trials')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--include_benign_contrast', action='store_true',
                        help='Include benign contrast baseline (optional)')
    parser.add_argument('--output_dir', type=str, default='artifacts/baselines',
                        help='Output directory')
    parser.add_argument('--config', type=str, default='configs/gpt2_small_alpha_sweep.yaml',
                        help='Config file to load settings from')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)

    print("="*80)
    print("BASELINE PACK EXPERIMENT")
    print("="*80)
    print(f"Layer: {args.layer}")
    print(f"Alphas: {args.alphas}")
    print(f"Prompts: {args.n_harm_test} harm_test, {args.n_benign} benign")
    print(f"Random trials: {args.n_random}")
    print(f"Include benign contrast: {args.include_benign_contrast}")
    print()

    # Load model
    print("Loading model...")
    model = load_model(
        model_name=config.model.name,
        device=config.model.device,
        dtype=config.model.dtype
    )
    d_model = model.cfg.d_model

    # Load prompts (subsample)
    print("Loading prompts...")
    harm_train_raw = load_prompts(config.data.harm_train_path, max_prompts=args.n_harm_test)
    harm_test_raw = load_prompts(config.data.harm_test_path, max_prompts=args.n_harm_test)
    benign_raw = load_prompts(config.data.benign_path, max_prompts=args.n_benign)

    # Extract prompt strings from loaded dictionaries
    harm_train_data = {'prompts': [p['prompt'] for p in harm_train_raw]}
    harm_test_data = {'prompts': [p['prompt'] for p in harm_test_raw]}
    benign_data = {'prompts': [p['prompt'] for p in benign_raw]}

    print(f"✓ Loaded {len(harm_train_data['prompts'])} harm_train prompts")
    print(f"✓ Loaded {len(harm_test_data['prompts'])} harm_test prompts")
    print(f"✓ Loaded {len(benign_data['prompts'])} benign prompts")
    print()

    # Build or load learned direction
    print("="*80)
    print("BUILDING DIRECTIONS")
    print("="*80)

    learned_dir_path = output_dir / f"learned_layer{args.layer}.pt"

    if learned_dir_path.exists():
        print(f"Loading learned direction from {learned_dir_path}...")
        learned_direction, learned_metadata = load_direction(learned_dir_path)
    else:
        print("Building learned direction...")
        learned_direction, learned_metadata = build_contrastive_direction(
            model=model,
            prompts=harm_train_data['prompts'],
            refusal_prefix=config.direction.refusal_prefix,
            compliance_prefix=config.direction.compliance_prefix,
            layer=args.layer,
            hook_point=config.direction.hook_point,
            token_position=config.direction.token_position,
            normalize=config.direction.normalize,
            show_progress=True
        )
        learned_metadata['direction_type'] = 'learned'
        save_direction(learned_direction, learned_metadata, learned_dir_path)

    # Build shuffled contrast direction
    shuffled_dir_path = output_dir / f"shuffled_layer{args.layer}.pt"

    if shuffled_dir_path.exists():
        print(f"Loading shuffled direction from {shuffled_dir_path}...")
        shuffled_direction, shuffled_metadata = load_direction(shuffled_dir_path)
    else:
        print("Building shuffled contrast direction...")
        shuffled_direction, shuffled_metadata = build_shuffled_contrast_direction(
            model=model,
            prompts=harm_train_data['prompts'],
            refusal_prefix=config.direction.refusal_prefix,
            compliance_prefix=config.direction.compliance_prefix,
            layer=args.layer,
            hook_point=config.direction.hook_point,
            token_position=config.direction.token_position,
            normalize=config.direction.normalize,
            shuffle_seed=42,
            show_progress=True
        )
        save_direction(shuffled_direction, shuffled_metadata, shuffled_dir_path)

    # Generate random directions
    print(f"Generating {args.n_random} random directions...")
    random_directions = generate_random_direction_ensemble(
        d_model=d_model,
        n_trials=args.n_random,
        base_seed=args.seed,
        normalize=True
    )
    print(f"✓ Generated {len(random_directions)} random directions")

    # Optionally build benign contrast direction
    benign_direction = None
    if args.include_benign_contrast:
        benign_dir_path = output_dir / f"benign_contrast_layer{args.layer}.pt"

        if benign_dir_path.exists():
            print(f"Loading benign contrast direction from {benign_dir_path}...")
            benign_direction, benign_metadata = load_direction(benign_dir_path)
        else:
            print("Building benign contrast direction...")
            benign_direction, benign_metadata = build_benign_contrast_direction(
                model=model,
                benign_prompts=benign_data['prompts'],
                suffix_a=" Here's a helpful answer:",
                suffix_b=" Let me think step by step:",
                layer=args.layer,
                hook_point=config.direction.hook_point,
                token_position=config.direction.token_position,
                normalize=True,
                show_progress=True
            )
            save_direction(benign_direction, benign_metadata, benign_dir_path)

    print()

    # Run experiments
    print("="*80)
    print("RUNNING BASELINE PACK")
    print("="*80)

    all_results = []

    # Calculate total number of conditions
    n_directions = 1 + 1 + args.n_random  # learned + shuffled + random trials
    if args.include_benign_contrast:
        n_directions += 1
    n_alphas = len(args.alphas)
    n_splits = 2  # harm_test + benign
    total = n_directions * n_alphas * n_splits

    pbar = tqdm(total=total, desc="Running conditions")

    # Test each direction type
    for alpha in args.alphas:
        # 1. Learned direction
        for split_name, prompts_data in [('harm_test', harm_test_data), ('benign', benign_data)]:
            results = run_baseline_condition(
                model=model,
                prompts_data=prompts_data,
                direction=learned_direction,
                direction_type='learned',
                layer=args.layer,
                alpha=alpha,
                split_name=split_name,
                config=config,
                random_trial=None
            )
            all_results.extend(results)
            pbar.update(1)

        # 2. Shuffled contrast direction
        for split_name, prompts_data in [('harm_test', harm_test_data), ('benign', benign_data)]:
            results = run_baseline_condition(
                model=model,
                prompts_data=prompts_data,
                direction=shuffled_direction,
                direction_type='shuffled',
                layer=args.layer,
                alpha=alpha,
                split_name=split_name,
                config=config,
                random_trial=None
            )
            all_results.extend(results)
            pbar.update(1)

        # 3. Random directions (all trials)
        for trial_idx, (random_dir, random_meta) in enumerate(random_directions):
            for split_name, prompts_data in [('harm_test', harm_test_data), ('benign', benign_data)]:
                results = run_baseline_condition(
                    model=model,
                    prompts_data=prompts_data,
                    direction=random_dir,
                    direction_type='random',
                    layer=args.layer,
                    alpha=alpha,
                    split_name=split_name,
                    config=config,
                    random_trial=trial_idx
                )
                all_results.extend(results)
                pbar.update(1)

        # 4. Benign contrast direction (optional)
        if args.include_benign_contrast:
            for split_name, prompts_data in [('harm_test', harm_test_data), ('benign', benign_data)]:
                results = run_baseline_condition(
                    model=model,
                    prompts_data=prompts_data,
                    direction=benign_direction,
                    direction_type='benign_contrast',
                    layer=args.layer,
                    alpha=alpha,
                    split_name=split_name,
                    config=config,
                    random_trial=None
                )
                all_results.extend(results)
                pbar.update(1)

    pbar.close()

    # Save results
    df = pd.DataFrame(all_results)

    # Add timestamp
    df['timestamp'] = datetime.now().isoformat()
    df['seed'] = args.seed
    df['temperature'] = config.generation.temperature
    df['top_p'] = config.generation.top_p
    df['max_new_tokens'] = config.generation.max_new_tokens

    # Save parquet
    parquet_path = output_dir / 'baseline_pack_results.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"\n✓ Saved results to {parquet_path}")

    # Save summary CSV
    summary = df.groupby(['direction_type', 'alpha', 'split', 'random_trial']).agg({
        'is_refusal': 'mean',
        'is_helpful': 'mean',
        'char_length': 'mean'
    }).reset_index()

    summary_path = output_dir / 'baseline_pack_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to {summary_path}")

    print("\n" + "="*80)
    print("BASELINE PACK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext: Run analysis script to generate figures")
    print(f"  python -m analysis.plot_baseline_pack --in_parquet {parquet_path}")


if __name__ == '__main__':
    main()
