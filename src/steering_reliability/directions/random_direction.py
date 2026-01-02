"""
Generate random direction baselines for testing direction specificity.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple


def sample_random_direction(
    d_model: int,
    seed: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Sample a random direction from N(0, I) for baseline comparison.

    This tests whether ablation effects are direction-specific or
    whether ablating any random direction produces similar results.

    Args:
        d_model: Dimension of the model's hidden states
        seed: Random seed for reproducibility
        normalize: Whether to normalize to unit norm (default: True)

    Returns:
        Tuple of (random_direction, metadata_dict)
    """
    # Set seed for reproducibility
    rng = torch.Generator().manual_seed(seed)

    # Sample from standard normal
    direction = torch.randn(d_model, generator=rng)

    # Compute metadata
    metadata = {
        "direction_type": "random",
        "d_model": d_model,
        "seed": seed,
        "direction_norm_before": direction.norm().item(),
    }

    # Normalize if requested
    if normalize:
        direction = direction / direction.norm()
        metadata["normalized"] = True
        metadata["final_norm"] = direction.norm().item()
    else:
        metadata["normalized"] = False

    return direction, metadata


def generate_random_direction_ensemble(
    d_model: int,
    n_trials: int = 10,
    base_seed: int = 0,
    normalize: bool = True,
) -> list:
    """
    Generate multiple random directions for robust baseline comparison.

    Args:
        d_model: Dimension of model hidden states
        n_trials: Number of random trials
        base_seed: Base seed (each trial uses base_seed + trial_idx)
        normalize: Whether to normalize directions

    Returns:
        List of (direction, metadata) tuples
    """
    random_directions = []

    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx
        direction, metadata = sample_random_direction(
            d_model=d_model,
            seed=seed,
            normalize=normalize
        )
        metadata["trial_idx"] = trial_idx
        metadata["n_trials"] = n_trials
        metadata["base_seed"] = base_seed

        random_directions.append((direction, metadata))

    return random_directions
