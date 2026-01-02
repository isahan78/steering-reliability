"""
Build steering directions using contrastive forced-prefix method.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np


def get_residual_at_position(
    cache: Dict[str, torch.Tensor],
    layer: int,
    position: int,
    hook_point: str = "resid_post"
) -> torch.Tensor:
    """
    Extract residual stream activation at a specific layer and position.

    Args:
        cache: Activation cache from model.run_with_cache()
        layer: Layer index
        position: Token position
        hook_point: Hook point name (resid_post or resid_pre)

    Returns:
        Residual stream tensor of shape (d_model,)
    """
    # Hook name format: "blocks.{layer}.hook_{hook_point}"
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    if hook_name not in cache:
        raise ValueError(f"Hook {hook_name} not found in cache. Available: {list(cache.keys())}")

    # Shape: (batch, seq_len, d_model)
    resid = cache[hook_name]

    # Extract at position (assume batch size 1)
    if resid.shape[0] == 1:
        return resid[0, position, :]
    else:
        # If batch > 1, return mean across batch
        return resid[:, position, :].mean(dim=0)


def build_contrastive_direction(
    model: HookedTransformer,
    prompts: List[str],
    refusal_prefix: str,
    compliance_prefix: str,
    layer: int,
    hook_point: str = "resid_post",
    token_position: str = "last_prompt_token",
    normalize: bool = True,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build steering direction using contrastive forced-prefix method.

    For each prompt p:
    1. Create two sequences:
       - p + refusal_prefix (teacher-forced)
       - p + compliance_prefix (teacher-forced)
    2. Capture residual stream at token_position in layer
    3. Compute difference: d_i = resid_refusal_i - resid_compliance_i
    4. Average: v = mean(d_i)
    5. Optionally normalize: v = v / ||v||

    Args:
        model: HookedTransformer model
        prompts: List of prompts to use for direction construction
        refusal_prefix: Refusal prefix (e.g., " I can't help with that.")
        compliance_prefix: Compliance prefix (e.g., " Sure, here's how:")
        layer: Layer index to extract activations from
        hook_point: Hook point (resid_post or resid_pre)
        token_position: Token position to extract ("last_prompt_token" or "first_prefix_token")
        normalize: Whether to normalize direction to unit norm
        batch_size: Batch size for processing
        show_progress: Show progress bar

    Returns:
        Tuple of (direction_vector, metadata_dict)
    """
    differences = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Building direction", disable=not show_progress):
        batch = prompts[i:i + batch_size]

        for prompt in batch:
            # Create refusal sequence
            refusal_text = prompt + refusal_prefix
            refusal_tokens = model.to_tokens(refusal_text, prepend_bos=True)

            # Create compliance sequence
            compliance_text = prompt + compliance_prefix
            compliance_tokens = model.to_tokens(compliance_text, prepend_bos=True)

            # Determine token position
            prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
            prompt_len = prompt_tokens.shape[1]

            if token_position == "last_prompt_token":
                pos = prompt_len - 1
            elif token_position == "first_prefix_token":
                pos = prompt_len
            else:
                raise ValueError(f"Invalid token_position: {token_position}")

            # Run forward passes with caching
            with torch.no_grad():
                _, refusal_cache = model.run_with_cache(refusal_tokens)
                _, compliance_cache = model.run_with_cache(compliance_tokens)

            # Extract residuals
            refusal_resid = get_residual_at_position(
                refusal_cache, layer, pos, hook_point
            )
            compliance_resid = get_residual_at_position(
                compliance_cache, layer, pos, hook_point
            )

            # Compute difference
            diff = refusal_resid - compliance_resid
            differences.append(diff.cpu())

    # Stack and compute mean
    differences = torch.stack(differences)
    direction = differences.mean(dim=0)

    # Compute metadata
    metadata = {
        "layer": layer,
        "hook_point": hook_point,
        "token_position": token_position,
        "n_prompts": len(prompts),
        "refusal_prefix": refusal_prefix,
        "compliance_prefix": compliance_prefix,
        "direction_norm": direction.norm().item(),
        "mean_diff_norm": differences.norm(dim=1).mean().item(),
        "std_diff_norm": differences.norm(dim=1).std().item(),
    }

    # Normalize if requested
    if normalize:
        direction = direction / direction.norm()
        metadata["normalized"] = True
        metadata["final_norm"] = direction.norm().item()
    else:
        metadata["normalized"] = False

    return direction, metadata


def build_shuffled_contrast_direction(
    model: HookedTransformer,
    prompts: List[str],
    refusal_prefix: str,
    compliance_prefix: str,
    layer: int,
    hook_point: str = "resid_post",
    token_position: str = "last_prompt_token",
    normalize: bool = True,
    batch_size: int = 8,
    shuffle_seed: int = 42,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build direction using shuffled contrast baseline.

    This tests whether per-prompt alignment matters for the contrastive effect.
    Instead of computing:
        d_i = resid_refusal(prompt_i) - resid_compliance(prompt_i)

    We compute:
        d_i = resid_refusal(prompt_i) - resid_compliance(prompt_perm(i))

    Where perm is a fixed random permutation. This preserves:
    - Same two suffix templates
    - Same model forward passes
    - Same prompts

    But breaks the per-prompt pairing that should matter if the direction
    captures a consistent contrastive feature.

    Args:
        model: HookedTransformer model
        prompts: List of prompts
        refusal_prefix: Refusal prefix
        compliance_prefix: Compliance prefix
        layer: Layer index
        hook_point: Hook point name
        token_position: Token position to extract
        normalize: Whether to normalize direction
        batch_size: Batch size for processing
        shuffle_seed: Seed for shuffling compliance prompts
        show_progress: Show progress bar

    Returns:
        Tuple of (shuffled_direction, metadata_dict)
    """
    # Create shuffled permutation of compliance prompts
    rng = np.random.RandomState(shuffle_seed)
    shuffled_indices = rng.permutation(len(prompts))
    compliance_prompts = [prompts[i] for i in shuffled_indices]

    differences = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="Building shuffled direction", disable=not show_progress):
        batch_refusal = prompts[i:i + batch_size]
        batch_compliance = compliance_prompts[i:i + batch_size]

        for ref_prompt, comp_prompt in zip(batch_refusal, batch_compliance):
            # Create refusal sequence (aligned)
            refusal_text = ref_prompt + refusal_prefix
            refusal_tokens = model.to_tokens(refusal_text, prepend_bos=True)

            # Create compliance sequence (shuffled - different prompt!)
            compliance_text = comp_prompt + compliance_prefix
            compliance_tokens = model.to_tokens(compliance_text, prepend_bos=True)

            # Determine token position (use refusal prompt length)
            prompt_tokens = model.to_tokens(ref_prompt, prepend_bos=True)
            prompt_len = prompt_tokens.shape[1]

            if token_position == "last_prompt_token":
                pos_ref = prompt_len - 1
            elif token_position == "first_prefix_token":
                pos_ref = prompt_len
            else:
                raise ValueError(f"Invalid token_position: {token_position}")

            # For compliance, use same position strategy
            comp_prompt_tokens = model.to_tokens(comp_prompt, prepend_bos=True)
            comp_prompt_len = comp_prompt_tokens.shape[1]

            if token_position == "last_prompt_token":
                pos_comp = comp_prompt_len - 1
            elif token_position == "first_prefix_token":
                pos_comp = comp_prompt_len
            else:
                raise ValueError(f"Invalid token_position: {token_position}")

            # Run forward passes with caching
            with torch.no_grad():
                _, refusal_cache = model.run_with_cache(refusal_tokens)
                _, compliance_cache = model.run_with_cache(compliance_tokens)

            # Extract residuals
            refusal_resid = get_residual_at_position(
                refusal_cache, layer, pos_ref, hook_point
            )
            compliance_resid = get_residual_at_position(
                compliance_cache, layer, pos_comp, hook_point
            )

            # Compute difference (now misaligned!)
            diff = refusal_resid - compliance_resid
            differences.append(diff.cpu())

    # Stack and compute mean
    differences = torch.stack(differences)
    direction = differences.mean(dim=0)

    # Compute metadata
    metadata = {
        "direction_type": "shuffled_contrast",
        "layer": layer,
        "hook_point": hook_point,
        "token_position": token_position,
        "n_prompts": len(prompts),
        "refusal_prefix": refusal_prefix,
        "compliance_prefix": compliance_prefix,
        "shuffle_seed": shuffle_seed,
        "direction_norm": direction.norm().item(),
        "mean_diff_norm": differences.norm(dim=1).mean().item(),
        "std_diff_norm": differences.norm(dim=1).std().item(),
    }

    # Normalize if requested
    if normalize:
        direction = direction / direction.norm()
        metadata["normalized"] = True
        metadata["final_norm"] = direction.norm().item()
    else:
        metadata["normalized"] = False

    return direction, metadata


def save_direction(
    direction: torch.Tensor,
    metadata: Dict[str, Any],
    save_path: str,
):
    """
    Save direction vector and metadata.

    Args:
        direction: Direction tensor
        metadata: Metadata dictionary
        save_path: Path to save (will save .pt for tensor and .json for metadata)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save direction tensor
    torch.save(direction, save_path)

    # Save metadata
    metadata_path = save_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved direction to {save_path}")
    print(f"✓ Saved metadata to {metadata_path}")


def load_direction(load_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load direction vector and metadata.

    Args:
        load_path: Path to direction file

    Returns:
        Tuple of (direction_tensor, metadata_dict)
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Direction file not found: {load_path}")

    # Load direction
    direction = torch.load(load_path)

    # Load metadata
    metadata_path = load_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return direction, metadata
