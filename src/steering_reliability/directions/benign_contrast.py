"""
Build benign contrast direction for testing specificity to harmful content.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .build_direction import get_residual_at_position


def build_benign_contrast_direction(
    model: HookedTransformer,
    benign_prompts: List[str],
    suffix_a: str = " Here's a helpful answer:",
    suffix_b: str = " Let me think step by step:",
    layer: int = 10,
    hook_point: str = "resid_post",
    token_position: str = "last_prompt_token",
    normalize: bool = True,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build a contrast direction on benign prompts using neutral suffixes.

    This tests whether ablation along *any* contrast direction produces refusal,
    or whether the effect is specific to harmful compliance/refusal geometry.

    The benign direction contrasts two neutral continuations:
    - Suffix A: Direct helpful response
    - Suffix B: Step-by-step thinking

    If ablation along v_benign produces high refusal on harm_test,
    that undermines the claim that the learned direction captures
    something specific about harmful compliance.

    Args:
        model: HookedTransformer model
        benign_prompts: List of benign prompts
        suffix_a: First suffix (e.g., direct helpful response)
        suffix_b: Second suffix (e.g., thinking step-by-step)
        layer: Layer index
        hook_point: Hook point name
        token_position: Token position to extract
        normalize: Whether to normalize direction
        batch_size: Batch size for processing
        show_progress: Show progress bar

    Returns:
        Tuple of (benign_direction, metadata_dict)
    """
    differences = []

    # Process in batches
    for i in tqdm(range(0, len(benign_prompts), batch_size),
                  desc="Building benign contrast direction", disable=not show_progress):
        batch = benign_prompts[i:i + batch_size]

        for prompt in batch:
            # Create suffix A sequence
            text_a = prompt + suffix_a
            tokens_a = model.to_tokens(text_a, prepend_bos=True)

            # Create suffix B sequence
            text_b = prompt + suffix_b
            tokens_b = model.to_tokens(text_b, prepend_bos=True)

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
                _, cache_a = model.run_with_cache(tokens_a)
                _, cache_b = model.run_with_cache(tokens_b)

            # Extract residuals
            resid_a = get_residual_at_position(cache_a, layer, pos, hook_point)
            resid_b = get_residual_at_position(cache_b, layer, pos, hook_point)

            # Compute difference
            diff = resid_a - resid_b
            differences.append(diff.cpu())

    # Stack and compute mean
    differences = torch.stack(differences)
    direction = differences.mean(dim=0)

    # Compute metadata
    metadata = {
        "direction_type": "benign_contrast",
        "layer": layer,
        "hook_point": hook_point,
        "token_position": token_position,
        "n_prompts": len(benign_prompts),
        "suffix_a": suffix_a,
        "suffix_b": suffix_b,
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
