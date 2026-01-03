"""
Text generation utilities with batching and progress tracking.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def generate_completions(
    model: HookedTransformer,
    prompts: List[str],
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 8,
    seed: Optional[int] = None,
    show_progress: bool = True,
    hook_fn: Optional[Callable] = None,
) -> List[str]:
    """
    Generate completions for a list of prompts.

    Args:
        model: HookedTransformer model
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for generation
        seed: Random seed for reproducibility
        show_progress: Show progress bar
        hook_fn: Optional hook function for interventions

    Returns:
        List of generated completion strings (without prompts)
    """
    if seed is not None:
        set_seed(seed)

    completions = []
    batches = batch_items(prompts, batch_size)

    iterator = tqdm(batches, desc="Generating") if show_progress else batches

    for batch in iterator:
        # Tokenize batch
        tokens = model.to_tokens(batch, prepend_bos=True)

        # Generate with optional hooks
        if hook_fn is not None:
            with model.hooks(fwd_hooks=hook_fn):
                generated = model.generate(
                    input=tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    stop_at_eos=False,  # Don't stop early
                )
        else:
            generated = model.generate(
                input=tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stop_at_eos=False,
            )

        # Decode and extract only the new tokens (not the prompt)
        for i, prompt in enumerate(batch):
            full_text = model.to_string(generated[i])

            # Remove BOS token if present
            bos_token = model.tokenizer.bos_token or "<|endoftext|>"
            if full_text.startswith(bos_token):
                full_text = full_text[len(bos_token):]

            # Remove the prompt from the beginning
            if full_text.startswith(prompt):
                completion = full_text[len(prompt):]
            else:
                # Fallback: just use the generated text as-is
                completion = full_text

            completions.append(completion)

    return completions


def generate_with_forced_prefix(
    model: HookedTransformer,
    prompts: List[str],
    prefix: str,
    batch_size: int = 8,
    show_progress: bool = False,
) -> List[torch.Tensor]:
    """
    Generate with a forced prefix (for direction construction).

    This function runs teacher forcing with the prefix to get
    residual stream activations.

    Args:
        model: HookedTransformer model
        prompts: List of prompt strings
        prefix: Prefix to force (e.g., " I can't help with that.")
        batch_size: Batch size for processing
        show_progress: Show progress bar

    Returns:
        List of residual stream activations (one per prompt)
    """
    activations = []
    batches = batch_items(prompts, batch_size)

    iterator = tqdm(batches, desc=f"Forced prefix") if show_progress else batches

    for batch in iterator:
        # Create prompts with forced prefix
        full_texts = [prompt + prefix for prompt in batch]

        # Tokenize
        tokens = model.to_tokens(full_texts, prepend_bos=True)

        # Run forward pass (teacher forcing)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        # Extract activations for this batch
        for i in range(len(batch)):
            activations.append(cache)

    return activations


def generate_batch(
    model: HookedTransformer,
    prompts: List[str],
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    hook_fn: Optional[Callable] = None,
) -> List[str]:
    """
    Generate completions for a single batch of prompts.

    Useful for notebook experimentation where you want to process
    a small batch without batching logic.

    Args:
        model: HookedTransformer model
        prompts: List of prompts (single batch)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        hook_fn: Optional hook function

    Returns:
        List of completions
    """
    tokens = model.to_tokens(prompts, prepend_bos=True)

    if hook_fn is not None:
        with model.hooks(fwd_hooks=hook_fn):
            generated = model.generate(
                input=tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stop_at_eos=False,
            )
    else:
        generated = model.generate(
            input=tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            stop_at_eos=False,
        )

    # Decode and remove prompts
    completions = []
    for i, prompt in enumerate(prompts):
        full_text = model.to_string(generated[i])

        # Remove BOS token if present
        bos_token = model.tokenizer.bos_token or "<|endoftext|>"
        if full_text.startswith(bos_token):
            full_text = full_text[len(bos_token):]

        if full_text.startswith(prompt):
            completion = full_text[len(prompt):]
        else:
            completion = full_text
        completions.append(completion)

    return completions
