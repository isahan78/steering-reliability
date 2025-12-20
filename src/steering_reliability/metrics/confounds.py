"""
Confound metrics: length, entropy, and log probability measurements.
"""

from typing import Dict, List, Optional
import torch
import numpy as np


def compute_token_length(tokens: List[int]) -> int:
    """
    Compute token length.

    Args:
        tokens: List of token IDs

    Returns:
        Number of tokens
    """
    return len(tokens)


def compute_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Compute average token-level entropy from logits.

    Args:
        logits: Tensor of shape (seq_len, vocab_size) or (batch, seq_len, vocab_size)

    Returns:
        Average entropy across sequence
    """
    # Handle batch dimension
    if logits.dim() == 3:
        # Average across batch
        logits = logits.mean(dim=0)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)

    # Average across sequence
    avg_entropy = entropy.mean().item()

    return avg_entropy


def compute_avg_logprob(logprobs: torch.Tensor) -> float:
    """
    Compute average log probability.

    Args:
        logprobs: Tensor of log probabilities (seq_len,) or (batch, seq_len)

    Returns:
        Average log probability
    """
    # Handle batch dimension
    if logprobs.dim() == 2:
        logprobs = logprobs.mean(dim=0)

    return logprobs.mean().item()


def compute_perplexity(logprobs: torch.Tensor) -> float:
    """
    Compute perplexity from log probabilities.

    Perplexity = exp(-mean(log_probs))

    Args:
        logprobs: Tensor of log probabilities

    Returns:
        Perplexity value
    """
    avg_logprob = compute_avg_logprob(logprobs)
    return np.exp(-avg_logprob)


def compute_confounds(
    text: str,
    tokens: Optional[List[int]] = None,
    logits: Optional[torch.Tensor] = None,
    logprobs: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all confound metrics.

    Args:
        text: Generated text (for character count)
        tokens: Token IDs (for token count)
        logits: Logits tensor (for entropy calculation)
        logprobs: Log probabilities (for avg logprob)

    Returns:
        Dictionary with confound metrics:
            - char_length: character count
            - token_length: token count (if tokens provided)
            - entropy: average token entropy (if logits provided)
            - avg_logprob: average log probability (if logprobs provided)
            - perplexity: perplexity (if logprobs provided)
    """
    metrics = {
        "char_length": len(text),
    }

    if tokens is not None:
        metrics["token_length"] = compute_token_length(tokens)

    if logits is not None:
        metrics["entropy"] = compute_entropy_from_logits(logits)

    if logprobs is not None:
        metrics["avg_logprob"] = compute_avg_logprob(logprobs)
        metrics["perplexity"] = compute_perplexity(logprobs)

    return metrics


def batch_compute_confounds(
    texts: List[str],
    tokens_list: Optional[List[List[int]]] = None,
    logits_list: Optional[List[torch.Tensor]] = None,
    logprobs_list: Optional[List[torch.Tensor]] = None,
) -> List[Dict[str, float]]:
    """
    Compute confounds for a batch of texts.

    Args:
        texts: List of texts
        tokens_list: List of token sequences
        logits_list: List of logits tensors
        logprobs_list: List of log probability tensors

    Returns:
        List of confound metric dictionaries
    """
    results = []

    for i, text in enumerate(texts):
        tokens = tokens_list[i] if tokens_list is not None else None
        logits = logits_list[i] if logits_list is not None else None
        logprobs = logprobs_list[i] if logprobs_list is not None else None

        result = compute_confounds(text, tokens, logits, logprobs)
        results.append(result)

    return results
