"""
Model loading utilities with automatic GPU detection.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional
import warnings


def get_device(device_str: str = "auto") -> str:
    """
    Get the best available device.

    Args:
        device_str: Device specification. Options:
            - "auto": automatically detect best device (cuda > mps > cpu)
            - "cuda": force CUDA GPU
            - "mps": force Apple Silicon GPU
            - "cpu": force CPU

    Returns:
        Device string (cuda, mps, or cpu)
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        if device_str == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            return "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            warnings.warn("MPS not available, falling back to CPU")
            return "cpu"
        return device_str


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch dtype.

    Args:
        dtype_str: One of "float32", "float16", "bfloat16"

    Returns:
        torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Invalid dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def load_model(
    model_name: str = "EleutherAI/pythia-160m",
    device: str = "auto",
    dtype: str = "float32",
) -> HookedTransformer:
    """
    Load a model using TransformerLens.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("float32", "float16", "bfloat16")

    Returns:
        HookedTransformer model instance

    Example:
        >>> model = load_model("EleutherAI/pythia-160m", device="auto")
        >>> print(model.cfg.n_layers)
        12
    """
    device = get_device(device)
    torch_dtype = get_dtype(dtype)

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Load model with TransformerLens
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
    )

    # Print model info
    print(f"\nModel loaded successfully!")
    print(f"- Layers: {model.cfg.n_layers}")
    print(f"- Hidden size: {model.cfg.d_model}")
    print(f"- Vocabulary size: {model.cfg.d_vocab}")

    return model


def get_model_info(model: HookedTransformer) -> dict:
    """
    Get basic information about a loaded model.

    Args:
        model: HookedTransformer instance

    Returns:
        Dictionary with model information
    """
    return {
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "d_vocab": model.cfg.d_vocab,
        "n_heads": model.cfg.n_heads,
        "device": str(model.cfg.device),
    }
