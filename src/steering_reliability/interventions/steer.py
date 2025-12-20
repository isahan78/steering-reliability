"""
Steering interventions using TransformerLens hooks.
"""

import torch
from typing import Callable, List, Tuple


def make_additive_steering_hook(
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    hook_point: str = "resid_post",
) -> List[Tuple[str, Callable]]:
    """
    Create hook for additive steering: resid += alpha * direction

    Args:
        direction: Steering direction vector (d_model,)
        alpha: Steering strength
        layer: Layer to apply intervention
        hook_point: Hook point (resid_post or resid_pre)

    Returns:
        List of (hook_name, hook_function) tuples for model.hooks()
    """
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    def steering_hook(resid: torch.Tensor, hook):
        """
        Add steering vector to residual stream.

        Args:
            resid: shape (batch, seq, d_model)
            hook: hook object
        """
        # Ensure direction is on the same device
        device = resid.device
        direction_device = direction.to(device)

        # Apply steering: resid += alpha * direction
        # Broadcast across batch and sequence dimensions
        resid = resid + alpha * direction_device

        return resid

    return [(hook_name, steering_hook)]


def make_ablation_steering_hook(
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    hook_point: str = "resid_post",
) -> List[Tuple[str, Callable]]:
    """
    Create hook for projection ablation: resid -= alpha * proj_v(resid)

    This removes the component of the residual stream in the direction
    of the steering vector.

    Args:
        direction: Steering direction vector (d_model,) - should be normalized
        alpha: Ablation strength (1.0 = full removal)
        layer: Layer to apply intervention
        hook_point: Hook point (resid_post or resid_pre)

    Returns:
        List of (hook_name, hook_function) tuples for model.hooks()
    """
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    def ablation_hook(resid: torch.Tensor, hook):
        """
        Remove projection onto steering direction.

        Args:
            resid: shape (batch, seq, d_model)
            hook: hook object
        """
        # Ensure direction is on the same device and normalized
        device = resid.device
        direction_device = direction.to(device)

        # Normalize direction (in case it wasn't already)
        direction_norm = direction_device / direction_device.norm()

        # Compute projection: proj = (resid · v) * v
        # resid shape: (batch, seq, d_model)
        # direction shape: (d_model,)

        # Dot product: (batch, seq, d_model) · (d_model,) -> (batch, seq)
        dot_product = torch.sum(resid * direction_norm, dim=-1, keepdim=True)

        # Projection: (batch, seq, 1) * (d_model,) -> (batch, seq, d_model)
        proj = dot_product * direction_norm

        # Remove projection: resid -= alpha * proj
        resid = resid - alpha * proj

        return resid

    return [(hook_name, ablation_hook)]


def make_steering_hooks(
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    intervention_type: str = "add",
    hook_point: str = "resid_post",
) -> List[Tuple[str, Callable]]:
    """
    Create steering hooks based on intervention type.

    Args:
        direction: Steering direction vector
        alpha: Steering/ablation strength
        layer: Layer to apply intervention
        intervention_type: "add" or "ablate"
        hook_point: Hook point (resid_post or resid_pre)

    Returns:
        List of (hook_name, hook_function) tuples
    """
    if intervention_type == "add":
        return make_additive_steering_hook(direction, alpha, layer, hook_point)
    elif intervention_type == "ablate":
        return make_ablation_steering_hook(direction, alpha, layer, hook_point)
    else:
        raise ValueError(f"Unknown intervention_type: {intervention_type}. Use 'add' or 'ablate'.")


def create_steering_hook_fn(
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    intervention_type: str = "add",
    hook_point: str = "resid_post",
) -> Callable:
    """
    Create a hook function compatible with generation.generate_completions.

    This returns a callable that can be passed to generate_completions(hook_fn=...).

    Args:
        direction: Steering direction vector
        alpha: Steering strength
        layer: Layer to apply intervention
        intervention_type: "add" or "ablate"
        hook_point: Hook point

    Returns:
        Hook function callable
    """
    hooks = make_steering_hooks(direction, alpha, layer, intervention_type, hook_point)
    return hooks
