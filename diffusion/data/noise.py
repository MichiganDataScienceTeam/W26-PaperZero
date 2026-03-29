from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_schedule(num_steps: int) -> tuple[Tensor, Tensor]:
    """
    Build cosine alpha/sigma schedules indexed by timestep.

    Args:
        num_steps: Total diffusion step count `T`.

    Returns:
        Tuple `(alphas, sigmas)` each with shape `(T + 1,)`.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    steps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float32)
    alphas = torch.cos(steps * math.pi / 2)
    sigmas = torch.sin(steps * math.pi / 2)
    return alphas, sigmas


def sample_timesteps(batch_size: int, num_steps: int, device: torch.device | str | None = None) -> Tensor:
    """
    Sample integer diffusion timesteps uniformly from `[1, T]`.

    Args:
        batch_size: Number of timesteps to sample.
        num_steps: Total diffusion step count `T`.
        device: Optional destination device.

    Returns:
        Tensor of shape `(batch_size,)` with integer timestep values.
    """
    return torch.randint(1, num_steps + 1, (batch_size,), device=device)


def _v_target(x_0: Tensor, eps: Tensor, alpha_t: Tensor, sigma_t: Tensor) -> Tensor:
    """
    Compute velocity target `v = alpha_t * eps - sigma_t * x_0`.
    """
    return alpha_t * eps - sigma_t * x_0


def add_masked_noise(
    x_0: Tensor,
    denoise_mask: Tensor,
    alpha_t: Tensor | float,
    sigma_t: Tensor | float,
    target_type: str = "v",
) -> tuple[Tensor, Tensor]:
    """
    Add Gaussian noise only on denoise-enabled elements.

    Args:
        x_0: Clean tokens with shape `(..., D)`.
        denoise_mask: Elementwise denoise mask with same shape as `x_0`.
        alpha_t: Cosine alpha value at timestep `t`.
        sigma_t: Cosine sigma value at timestep `t`.
        target_type: Training target type, either `"eps"` or `"v"`.

    Returns:
        Tuple `(x_t, target)` with same shape as `x_0`.
    """
    if x_0.shape != denoise_mask.shape:
        raise ValueError(f"x_0 and denoise_mask must have the same shape, got {x_0.shape} vs {denoise_mask.shape}")

    alpha = torch.as_tensor(alpha_t, dtype=x_0.dtype, device=x_0.device)
    sigma = torch.as_tensor(sigma_t, dtype=x_0.dtype, device=x_0.device)
    while alpha.ndim < x_0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)

    eps = torch.randn_like(x_0)
    noised = alpha * x_0 + sigma * eps
    x_t = (1.0 - denoise_mask) * x_0 + denoise_mask * noised

    if target_type == "eps":
        target = eps
    elif target_type == "v":
        target = _v_target(x_0=x_0, eps=eps, alpha_t=alpha, sigma_t=sigma)
    else:
        raise ValueError("target_type must be 'eps' or 'v'")

    return x_t, denoise_mask * target


__all__ = ["cosine_schedule", "sample_timesteps", "add_masked_noise"]
