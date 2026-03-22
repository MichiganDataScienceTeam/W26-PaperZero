from __future__ import annotations

import math

import torch
from torch import Tensor

import diffusion.config as config


def cosine_schedule(T: int) -> tuple[Tensor, Tensor]:
    steps = torch.linspace(0, 1, T + 1, dtype=torch.float32)
    alphas = torch.cos(steps * math.pi / 2)
    sigmas = torch.sin(steps * math.pi / 2)
    return alphas, sigmas


def sample_timesteps(batch_size: int, T: int, device: torch.device | str | None = None) -> Tensor:
    return torch.randint(1, T + 1, (batch_size,), device=device)


def _v_target(x_0: Tensor, eps: Tensor, alpha_t: Tensor, sigma_t: Tensor) -> Tensor:
    return alpha_t * eps - sigma_t * x_0


def add_masked_noise(
    x_0: Tensor,
    denoise_mask: Tensor,
    alpha_t: Tensor | float,
    sigma_t: Tensor | float,
    target_type: str = config.DATA.noise_target,
) -> tuple[Tensor, Tensor]:
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
        raise ValueError(f"Unsupported target_type: {target_type}. Expected 'eps' or 'v'.")

    return x_t, denoise_mask * target
