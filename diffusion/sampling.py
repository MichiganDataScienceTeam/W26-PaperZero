from __future__ import annotations

import torch
from torch import Tensor

from diffusion.data.noise import cosine_schedule
from diffusion.models.denoiser import DenoiserModel


def _dynamic_threshold(x0_hat: Tensor, quantile: float = 0.995) -> Tensor:
    """
    Dynamic thresholding on x0 prediction, applied per sample.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    flat = x0_hat.abs().flatten(start_dim=1)
    s = torch.quantile(flat, quantile, dim=1, keepdim=True).clamp_min(1.0)
    while s.ndim < x0_hat.ndim:
        s = s.unsqueeze(-1)
    return torch.clamp(x0_hat, -s, s) / s


def sample_pure_noise_v(
    model: DenoiserModel,
    token_dim: int,
    state_dim: int,
    horizon: int,
    timesteps: int,
    device: torch.device | str,
    pin_start_state: Tensor | None = None,
    pin_goal_state: Tensor | None = None,
    dynamic_threshold_q: float | None = 0.995,
    clip_x0: float | None = None,
) -> tuple[Tensor, dict[str, Tensor | int]]:
    """
    Reverse-sample from x_T ~ N(0, I) with optional endpoint state pinning.
    """
    dev = torch.device(device)
    model.eval()

    alphas, sigmas = cosine_schedule(timesteps)
    L = horizon + 1
    x_t = torch.randn(1, L, token_dim, device=dev)

    slot_valid = torch.ones(1, L, device=dev)
    slot_pin = torch.zeros(1, L, device=dev)
    x0_cond = torch.zeros(1, L, state_dim, device=dev)

    if pin_start_state is not None:
        slot_pin[:, 0] = 1.0
        x0_cond[:, 0, :] = pin_start_state.to(device=dev, dtype=x_t.dtype)
    if pin_goal_state is not None:
        slot_pin[:, -1] = 1.0
        x0_cond[:, -1, :] = pin_goal_state.to(device=dev, dtype=x_t.dtype)

    probe: dict[str, Tensor | int] = {}
    probe_t = max(2, timesteps // 2)

    with torch.no_grad():
        for t in range(timesteps, 0, -1):
            alpha_t = alphas[t].to(device=dev, dtype=x_t.dtype)
            sigma_t = sigmas[t].to(device=dev, dtype=x_t.dtype)

            # Keep known endpoints exact and noise-free.
            if slot_pin[0, 0] > 0:
                x_t[:, 0, :state_dim] = x0_cond[:, 0, :]
            if slot_pin[0, -1] > 0:
                x_t[:, -1, :state_dim] = x0_cond[:, -1, :]

            t_batch = torch.tensor([t], dtype=torch.long, device=dev)
            v_hat = model(x_t, t_batch, slot_valid, slot_pin)

            eps_hat = sigma_t * x_t + alpha_t * v_hat
            x0_hat = alpha_t * x_t - sigma_t * v_hat

            if dynamic_threshold_q is not None:
                x0_hat = _dynamic_threshold(x0_hat, quantile=dynamic_threshold_q)
            if clip_x0 is not None:
                x0_hat = x0_hat.clamp(-float(clip_x0), float(clip_x0))

            sigma_safe = sigma_t.clamp_min(1e-6)
            eps_hat = (x_t - alpha_t * x0_hat) / sigma_safe

            if t == probe_t:
                probe = {
                    "t": t,
                    "x_t": x_t.detach().cpu()[0],
                    "eps_hat": eps_hat.detach().cpu()[0],
                    "x0_hat": x0_hat.detach().cpu()[0],
                }

            if t > 1:
                alpha_prev = alphas[t - 1].to(device=dev, dtype=x_t.dtype)
                sigma_prev = sigmas[t - 1].to(device=dev, dtype=x_t.dtype)
                x_t = alpha_prev * x0_hat + sigma_prev * eps_hat
            else:
                x_t = x0_hat

    if slot_pin[0, 0] > 0:
        x_t[:, 0, :state_dim] = x0_cond[:, 0, :]
    if slot_pin[0, -1] > 0:
        x_t[:, -1, :state_dim] = x0_cond[:, -1, :]

    return x_t[0].cpu(), probe


__all__ = ["sample_pure_noise_v"]
