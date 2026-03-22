from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import diffusion.config as config
from diffusion.data_noise import add_masked_noise, cosine_schedule
from diffusion.vae import VAE, load_pretrained_vae
from paper import Paper


STATE_DIM = config.VAE.latent_scalars
ACTION_DIM = config.DATA.action_dim
TOKEN_DIM = STATE_DIM + ACTION_DIM


@dataclass
class TrajectoryExample:
    papers: List[Paper]
    actions: np.ndarray


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_states(example: TrajectoryExample, vae: VAE, device: torch.device) -> tuple[Tensor, np.ndarray, int]:
    actions = np.asarray(example.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (L, {ACTION_DIM}), got {actions.shape}")

    L = actions.shape[0]
    if len(example.papers) != L + 1:
        raise ValueError(f"papers length must equal L+1={L + 1}, got {len(example.papers)}")
    if L > config.DATA.horizon:
        raise ValueError(f"trajectory length L={L} exceeds horizon H={config.DATA.horizon}")

    rasters = np.stack(
        [paper.rasterize(config.VAE.img_size, config.VAE.img_size).astype(np.float32)[None, :, :] for paper in example.papers],
        axis=0,
    )
    x = torch.from_numpy(rasters).to(device)
    with torch.no_grad():
        mu, _ = vae.encode_stats(x)
    return mu.flatten(start_dim=1).cpu(), actions, L


def _pack_one(example: TrajectoryExample, vae: VAE, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    states_flat, actions, L = _encode_states(example, vae=vae, device=device)
    slots = config.DATA.horizon + 1

    states = torch.full((slots, STATE_DIM), float(config.DATA.pad_state_value), dtype=torch.float32)
    acts = torch.full((slots, ACTION_DIM), float(config.DATA.pad_action_value), dtype=torch.float32)
    states[: L + 1] = states_flat[: L + 1]
    if L > 0:
        acts[:L] = torch.from_numpy(actions)
    x_0 = torch.cat([states, acts], dim=1)

    slot_valid = torch.zeros(slots, dtype=torch.float32)
    slot_valid[: L + 1] = 1.0

    slot_pin = torch.zeros(slots, dtype=torch.float32)
    slot_pin[0] = 1.0
    slot_pin[L] = 1.0

    action_valid = torch.zeros(slots, dtype=torch.float32)
    if L > 0:
        action_valid[:L] = 1.0

    denoise = torch.zeros(slots, TOKEN_DIM, dtype=torch.float32)
    denoise[:, :STATE_DIM] = (slot_valid * (1.0 - slot_pin)).unsqueeze(-1)
    denoise[:, STATE_DIM:] = action_valid.unsqueeze(-1)
    return x_0, denoise, slot_valid, slot_pin


class PackedTrajectoryDataset(Dataset):
    """
    Dataset item:
    (x_t, t, target, slot_valid_mask, slot_pin_mask)
    """

    def __init__(
        self,
        trajectories: List[TrajectoryExample],
        T: int,
        target_type: str = config.DATA.noise_target,
        device: str | torch.device | None = None,
        checkpoint_path: str | None = None,
    ):
        dev = _device(device)
        vae = load_pretrained_vae(device=dev, checkpoint_path=checkpoint_path).eval()
        packed = [_pack_one(example, vae=vae, device=dev) for example in trajectories]

        self.x_0 = torch.stack([row[0] for row in packed], dim=0)
        self.denoise = torch.stack([row[1] for row in packed], dim=0)
        self.slot_valid = torch.stack([row[2] for row in packed], dim=0)
        self.slot_pin = torch.stack([row[3] for row in packed], dim=0)

        self.T = int(T)
        self.target_type = target_type
        self.alphas, self.sigmas = cosine_schedule(self.T)

    def __len__(self) -> int:
        return int(self.x_0.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor, Tensor, Tensor]:
        x_0 = self.x_0[idx]
        denoise = self.denoise[idx]
        t = int(torch.randint(1, self.T + 1, (1,)).item())
        x_t, target = add_masked_noise(
            x_0=x_0,
            denoise_mask=denoise,
            alpha_t=self.alphas[t],
            sigma_t=self.sigmas[t],
            target_type=self.target_type,
        )
        return x_t, t, target, self.slot_valid[idx], self.slot_pin[idx]


def build_training_dataset(
    trajectories: List[TrajectoryExample],
    T: int,
    target_type: str = config.DATA.noise_target,
    device: str | torch.device | None = None,
    checkpoint_path: str | None = None,
) -> PackedTrajectoryDataset:
    return PackedTrajectoryDataset(
        trajectories=trajectories,
        T=T,
        target_type=target_type,
        device=device,
        checkpoint_path=checkpoint_path,
    )


def sample_training_tuple(
    trajectories: List[TrajectoryExample],
    T: int,
    item_index: int = 0,
    target_type: str = config.DATA.noise_target,
    device: str | torch.device | None = None,
    checkpoint_path: str | None = None,
) -> Tuple[Tensor, int, Tensor, Tensor, Tensor]:
    ds = build_training_dataset(
        trajectories=trajectories,
        T=T,
        target_type=target_type,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    return ds[item_index]


__all__ = ["TrajectoryExample", "PackedTrajectoryDataset", "build_training_dataset", "sample_training_tuple"]
