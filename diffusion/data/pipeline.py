from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.origami_sampler import OrigamiSampler
from diffusion.data.noise import add_masked_noise, cosine_schedule, min_snr_weight
from diffusion.models.vae import DEFAULT_VAE_CONFIG, VAE, VAEConfig, load_vae
from diffusion.utils import get_device
from paper import Paper, Segment, Vec2


StateMode = Literal["vae", "raw"]
NoiseTarget = Literal["eps", "v"]


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for trajectory packing and diffusion dataset sampling.
    """
    action_dim: int = 4
    horizon: int = 15
    timesteps: int = 1000
    noise_target: NoiseTarget = "v"
    state_mode: StateMode = "vae"
    image_size: int = 128
    pad_state_value: float = 0.0
    pad_action_value: float = 0.0
    min_snr_gamma: float | None = 5.0
    vae_config: VAEConfig = DEFAULT_VAE_CONFIG
    vae_checkpoint_path: str | Path | None = None


@dataclass
class TrajectoryExample:
    """
    One trajectory example with paper states and fold actions.

    Args:
        papers: List of paper states `[paper_0, ..., paper_L]` with length `L + 1`.
        actions: Fold actions with shape `(L, action_dim)`.
    """

    papers: list[Paper]
    actions: npt.NDArray[np.float32]


def _state_dim(cfg: PipelineConfig) -> int:
    """
    Resolve flattened state dimension from pipeline configuration.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Flattened state dimension.
    """
    if cfg.state_mode == "raw":
        return cfg.image_size * cfg.image_size
    if cfg.state_mode == "vae":
        return cfg.vae_config.latent_scalars
    raise ValueError(f"Unsupported state_mode: {cfg.state_mode}")


def _encode_states(
    papers: Sequence[Paper],
    cfg: PipelineConfig,
    vae: VAE | None,
    device: torch.device,
) -> Tensor:
    """
    Encode paper states into flattened vectors for the configured state mode.

    Args:
        papers: Sequence of paper states.
        cfg: Pipeline configuration.
        vae: Optional VAE model for latent-state encoding.
        device: Device used for VAE encoding.

    Returns:
        Tensor with shape `(num_states, state_dim)`.
    """
    if cfg.state_mode == "raw":
        rasters = np.stack(
            [paper.rasterize(cfg.image_size, cfg.image_size).astype(np.float32) for paper in papers],
            axis=0,
        )
        return torch.from_numpy(rasters).flatten(start_dim=1)

    if cfg.state_mode == "vae":
        if vae is None:
            raise ValueError("VAE must be loaded when state_mode='vae'.")
        rasters = np.stack(
            [paper.rasterize(cfg.image_size, cfg.image_size).astype(np.float32)[None, :, :] for paper in papers],
            axis=0,
        )
        x = torch.from_numpy(rasters).to(device)
        with torch.no_grad():
            mu, _ = vae.encode_stats(x)
        return mu.flatten(start_dim=1).cpu()

    raise ValueError(f"Unsupported state_mode: {cfg.state_mode}")


def _validate_trajectory(example: TrajectoryExample, cfg: PipelineConfig) -> int:
    """
    Validate trajectory shapes and return its action length.

    Args:
        example: Input trajectory example.
        cfg: Pipeline configuration.

    Returns:
        Action count `L` for the trajectory.
    """
    actions = np.asarray(example.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != cfg.action_dim:
        raise ValueError(f"actions must have shape (L, {cfg.action_dim}), got {actions.shape}")

    length = int(actions.shape[0])
    if len(example.papers) != length + 1:
        raise ValueError(f"papers length must equal L+1={length + 1}, got {len(example.papers)}")
    if length > cfg.horizon:
        raise ValueError(f"trajectory length L={length} exceeds horizon H={cfg.horizon}")
    return length


def _pack_one(
    example: TrajectoryExample,
    cfg: PipelineConfig,
    vae: VAE | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Pack one trajectory into fixed-horizon tensors and masks.

    Args:
        example: Input trajectory.
        cfg: Pipeline configuration.
        vae: Optional VAE model used for `state_mode='vae'`.
        device: Device used for VAE encoding.

    Returns:
        Tuple `(x_0, denoise, slot_valid, slot_pin)` where:
        - `x_0` has shape `(H+1, token_dim)`
        - `denoise` has shape `(H+1, token_dim)`
        - `slot_valid` has shape `(H+1,)`
        - `slot_pin` has shape `(H+1,)`
    """
    length = _validate_trajectory(example, cfg)
    actions = torch.from_numpy(np.asarray(example.actions, dtype=np.float32))

    states_flat = _encode_states(example.papers, cfg=cfg, vae=vae, device=device)
    state_dim = _state_dim(cfg)
    slots = cfg.horizon + 1
    token_dim = state_dim + cfg.action_dim

    states = torch.full((slots, state_dim), float(cfg.pad_state_value), dtype=torch.float32)
    acts = torch.full((slots, cfg.action_dim), float(cfg.pad_action_value), dtype=torch.float32)

    states[: length + 1] = states_flat[: length + 1]
    if length + 1 < slots:
        states[length + 1 :] = states[length].unsqueeze(0)

    if length > 0:
        acts[:length] = actions
        acts[length:] = actions[length - 1].unsqueeze(0)

    x_0 = torch.cat([states, acts], dim=1)

    slot_valid = torch.zeros(slots, dtype=torch.float32)
    slot_valid[: length + 1] = 1.0

    slot_pin = torch.zeros(slots, dtype=torch.float32)
    slot_pin[0] = 1.0
    slot_pin[-1] = 1.0

    action_valid = torch.zeros_like(slot_valid)
    action_valid[:-1] = slot_valid[1:]

    denoise = torch.zeros(slots, token_dim, dtype=torch.float32)
    denoise[:, :state_dim] = (1.0 - slot_pin).unsqueeze(-1)
    denoise[:, state_dim:] = action_valid.unsqueeze(-1)

    return x_0, denoise, slot_valid, slot_pin


class PackedTrajectoryDataset(Dataset):
    """
    Dataset that samples one noised diffusion tuple per item access.

    Dataset item:
    `(x_t, t, target, slot_valid_mask, slot_pin_mask, denoise_mask, loss_weight)`
    """

    def __init__(
        self,
        trajectories: Sequence[TrajectoryExample],
        cfg: PipelineConfig = PipelineConfig(),
        device: str | torch.device | None = None,
    ) -> None:
        """
        Build packed trajectory tensors and diffusion schedules.

        Args:
            trajectories: Trajectory examples to pre-pack.
            cfg: Pipeline configuration.
            device: Optional explicit device for encoding.
        """
        if len(trajectories) == 0:
            raise ValueError("trajectories must not be empty")

        self.cfg = cfg
        self.device = get_device(device)
        self.vae = None
        if cfg.state_mode == "vae":
            self.vae = load_vae(
                checkpoint_path=cfg.vae_checkpoint_path,
                device=self.device,
                cfg=cfg.vae_config,
            ).eval()

        packed = [
            _pack_one(example, cfg=cfg, vae=self.vae, device=self.device)
            for example in trajectories
        ]

        self.x_0 = torch.stack([row[0] for row in packed], dim=0)
        self.denoise = torch.stack([row[1] for row in packed], dim=0)
        self.slot_valid = torch.stack([row[2] for row in packed], dim=0)
        self.slot_pin = torch.stack([row[3] for row in packed], dim=0)
        # For endpoint-conditioned generation, inference uses full-horizon validity.
        # Train with the same validity mask to avoid train/inference mismatch.
        self.model_slot_valid = torch.ones_like(self.slot_valid)

        self.state_dim = _state_dim(cfg)
        self.token_dim = self.state_dim + self.cfg.action_dim

        self.alphas, self.sigmas = cosine_schedule(self.cfg.timesteps)

    def __len__(self) -> int:
        """
        Returns:
            Dataset length.
        """
        return int(self.x_0.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample one noised training tuple for a trajectory.

        Args:
            idx: Trajectory index.

        Returns:
            Tuple `(x_t, t, target, slot_valid_mask, slot_pin_mask, denoise_mask, loss_weight)`.
        """
        x_0 = self.x_0[idx]
        denoise = self.denoise[idx]
        t = int(torch.randint(1, self.cfg.timesteps + 1, (1,)).item())

        x_t, target = add_masked_noise(
            x_0=x_0,
            denoise_mask=denoise,
            alpha_t=self.alphas[t],
            sigma_t=self.sigmas[t],
            target_type=self.cfg.noise_target,
        )
        # Keep pinned states noise-free during training.
        pin_state = self.slot_pin[idx].unsqueeze(-1)
        x_t[:, : self.state_dim] = (1.0 - pin_state) * x_t[:, : self.state_dim] + pin_state * x_0[:, : self.state_dim]
        target[:, : self.state_dim] = (1.0 - pin_state) * target[:, : self.state_dim]

        loss_weight = torch.tensor(1.0, dtype=torch.float32)
        if self.cfg.min_snr_gamma is not None:
            loss_weight = min_snr_weight(
                alpha_t=self.alphas[t],
                sigma_t=self.sigmas[t],
                gamma=float(self.cfg.min_snr_gamma),
            ).to(torch.float32)

        return x_t, t, target, self.model_slot_valid[idx], self.slot_pin[idx], denoise, loss_weight


__all__ = [
    "StateMode",
    "NoiseTarget",
    "PipelineConfig",
    "TrajectoryExample",
    "PackedTrajectoryDataset",
]


if __name__ == "__main__":
    level = 3
    sampler = OrigamiSampler(max_fold_attempts=80, max_paper_retries=30)

    sample = None
    while sample is None or sample["actual_folds"] < 2:
        sample = sampler.sample(level=level)

    actual_folds = int(sample["actual_folds"])
    actions = sample["total_action"].reshape(actual_folds, 4).astype(np.float32)

    paper = Paper()
    papers = [paper.copy()]
    for action in actions:
        fold = Segment(
            Vec2(float(action[0]), float(action[1])),
            Vec2(float(action[2]), float(action[3])),
        )
        paper.fold(fold)
        papers.append(paper.copy())

    trajectory = TrajectoryExample(papers=papers, actions=actions)
    cfg = PipelineConfig(state_mode="vae")
    dataset = PackedTrajectoryDataset([trajectory], cfg=cfg, device="cpu")
    x_t, t, target, slot_valid_mask, slot_pin_mask, denoise_mask, loss_weight = dataset[0]

    print("x_t:", tuple(x_t.shape))
    print("t:", int(t))
    print("target:", tuple(target.shape))
    print("slot_valid_mask:", tuple(slot_valid_mask.shape))
    print("slot_pin_mask:", tuple(slot_pin_mask.shape))
    print("denoise_mask:", tuple(denoise_mask.shape))
    print("loss_weight:", float(loss_weight))
