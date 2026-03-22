from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DIFFUSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIFFUSION_DIR.parent


@dataclass(frozen=True)
class VAEConfig:
    in_channels: int = 1
    img_size: int = 128
    base_ch: int = 32
    latent_channels: int = 4
    downsample_factor: int = 8
    checkpoint_filename: str = "checkpoint.pt"
    canonical_checkpoint: str = "checkpoint.pt"
    fallback_checkpoint: str = "checkpoint.pt"

    @property
    def latent_h(self) -> int:
        return self.img_size // self.downsample_factor

    @property
    def latent_w(self) -> int:
        return self.img_size // self.downsample_factor

    @property
    def latent_scalars(self) -> int:
        return self.latent_channels * self.latent_h * self.latent_w


@dataclass(frozen=True)
class DataConfig:
    action_dim: int = 4
    horizon: int = 15
    default_timesteps: int = 1000
    pad_state_value: float = 0.0
    pad_action_value: float = 0.0
    noise_target: str = "v"


@dataclass(frozen=True)
class PipelineConfig:
    artifact_dir: str = "diffusion/artifacts"
    default_split_seed: int = 2026
    default_train_ratio: float = 0.9


@dataclass(frozen=True)
class DenoiserConfig:
    hidden_dim: int = 256
    time_embed_dim: int = 128
    num_blocks: int = 6


VAE = VAEConfig()
DATA = DataConfig()
PIPELINE = PipelineConfig()
DENOISER = DenoiserConfig()


def resolve_vae_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    canonical = PROJECT_ROOT / VAE.canonical_checkpoint
    if canonical.exists():
        return canonical

    primary = DIFFUSION_DIR / VAE.checkpoint_filename
    if primary.exists():
        return primary

    fallback = PROJECT_ROOT / VAE.fallback_checkpoint
    return fallback
