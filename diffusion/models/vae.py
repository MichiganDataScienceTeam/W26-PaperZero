from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from diffusion.models.blocks import Downsample2D, ResidualBlock2D, Upsample2D, channel_schedule, group_count
from diffusion.utils import get_device


@dataclass(frozen=True)
class VAEConfig:
    """
    Configuration for the spatial latent VAE.
    """
    
    in_channels: int = 1
    img_size: int = 128
    base_ch: int = 32
    latent_channels: int = 4
    downsample_factor: int = 8

    @property
    def latent_h(self) -> int:
        return self.img_size // self.downsample_factor

    @property
    def latent_w(self) -> int:
        return self.img_size // self.downsample_factor

    @property
    def latent_scalars(self) -> int:
        return self.latent_channels * self.latent_h * self.latent_w


DEFAULT_VAE_CONFIG = VAEConfig()
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "checkpoint.pt"


class VAE(nn.Module):
    """
    Compact spatial-latent VAE used by the data pipeline.

    Args:
        cfg: VAE architecture configuration.
    """

    def __init__(self, cfg: VAEConfig = DEFAULT_VAE_CONFIG) -> None:
        super().__init__()
        if cfg.img_size % cfg.downsample_factor != 0:
            raise ValueError("img_size must be divisible by downsample_factor")

        self.cfg = cfg

        downsample_steps = int(math.log2(cfg.downsample_factor))
        channels = channel_schedule(cfg.base_ch, downsample_steps)
        final_ch = channels[-1]

        stem_groups = group_count(cfg.base_ch)
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(cfg.in_channels, cfg.base_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(stem_groups, cfg.base_ch),
            nn.SiLU(inplace=True),
            ResidualBlock2D(cfg.base_ch, cfg.base_ch),
        ]

        for idx in range(downsample_steps):
            in_ch = channels[idx]
            out_ch = channels[idx + 1]
            encoder_layers.append(Downsample2D(in_ch))
            encoder_layers.append(ResidualBlock2D(in_ch, out_ch))

        encoder_layers.append(ResidualBlock2D(final_ch, final_ch))
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_head = nn.Conv2d(final_ch, cfg.latent_channels, kernel_size=1)
        self.log_var_head = nn.Conv2d(final_ch, cfg.latent_channels, kernel_size=1)
        self.latent_to_dec = nn.Conv2d(cfg.latent_channels, final_ch, kernel_size=1)

        decoder_layers: list[nn.Module] = []
        curr_ch = final_ch
        for idx in range(downsample_steps, 0, -1):
            next_ch = channels[idx - 1]
            decoder_layers.append(ResidualBlock2D(curr_ch, curr_ch))
            decoder_layers.append(Upsample2D(curr_ch))
            decoder_layers.append(ResidualBlock2D(curr_ch, next_ch))
            curr_ch = next_ch

        decoder_layers.append(ResidualBlock2D(curr_ch, curr_ch))
        decoder_layers.append(nn.Conv2d(curr_ch, cfg.in_channels, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode inputs into latent Gaussian parameters.

        Args:
            x: Input images with shape `(B, C, H, W)`.

        Returns:
            Tuple `(mu, log_var)` with shape `(B, latent_c, latent_h, latent_w)`.
        """

        h = self.encoder(x)
        return self.mu_head(h), self.log_var_head(h)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent samples back to image space.

        Args:
            z: Latent tensor with shape `(B, latent_c, latent_h, latent_w)`.

        Returns:
            Reconstructed images with shape `(B, C, H, W)`.
        """

        return self.decoder(self.latent_to_dec(z))


def load_vae(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    cfg: VAEConfig = DEFAULT_VAE_CONFIG,
) -> VAE:
    """
    Load VAE weights from one explicit checkpoint path.

    Args:
        checkpoint_path: Optional checkpoint path. Uses default if omitted.
        device: Optional device override.
        cfg: VAE architecture configuration.

    Returns:
        Loaded VAE in eval mode on the requested device.
    """

    resolved_device = get_device(device)
    path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {path}")

    payload = torch.load(path, map_location=resolved_device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    model = VAE(cfg=cfg).to(resolved_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


__all__ = ["VAEConfig", "VAE", "load_vae", "DEFAULT_VAE_CONFIG", "DEFAULT_CHECKPOINT"]
