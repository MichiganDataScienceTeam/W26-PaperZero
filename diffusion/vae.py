import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion.config import VAE, resolve_vae_checkpoint_path

TUNED_IN_CHANNELS = VAE.in_channels
TUNED_IMG_SIZE = VAE.img_size
TUNED_BASE_CH = VAE.base_ch
TUNED_LATENT_CHANNELS = VAE.latent_channels
TUNED_DOWNSAMPLE_FACTOR = VAE.downsample_factor
TUNED_LATENT_H = TUNED_IMG_SIZE // TUNED_DOWNSAMPLE_FACTOR
TUNED_LATENT_W = TUNED_IMG_SIZE // TUNED_DOWNSAMPLE_FACTOR
TUNED_LATENT_SCALARS = TUNED_LATENT_CHANNELS * TUNED_LATENT_H * TUNED_LATENT_W
DEFAULT_CHECKPOINT_PATH = resolve_vae_checkpoint_path()


def _valid_groups(channels: int, requested: int) -> int:
    groups = min(requested, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _group_count(channels: int) -> int:
    if channels % 8 == 0:
        return 8
    if channels % 4 == 0:
        return 4
    return 1


def _channel_schedule(base_ch: int, downsample_steps: int) -> list[int]:
    channels = [base_ch]
    for step_idx in range(downsample_steps):
        mult = min(2 ** (step_idx + 1), 4)
        channels.append(base_ch * mult)
    return channels


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        out_groups = _valid_groups(out_channels, groups)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(out_groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(out_groups, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        return self.act(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class VAE(nn.Module):
    """
    Spatial latent VAE matching the ablation checkpoint architecture.
    mu/log_var shape: [B, latent_channels, img_size/downsample_factor, img_size/downsample_factor]
    """

    def __init__(
        self,
        in_channels: int = TUNED_IN_CHANNELS,
        img_size: int = TUNED_IMG_SIZE,
        base_ch: int = TUNED_BASE_CH,
        latent_channels: int = TUNED_LATENT_CHANNELS,
        downsample_factor: int = TUNED_DOWNSAMPLE_FACTOR,
    ) -> None:
        super().__init__()
        if downsample_factor not in {8, 16}:
            raise ValueError(f"downsample_factor must be 8 or 16, got {downsample_factor}")
        if img_size % downsample_factor != 0:
            raise ValueError(f"img_size={img_size} must be divisible by downsample_factor={downsample_factor}")

        self.in_channels = in_channels
        self.img_size = img_size
        self.base_ch = base_ch
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.latent_h = img_size // downsample_factor
        self.latent_w = img_size // downsample_factor

        downsample_steps = int(math.log2(downsample_factor))
        channels = _channel_schedule(base_ch, downsample_steps)
        final_ch = channels[-1]

        stem_groups = _group_count(base_ch)
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(stem_groups, base_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(base_ch, base_ch),
        ]

        for idx in range(downsample_steps):
            in_ch = channels[idx]
            out_ch = channels[idx + 1]
            encoder_layers.append(Downsample(in_ch))
            encoder_layers.append(ResidualBlock(in_ch, out_ch))
        encoder_layers.append(ResidualBlock(final_ch, final_ch))
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_head = nn.Conv2d(final_ch, latent_channels, kernel_size=1)
        self.log_var_head = nn.Conv2d(final_ch, latent_channels, kernel_size=1)
        self.latent_to_dec = nn.Conv2d(latent_channels, final_ch, kernel_size=1)

        decoder_layers: list[nn.Module] = []
        curr_ch = final_ch
        for idx in range(downsample_steps, 0, -1):
            next_ch = channels[idx - 1]
            decoder_layers.append(ResidualBlock(curr_ch, curr_ch))
            decoder_layers.append(Upsample(curr_ch))
            decoder_layers.append(ResidualBlock(curr_ch, next_ch))
            curr_ch = next_ch
        decoder_layers.append(ResidualBlock(curr_ch, curr_ch))
        decoder_layers.append(nn.Conv2d(curr_ch, in_channels, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.log_var_head(h)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.latent_to_dec(z))

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode_stats(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> Tensor:
        z = torch.randn(n, self.latent_channels, self.latent_h, self.latent_w, device=device)
        return self.decode(z)


def vae_loss(
    x_hat: Tensor,
    x: Tensor,
    mu: Tensor,
    log_var: Tensor,
    kl_coeff: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon + kl_coeff * kl
    return total, recon, kl


def load_pretrained_vae(
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
) -> VAE:
    """
    Load the tuned spatial VAE weights from diffusion/checkpoint.pt by default.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    path = resolve_vae_checkpoint_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "run" in payload and "training" in payload:
        run = payload["run"]
        training = payload["training"]
        if int(run.get("downsample_factor", TUNED_DOWNSAMPLE_FACTOR)) != TUNED_DOWNSAMPLE_FACTOR:
            raise ValueError(f"Checkpoint downsample_factor mismatch: expected {TUNED_DOWNSAMPLE_FACTOR}, got {run.get('downsample_factor')}")
        if int(run.get("latent_c", TUNED_LATENT_CHANNELS)) != TUNED_LATENT_CHANNELS:
            raise ValueError(f"Checkpoint latent_c mismatch: expected {TUNED_LATENT_CHANNELS}, got {run.get('latent_c')}")
        if int(training.get("img_size", TUNED_IMG_SIZE)) != TUNED_IMG_SIZE:
            raise ValueError(f"Checkpoint img_size mismatch: expected {TUNED_IMG_SIZE}, got {training.get('img_size')}")

    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    model = VAE().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
