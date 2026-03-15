import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Any


TUNED_IN_CHANNELS = 1
TUNED_IMG_SIZE = 128
TUNED_LATENT_DIM = 1024
TUNED_BASE_CH = 32


def _valid_groups(channels: int, requested: int) -> int:
    groups = min(requested, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


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


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = TUNED_IN_CHANNELS,
        img_size: int = TUNED_IMG_SIZE,
        latent_dim: int = TUNED_LATENT_DIM,
        base_ch: int = TUNED_BASE_CH,
    ):
        super().__init__()
        if img_size % 8 != 0:
            raise ValueError(f"img_size must be divisible by 8, got {img_size}")

        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        self.spatial_dim = img_size // 8

        stem_groups = _valid_groups(base_ch, 8)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(stem_groups, base_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(base_ch, base_ch),
            Downsample(base_ch),
            ResidualBlock(base_ch, base_ch * 2),
            Downsample(base_ch * 2),
            ResidualBlock(base_ch * 2, base_ch * 4),
            Downsample(base_ch * 4),
            ResidualBlock(base_ch * 4, base_ch * 4),
        )

        self.enc_feat_dim = base_ch * 4 * self.spatial_dim * self.spatial_dim
        self.fc_enc = nn.Linear(self.enc_feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(base_ch * 4, base_ch * 4),
            Upsample(base_ch * 4),
            ResidualBlock(base_ch * 4, base_ch * 2),
            Upsample(base_ch * 2),
            ResidualBlock(base_ch * 2, base_ch),
            Upsample(base_ch),
            ResidualBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_enc(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.base_ch * 4, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)

    def forward(self, x: Tensor) -> Any:
        return self.decode(self.encode(x))


class VAE(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim, self.latent_dim)

    def encode_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = super().encode(x)
        return self.fc_mu(h), self.fc_log_var(h)

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode_stats(x)
        z = self.reparameterize(mu, log_var)
        x_hat = super().decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return super().decode(z)


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
