# I asked chat to clean up code from encoder.py

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        assert out_channels % groups == 0

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.gn2(x)

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
        self.deconv = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 128,
        latent_dim: int = 128,
        base_ch: int = 32,
    ):
        super().__init__()

        self.base_ch = base_ch
        self.latent_dim = latent_dim
        self.spatial_dim = img_size // 8  # 128 → 16

        # ---------------- Encoder ----------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),

            ResidualBlock(base_ch, base_ch),
            Downsample(base_ch),            # 128 → 64

            ResidualBlock(base_ch, base_ch * 2),
            Downsample(base_ch * 2),         # 64 → 32

            ResidualBlock(base_ch * 2, base_ch * 4),
            Downsample(base_ch * 4),         # 32 → 16

            ResidualBlock(base_ch * 4, base_ch * 4),
        )

        enc_feat_dim = base_ch * 4 * self.spatial_dim * self.spatial_dim
        self.fc_enc = nn.Linear(enc_feat_dim, latent_dim)

        # ---------------- Decoder ----------------
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(base_ch * 4, base_ch * 4),
            Upsample(base_ch * 4),           # 16 → 32

            ResidualBlock(base_ch * 4, base_ch * 2),
            Upsample(base_ch * 2),           # 32 → 64

            ResidualBlock(base_ch * 2, base_ch),
            Upsample(base_ch),               # 64 → 128

            ResidualBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    # -------- API --------
    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_enc(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_dec(z)
        h = h.view(
            z.size(0),
            self.base_ch * 4,
            self.spatial_dim,
            self.spatial_dim,
        )
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

class VAE(AutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(self.latent_dim, self.latent_dim)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        h = self.encode(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)
    
    def vae_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_coeff: float = 1.0,
    ):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")

        # KL divergence (averaged over batch)
        kl = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        loss = recon_loss + kl_coeff * kl
        return loss, recon_loss, kl

