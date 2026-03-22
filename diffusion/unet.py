# Remove the time emdeddings such that its not for EVERY res block. NVM keep time embeddings for now bc Jeffery says it might be ok
# If the architecture is slow, remove the two nn.Linears()


import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Based on vae.py, but for diffusion UNet with residual blocks and vertical skip connections (residual connections within blocks)

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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, groups: int = 8):
        super().__init__()
        out_groups = _valid_groups(out_channels, groups)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(out_groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)

        self.time_mlp = nn.Linear(time_embed_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 kernel
                nn.GroupNorm(out_groups, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + time_emb
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


class UNet(nn.Module):
    """
    UNet with residual blocks and vertical skip connections (residual connections).
    No horizontal skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 128,
        base_ch: int = 32,
        time_embed_dim: int = 128,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        if downsample_factor not in {8, 16}:
            raise ValueError(f"downsample_factor must be 8 or 16, got {downsample_factor}")
        if img_size % downsample_factor != 0:
            raise ValueError(f"img_size={img_size} must be divisible by downsample_factor={downsample_factor}")

        self.in_channels = in_channels
        self.img_size = img_size
        self.base_ch = base_ch
        self.time_embed_dim = time_embed_dim
        self.downsample_factor = downsample_factor

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        downsample_steps = int(math.log2(downsample_factor))
        channels = _channel_schedule(base_ch, downsample_steps)
        final_ch = channels[-1]

        stem_groups = _group_count(base_ch)
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(stem_groups, base_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(base_ch, base_ch, time_embed_dim),
        ])

        for idx in range(downsample_steps):
            in_ch = channels[idx]
            out_ch = channels[idx + 1]
            self.encoder_layers.append(Downsample(in_ch))
            self.encoder_layers.append(ResidualBlock(in_ch, out_ch, time_embed_dim))
        self.encoder_layers.append(ResidualBlock(final_ch, final_ch, time_embed_dim))

        self.decoder_layers = nn.ModuleList()
        curr_ch = final_ch
        for idx in range(downsample_steps, 0, -1):
            next_ch = channels[idx - 1]
            self.decoder_layers.append(ResidualBlock(curr_ch, curr_ch, time_embed_dim))
            self.decoder_layers.append(Upsample(curr_ch))
            self.decoder_layers.append(ResidualBlock(curr_ch, next_ch, time_embed_dim))
            curr_ch = next_ch
        self.decoder_layers.append(ResidualBlock(curr_ch, curr_ch, time_embed_dim))
        self.decoder_layers.append(nn.Conv2d(curr_ch, in_channels, kernel_size=3, padding=1))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embed(t)

        # Encoder
        for layer in self.encoder_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)

        # Decoder
        for layer in self.decoder_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)

        return x
