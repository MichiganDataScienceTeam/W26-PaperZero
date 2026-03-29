from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _valid_groups(channels: int, requested: int) -> int:
    """
    Select a valid GroupNorm group count that divides `channels`.

    Args:
        channels: Channel count for normalization.
        requested: Requested group count.

    Returns:
        Group count that divides `channels`.
    """

    groups = min(requested, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def group_count(channels: int) -> int:
    """
    Pick a small default group count for GroupNorm.

    Args:
        channels: Channel count for normalization.

    Returns:
        Suggested GroupNorm group count.
    """

    if channels % 8 == 0:
        return 8
    if channels % 4 == 0:
        return 4
    return 1


def channel_schedule(base_ch: int, downsample_steps: int) -> list[int]:
    """
    Build encoder/decoder channel widths across downsample stages.

    Args:
        base_ch: Base channel width.
        downsample_steps: Number of downsample stages.

    Returns:
        Channel list of length `downsample_steps + 1`.
    """

    channels = [base_ch]
    for step_idx in range(downsample_steps):
        channels.append(base_ch * min(2 ** (step_idx + 1), 4))
    return channels


class ResidualBlock2D(nn.Module):
    """
    2D residual block with optional time conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        out_groups = _valid_groups(out_channels, groups)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(out_groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)

        self.time_mlp = nn.Linear(time_embed_dim, out_channels) if time_embed_dim is not None else None

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(out_groups, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> Tensor:
        """
        Apply residual 2D processing.

        Args:
            x: Input tensor `(B, C, H, W)`.
            t_emb: Optional time embedding `(B, time_dim)`.

        Returns:
            Tensor with shape `(B, out_channels, H, W)`.
        """

        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))

        if self.time_mlp is not None:
            if t_emb is None:
                raise ValueError("t_emb is required for time-conditioned ResidualBlock2D")
            x = x + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)

        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class Downsample2D(nn.Module):
    """
    Strided 2D downsampling block.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C, H, W)`.

        Returns:
            Tensor `(B, C, H/2, W/2)`.
        """

        return self.conv(x)


class Upsample2D(nn.Module):
    """
    Transposed-convolution 2D upsampling block.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C, H, W)`.

        Returns:
            Tensor `(B, C, 2H, 2W)`.
        """

        return self.deconv(x)


class ResidualBlock1D(nn.Module):
    """
    1D residual block with additive time conditioning.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C, L)`.
            t_emb: Time embedding `(B, time_dim)`.

        Returns:
            Tensor `(B, C, L)`.
        """

        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Down1D(nn.Module):
    """
    Strided 1D downsampling block.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C_in, L)`.

        Returns:
            Tensor `(B, C_out, L/2)`.
        """

        return self.conv(x)


class Up1D(nn.Module):
    """
    Transposed-convolution 1D upsampling block.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C_in, L)`.

        Returns:
            Tensor `(B, C_out, 2L)`.
        """
        
        return self.deconv(x)
