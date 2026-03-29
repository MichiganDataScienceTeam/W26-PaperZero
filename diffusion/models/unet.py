from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion.models.blocks import Down1D, Up1D
from diffusion.models.denoiser import DenoiserModel


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """
    Build sinusoidal timestep embeddings.

    Args:
        t: Timesteps with shape `(B,)`.
        dim: Embedding dimension.

    Returns:
        Tensor with shape `(B, dim)`.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _ResBlock1D(nn.Module):
    """
    Time-conditioned residual block with temporal convolutions.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        time_dim: Time embedding width.
        groups: GroupNorm groups (auto-adjusted to channel divisibility).
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()

        g1 = min(groups, in_ch)
        while in_ch % g1 != 0 and g1 > 1:
            g1 -= 1

        g2 = min(groups, out_ch)
        while out_ch % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C_in, L)`.
            t_emb: Time embedding `(B, time_dim)`.

        Returns:
            Tensor `(B, C_out, L)`.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _SelfAttention1D(nn.Module):
    """
    Temporal self-attention block over sequence dimension.

    Args:
        channels: Channel width.
        num_heads: Attention head count.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor `(B, C, L)`.

        Returns:
            Tensor `(B, C, L)`.
        """
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class UNetDenoiser(DenoiserModel):
    """
    DDPM-style temporal-convolution U-Net denoiser for trajectory tokens.

    Notes:
        Attention is applied at deeper levels and bottleneck, which is typically
        where it brings the most gain for sequence diffusion while limiting cost.

    Args:
        token_dim: Token feature dimension `D`.
        state_dim: Leading state feature dimension in each token.
        base_ch: Base hidden channel width.
        time_dim: Timestep embedding width.
        channel_mults: Per-level channel multipliers.
        num_res_blocks: Residual blocks per level.
        attn_levels: Down/up level indices where attention is applied.
    """

    def __init__(
        self,
        token_dim: int,
        state_dim: int,
        base_ch: int = 128,
        time_dim: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim >= token_dim:
            raise ValueError("state_dim must be in (0, token_dim)")
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must include at least two levels")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")

        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.time_dim = int(time_dim)
        self.levels = len(channel_mults)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_proj = nn.Conv1d(token_dim, base_ch, kernel_size=3, padding=1)

        level_channels = [base_ch * m for m in channel_mults]

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_ch = base_ch
        for level_idx, out_ch in enumerate(level_channels):
            blocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                in_ch = curr_ch if block_idx == 0 else out_ch
                blocks.append(_ResBlock1D(in_ch=in_ch, out_ch=out_ch, time_dim=time_dim))
                curr_ch = out_ch
            self.down_blocks.append(blocks)
            self.down_attn.append(_SelfAttention1D(curr_ch) if level_idx in attn_levels else nn.Identity())

            if level_idx < self.levels - 1:
                self.downsamples.append(Down1D(curr_ch, curr_ch))

        self.mid_block1 = _ResBlock1D(curr_ch, curr_ch, time_dim=time_dim)
        self.mid_attn = _SelfAttention1D(curr_ch)
        self.mid_block2 = _ResBlock1D(curr_ch, curr_ch, time_dim=time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level_idx in reversed(range(self.levels)):
            out_ch = level_channels[level_idx]
            blocks = nn.ModuleList()

            # Horizontal skip connection via channel concat.
            blocks.append(_ResBlock1D(in_ch=curr_ch + out_ch, out_ch=out_ch, time_dim=time_dim))
            curr_ch = out_ch
            for _ in range(1, num_res_blocks):
                blocks.append(_ResBlock1D(in_ch=curr_ch, out_ch=out_ch, time_dim=time_dim))

            self.up_blocks.append(blocks)
            self.up_attn.append(_SelfAttention1D(curr_ch) if level_idx in attn_levels else nn.Identity())
            if level_idx > 0:
                self.upsamples.append(Up1D(curr_ch, curr_ch))

        self.out_norm = nn.GroupNorm(8 if curr_ch % 8 == 0 else 1, curr_ch)
        self.out_proj = nn.Conv1d(curr_ch, token_dim, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        slot_valid_mask: Tensor,
        slot_pin_mask: Tensor,
    ) -> Tensor:
        """
        Predict denoising target for one diffusion step.

        Args:
            x_t: Noised tokens with shape `(B, L, token_dim)`.
            t: Diffusion timesteps with shape `(B,)`.
            slot_valid_mask: Valid-slot mask with shape `(B, L)`.
            slot_pin_mask: Pinned-slot mask with shape `(B, L)`.

        Returns:
            Predicted denoising target with shape `(B, L, token_dim)`.
        """

        if x_t.ndim != 3:
            raise ValueError(f"x_t must have shape (B, L, D), got {tuple(x_t.shape)}")
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError("t must have shape (B,)")
        if slot_valid_mask.shape != x_t.shape[:2] or slot_pin_mask.shape != x_t.shape[:2]:
            raise ValueError("slot masks must have shape (B, L)")

        required = 2 ** (self.levels - 1)
        length = x_t.shape[1]
        if length % required != 0:
            raise ValueError(
                f"sequence length L={length} must be divisible by {required} for {self.levels} U-Net levels"
            )

        h = x_t.transpose(1, 2)
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))

        h = self.in_proj(h)

        skips: list[Tensor] = []
        for level_idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            h = self.down_attn[level_idx](h)
            skips.append(h)
            if level_idx < self.levels - 1:
                h = self.downsamples[level_idx](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for up_idx, blocks in enumerate(self.up_blocks):
            skip = skips[-(up_idx + 1)]
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            h = self.up_attn[up_idx](h)
            if up_idx < len(self.upsamples):
                h = self.upsamples[up_idx](h)

        y = self.out_proj(F.silu(self.out_norm(h))).transpose(1, 2)

        valid = slot_valid_mask.unsqueeze(-1).to(y.dtype)
        pin = slot_pin_mask.unsqueeze(-1).to(y.dtype)

        state_gate = 1.0 - pin
        action_gate = torch.zeros_like(slot_valid_mask, dtype=y.dtype)
        action_gate[:, :-1] = slot_valid_mask[:, 1:].to(y.dtype)

        y_state = y[..., : self.state_dim] * state_gate
        y_action = y[..., self.state_dim :] * action_gate.unsqueeze(-1)
        return torch.cat([y_state, y_action], dim=-1) * valid
