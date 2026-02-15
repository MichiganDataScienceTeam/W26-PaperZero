import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False) # bias parame keep?
        self.gn2 = nn.GroupNorm(groups, out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu(x)
        
        x = self.conv2(x)
        x = self.gn2(x)

        x += identity
        x = self.silu(x)
        return x