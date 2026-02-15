import torch.nn as nn
from paper import Paper, Segment, Vec2
import numpy as np
import matplotlib.pyplot as plt
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
    
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=1,
            padding="same",
            bias = False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()

        conv_kwargs = {
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": kernel,
            "stride": 1,
            "padding": "same",
            "bias" : False
        }

        self.conv1 = nn.Conv2d(**conv_kwargs)
        self.norm1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(**conv_kwargs)
        self.norm2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += x

        out = self.relu(out)

        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding="same"
        )
        
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.policy = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )


    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.policy(out)
        return out


    """def spatial_softmax(logits):
        B, C, H, W = logits.shape
        logits = logits.view(B, C, -1)
        probs = torch.softmax(logits, dim=2)
        return probs.view(B, C, H, W)"""


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, kernel: int):
        super().__init__()

    def forward(self, x):
        return NotImplementedError


class ThinkArchitecture(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel)
        
        # 5-9 ResBlocks
        self.res_tower = nn.ModuleList([
            ResBlock(out_channels, kernel) for _ in range(9)
        ])

        self.policy = PolicyHead(out_channels)

    def forward(self, x):
        out = self.conv(x)

        for res in self.res_tower:
            out = res(out)

        # add policy + value head...?
        out = self.policy(out)

        return out


if __name__ == "__main__":
    paper = Paper()
    seg = Segment(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
    fold = paper.fold(seg)

    img = paper.rasterize(128, 128, 0.0)
    img = np.array(img).reshape(128, 128)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    network = ThinkArchitecture(1, 128, 3)

    out = network.forward(img_tensor)
    out_np = out.detach().squeeze().numpy() 
    
    out1 = out_np[0] # Channel 0: Start Point Map
    out2 = out_np[1] # Channel 1: End Point Map

    # Plot
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(out1, cmap='Reds', origin='lower')
    ax[0].set_title("Start Point Policy")
    
    ax[1].imshow(out2, cmap='Blues', origin='lower')
    ax[1].set_title("End Point Policy")
    
    plt.show()
