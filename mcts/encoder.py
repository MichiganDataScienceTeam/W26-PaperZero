import torch.nn as nn
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
            bias=False,
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


class ValueHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_features=1),
        )

    def forward(self, x):
        return self.network(x)


class ThinkArchitecture(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, num_res_blocks: int = 9):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel)
        self.res_tower = nn.ModuleList(
            [ResBlock(out_channels, kernel) for _ in range(num_res_blocks)]
        )

        self.policy = PolicyHead(out_channels)
        self.value = ValueHead(out_channels)

    def forward(self, x):
        out = self.conv(x)

        for res in self.res_tower:
            out = res(out)

        policy = self.policy(out)
        value = self.value(out)

        return (policy, value)
