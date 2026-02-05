import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        return x


# class DiffusionEncoder(nn.module):
#     def __init__(self, classes_num: int, in_channels: int = 1, image_size: tuple = (128, 128)):
#         #declare some residual blocks
#         self.conv = nn.Conv2d(
#             in_channels=in_channels, 
#             out_channels=16, 
#             kernel_size=(3, 3), 
#             padding='same', 
#             bias=False
#         )
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU()
        
#         self.res_blocks = nn.Sequential(
#             ResidualBlock(16, 16),
#             ResidualBlock(16, 16),
#             ResidualBlock(16, 32),
#             ResidualBlock(32, 32),
#             ResidualBlock(32, 32),
#             ResidualBlock(32, 64),
#             ResidualBlock(64, 64),
#             ResidualBlock(64, 64)
#         )
        
#     def forward():
        