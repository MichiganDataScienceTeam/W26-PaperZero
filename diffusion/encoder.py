import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int=8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False) 
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
    









class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)




class AutoEncoder(nn.Module):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=2, 
                out_channels=16, 
                kernel_size=(3, 3), 
                padding='same', 
                bias=False
            ),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            ResidualBlock(16, 32),
            Downsample(32),
            ResidualBlock(32, 64),
            Downsample(64),
            ResidualBlock(64, 64),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )
        base_ch = 32

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=2, base_ch=32, kernel_size=(3, 3), padding='same', bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_ch),
            nn.SiLU(),
            ResidualBlock(base_ch * 2, base_ch * 2),
            Upsample(base_ch * 2),
            ResidualBlock(base_ch * 2, base_ch),
            ResidualBlock(base_ch * 4, base_ch * 4),
            Upsample(base_ch * 4),
            ResidualBlock(base_ch * 4, base_ch * 2),
            ResidualBlock(self.enc_feat_ch, self.enc_feat_ch),
            Upsample(self.enc_feat_ch),
            ResidualBlock(self.enc_feat_ch, base_ch * 4),
            nn.Conv2d(base_ch, in_channels=2, kernel_size=(3, 3), padding='same'),
            nn.Sigmoid()
        )

class VAE(AutoEncoder):
    def __init__(self, in_channels: int = 2, img_size: int = 128, latent_dim: int = 128, base_ch: int = 32):
        super().__init__(in_channels=in_channels, img_size=img_size, latent_dim=latent_dim, base_ch=base_ch)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        h = super().encode(x)     # [B, latent_dim]
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = super().decode(z) 
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return super().decode(z)














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
        