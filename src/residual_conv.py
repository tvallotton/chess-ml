import torch
from torch import nn


class ResidualConv(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )
        self.relu = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm((8, 8))

    def forward(self, x: torch.Tensor):
        y = self.relu(self.conv(x))
        return self.layer_norm(y + x)
