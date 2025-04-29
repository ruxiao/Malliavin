import torch, torch.nn as nn, torch.nn.functional as F
from .malliavin_block import MalliavinBlock


def positive_init_sigma(i, blocks, base=0.4, floor=5e-3):
    return max(floor, base * (i + 1) / blocks)

class MalliavinResNet(nn.Module):
    def __init__(self, blocks=6, dim=64, num_cls=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.blocks = nn.ModuleList([
            MalliavinBlock(dim, positive_init_sigma(i, blocks)) for i in range(blocks)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_cls)

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.size()
        z_shared = torch.randn(B, C, H, W, device=x.device)
        for blk in self.blocks:
            x = blk(x, z_shared)
        x = self.pool(x).flatten(1)
        return self.head(x)
