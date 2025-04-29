import math, torch, torch.nn as nn, torch.nn.functional as F

class MalliavinBlock(nn.Module):
    """Residual block with learnable σ and pathwise Malliavin gradient."""
    def __init__(self, dim: int, init_sigma: float, alpha: float = 0.5, decay: float = 0.99):
        super().__init__()
        self.alpha, self.decay = alpha, decay
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))
        self.register_buffer("baseline", torch.zeros(1))
        self.baseline_pred = nn.Linear(dim, 1)

    def forward(self, x, z_shared, antithetic: bool = True):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        mu = x + h
        eps_ind = torch.randn_like(mu)
        noise = self.alpha * z_shared + math.sqrt(1 - self.alpha ** 2) * eps_ind
        if antithetic:
            B = noise.size(0)
            noise[B // 2:] = -noise[B // 2:]
        sigma = F.softplus(self.log_sigma).clamp(max=5.0)
        out = F.relu(mu + sigma * noise)
        feat = x.mean(dim=(2, 3))
        self.block_pred = self.baseline_pred(feat).squeeze(1)
        return out
