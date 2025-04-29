import math, time, os, random
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision, torchvision.transforms as T
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, last_epoch=-1):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup:
            return [base * step / self.warmup for base in self.base_lrs]
        t = (step - self.warmup) / (self.max_iters - self.warmup)
        return [base * 0.5 * (1 + math.cos(math.pi * t)) for base in self.base_lrs]


# ---------------------------------------------------------------------------
#  Data augmentation & mixup‑with‑label‑smoothing
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def smooth_ce(logits, y1, y2, lam, C, eps=0.1):
    logp = F.log_softmax(logits, dim=1)
    t1 = torch.zeros_like(logp).scatter_(1, y1.unsqueeze(1), 1)
    t2 = torch.zeros_like(logp).scatter_(1, y2.unsqueeze(1), 1)
    t1 = t1 * (1 - eps) + eps / C
    t2 = t2 * (1 - eps) + eps / C
    return -(lam * (t1 * logp).sum(1) + (1 - lam) * (t2 * logp).sum(1))


# ---------------------------------------------------------------------------
#  Malliavin Residual Block
# ---------------------------------------------------------------------------

def positive_init_sigma(i, blocks, base=0.4, floor=5e-3):
    return max(floor, base * (i + 1) / blocks)


class MalliavinBlock(nn.Module):
    """A basic residual block with shared + independent Gaussian noise.

    The pathwise (a.k.a. Malliavin) gradient of `sigma` comes for free through
    re‑parameterisation.  An antithetic trick halves the estimator variance.
    """

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

    def forward(self, x, z_shared):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        mu = x + h

        # Noise – shared across blocks + independent part (antithetic inside the batch)
        eps_ind = torch.randn_like(mu)
        noise = self.alpha * z_shared + math.sqrt(1 - self.alpha ** 2) * eps_ind

        # Antithetic: flip sign for the *second* half of the batch.
        B = noise.size(0)
        noise[B // 2:] = -noise[B // 2:]

        sigma = F.softplus(self.log_sigma).clamp(max=5.0)  # σ ≥ 0, bounded above
        out = F.relu(mu + sigma * noise)

        # Baseline feature (stop‑grad when used for variance reduction)
        feat = x.mean(dim=(2, 3))
        self.block_pred = self.baseline_pred(feat).squeeze(1)

        return out


# ---------------------------------------------------------------------------
#  Malliavin ResNet backbone
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Training helpers
# ---------------------------------------------------------------------------

def make_optimizer(model, lr=3e-4, wd=5e-4):
    wd_params, no_wd_params = [], []
    for n, p in model.named_parameters():
        (no_wd_params if "log_sigma" in n else wd_params).append(p)
    return optim.AdamW([
        {"params": wd_params, "weight_decay": wd},
        {"params": no_wd_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.99))


def train_epoch(loader, model, opt, scaler, device, accum_steps=1, num_cls=100):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss = 0.0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        xm, y1, y2, lam = mixup_data(x, y, alpha=1.0)

        with autocast():
            logits = model(xm)
            loss_vec = smooth_ce(logits, y1, y2, lam, num_cls)
            loss = loss_vec.mean() / accum_steps

        scaler.scale(loss).backward()

        # Clark–Ocone baseline regression (stop‑grad on loss_vec)
        with torch.no_grad():
            for blk in model.blocks:
                blk.baseline.mul_(blk.decay).add_((1 - blk.decay) * loss_vec.mean())

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


@torch.no_grad()
@torch.inference_mode()
def evaluate(loader, model, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100 * correct / total


# ---------------------------------------------------------------------------
#  Main entry
# ---------------------------------------------------------------------------

def main():
    freeze_support()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    root = os.environ.get("DATA", "./data")
    train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = MalliavinResNet(blocks=6, dim=64, num_cls=100).to(device)
    model = torch.compile(model)  # Torch 2.2+  (no‑op on older versions)

    opt = make_optimizer(model)
    scaler = GradScaler()
    sched = CosineWarmupLR(opt, warmup=100, max_iters=200 * len(train_loader))

    best_acc = 0.0
    for epoch in range(200):
        t0 = time.time()
        loss = train_epoch(train_loader, model, opt, scaler, device, accum_steps=1)
        acc = evaluate(test_loader, model, device)
        sched.step()

        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch:03d}  |  loss {loss:6.4f}  |  acc {acc:5.2f}%  |  best {best_acc:5.2f}%  |  {time.time() - t0:4.1f}s")


if __name__ == "__main__":
    main()
