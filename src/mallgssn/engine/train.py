import torch, time
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf
from .schedulers import CosineWarmupLR
from .optim_factory import build_optimizer
from ..models import MalliavinResNet
from ..datasets.cifar import build_cifar


def train(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MalliavinResNet(**cfg.model).to(device)
    opt = build_optimizer(model, cfg.optim)
    scaler = GradScaler(enabled=cfg.train.amp)
    train_loader, test_loader = build_cifar(cfg.train)
    sched = CosineWarmupLR(opt, warmup=100, max_iters=cfg.train.epochs * len(train_loader))

    for epoch in range(cfg.train.epochs):
        model.train(); t0 = time.time(); total = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            with autocast(enabled=cfg.train.amp):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            total += loss.item()
        sched.step()
        acc = evaluate(model, test_loader, device)
        print(f"{epoch:03d}  loss {total/len(train_loader):.4f} acc {acc:.2f}%  {time.time()-t0:.1f}s")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); corr=tot=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        corr += (preds==y).sum().item(); tot+=y.size(0)
    return 100*corr/tot

if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser(); p.add_argument("--cfg", default="configs/cifar100_small.yaml")
    args = p.parse_args()
    train(args.cfg)
