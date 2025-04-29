import math, torch.optim as optim

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
