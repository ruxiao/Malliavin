import torch.optim as optim

def build_optimizer(model, opt_cfg):
    wd_params, no_wd_params = [], []
    for n, p in model.named_parameters():
        (no_wd_params if "log_sigma" in n else wd_params).append(p)
    if opt_cfg.name == "sam_adamw":
        from torch_optimizer import SAM  # optional third‑party
        base_opt = optim.AdamW([
            {"params": wd_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ], lr=opt_cfg.lr, betas=(0.9, 0.99))
        return SAM(base_opt, rho=opt_cfg.rho)
    return optim.AdamW([
        {"params": wd_params, "weight_decay": opt_cfg.weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ], lr=opt_cfg.lr)
