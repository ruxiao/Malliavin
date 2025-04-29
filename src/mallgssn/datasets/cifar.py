import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

def build_cifar(loader_cfg):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    root = loader_cfg.get("root", "./data")
    train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=loader_cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=loader_cfg.batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader
