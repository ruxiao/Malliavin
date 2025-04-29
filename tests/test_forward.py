import torch
from mallgssn.models import MalliavinResNet

def test_forward():
    model = MalliavinResNet(blocks=2, dim=16, num_cls=10)
    x = torch.randn(4,3,32,32)
    y = model(x)
    assert y.shape == (4,10)
