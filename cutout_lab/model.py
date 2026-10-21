"""CIFAR ResNet-20 with original parameter-free option-A shortcuts.

Independently implemented from He et al., CVPR 2016, arXiv:1512.03385.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.stride = stride
        self.pad = (outgoing - incoming) // 2

    def forward(self, x):
        identity = x
        if self.stride != 1 or self.pad:
            identity = F.pad(x[:, :, ::self.stride, ::self.stride],
                             (0, 0, 0, 0, self.pad, self.pad))
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + identity)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        blocks = []
        incoming = 16
        for outgoing in (16, 32, 64):
            for index in range(3):
                stride = 2 if index == 0 and outgoing != incoming else 1
                blocks.append(Block(incoming, outgoing, stride))
                incoming = outgoing
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.blocks(F.relu(self.bn(self.conv(x))))
        return self.fc(x.mean((2, 3)))


def parameter_count():
    return sum(p.numel() for p in ResNet20().parameters())
