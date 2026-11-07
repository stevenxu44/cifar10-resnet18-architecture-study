"""CIFAR-sized ResNet-18 with optional depthwise convolutions and SE blocks."""
import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1, depthwise=False):
    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 depthwise=False, se=False):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, depthwise)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1, depthwise)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels) if se else nn.Identity()
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride,
                                    bias=False), nn.BatchNorm2d(out_channels))
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + identity)


class CIFARResNet18(nn.Module):
    def __init__(self, depthwise=False, se=False, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        channels = [64, 128, 256, 512]
        stages = []
        prev = 64
        for stage, current in enumerate(channels):
            blocks = []
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(BasicBlock(prev, current, stride, depthwise, se))
                prev = current
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def build_model(variant):
    configs = {
        "baseline": (False, False),
        "depthwise": (True, False),
        "se": (False, True),
        "depthwise_se": (True, True),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")
    return CIFARResNet18(*configs[variant])
