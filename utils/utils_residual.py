import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual module based on convolutional layers, with an optional global average pooling layer."""

    def __init__(self, in_channels, out_channels, stride=1, use_gap=False):
        super(ResidualBlock, self).__init__()
        self.use_gap = use_gap

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels))

        # Global average pooling layer
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_gap:
            out = self.gap(out)
            out = out.view(out.size(0), -1)
        return out
