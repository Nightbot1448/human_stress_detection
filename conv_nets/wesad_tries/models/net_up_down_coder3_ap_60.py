import torch
import torch.nn as nn

from .modules.conv_x import ConvX


class NetUpDownCoder3AP_60(nn.Module):
    def __init__(self, numeric_derivative: bool = False):
        super(NetUpDownCoder3AP_60, self).__init__()
        in_channels = 2 if numeric_derivative else 1
        self.seq = torch.nn.Sequential(
            ConvX(in_channels, 4, kernel=3),
            ConvX(4, 8, kernel=3),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvX(8, 16, kernel=3),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvX(16, 32, kernel=3),
            ConvX(32, 16, kernel=3),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvX(16, 8, kernel=3),
            nn.AvgPool1d(kernel_size=1, stride=2),
            ConvX(8, 4, kernel=3),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvX(4, 2, kernel=3),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.seq(x)
        x = torch.flatten(x, 1)
        return self.softmax(x)
