import torch.nn as nn


class ConvX(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel=3, stride=1, padding=None
    ):
        super(ConvX, self).__init__()
        padding = kernel // 2 if padding is None else padding
        self.conv = nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
