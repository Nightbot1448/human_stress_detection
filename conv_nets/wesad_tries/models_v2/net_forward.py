import torch
from torch import nn


class NetBaseForward(nn.Module):
    def __init__(self):
        super(NetBaseForward, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.seq(x)
        x = torch.flatten(x, 1)
        return self.softmax(x)
