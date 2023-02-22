from torch import nn

from conv_nets.wesad_tries.models.modules.conv_x import ConvX
from conv_nets.wesad_tries.models_v2.get_reduce import get_reduce
from conv_nets.wesad_tries.models_v2.net_forward import NetBaseForward


class Net30Base(NetBaseForward):
    def __init__(self, reduce_cls=nn.Conv1d, numeric_derivative=False):
        super(Net30Base, self).__init__()
        in_channels = 2 if numeric_derivative else 1
        self.seq = nn.Sequential(
            ConvX(in_channels, 4, kernel=3),
            ConvX(4, 8, kernel=3),
            get_reduce(reduce_cls, 8, kernel_size=2, stride=2),
            ConvX(8, 16, kernel=3),
            ConvX(16, 32, kernel=3),
            get_reduce(reduce_cls, 32, kernel_size=2, stride=2),
            ConvX(32, 16, kernel=3),
            ConvX(16, 8, kernel=3),
            get_reduce(reduce_cls, 8, kernel_size=2, stride=2),
            ConvX(8, 4, kernel=3),
            ConvX(4, 2, kernel=3),
            nn.Conv1d(2, 2, kernel_size=3, stride=1),
        )
