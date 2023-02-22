from torch import nn

from .net_60_base import Net60Base


class NetUpDownCoder4Conv60(Net60Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Conv60, self).__init__(
            reduce_cls=nn.Conv1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Avg60(Net60Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Avg60, self).__init__(
            reduce_cls=nn.AvgPool1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Max60(Net60Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Max60, self).__init__(
            reduce_cls=nn.MaxPool1d, numeric_derivative=numeric_derivative
        )
