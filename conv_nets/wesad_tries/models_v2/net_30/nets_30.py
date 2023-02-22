from torch import nn

from .net_30_base import Net30Base


class NetUpDownCoder4Conv30(Net30Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Conv30, self).__init__(
            reduce_cls=nn.Conv1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Avg30(Net30Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Avg30, self).__init__(
            reduce_cls=nn.AvgPool1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Max30(Net30Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Max30, self).__init__(
            reduce_cls=nn.MaxPool1d, numeric_derivative=numeric_derivative
        )
