from torch import nn

from .net_45_base import Net45Base


class NetUpDownCoder4Conv45(Net45Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Conv45, self).__init__(
            reduce_cls=nn.Conv1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Avg45(Net45Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Avg45, self).__init__(
            reduce_cls=nn.AvgPool1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Max45(Net45Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Max45, self).__init__(
            reduce_cls=nn.MaxPool1d, numeric_derivative=numeric_derivative
        )
