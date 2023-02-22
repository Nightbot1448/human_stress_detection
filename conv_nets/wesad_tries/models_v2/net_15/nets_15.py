from torch import nn

from .net_15_base import Net15Base


class NetUpDownCoder4Conv15(Net15Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Conv15, self).__init__(
            reduce_cls=nn.Conv1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Avg15(Net15Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Avg15, self).__init__(
            reduce_cls=nn.AvgPool1d, numeric_derivative=numeric_derivative
        )


class NetUpDownCoder4Max15(Net15Base):
    def __init__(self, numeric_derivative=False):
        super(NetUpDownCoder4Max15, self).__init__(
            reduce_cls=nn.MaxPool1d, numeric_derivative=numeric_derivative
        )
