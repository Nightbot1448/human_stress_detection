from typing import Union, Type

from torch import nn


_RCType = Union[Type[nn.Conv1d], Type[nn.MaxPool1d], Type[nn.AvgPool1d]]


def get_reduce(_reduce_cls: _RCType, channels: int, **kwargs):
    if _reduce_cls == nn.Conv1d:
        return nn.Conv1d(channels, channels, **kwargs)
    return _reduce_cls(**kwargs)
