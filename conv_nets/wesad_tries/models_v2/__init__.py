from .net_15 import (
    NetUpDownCoder4Avg15,
    NetUpDownCoder4Max15,
    NetUpDownCoder4Conv15,
)
from .net_30 import (
    NetUpDownCoder4Avg30,
    NetUpDownCoder4Max30,
    NetUpDownCoder4Conv30,
)
from .net_45 import (
    NetUpDownCoder4Avg45,
    NetUpDownCoder4Max45,
    NetUpDownCoder4Conv45,
)
from .net_60 import (
    NetUpDownCoder4Avg60,
    NetUpDownCoder4Max60,
    NetUpDownCoder4Conv60,
)

_allowed_lens = (15, 30, 45, 60)

_mapping = {
    "conv": {
        15: NetUpDownCoder4Conv15,
        30: NetUpDownCoder4Conv30,
        45: NetUpDownCoder4Conv45,
        60: NetUpDownCoder4Conv60,
    },
    "avg": {
        15: NetUpDownCoder4Avg15,
        30: NetUpDownCoder4Avg30,
        45: NetUpDownCoder4Avg45,
        60: NetUpDownCoder4Avg60,
    },
    "max": {
        15: NetUpDownCoder4Max15,
        30: NetUpDownCoder4Max30,
        45: NetUpDownCoder4Max45,
        60: NetUpDownCoder4Max60,
    },
}


def get_model(signal_len=30, agg="conv"):
    if agg not in _mapping.keys() or signal_len not in _allowed_lens:
        raise ValueError("Unsupported aggregation type or signal len")
    return _mapping.get(agg).get(signal_len)
