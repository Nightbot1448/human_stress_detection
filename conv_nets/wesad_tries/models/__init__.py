from .net_up_down_coder3_15 import NetUpDownCoder3_15
from .net_up_down_coder3_30 import NetUpDownCoder3_30
from .net_up_down_coder3_45 import NetUpDownCoder3_45
from .net_up_down_coder3_60 import NetUpDownCoder3_60
from .net_up_down_coder3_mp_15 import NetUpDownCoder3MP_15
from .net_up_down_coder3_mp_30 import NetUpDownCoder3MP_30
from .net_up_down_coder3_mp_45 import NetUpDownCoder3MP_45
from .net_up_down_coder3_mp_60 import NetUpDownCoder3MP_60
from .net_up_down_coder3_ap_15 import NetUpDownCoder3AP_15
from .net_up_down_coder3_ap_30 import NetUpDownCoder3AP_30
from .net_up_down_coder3_ap_45 import NetUpDownCoder3AP_45
from .net_up_down_coder3_ap_60 import NetUpDownCoder3AP_60

_allowed_lens = (15, 30, 45, 60)
_mapping_sig_len = {
    15: NetUpDownCoder3_15,
    30: NetUpDownCoder3_30,
    45: NetUpDownCoder3_45,
    60: NetUpDownCoder3_60,
}
_mapping_sig_len_avg = {
    15: NetUpDownCoder3AP_15,
    30: NetUpDownCoder3AP_30,
    45: NetUpDownCoder3AP_45,
    60: NetUpDownCoder3AP_60,
}
_mapping_sig_len_max = {
    15: NetUpDownCoder3MP_15,
    30: NetUpDownCoder3MP_30,
    45: NetUpDownCoder3MP_45,
    60: NetUpDownCoder3MP_60,
}


def get_model(singal_len=30, agg="conv"):
    if agg == "conv":
        return _mapping_sig_len[singal_len]
    if agg == "max":
        return _mapping_sig_len_max[singal_len]
    if agg == "avg":
        return _mapping_sig_len_avg[singal_len]
    raise ValueError("Unknown aggregation type")
