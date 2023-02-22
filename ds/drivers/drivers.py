import os
import pickle
from math import ceil
from copy import deepcopy

import wfdb
import numpy as np
import heartpy as hp
from scipy.signal import savgol_filter

savgol_filter_params = (13, 9)
excluded_ranges = {
    5: [(29000, 34000), (40200, 40300)],
    7: [(82400, 82442)],
    8: [(75400, 75509)],
    9: [(65800, 65912)],
    10: [
        (25400, 25520),
        (33550, 34200),
        (50100, 50300),
    ],
}

# 0 - no stress
# 1 - medium stress
# 2 - high stress
action_stress_level_mapping = {
    "Initial Rest": 0,
    "City 1": 2,
    "Highway 1": 1,
    "City 2": 2,
    "Highway 2": 1,
    "City 3": 2,
    "Final Rest": 0,
}

# noqa: E501
drivers_sessions_info = """
Driver id,Initial Rest,City 1,Highway 1,City 2,Highway 2,City 3,Final Rest,Total duration
Driver05,15.13,16.00,7.74,6.06,7.56,14.96,15.75,83.23
Driver06,15.05,14.49,7.32,6.53,7.64,12.29,15.05,78.38
Driver07,15.04,16.23,10.96,9.83,7.64,10.15,15.03,84.87
Driver08,15.00,12.31,7.23,9.51,7.64,13.43,15.07,80.19
Driver09,15.66,19.21,8.47,5.20,7.06,13.21,0.00,68.82
Driver10,15.04,15.30,8.66,5.27,7.04,12.06,14.79,78.15
Driver11,15.02,15.81,7.43,7.15,6.96,11.72,14.99,79.08
Driver12,15.01,13.41,7.56,6.50,8.06,11.68,15.01,77.23
Driver15,15.00,12.54,7.24,5.99,6.82,12.12,15.00,74.70
Driver16,15.01,16.12,7.14,5.12,6.81,13.91,0.00,64.10
"""

drivers_sessions_info = list(
    map(
        lambda x: x.split(","),
        filter(bool, drivers_sessions_info.splitlines()),
    )
)
titles = drivers_sessions_info[0][1:]
drivers_sessions_info = drivers_sessions_info[1:]
drivers_sessions_info = {
    int(el[0][6:]): list(map(float, el[1:])) for el in drivers_sessions_info
}
drivers_sessions_info = {
    key: dict(zip(titles, val)) for key, val in drivers_sessions_info.items()
}

records = {
    u: wfdb.rdrecord(
        os.path.abspath(
            os.path.join(
                "..",
                "data",
                "stress-recognition-in-automobile-drivers",
                "data",
                f"drive{str(u).zfill(2)}",
            )
        )
    )
    for u in drivers_sessions_info.keys()
}
requested_fields = ("fs", "sig_len", "sig_name", "p_signal")
records = {
    u_id: {
        v_name: v_value
        for v_name, v_value in val.__dict__.items()
        if v_name in requested_fields
    }
    for u_id, val in records.items()
}

time_shift_for_ranges = {
    driver_id: ceil(
        (
            records.get(driver_id).get("sig_len")
            - drivers_sessions_info.get(driver_id).get("Total duration")
            * 15.5
            * 60
        )
        / 2
    )
    for driver_id in records
}


def get_indices_of_subtasks(driver_id):
    indices_of_subtasks = np.ceil(
        np.fromiter(drivers_sessions_info.get(driver_id).values(), dtype=float)
        * 15.5
        * 60
    ).astype(int)
    indices_of_subtasks[-1] = 0
    indices_of_subtasks = np.cumsum(np.roll(indices_of_subtasks, 1))
    delta = time_shift_for_ranges.get(driver_id)
    ranges = np.vstack((indices_of_subtasks[:-1], indices_of_subtasks[1:])).T
    # don't add if task duration == 0
    add_mask = np.ravel(np.diff(ranges, axis=1).astype(bool))
    ranges[add_mask] += np.array([1, -1]) * delta
    return ranges


drivers_sessions_stress_ranges = {
    driver_id: dict(
        zip(
            map(tuple, get_indices_of_subtasks(driver_id)),
            map(lambda x: action_stress_level_mapping.get(x), sessions.keys()),
        )
    )
    for driver_id, sessions in drivers_sessions_info.items()
}


# remove strange signals parts
cleared_drivers_sessions_stress_ranges = deepcopy(
    drivers_sessions_stress_ranges
)
for driver_id, driver_excluded_ranges in excluded_ranges.items():
    driver_sessions = drivers_sessions_stress_ranges.get(driver_id)
    new_driver_sessions = deepcopy(driver_sessions)
    for ex_left_bound, ex_right_bound in driver_excluded_ranges:
        for (l, r), v in driver_sessions.items():
            if l >= ex_left_bound and r < ex_right_bound:
                # зона полностью входит в исключаемый диапазон
                new_driver_sessions.pop((l, r), None)
                continue
            if l <= ex_left_bound < r:
                new_driver_sessions.pop((l, r), None)
                if l < ex_left_bound:
                    new_driver_sessions.update({(l, ex_left_bound): v})
            if l <= ex_right_bound < r:
                new_driver_sessions.pop((l, r), None)
                if ex_right_bound < r:
                    new_driver_sessions.update({(ex_right_bound, r): v})
    cleared_drivers_sessions_stress_ranges.update(
        {driver_id: new_driver_sessions}
    )


# Initial Rest,City 1,Highway 1,City 2,Highway 2,City 3,Final Rest
result_dict = {}
for driver_id, record in records.items():
    print("Driver", driver_id)
    driver_list = []
    for range_, stress_label in cleared_drivers_sessions_stress_ranges.get(
        driver_id
    ).items():
        if range_[1] - range_[0]:
            try:
                # noqa: E203
                ecg_signal = record.get("p_signal")[range_[0] : range_[1], 0]
                filtered = savgol_filter(ecg_signal, *savgol_filter_params)
                print(range_, len(ecg_signal), len(filtered))
                fs = record.get("fs")
                wd, m = hp.process(ecg_signal, fs)
                wd_f, m_f = hp.process(filtered, fs)
                driver_list.append(
                    {
                        "stress_label": stress_label,
                        "ECG_signal": ecg_signal,
                        "original_range": range_,
                        "fs": fs,
                        "working_data": wd,
                        "measurements": m,
                        "working_data_filtered": wd_f,
                        "measurements_filtered": m_f,
                    }
                )
            except Exception:
                print(range_, "failed to be proccesed")
    result_dict.update({driver_id: driver_list})

with open(
    (
        "../../data/stress-recognition-in-automobile-drivers/"
        "cleared_drivers_session_info.pkl"
    ),
    "wb",
) as f:
    pickle.dump(result_dict, f)
