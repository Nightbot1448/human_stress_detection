import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset

# data_field = "working_data"
# data_field = "working_data_filtered"


with open(
    os.path.abspath(
        os.path.join(
            __file__,
            "..",
            "..",
            "..",
            "data",
            "stress-recognition-in-automobile-drivers",
            "cleared_drivers_session_info.pkl",
        )
    ),
    "rb",
) as f:
    all_data = pickle.load(f)


def get_driver_train_test_items(
    input_data,
    step=15,
    window_size=30,
    unify_stress=True,
    test_percentage=0.2,
    data_field="working_data",
):
    stress_labeled_data = []
    for state_data in input_data:
        rr_list_corr = np.array(state_data.get(data_field).get("RR_list_cor"))
        count = ((rr_list_corr.shape[0] - window_size) // step) + 1
        iterable = (
            list(x * step + i for i in range(window_size))
            for x in range(count)
        )
        indices = np.fromiter(iterable, dtype=np.dtype((int, window_size)))
        stress_label = state_data.get("stress_label")
        if unify_stress:
            stress_label = int(bool(stress_label))
        stress_labeled_data.extend(
            zip(rr_list_corr.take(indices), [stress_label] * indices.shape[0])
        )
    train_count = int(len(stress_labeled_data) * (1 - test_percentage))
    return stress_labeled_data[:train_count], stress_labeled_data[train_count:]


class DriversBaseDataset(Dataset):
    def __init__(
        self,
        step,
        window_size,
        unify_stress,
        ds_type: str = "train",
        *args,
        **kwargs,
    ):
        super(DriversBaseDataset, self).__init__(*args, **kwargs)
        assert ds_type in ("train", "test")
        self.ds_type = ds_type
        self.step = step
        self.window_size = window_size
        self.unify_stress = unify_stress
        self._input_data = None
        self._train_items = None
        self._test_items = None
        self._items = None

    def _set_items(self):
        self._items = (
            self._train_items if self.ds_type == "train" else self._test_items
        )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        _data, _label = self._items[idx]
        return torch.tensor(_data), _label


class DriverDataset(DriversBaseDataset):
    def __init__(
        self,
        driver_data,
        step=15,
        window_size=30,
        unify_stress=True,
        test_percentage=0.2,
        ds_type: str = "train",
        data_field: str = "working_data",
        *args,
        **kwargs,
    ):
        super(DriverDataset, self).__init__(
            *args,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            unify_stress=unify_stress,
            **kwargs,
        )
        self._train_items, self._test_items = get_driver_train_test_items(
            driver_data,
            step,
            window_size,
            unify_stress,
            test_percentage,
            data_field=data_field,
        )
        self._set_items()


class DriversDataset(DriversBaseDataset):
    def __init__(
        self,
        drivers_data,
        step=15,
        window_size=30,
        unify_stress=True,
        test_percentage=0.2,
        ds_type: str = "train",
        data_field: str = "working_data",
        *args,
        **kwargs,
    ):
        super(DriversDataset, self).__init__(
            *args,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            unify_stress=unify_stress,
            **kwargs,
        )
        self._train_items = []
        self._test_items = []
        for driver_data in drivers_data.values():
            train_data, test_data = get_driver_train_test_items(
                driver_data,
                step,
                window_size,
                unify_stress,
                test_percentage,
                data_field=data_field,
            )
            self._train_items.extend(train_data)
            self._test_items.extend(test_data)
        self._set_items()
