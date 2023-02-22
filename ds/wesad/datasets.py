import pickle
from pathlib import Path
from itertools import islice

import numpy as np

import torch
from torch.utils.data import Dataset

from ds.wesad.labels import Labels  # noqa: F401

with open(
    Path(__file__).parent.parent.parent.joinpath(
        "data", "WESAD", "wesad_signal_and_normalised_rr_intervals.pkl"
    ),
    "rb",
) as f:
    subjects_data = pickle.load(f)


def _data_with_numeric_derivative(data):
    derivative = np.hstack(
        (np.zeros((data.shape[0], 1)), np.diff(data, axis=1))
    )
    return np.hstack((np.expand_dims(data, 1), np.expand_dims(derivative, 1)))


def get_subject_train_test_items(
    input_data,
    step=15,
    window_size=30,
    test_percentage=0.2,
    key="rr_intervals",
    numeric_derivative=False,
):
    train_data = []
    test_data = []
    for state_data in input_data:
        rr_list_corr = state_data.get(key)
        count = ((rr_list_corr.shape[0] - window_size) // step) + 1
        iterable = (
            list(x * step + i for i in range(window_size))
            for x in range(count)
        )
        indices = np.fromiter(iterable, dtype=np.dtype((int, window_size)))
        stress_label: Labels = state_data.get("label")
        stress_label = stress_label.value - 1
        data = rr_list_corr.take(indices)
        if numeric_derivative:
            data = _data_with_numeric_derivative(data)
        else:
            data = np.expand_dims(data, 1)
        labeled = zip(
            data, [stress_label] * indices.shape[0]
        )
        train_data.extend(
            islice(labeled, int(indices.shape[0] * (1 - test_percentage)))
        )
        test_data.extend(
            islice(labeled, int(indices.shape[0] * test_percentage))
        )
    return train_data, test_data


class SubjectsBaseDataset(Dataset):
    def __init__(
        self,
        items,
        step,
        window_size,
        ds_type: str = "train",
        test_percentage: float = 0.2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert ds_type in ("train", "test")
        self._items = items
        self.ds_type = ds_type
        self.step = step
        self.window_size = window_size
        self.test_percentage = test_percentage

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        _data, _label = self._items[idx]
        return torch.tensor(_data), _label


class SubjectDataset(SubjectsBaseDataset):
    def __init__(
        self,
        items,
        subject_id: int | None = None,
        step=15,
        window_size=30,
        test_percentage=0.2,
        ds_type: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            items=items,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            test_percentage=test_percentage,
            **kwargs,
        )
        self.subject_id = subject_id


class SubjectsDataset(SubjectsBaseDataset):
    def __init__(
        self,
        items,
        step=15,
        window_size=30,
        test_percentage=0.2,
        ds_type: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            items=items,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            test_percentage=test_percentage,
            **kwargs,
        )


def _get_data(
    subject_id: int | None,
    step=15,
    window_size=30,
    test_percentage=0.2,
    key="rr_intervals",
    numeric_derivative=False,
) -> tuple[list, list]:
    """
    :param subject_id: If None returns datasets for all subjects
    :param test_percentage:
    :param window_size:
    :param step:
    :return:
    """
    if subject_id is not None:
        return get_subject_train_test_items(
            subjects_data.get(subject_id),
            step,
            window_size,
            test_percentage,
            key=key,
            numeric_derivative=numeric_derivative
        )
    train_data, test_data = [], []
    for subject_data in subjects_data.values():
        _train_data, _test_data = get_subject_train_test_items(
            subject_data,
            step,
            window_size,
            test_percentage,
            key=key,
            numeric_derivative=numeric_derivative
        )
        train_data.extend(train_data)
        test_data.extend(test_data)
    return train_data, test_data


def get_dataset(
    ds_type: str,
    subject_id: int | None = None,
    step=15,
    window_size=30,
    test_percentage=0.2,
    key="rr_intervals",
    numeric_derivative=False,
):
    ds_train, ds_test = _get_data(
        subject_id,
        step,
        window_size,
        test_percentage,
        key=key,
        numeric_derivative=numeric_derivative,
    )
    kwargs = {
        "ds_type": ds_type,
        "step": step,
        "window_size": window_size,
        "test_percentage": test_percentage,
    }
    if subject_id is None:
        _Dataset = SubjectsDataset
    else:
        _Dataset = SubjectDataset
        kwargs["subject_id"] = subject_id
    kwargs["items"] = ds_train if ds_type == "train" else ds_test
    return _Dataset(**kwargs)


def get_train_test_dataset(
    subject_id: int | None = None,
    step=15,
    window_size=30,
    test_percentage=0.2,
    key="rr_intervals",
    numeric_derivative=False,
) -> tuple[SubjectsBaseDataset, SubjectsBaseDataset]:
    ds_train, ds_test = _get_data(
        subject_id,
        step,
        window_size,
        test_percentage,
        key=key,
        numeric_derivative=numeric_derivative
    )
    kwargs = {
        "step": step,
        "window_size": window_size,
        "test_percentage": test_percentage,
    }
    if subject_id is None:
        _Dataset = SubjectsDataset
    else:
        _Dataset = SubjectDataset
        kwargs["subject_id"] = subject_id
    return (
        _Dataset(items=ds_train, ds_type="train", **kwargs),
        _Dataset(items=ds_test, ds_type="test", **kwargs),
    )
