import torch
from torch.utils.data import Dataset

from ds.wesad.datasets import get_subject_train_test_items


class SubjectsBaseDataset(Dataset):
    def __init__(
        self, step, window_size, ds_type: str = "train", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert ds_type in ("train", "test")
        self.ds_type = ds_type
        self.step = step
        self.window_size = window_size
        self._input_data = None
        self._train_items = None
        self._test_items = None
        self._items = None

    def _set_items(self):
        if self.ds_type == "train":
            self._items = self._train_items
            del self._test_items
            return
        self._items = self._test_items
        del self._train_items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        _data, _label = self._items[idx]
        return torch.tensor(_data), _label


class SubjectDataset(SubjectsBaseDataset):
    def __init__(
        self,
        subject_data,
        step=15,
        window_size=30,
        test_percentage: float = 0.2,
        key="rr_intervals",
        ds_type: str = "train",
        numeric_derivative=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            **kwargs,
        )
        self._train_items, self._test_items = get_subject_train_test_items(
            subject_data,
            step,
            window_size,
            test_percentage,
            key=key,
            numeric_derivative=numeric_derivative,
        )
        self._set_items()


class SubjectsDataset(SubjectsBaseDataset):
    def __init__(
        self,
        subjects_data,
        step=15,
        window_size=30,
        test_percentage=0.2,
        skip_users: list[int] | None = None,
        ds_type: str = "train",
        key="rr_intervals",
        numeric_derivative=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            ds_type=ds_type,
            step=step,
            window_size=window_size,
            **kwargs,
        )
        if skip_users is None:
            skip_users = []
        self._train_items = []
        self._test_items = []
        for subject_id, subject_data in subjects_data.items():
            if subject_id in skip_users:
                continue
            train_items, test_items = get_subject_train_test_items(
                subject_data,
                step,
                window_size,
                test_percentage,
                key=key,
                numeric_derivative=numeric_derivative,
            )
            self._train_items.extend(train_items)
            self._test_items.extend(test_items)
        self._set_items()
