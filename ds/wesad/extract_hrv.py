#!/usr/bin/env python
# coding: utf-8
import pickle
from pathlib import Path

import numpy as np
import heartpy as hp

from tqdm import tqdm

from ds.wesad.labels import Labels

rate = 700
wesad = Path(__file__).parent.parent.parent / "data" / "WESAD"


def state_has_single_interval(labels: np.ndarray):
    indices = np.where(labels == Labels.BASELINE)
    single_baseline = (indices[0][-1] + 1 - indices[0][0]) == (
        labels == Labels.BASELINE
    ).sum()
    indices = np.where(labels == Labels.STRESS)
    single_stress = (indices[0][-1] + 1 - indices[0][0]) == (
        labels == Labels.STRESS
    ).sum()
    return single_baseline and single_stress


def subject_ecg_ranges(ecg, ecg_labels, requested_labels):
    subject_data = []
    for label in requested_labels:
        signal = ecg[ecg_labels == label]
        wd, m = hp.process(signal, rate)
        subject_data.append(
            {
                "label": label,
                "signal": signal,
                "working_data": wd,
                "measurements": m,
            }
        )
    return subject_data


def get_baseline_and_stress(wesad_path):
    subjects_info = {}
    for subject in tqdm(wesad_path.glob("S*")):
        path = (subject / subject.name).with_suffix(".pkl")
        with open(path, "rb") as f:
            subject_id = int(subject.name[1:])
            data = pickle.load(f, encoding="latin1")
            # проверяем, что только по одному промежутку
            # в состоянии покоя (начальном) и стресса
            if not state_has_single_interval(data.get("label")):
                print(
                    f"Subject {subject_id} has more than "
                    "1 period of baseline and stress"
                )
            subjects_info[subject_id] = subject_ecg_ranges(
                data.get("signal").get("chest").get("ECG").squeeze(),
                data.get("label"),
                [Labels.BASELINE, Labels.STRESS],
            )
    return subjects_info


if __name__ == "__main__":
    subjects_baseline_and_stress = get_baseline_and_stress(wesad)
    with open(wesad / "wesad_computed.pkl", "wb") as out:
        pickle.dump(subjects_baseline_and_stress, out)
