import sys
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.extend(["/home/dmo/Documents/human_func_state/human_func_state/"])


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def val(model, dl, device="cuda"):
    with torch.no_grad():
        sum_ = 0
        for i, data in enumerate(dl):
            inputs, labels = data
            inputs = inputs.to(torch.float32).to(device=device)

            outputs = model(inputs).cpu()
            eq = outputs.max(1).indices == labels
            sum_ += eq.sum()
    return sum_


def common_validate(
    model,
    dl_val,
    len_ds_val,
    skip_user,
    subjects_test_datasets,
    each_user_rate_history,
):
    subject_accuracies = None
    if each_user_rate_history:
        subject_accuracies = {}
        for subject_id, subject_ds_test in subjects_test_datasets.items():
            if subject_id == skip_user:
                continue
            subject_dl_test = DataLoader(
                subject_ds_test,
                batch_size=1,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=True,
            )
            accuracy = val(model, subject_dl_test)
            subject_accuracies[subject_id] = accuracy / len(subject_ds_test)
    common_accuracy = val(model, dl_val) / len_ds_val
    return common_accuracy, subject_accuracies


def save_data(
    path,
    net_class,
    net_state_dict,
    optimizer,
    epoch,
    rate,
    train_batch_size,
    test_batch_size,
    rate_subject=None,
    device="cuda",
):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "net_state_dict": net_state_dict,
                "net_class": net_class,
                "current_epoch": epoch,
                "rate": rate,
                "rate_subject": rate_subject,
                "optimizer": optimizer.__class__.__name__,
                "optimizer_params": optimizer.param_groups,
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                "device": device,
            },
            f,
        )


def train_epoch(
    model,
    criterion,
    optimizer,
    dl,
    epoch,
    min_loss,
    out_f,
    print_step=50,
    device="cuda",
):
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(dl):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(torch.float32).to(device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.cpu(), labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % print_step == print_step - 1:
            mean_loss = running_loss / print_step
            if mean_loss < min_loss[0]:
                min_loss = (mean_loss, (epoch, i))
            print(f"[{epoch:3d}, {i:4d}] loss: {mean_loss:.3f}", file=out_f)
            epoch_loss += running_loss
            running_loss = 0.0
    epoch_loss += running_loss
    return epoch_loss, min_loss
