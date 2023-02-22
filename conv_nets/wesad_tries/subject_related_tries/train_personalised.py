#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

sys.path.extend(["/home/dmo/Documents/human_func_state/human_func_state"])

from conv_nets.wesad_tries.models_v2 import get_model
from conv_nets.wesad_tries.utils import (
    set_seed,
    val,
    save_data,
    train_epoch,
)
from ds.wesad.datasets import subjects_data
from ds.wesad.datasets_users import SubjectDataset


subjects_was_skipped_while_training = True
data_key = "rr_intervals"
numeric_derivative = True
agg_type = "conv"

epoch_count = 50
signal_len = 60
optim_lr = 1e-6
ds_step_size = 5
train_batch_size = 8
test_batch_size = 1
device = "cuda"
Model = get_model(signal_len, agg_type)
Optimizer = optim.ASGD
net_with_optim_name = f"{Model.__name__}_{Optimizer.__name__}_lr_{optim_lr}"

base_path = Path("/home/dmo/Documents/human_func_state/human_func_state")
wesad = base_path.joinpath("models_dumps", "wesad")


def get_out_and_common_base_paths(
    skip_user_on_train=False,
) -> tuple[Path, Path]:
    global net_with_optim_name
    return (
        wesad.joinpath(
            *[
                "subjects_related",
                *(["derivative"] if numeric_derivative else []),
                *(["skip_users"] if skip_user_on_train else []),
                net_with_optim_name,
            ]
        ),
        wesad.joinpath(
            *[
                "steps",
                *(["derivative"] if numeric_derivative else []),
                *(["skip_users"] if skip_user_on_train else []),
                net_with_optim_name,
            ]
        ),
    )


out_base_path, common_base_net_home = get_out_and_common_base_paths(
    subjects_was_skipped_while_training
)


def get_ds_and_dl(subject_id):
    ds_subj_train = SubjectDataset(
        subjects_data.get(subject_id),
        ds_type="train",
        window_size=signal_len,
        step=ds_step_size,
        key=data_key,
        numeric_derivative=numeric_derivative,
    )
    ds_subj_test = SubjectDataset(
        subjects_data.get(subject_id),
        ds_type="test",
        window_size=signal_len,
        step=ds_step_size,
        key=data_key,
        numeric_derivative=numeric_derivative,
    )
    return {
        "ds": {
            "train": ds_subj_train,
            "test": ds_subj_test,
        },
        "dl": {
            "train": DataLoader(
                ds_subj_train,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=True,
            ),
            "test": DataLoader(
                ds_subj_test,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=True,
            ),
        },
    }


def get_net_state_dict(subject_id: int | None = None):
    """
    Get net state dict. If `subject_id` specified then returns network
    state with skipped subject while training

    :param subject_id: skipped subject while training
    :return: Common trained
    """
    global net_with_optim_name
    net_home = common_base_net_home / net_with_optim_name
    if subject_id is not None:
        net_home = common_base_net_home.joinpath(
            f"subj_{str(subject_id).zfill(2)}_{net_with_optim_name}"
        )
    net_state_file = next(net_home.glob("*_best.pkl"))
    with open(net_state_file, "rb") as f:
        data = pickle.load(f)
        # print(data.keys())
        # print(f"class = {data.get('net_class')}")
        return data.get("net_state_dict")


def get_net_criterion_optimizer(
    net_state: dict | None = None,
) -> tuple[Model, nn.CrossEntropyLoss, optim.Optimizer]:
    global device
    net = Model(numeric_derivative=numeric_derivative).to(device=device)
    if net_state is not None:
        net.load_state_dict(net_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.ASGD(net.parameters())
    return net, criterion, optimizer


def get_writer_and_dump_names(subject_id):
    global net_with_optim_name
    global out_base_path
    subj_mod_name = f"subj_{subject_id}_{net_with_optim_name}"

    write_path = out_base_path / subj_mod_name
    writer = SummaryWriter(log_dir=str(write_path / "hist"))
    dump_name = (write_path / f"{subj_mod_name}_last").with_suffix(".pkl")
    best_name = (write_path / f"{subj_mod_name}_best").with_suffix(".pkl")

    return writer, dump_name, best_name


def train_user_related(
    subj_id,
    out_f,
    epoch_count=50,
    start_epoch=0,
):
    global subjects_was_skipped_while_training
    set_seed()
    ds_and_dl = get_ds_and_dl(subj_id)
    dl_train = ds_and_dl.get("dl").get("train")
    dl_test = ds_and_dl.get("dl").get("test")
    ds_train_len = len(ds_and_dl.get("ds").get("train"))
    ds_test_len = len(ds_and_dl.get("ds").get("test"))
    writer, dump_name, best_name = get_writer_and_dump_names(subj_id)
    min_loss = (torch.tensor(torch.inf), 0)
    best_rate = (torch.tensor(0.0), 0)
    worst_rate = (torch.tensor(1.0), 0)
    net_state = get_net_state_dict(
        subj_id if subjects_was_skipped_while_training else None
    )
    model, criterion, optimizer = get_net_criterion_optimizer(net_state)
    model.eval()
    print("Rate before: ", val(model, dl_test) / ds_test_len, file=out_f)
    for epoch in trange(start_epoch, start_epoch + epoch_count):
        model.train()
        epoch_loss, min_loss = train_epoch(
            model,
            criterion,
            optimizer,
            dl_train,
            epoch,
            min_loss,
            out_f,
            print_step=ds_train_len + 1,
        )
        mean_loss = epoch_loss / (ds_train_len / ds_step_size)
        if mean_loss < min_loss[0]:
            min_loss = (mean_loss, epoch)
        model.eval()
        common_acc = val(model, dl_test) / ds_test_len
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", common_acc, epoch)
        if common_acc > best_rate[0]:
            best_rate = (common_acc, epoch)
            save_data(
                best_name,
                Model,
                model.state_dict(),
                optimizer,
                epoch,
                common_acc,
                train_batch_size,
                test_batch_size,
            )
        if common_acc < worst_rate[0]:
            worst_rate = (common_acc, epoch)
        save_data(
            dump_name,
            Model,
            model.state_dict(),
            optimizer,
            epoch,
            common_acc,
            train_batch_size,
            test_batch_size,
        )

        print(
            f"[subject {subj_id:2d}, {epoch:3d}] "
            f"rate: {common_acc:.4f}; {best_rate = }, {worst_rate = }",
            file=out_f,
        )
    return worst_rate, best_rate, min_loss


if __name__ == "__main__":
    logs = out_base_path / "logs"
    logs.mkdir(exist_ok=True, parents=True)
    for subject_id in tqdm(subjects_data):
        with open(
            logs / f"subject_{str(subject_id).zfill(2)}.log", "w"
        ) as out:
            worst_r, best_r, min_loss = train_user_related(
                subject_id, out, epoch_count=epoch_count
            )
            print("Results:", file=out)
            print("Loss", min_loss, file=out)
            print("Worst rate", worst_r, file=out)
            print("Best rate", best_r, file=out)
