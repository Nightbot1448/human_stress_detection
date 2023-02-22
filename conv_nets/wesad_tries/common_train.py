import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


sys.path.extend(["/home/dmo/Documents/human_func_state/human_func_state/"])


from conv_nets.wesad_tries.models_v2 import get_model
from conv_nets.wesad_tries.utils import (
    set_seed,
    common_validate,
    save_data,
    train_epoch,
)

from ds.wesad.datasets import subjects_data
from ds.wesad.datasets_users import SubjectDataset, SubjectsDataset


data_key = "rr_intervals"
numeric_derivative = False
agg_type = "conv"

epoch_count = 50
signal_len = 30
optim_lr = 1e-6
each_user_rate_history = True
ds_step_size = 5
train_batch_size = 8
test_batch_size = 1
device = "cuda"
Model = get_model(signal_len, agg_type)
Optimizer = optim.ASGD

base_path = Path("/home/dmo/Documents/human_func_state/human_func_state")
base_path = base_path.joinpath(
    "models_dumps",
    "wesad",
    "steps",
    *(["derivative"] if numeric_derivative else []),
)


net_with_optim_name = f"{Model.__name__}_{Optimizer.__name__}_lr_{optim_lr}"
base_path = base_path / net_with_optim_name


subjects_test_datasets = {
    subject_id: SubjectDataset(
        subject_data,
        ds_type="test",
        window_size=signal_len,
        step=ds_step_size,
        key=data_key,
        numeric_derivative=numeric_derivative,
    )
    for subject_id, subject_data in subjects_data.items()
}


def get_ds_and_dl(
    window_size=30,
    key="rr_intervals",
    numeric_derivative=False,
):
    ds_all_train = SubjectsDataset(
        subjects_data,
        ds_type="train",
        window_size=window_size,
        step=ds_step_size,
        key=key,
        numeric_derivative=numeric_derivative,
    )
    ds_all_test = SubjectsDataset(
        subjects_data,
        ds_type="test",
        window_size=window_size,
        step=ds_step_size,
        key=key,
        numeric_derivative=numeric_derivative,
    )
    return {
        "ds": {
            "train": ds_all_train,
            "test": ds_all_test,
        },
        "dl": {
            "train": DataLoader(
                ds_all_train,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=True,
            ),
            "test": DataLoader(
                ds_all_test,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=True,
            ),
        },
    }


def get_writer_and_dump_names():
    global net_with_optim_name
    write_path = base_path / net_with_optim_name
    writer = SummaryWriter(log_dir=write_path / "hist")
    dump_name = (write_path / f"{net_with_optim_name}_last").with_suffix(
        ".pkl"
    )
    best_name = (write_path / f"{net_with_optim_name}_best").with_suffix(
        ".pkl"
    )
    return writer, dump_name, best_name


def get_net_criterion_optimizer(
    device="cuda",
) -> tuple[Model, nn.CrossEntropyLoss, optim.Optimizer]:
    net = Model(numeric_derivative=numeric_derivative).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.ASGD(net.parameters())
    return net, criterion, optimizer


def train(
    out_f,
    epoch_count=50,
    print_step=50,
    start_epoch=0,
    window_size=30,
    key="rr_intervals",
    numeric_derivative=False,
):
    global each_user_rate_history
    global subjects_test_datasets
    set_seed()
    ds_and_dl = get_ds_and_dl(
        window_size=window_size, key=key, numeric_derivative=numeric_derivative
    )
    dl_train = ds_and_dl.get("dl").get("train")
    dl_test = ds_and_dl.get("dl").get("test")
    ds_test_len = len(ds_and_dl.get("ds").get("test"))
    writer, dump_name, best_name = get_writer_and_dump_names()
    min_loss = (torch.tensor(np.inf), 0)
    best_common_rate = (torch.tensor(0.0), 0)
    worst_common_rate = (torch.tensor(1.0), 0)
    model, criterion, optimizer = get_net_criterion_optimizer()
    # loop over the dataset multiple times
    for epoch in trange(start_epoch, epoch_count):
        model.train()
        epoch_loss, min_loss = train_epoch(
            model,
            criterion,
            optimizer,
            dl_train,
            epoch,
            min_loss,
            out_f,
            print_step,
        )
        model.eval()
        common_acc, subjects_accs = common_validate(
            model,
            dl_test,
            ds_test_len,
            skip_user=None,
            subjects_test_datasets=subjects_test_datasets,
            each_user_rate_history=each_user_rate_history,
        )
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", common_acc, epoch)
        if each_user_rate_history:
            rates = dict(
                zip(
                    map(
                        lambda _id: f"subject_{_id}",
                        map(str, subjects_accs.keys()),
                    ),
                    subjects_accs.values(),
                )
            )
            writer.add_scalars("Accuracy_subjects/train", rates, epoch)
        dump_args = [
            optimizer,
            epoch,
            common_acc,
            train_batch_size,
            test_batch_size,
        ]
        if common_acc > best_common_rate[0]:
            best_common_rate = (common_acc, epoch)
            save_data(
                best_name, model.__class__, model.state_dict(), *dump_args
            )
        if common_acc < worst_common_rate[0]:
            worst_common_rate = (common_acc, epoch)
        save_data(dump_name, model.__class__, model.state_dict(), *dump_args)

        print(
            (
                f"[{epoch:3d}] rate: {common_acc:.4f}; "
                f"{best_common_rate = }, {worst_common_rate = }"
            ),
            file=out_f,
        )
    return (
        worst_common_rate,
        best_common_rate,
        min_loss,
    )


logs = base_path / "logs"
logs.mkdir(exist_ok=True, parents=True)
with open(logs / f"train.log", "w") as out:
    worst_r, best_r, min_loss = train(
        out,
        epoch_count=epoch_count,
        window_size=signal_len,
        key=data_key,
        numeric_derivative=numeric_derivative,
    )
    print("Results:", file=out)
    print("Loss", min_loss, file=out)
    print("Worst common rate", worst_r, file=out)
    print("Best common rate", best_r, file=out)
