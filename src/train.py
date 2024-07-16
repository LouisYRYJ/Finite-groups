import torch as t
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from model import MLP2, MLP3
import wandb
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from utils import *
from group_data import *
import json
import argparse
import einops
from pprint import pprint
import numpy as np
import gc

# os.environ["WANDB_MODE"] = "disabled"

device = t.device("cuda" if t.cuda.is_available() else "cpu")


@dataclass
class Parameters:
    instances: int = 3
    embed_dim: int = 32
    hidden_size: int = 64
    num_epoch: int = 2000
    batch_size: int = 64
    max_batch: bool = True  # batch size is the whole data set
    activation: str = "relu"  # gelu or relu
    weight_decay: float = 2e-4
    lr: float = 0.01
    beta1: int = 0.9
    beta2: int = 0.98
    warmup_steps = 0
    optimizer: str = "adam"  # adamw or adam or sgd
    checkpoint: int = 3
    name: str = "experiment"
    seed: int = 42
    group_string: tuple[str] = (
        "twisted(cyclic(48))",
        "twisted(cyclic(48), lambda x: 25 * x))",
    )
    intersect_frac: float = 1.0
    delta_frac: tuple[float] = 0.0
    train_frac: float = 1.0


def train(model, group_dataset, params):
    t.manual_seed(params.seed)
    current_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="neural_fate",
        project="group generalization",
        name=f"{current_time}_{params.name}",
        config=params.__dict__,
    )

    if params.max_batch == True:
        batch_size = len(group_dataset)
    else:
        batch_size = params.batch_size

    train_loader = DataLoader(
        dataset=group_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    if params.optimizer == "sgd":
        optimizer = t.optim.SGD(model.parameters(), lr=params.lr)
    if params.optimizer == "adam":
        optimizer = t.optim.Adam(
            model.parameters(),
            weight_decay=params.weight_decay,
            lr=params.lr,
        )
    if params.optimizer == "adamw":
        optimizer = t.optim.AdamW(
            model.parameters(),
            weight_decay=params.weight_decay,
            lr=params.lr,
            betas=[params.beta1, params.beta2],
        )

    step = 0
    checkpoint_every = None
    directory_path = f"models/{current_time}_{params.name}"
    if params.checkpoint > 0:
        checkpoint_every = params.checkpoint

        if not os.path.exists(directory_path):
            os.makedirs(directory_path + "/losses")
            os.makedirs(directory_path + "/ckpts")

        with open(directory_path + "/params.json", "w") as f:
            json_str = json.dumps(dataclasses.asdict(params))
            f.write(json_str)

    # wandb.watch(model, log="all", log_freq=10)
    epoch_train_loss = t.zeros(params.instances, device=device)
    epoch_train_acc = t.zeros(params.instances, device=device)
    epoch_train_margin = t.full((params.instances,), np.inf, device=device)

    # TODO: fix train loss and acc and margin and move into a logging function
    for epoch in tqdm(range(params.num_epoch)):
        with t.no_grad():
            model.eval()
            loss_dict = test_loss(model, group_dataset)
            loss_dict["epoch_train_loss"] = epoch_train_loss
            loss_dict["epoch_train_acc"] = epoch_train_acc
            loss_dict["epoch_train_margin"] = epoch_train_margin
            log_dict = {}
            for inst in range(params.instances):
                for k in loss_dict:
                    log_dict[f"{k}_{inst:03d}"] = loss_dict[k][inst].item()
            for i in range(group_dataset.num_groups):
                log_dict[f"G{i}_grokked_count"] = (
                    (loss_dict[f"G{i}_accuracy"] >= 1 - 5e-3).sum().item()
                )
            if checkpoint_every is not None and epoch % checkpoint_every == 0:
                t.save(
                    model.state_dict(),
                    directory_path + f"/ckpts/{epoch:06d}.pt",
                )
                t.save(loss_dict, directory_path + f"/losses/{epoch:06d}.pt")

        epoch_train_loss.zero_()
        epoch_train_acc.zero_()
        nn.init.constant_(epoch_train_margin, np.inf)
        for x, z in train_loader:
            global_step = epoch * len(group_dataset) + step
            if global_step < params.warmup_steps:
                lr = global_step * params.lr / float(params.warmup_steps)
            else:
                lr = params.lr
            for g in optimizer.param_groups:
                g["lr"] = lr

            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            z = z.to(device)
            output = model(x)
            loss = get_cross_entropy(output, z)
            acc = get_accuracy(output, z)
            margin = get_margin(output, z)
            epoch_train_loss += loss * z.shape[0]  # batch_size
            epoch_train_acc += acc * z.shape[0]
            epoch_train_margin = t.min(epoch_train_margin, margin)

            loss.sum().backward()
            optimizer.step()
            step += 1

        epoch_train_loss /= len(group_dataset)
        epoch_train_acc /= len(group_dataset)

        wandb.log(log_dict)
    wandb.finish()

    if checkpoint_every is not None:
        print(os.path.abspath(directory_path))
        print("============SUMMARY STATS============")
        traj = load_loss_trajectory(directory_path)
        is_grokked_summary(traj, params.instances)


if __name__ == "__main__":
    params = Parameters()
    parser = argparse.ArgumentParser()
    for k in params.__dict__:
        parser.add_argument(f"--{k}")
    args = parser.parse_args()
    arg_vars = {k: autocast(v) for k, v in vars(args).items() if v is not None}
    params.__dict__.update(arg_vars)
    group_dataset = GroupData(params=params)
    model = MLP3(group_dataset.N, params=params).to(device)
    train(model=model, group_dataset=group_dataset, params=params)
