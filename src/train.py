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
from groups_data import GroupData
from datetime import datetime
from utils import *
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
    N_1: int = 48
    N: int = N_1 * 2  # cardinality of group
    embed_dim: int = 32
    hidden_size: int = 64
    num_epoch: int = 2000
    batch_size: int = 64
    max_batch: bool = True  # batch size is the whole data set
    activation: str = "relu"  # gelu or relu
    max_steps_per_epoch: int = N * N // batch_size
    train_frac: float = 1.0
    weight_decay: float = 0.0002
    lr: float = 0.01
    beta1: int = 0.9
    beta2: int = 0.98
    warmup_steps = 0
    optimizer: str = "adam"  # adamw or adam or sgd
    data_group1: bool = True  # training data G1
    data_group2: bool = True  # training data G2
    add_points_group1: int = 0  # add points from G1 only
    add_points_group2: int = 0  # add points from G2 only
    checkpoint: int = 3
    random: bool = False
    name: str = "experiment"
    seed: int = 42

def train(model, params):
    t.manual_seed(params.seed)
    current_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="neural_fate",
        project="group generalization",
        name=f"{current_time}_{params.name}",
        config=params.__dict__,
    )
    group_dataset = GroupData(params=params)

    train_data = t.utils.data.Subset(
        group_dataset, random_indices(group_dataset, params)
    )

    if params.max_batch == True:
        batch_size = len(train_data)
    else:
        batch_size = params.batch_size

    train_loader = DataLoader(
        dataset=train_data,
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
    list_of_figures = []

    checkpoint_every = None
    if params.checkpoint > 0:
        checkpoint_every = params.checkpoint

        directory_path = f"models/{current_time}_{params.name}"
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
            loss_dict = test_loss(model, params.N, group_dataset)
            loss_dict["epoch_train_loss"] = epoch_train_loss
            loss_dict["epoch_train_acc"] = epoch_train_acc
            loss_dict["epoch_train_margin"] = epoch_train_margin
            log_dict = {}
            for inst in range(params.instances):
                for k in loss_dict:
                    log_dict[f"{k}_{inst:03d}"] = loss_dict[k][inst].item()
            g1_grokked = (loss_dict["G1_accuracy"] >= 1 - 5e-3).sum()
            g2_grokked = (loss_dict["G2_accuracy"] >= 1 - 5e-3).sum()
            log_dict["G1_grokked_count"] = g1_grokked.item()
            log_dict["G2_grokked_count"] = g2_grokked.item()
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
            global_step = epoch * len(train_data) + step
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
            gc.collect()
            t.cuda.empty_cache()

        epoch_train_loss /= len(train_data)
        epoch_train_acc /= len(train_data)

        wandb.log(log_dict)
    wandb.finish()
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
    # Need to update these manually...
    params.N = params.N_1 * 2
    params.max_steps_per_epoch = params.N * params.N // params.batch_size
    model = MLP3(params).to(device)
    train(model=model, params=params)
