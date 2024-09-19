#!/usr/bin/env python
import torch as t
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import random
from model import MODEL_DICT
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
from typing import Optional, Union
import re
import warnings
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()  # repo root
device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class Parameters:
    instances: int = 3
    embed_dim: int = 32
    hidden_size: int = 64
    epochs: int = 2000
    batch_size: int = 64
    batched: bool = False  # if false, batch is entire data set
    activation: str = "relu"  # gelu or relu
    weight_decay: float = 2e-4
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps = 0
    optimizer: str = "adam"  # adamw or adam or sgd
    checkpoint: int = 3
    name: str = ""
    seed: int = 42
    group_string: str = "Z(48,2);twZ(48)"
    intersect_frac: float = 1.0
    delta_frac: Union[float, list[float]] = 0.0
    train_frac: float = 1.0
    save_weights: bool = False
    save_losses: bool = False
    load_weights: str = ""
    wandb: bool = False
    thresh_grok: float = 0.95
    project: str = "group generalization"
    model: str = "MLP3"
    unembed_bias: bool = False
    init_func: str = "kaiming_uniform"
    correct_embed: bool = False
    replacement: bool = False


def train(model, group_dataset, params):
    # import pdb; pdb.set_trace()
    if params.load_weights:
        model.load_state_dict(t.load(params.load_weights))
    t.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    current_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if not params.wandb:
        os.environ["WANDB_MODE"] = "disabled"
        warnings.warn("Wandb is disabled!")
    if isinstance(params.name, Union[list, tuple]):
        # autocast splits all strings containing ';'
        # but we don't want params.name to be split
        # TODO: find a less annoying way to do params parsing
        params.name = ';'.join(params.name)
    wandb_config = params.__dict__
    wandb_config['groups'] = [group.name for group in group_dataset.groups]
    wandb_config['cmd'] = ' '.join(sys.argv)
    wandb.init(
        entity="neural_fate",
        project=params.project,
        name=f"{current_time}_{params.name}",
        config=wandb_config,
    )

    if not params.batched:
        batch_size = len(group_dataset)
    else:
        batch_size = params.batch_size

    sampler = RandomSampler(group_dataset, replacement=params.replacement)
    train_loader = DataLoader(
        dataset=group_dataset,
        batch_size=batch_size,
        # shuffle=True,
        drop_last=True,
        sampler=sampler,
    )


    if params.optimizer == "sgd":
        optimizer = t.optim.SGD(model.parameters(), lr=params.lr)
    # lower learning rate for unembed, following tensor programs V
    # unembed_params = {
    #     'params': [p for name, p in model.named_parameters() if 'unembed' in name],
    #     # 'lr': params.lr / params.hidden_size,
    #     'weight_decay': params.weight_decay,
    # }
    # other_params = {
    #     'params': [p for name, p in model.named_parameters() if 'unembed' not in name],
    #     # 'lr': params.lr,
    #     'weight_decay': params.weight_decay * 0.,
    # }
    # bias shouldn't have weight decay
    bias_params = {
        'params': [p for name, p in model.named_parameters() if 'bias' in name],
        'weight_decay': 0.,
    }
    weight_params = {
        'params': [p for name, p in model.named_parameters() if 'bias' not in name],
        'weight_decay': params.weight_decay,
    }
    if params.optimizer == "adam":
        optimizer = t.optim.Adam(
            # model.parameters(),
            [bias_params, weight_params],
            # weight_decay=params.weight_decay,
            lr=params.lr,
            betas=[params.beta1, params.beta2],
        )
    if params.optimizer == "adamw":
        optimizer = t.optim.AdamW(
            [bias_params, weight_params],
            # model.parameters(),
            weight_decay=params.weight_decay,
            lr=params.lr,
            betas=[params.beta1, params.beta2],
        )

    step = 0
    checkpoint_every = None
    directory_path = f"{ROOT}/models/{current_time}_{params.name}"
    directory_path = re.sub(r"[^a-zA-Z0-9_/\-]", "_", directory_path)
    if params.checkpoint > 0:
        checkpoint_every = params.checkpoint

        if not os.path.exists(directory_path):
            os.makedirs(directory_path + "/losses")
            os.makedirs(directory_path + "/ckpts")

        with open(directory_path + "/params.json", "w") as f:
            json_str = json.dumps(dataclasses.asdict(params))
            f.write(json_str)

    wandb.watch(model, log="all", log_freq=10)
    epoch_train_loss = t.zeros(params.instances, device=device)
    epoch_train_acc = t.zeros(params.instances, device=device)
    epoch_train_margin = t.full((params.instances,), np.inf, device=device)

    # TODO: fix train loss and acc and margin and move into a logging function
    for epoch in tqdm(range(params.epochs)):
        with t.no_grad():
            model.eval()
            loss_dict = test_loss(model, group_dataset)
            loss_dict["epoch_train_loss"] = epoch_train_loss
            loss_dict["epoch_train_acc"] = epoch_train_acc
            loss_dict["epoch_train_margin"] = epoch_train_margin
            log_dict = {}
            for k in loss_dict:
                for inst in range(params.instances):
                    log_dict[f"{k}_{inst:03d}"] = loss_dict[k][inst].item()
                log_dict[f"{k}_mean"] = loss_dict[k].mean().item()
                log_dict[f"{k}_median"] = loss_dict[k].median().item()
                log_dict[f"{k}_max"] = loss_dict[k].max().item()
                log_dict[f"{k}_min"] = loss_dict[k].min().item()
            for i, group in enumerate(group_dataset.groups):
                # TODO: Move into utils.py test_loss()
                log_dict[f"G{i}_grokked"] = (
                    (loss_dict[f"G{i}_acc"] >= params.thresh_grok)
                    .sum()
                    .item()
                )
            if checkpoint_every is not None and epoch % checkpoint_every == 0:
                if params.save_weights:
                    t.save(
                        model.state_dict(),
                        directory_path + f"/ckpts/{epoch:06d}.pt",
                    )
                if params.save_losses:
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

    t.save(
        model.state_dict(),
        directory_path + f"/ckpts/final.pt",
    )
    t.save(loss_dict, directory_path + f"/losses/final.pt")

    if checkpoint_every is not None:
        print(os.path.abspath(directory_path))
        print("============SUMMARY STATS============")
        traj = load_loss_trajectory(directory_path)
        is_grokked_summary(traj, params.instances, thresh_grok=params.thresh_grok)

        
def parse() -> Parameters:
    params = Parameters()
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(params):
        if field.type == bool:
            parser.add_argument(f"--{field.name}", action="store_true")
        else:
            parser.add_argument(f"--{field.name}", type=str)
    args = parser.parse_args()
    arg_vars = {k: autocast(v) for k, v in vars(args).items() if v is not None}
    params.__dict__.update(arg_vars)
    if not params.name:
        params.name = params.group_string
    return params


if __name__ == "__main__":
    params = parse()
    group_dataset = GroupData(params=params)
    model = MODEL_DICT[params.model](params=params).to(device)
    train(model=model, group_dataset=group_dataset, params=params)
