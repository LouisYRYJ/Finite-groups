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
from utils import get_cross_entropy, test_loss, random_indices, autocast
import json
import argparse
import einops

# os.environ["WANDB_MODE"] = "disabled"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class Parameters:
    instances: int = 3
    N_1: int = 24
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
    data_group1: bool = True# training data G_1
    data_group2: bool = True # training data G_2
    add_points_group1: int = 0  # add points from G_1 only
    add_points_group2: int = 0  # add points from G_2 only
    checkpoint: int = 3
    random: bool = False
    name: str = 'experiment'


def train(model, params):
    current_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="neural_fate",
        project="Dev Group (Specification vs determination)",
        name=f'{params.name}_{current_time}',
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

        directory_path = f"models/{params.name}_{current_time}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(directory_path + "/params.json", "w") as f:
            json_str = json.dumps(dataclasses.asdict(params))
            f.write(json_str)

    checkpoint_no = 0
    wandb.watch(model, log="all", log_freq=10)
    for epoch in tqdm(range(params.num_epoch)):
        with t.no_grad():
            model.eval()
            losses_test, accuracies_test = test_loss(model, params.N, group_dataset)
            for inst in range(params.instances):
                for group in range(2):
                    wandb.log({f"G_{group+1}_loss_{inst}": losses_test[group][inst].item()})
                    wandb.log({f"G_{group+1}_accuracy_{inst}": accuracies_test[group][inst].item()})
            if checkpoint_every is not None and epoch % checkpoint_every == 0:
                t.save(
                    model.state_dict(),
                    directory_path + f"/{checkpoint_no}.pt",
                )
                checkpoint_no += 1

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
            output = model(x.to(device))
            loss = get_cross_entropy(output, z.to(device))
            for inst in range(params.instances):
                wandb.log({f'train_loss_{inst}': loss[inst].item()})
            loss.sum().backward()
            optimizer.step()
            step += 1

    wandb.finish()


if __name__ == "__main__":
    params = Parameters()
    parser = argparse.ArgumentParser()
    for k in params.__dict__:
        parser.add_argument(f'--{k}')
    args = parser.parse_args()
    arg_vars = {k: autocast(v) for k, v in vars(args).items() if v is not None}
    params.__dict__.update(arg_vars)
    model = MLP3(params).to(device)
    train(model=model, params=params)
