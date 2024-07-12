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
from utils import test_loss, random_indices, autocast
import json
import argparse

os.environ["WANDB_MODE"] = "disabled"

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
    train_frac: float = 0.4
    weight_decay: float = 0.0002
    lr: float = 0.01
    beta_1: int = 0.9
    beta_2: int = 0.98
    warmup_steps = 0
    optimizer: str = "adam"  # adamw or adam or sgd
    data_group1: bool = True# training data G_1
    data_group2: bool = True # training data G_2
    add_points_group1: int = 0  # add points from G_1 only
    add_points_group2: int = 0  # add points from G_2 only
    checkpoint: int = 3
    random: bool = False
    name: str = ''


def train(model, params):
    current_time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    name = f'ex_{current_time}'
    if params.name:
        name += f'_{name}'
    wandb.init(
        entity="neural_fate",
        project="Dev Group (Specification vs determination)",
        name=name,
        config=params.__dict__,
    )
    Group_Dataset = GroupData(params=params)

    train_data = t.utils.data.Subset(
        Group_Dataset, random_indices(Group_Dataset, params)
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

    criterion = t.nn.CrossEntropyLoss()
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
            betas=[params.beta_1, params.beta_2],
        )

    average_loss_training = 0
    step = 0
    list_of_figures = []

    checkpoint_every = None
    if params.checkpoint > 0:
        checkpoint_every = params.checkpoint

        directory_path = f"models/model_{current_time}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(directory_path + "/params.json", "w") as f:
            json_str = json.dumps(dataclasses.asdict(params))
            f.write(json_str)

    checkpoint_no = 0
    # wandb.watch(model, criterion=criterion, log="all", log_freq=10)
    for epoch in tqdm(range(params.num_epoch)):

        losses_test, accuracies_test = test_loss(model, params, Group_Dataset, device)
        wandb.log({"Loss G_1": losses_test[0], "Loss G_2": losses_test[1]})
        wandb.log(
            {"Accuracy G_1": accuracies_test[0], "Accuracy G_2": accuracies_test[1]}
        )
        wandb.log({"Training loss": average_loss_training})

        with t.no_grad():
            model.eval()

            if checkpoint_every is not None and epoch % checkpoint_every == 0:

                t.save(
                    model.state_dict(),
                    directory_path + f"/{checkpoint_no}.pt",
                )
                checkpoint_no += 1

            average_loss_training = average_loss_training / (params.max_steps_per_epoch)

            average_loss_training = 0
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
            loss = criterion(output, z.to(device))
            average_loss_training += loss.item()
            loss.backward()
            optimizer.step()
            step += 1

            """progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}")"""

    wandb.finish()


random.seed(42)


if __name__ == "__main__":
    params = Parameters()
    parser = argparse.ArgumentParser()
    for k in params.__dict__:
        parser.add_argument(f'--{k}')
    parser.add_argument('name')
    arg_vars = {k: autocast(v) for k, v in vars(args).items() if v is not None}
    params.__dict__.update(arg_vars)
    for _ in range(1):
        model = MLP2(params).to(device)
        train(model=model, params=ExperimentsParameters)
