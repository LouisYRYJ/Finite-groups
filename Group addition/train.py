import torch as t
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import devinterp
import random
from model import MLP
import wandb
from dataclasses import dataclass
from groups_data import IntersectionData, group_1, group_2
import copy
from datetime import datetime


@dataclass
class ExperimentsParameters:
    N_1: int = 7
    N_2: int = 2
    N: int = N_1 * N_1 * N_2
    embed_dim: int = 16
    hidden_size: int = 64
    num_epoch: int = 100
    batch_size: int = 256
    activation: str = "gelu"
    checkpoint_every: int = 5


ExperimentsParameter = ExperimentsParameters()

IntersectionDataSet = IntersectionData()
model = MLP(ExperimentsParameter)

train_loader = DataLoader(
    dataset=IntersectionDataSet,
    batch_size=ExperimentsParameter.batch_size,
    shuffle=True,
    drop_last=True,
)


current_day = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
wandb.init(
    project="Grokking ambiguous data",
    name=f"experiment_{current_day}",
    config={
        "Epochs": ExperimentsParameter.num_epoch,
        "Cardinality": ExperimentsParameter.N,
        "Embedded dimension": ExperimentsParameter.embed_dim,
        "Hidden dimension": ExperimentsParameter.hidden_size,
    },
)

# Creating DataLoader object


def loss_fn(logits, labels):
    """
    Compute cross entropy loss.

    Args:
        logits (Tensor): (batch, group.order) tensor of logits
        labels (Tensor): (batch) tensor of labels

    Returns:
        float: cross entropy loss
    """
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def get_accuracy(logits, labels):
    """
    Compute accuracy of model.

    Args:
        logits (torch.tensor): (batch, group.order) tensor of logits
        labels (torch.tensor): (batch) tensor of labels

    Returns:
        float: accuracy
    """
    return ((logits.argmax(-1) == labels).sum() / len(labels)).item()


def test_loss():
    pass


def train(model, train_data, params, optimizer):
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=ExperimentsParameter.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model_history = []
    average_loss = 0
    criterion = t.nn.CrossEntropyLoss()

    for epoch in tqdm(range(params.num_epoch)):
        with t.no_grad():
            model.eval()
            if (
                params.checkpoint_every
                is not None & epoch % params.checkpoint_every
                == 0
            ):
                model.save(os.path.join(wandb.run.dir, f"Model_{epoch}"))
            test_labels_x = [
                num for num in range(N) for _ in range(ExperimentsParameter.N)
            ]
            test_labels_y = []
            logits = model(test_labels_x, test_labels_y)
            labels_group_1 = group_1(test_labels_x, test_labels_y)
            labels_group_2 = group_2(test_labels_x, test_labels_y)
            loss_group_1 = loss_fn(logits, labels_group_1)
            loss_group_2 = loss_fn(logits, labels_group_2)
            accuracy_group_1 = get_accuracy(logits, labels_group_2)
            accuracy_group_1 = get_accuracy(logits, labels_group_2)
        for x, y, z in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(x, y)
            loss = criterion(output, z)
            average_loss += loss.item()  # divide by 5488/256
            loss.backward()
            optimizer.step()

    wandb.finish()
