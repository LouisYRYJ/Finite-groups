import torch as t
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import devinterp
import random
from einops import rearrange
from model import MLP, MLP2
import wandb
from dataclasses import dataclass
from groups_data import GroupData
import copy
from datetime import datetime


@dataclass
class Parameters:
    N_1: int = 50
    N: int = N_1 * 2
    embed_dim: int = 32
    hidden_size: int = 64
    num_epoch: int = 2000
    batch_size: int = 512
    activation: str = "relu"  # gelu or relu
    checkpoint_every: int = 5
    max_steps_per_epoch: int = N * N // batch_size
    train_frac: float = 1
    weight_decay: float = 0.0002
    lr: float = 0.01
    beta_1: int = 0.9
    beta_2: int = 0.98
    warmup_steps = 0
    optimizer: str = "adam"  # adamw or adam or sgd
    data_group1: bool = True  # training data G_1
    data_group2: bool = True  # training data G_2
    add_points_group1: int = 0  # add points from G_1 only
    add_points_group2: int = 0  # add points from G_2 only


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


def group_1(i, j, params):
    g_1 = (i % params.N, (i - i % params.N) // params.N)
    g_2 = (j % params.N, (i - i % params.N) // params.N)
    product = (
        (g_1[0] + g_2[0]) % params.N,
        (g_1[1] + g_2[1]) % 2,
    )
    return product[0] + params.N * product[1]


def test_loss(model, params, Group_Dataset):
    """Create all possible pairs (x,y) and return loss and accuracy for G_1 and G_2"""
    test_labels_x = t.tensor([num for num in range(params.N) for _ in range(params.N)])
    test_labels_y = t.tensor([num % params.N for num in range(params.N * params.N)])

    logits = model([test_labels_x, test_labels_y])

    labels_group_test = group_1(test_labels_x, test_labels_y, params)

    labels_group_1 = rearrange(Group_Dataset.group1, "a b-> (a b)")
    labels_group_2 = rearrange(Group_Dataset.group2, "a b-> (a b)")

    loss_group_1 = loss_fn(logits, labels_group_1)
    loss_group_2 = loss_fn(logits, labels_group_2)

    accuracy_group_1 = get_accuracy(logits, labels_group_1)
    accuracy_group_2 = get_accuracy(logits, labels_group_2)

    return (loss_group_1, loss_group_2), (accuracy_group_1, accuracy_group_2)


def train(model, train_data, params):
    progress_bar = tqdm(total=params.num_epoch * params.max_steps_per_epoch)
    current_day = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="Grokking ambiguous data",
        name=f"experiment_{current_day}",
        config={
            "Epochs": params.num_epoch,
            "Batch size": params.batch_size,
            "Cardinality": params.N,
            "Embedded dimension": params.embed_dim,
            "Hidden dimension": params.hidden_size,
            "Training": (params.data_group1, params.data_group2),
            "Added points": (params.add_points_group1, params.add_points_group2),
            "Train frac": params.train_frac,
            "Weight decay": params.weight_decay,
            "Learning rate": params.lr,
            "Warm up steps": params.warmup_steps,
        },
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=len(train_data),
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
    for epoch in tqdm(range(params.num_epoch)):
        with t.no_grad():
            model.eval()

            average_loss_training = average_loss_training / (params.max_steps_per_epoch)

            losses_test, accuracies_test = test_loss(model, params, Group_Dataset)
            wandb.log({"Loss G_1": losses_test[0], "Loss G_2": losses_test[1]})
            wandb.log(
                {"Accuracy G_1": accuracies_test[0], "Accuracy G_2": accuracies_test[1]}
            )
            wandb.log({"Training loss": average_loss_training})
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
            output = model(x)
            loss = criterion(output, z)
            average_loss_training += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            """progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}")"""

    wandb.finish()


random.seed(42)


def random_indices(full_dataset, params):
    num_indices = int(len(full_dataset) * params.train_frac)
    picked_indices = random.sample(list(range(len(full_dataset))), num_indices)
    return picked_indices


if __name__ == "__main__":
    ExperimentsParameters = Parameters()

    Group_Dataset = GroupData(ExperimentsParameters)

    Training_Set = t.utils.data.Subset(
        Group_Dataset, random_indices(Group_Dataset, ExperimentsParameters)
    )

    for _ in range(10):
        model = MLP2(ExperimentsParameters)
        train(model=model, train_data=Training_Set, params=ExperimentsParameters)


"""            if (
                params.checkpoint_every is not None
                and epoch % params.checkpoint_every == 0
            ):
                model.save(os.path.join(wandb.run.dir, f"Model_{epoch}"))
"""
