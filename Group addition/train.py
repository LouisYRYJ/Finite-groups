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
class Parameters:
    N_1: int = 7
    N_2: int = 2
    N: int = N_1 * N_1 * N_2
    embed_dim: int = 16
    hidden_size: int = 64
    num_epoch: int = 20000
    batch_size: int = 256
    activation: str = "gelu"
    checkpoint_every: int = 5
    max_steps_per_epoch: int = N * N // batch_size


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


def test_loss(model, params):
    """Create all possible pairs (x,y) and return loss and accuracy for G_1 and G_2"""
    test_labels_x = t.tensor([num for num in range(params.N) for _ in range(params.N)])
    test_labels_y = t.tensor([num % params.N for num in range(params.N * params.N)])

    logits = model(test_labels_x, test_labels_y)

    labels_group_1 = group_1(test_labels_x, test_labels_y)
    labels_group_2 = group_2(test_labels_x, test_labels_y)

    loss_group_1 = loss_fn(logits, labels_group_1)
    loss_group_2 = loss_fn(logits, labels_group_2)

    accuracy_group_1 = get_accuracy(logits, labels_group_1)
    accuracy_group_2 = get_accuracy(logits, labels_group_2)

    return (loss_group_1, loss_group_2), (accuracy_group_1, accuracy_group_2)


def train(model, train_data, params, optimizer):
    progress_bar = tqdm(total=params.num_epoch * params.max_steps_per_epoch)
    current_day = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="Grokking ambiguous data",
        name=f"experiment_{current_day}",
        config={
            "Epochs": params.num_epoch,
            "Cardinality": params.N,
            "Embedded dimension": params.embed_dim,
            "Hidden dimension": params.hidden_size,
        },
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=False,
    )
    criterion = t.nn.CrossEntropyLoss()
    average_loss_training = 0
    for epoch in tqdm(range(params.num_epoch)):
        with t.no_grad():
            model.eval()

            average_loss_training = average_loss_training / (params.max_steps_per_epoch)

            losses_test, accuracies_test = test_loss(model, params)
            wandb.log({"Loss G_1": losses_test[0], "Loss G_2": losses_test[1]})
            wandb.log(
                {"Accuracy G_1": accuracies_test[0], "Accuracy G_2": accuracies_test[1]}
            )
            wandb.log({"Training loss": average_loss_training})

            average_loss_training = 0
        for x, y, z in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(x, y)
            loss = criterion(output, z)
            average_loss_training += loss.item()
            loss.backward()
            optimizer.step()

            """progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}")"""

    wandb.finish()


if __name__ == "__main__":
    ExperimentsParameters = Parameters()
    IntersectionDataSet = IntersectionData(ExperimentsParameters)
    model = MLP(ExperimentsParameters)
    optimizer = t.optim.Adam(model.parameters())

    train(
        model=model,
        train_data=IntersectionDataSet,
        params=ExperimentsParameters,
        optimizer=optimizer,
    )


"""            if (
                params.checkpoint_every is not None
                and epoch % params.checkpoint_every == 0
            ):
                model.save(os.path.join(wandb.run.dir, f"Model_{epoch}"))
"""
