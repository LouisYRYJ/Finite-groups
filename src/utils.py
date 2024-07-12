import einops
import random
import torch as t
from torch import nn
import torch.nn.functional as F
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary, estimate_learning_coeff
from groups_data import GroupData
from torch.utils.data import DataLoader
from jaxtyping import Int, Float, jaxtyped
from beartype import beartype
from groups_data import GroupData
from typing import Tuple
from itertools import product

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Easier to just use F.cross_entropy
# def loss_fn(logits, labels):
# """
# Compute cross entropy loss.

# Args:
# logits (Tensor): (batch, group.order) tensor of logits
# labels (Tensor): (batch) tensor of labels

# Returns:
# float: cross entropy loss
# """
# log_probs = logits.log_softmax(dim=-1)
# correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
# return -correct_log_probs.mean()


@jaxtyped(typechecker=beartype)
def get_accuracy(
    logits: Float[t.Tensor, "batch instance vocab"], labels: Int[t.Tensor, "batch"]
) -> Float[t.Tensor, "instance"]:
    """
    Compute instance-wise accuracy of model.
    """
    instances = logits.shape[1]
    labels = einops.repeat(labels, "batch -> batch n", n=instances)
    return (logits.argmax(-1) == labels).sum(dim=0) / labels.shape[0]

@jaxtyped(typechecker=beartype)
def get_cross_entropy(
    logits: Float[t.Tensor, "batch instance vocab"], labels: Int[t.Tensor, "batch"]
) -> Float[t.Tensor, "instance"]:
    """
    Compute instance-wise cross entropy loss of model.
    (Need to rearrange batch and instance to match the expected shape of F.cross_entropy)
    """
    instances = logits.shape[1]
    labels = einops.repeat(labels, "batch -> batch n", n=instances)
    logits = einops.rearrange(logits, "batch instance vocab -> batch vocab instance")
    return F.cross_entropy(logits, labels, reduction='none').mean(dim=0)

@jaxtyped(typechecker=beartype)
def test_loss(
    model: nn.Module,
    N: int,
    group_dataset: GroupData,
) -> Tuple[
    Tuple[Float[t.Tensor, "instance"], Float[t.Tensor, "instance"]],
    Tuple[Float[t.Tensor, "instance"], Float[t.Tensor, "instance"]],
]:
    """Create all possible pairs (x,y) and return loss and accuracy for G_1 and G_2"""
    test_inputs = t.tensor(list(product(range(N), repeat=2)), device=device)

    logits = model(test_inputs)
    labels_group_1 = einops.rearrange(group_dataset.group1, "a b-> (a b)").to(device)
    labels_group_2 = einops.rearrange(group_dataset.group2, "a b-> (a b)").to(device)
    if labels_group_1.shape[0] == 47:
        import pdb; pdb.set_trace()

    loss_group_1 = get_cross_entropy(logits, labels_group_1)
    loss_group_2 = get_cross_entropy(logits, labels_group_2)

    accuracy_group_1 = get_accuracy(logits, labels_group_1)
    accuracy_group_2 = get_accuracy(logits, labels_group_2)

    return (loss_group_1, loss_group_2), (accuracy_group_1, accuracy_group_2)


def random_indices(full_dataset, params):
    num_indices = int(len(full_dataset) * params.train_frac)
    picked_indices = random.sample(list(range(len(full_dataset))), num_indices)
    return picked_indices


def make_fourier_basis(params):
    group_order = params.N
    fourier_basis_values = t.ones(group_order, group_order)
    fourier_basis_names = ["Const"]
    for i in range(1, group_order // 2):
        # Define each of the cos and sin terms
        fourier_basis_values[2 * i - 1] = t.cos(
            2 * t.pi * t.arange(group_order) * i / group_order
        )
        fourier_basis_values[2 * i] = t.sin(
            2 * t.pi * t.arange(group_order) * i / group_order
        )
        fourier_basis_names.extend([f"cos {i}", f"sin {i}"])

    fourier_basis_values[group_order - 1] = t.cos(t.pi * t.arange(group_order))
    fourier_basis_names.append(f"cos {group_order//2}")
    # Normalize vectors, and return them
    fourier_basis_values /= fourier_basis_values.norm(dim=1, keepdim=True)
    return fourier_basis_values, fourier_basis_names


def fourier_transform_embedding(matrix, params):

    pass


def measure_llc(model, params, summary: bool):

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

    if summary:
        learning_coeff_stats = estimate_learning_coeff_with_summary(
            model,
            loader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=5e-4, localization=100.0),
            num_chains=5,  # How many independent chains to run
            num_draws=300,  # How many samples to draw per chain
            num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
            num_steps_bw_draws=1,  # How many steps to take between each sample
            online=True,
        )

    else:
        learning_coeff_stats = estimate_learning_coeff(
            model,
            loader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=2e-5, localization=100.0),
            num_chains=5,  # How many independent chains to run
            num_draws=300,  # How many samples to draw per chain
            num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
            num_steps_bw_draws=1,  # How many steps to take between each sample
        )

    return learning_coeff_stats


def autocast(x):
    try:
        return int(x)
    except:
        pass
    try:
        return float(x)
    except:
        return x
