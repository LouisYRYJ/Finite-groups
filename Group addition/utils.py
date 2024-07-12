from einops import rearrange
import random
import torch as t
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary, estimate_learning_coeff
from groups_data import GroupData
from torch.utils.data import DataLoader


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


def test_loss(model, params, Group_Dataset, device):
    """Create all possible pairs (x,y) and return loss and accuracy for G_1 and G_2"""
    test_labels_x = t.tensor(
        [num for num in range(params.N) for _ in range(params.N)]
    ).to(device)
    test_labels_y = t.tensor([num % params.N for num in range(params.N * params.N)]).to(
        device
    )

    test_labels = t.stack((test_labels_x, test_labels_y), dim=1)

    logits = model(test_labels)
    labels_group_1 = rearrange(Group_Dataset.group1, "a b-> (a b)").to(device)
    labels_group_2 = rearrange(Group_Dataset.group2, "a b-> (a b)").to(device)

    loss_group_1 = loss_fn(logits, labels_group_1)
    loss_group_2 = loss_fn(logits, labels_group_2)

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
            criterion=t.nn.CrossEntropyLoss(),
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
            criterion=t.nn.CrossEntropyLoss(),
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
