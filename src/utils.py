import einops
import random
import torch as t
from torch import nn
import torch.nn.functional as F
from model import MODEL_DICT
from devinterp.optim.sgld import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary, estimate_learning_coeff
from torch.utils.data import DataLoader
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from group_data import *
from typing import Tuple, Union, Any
from itertools import product
import glob
import numpy as np
from model import InstancedModule
import json
import os
import re

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@t.no_grad()
def ema(x, alpha, dim=-1):
    ret = [x.select(dim, 0)]
    for i in range(1, x.shape[dim]):
        ret.append(alpha * x.select(dim, i) + (1 - alpha) * ret[-1])
    return t.stack(ret, dim=dim)

def full_train_loss(
    model, dataset,
) -> Float[t.Tensor, 'instance']:
    loader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False
    )
    model.eval()
    loss = 0
    for x, z in loader:
        x = x.to(device)
        z = z.to(device)
        output = model(x)
        loss += get_cross_entropy(output, z)
    return loss / len(loader)

@t.no_grad()
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
    logits: Float[t.Tensor, "batch instance vocab"], labels: Int[t.Tensor, "batch"],
) -> Float[t.Tensor, "instance"]:
    """
    Compute instance-wise cross entropy loss of model.
    (Need to rearrange batch and instance to match the expected shape of F.cross_entropy)
    """
    instances = logits.shape[1]
    labels = einops.repeat(labels, "batch -> batch n", n=instances)
    logits = einops.rearrange(logits, "batch instance vocab -> batch vocab instance")
    return F.cross_entropy(logits, labels, reduction="none").mean(dim=0)

@jaxtyped(typechecker=beartype)
@t.no_grad()
def get_margin(
    logits: Float[t.Tensor, "batch instance vocab"], labels: Int[t.Tensor, "batch"]
) -> Float[t.Tensor, "instance"]:
    """
    Compute instance-wise margin (correct logit minus max incorrect logit; min across batches)
    """
    instances = logits.shape[1]
    labels = einops.repeat(labels, "batch -> batch n", n=instances)
    labels_onehot = F.one_hot(labels, num_classes=logits.shape[-1])
    label_logits = einops.einsum(logits, labels_onehot.float(),
        "batch instance vocab, batch instance vocab -> batch instance"
    )
    label_mask = t.where(labels_onehot > 0, -np.inf, 0)
    other_logits = (logits + label_mask).max(dim=2).values
    
    # # logit values at indices corresponding to labels
    # label_logits = logits.gather(dim=2, index=labels).squeeze(-1)
    # other_logits = logits.clone()
    # # set logit values at labels to -inf so we can then take max over non-labels
    # other_logits.scatter_(
    #     dim=2,
    #     index=labels,
    #     src=t.ones_like(logits) * -np.inf
    # )
    # other_logits = t.max(other_logits, dim=2).values
    return t.clamp(t.min(label_logits - other_logits, dim=0).values, min=0)

@jaxtyped(typechecker=beartype)
@t.no_grad()
def test_loss(
    model, group_dataset
) -> dict[str, Float[t.Tensor, "instance"]]:
    """Create all possible pairs (x,y) and return loss and accuracy for all groups in group_dataset."""
    N = group_dataset.N
    test_inputs = t.tensor(list(product(range(N), repeat=2)), device=device)

    logits = model(test_inputs)
    loss_dict = dict()
    for i, group in enumerate(group_dataset.groups):
        labels = einops.rearrange(group.cayley_table, "a b -> (a b)").to(device)
        loss = get_cross_entropy(logits, labels)
        accuracy = get_accuracy(logits, labels)
        # Don't add group name to wandb logs; it makes plot searching less convenient
        # Instead store group names in wandb config (in train.py)
        loss_dict[f"G{i}_loss"] = loss
        loss_dict[f"G{i}_acc"] = accuracy
        # loss_dict[f"G{i}_loss_{group.name}"] = loss
        # loss_dict[f"G{i}_acc_{group.name}"] = accuracy

    return loss_dict

@jaxtyped(typechecker=beartype)
def load_loss_trajectory(
    save_path: str,
) -> dict[str, Float[t.Tensor, "instance epoch"]]:
    """Load loss trajectory from a saved file."""
    loss_files = sorted(glob.glob(f"{save_path}/losses/*.pt"))
    losses = [t.load(f) for f in loss_files]
    return {k: t.stack([l[k] for l in losses], dim=1) for k in losses[0]}

@jaxtyped(typechecker=beartype)
def is_grokked(
    trajectory: dict[str, Float[t.Tensor, "instance epoch"]],
    thresh_grok: float = 1 - 5e-3,
    thresh_ungrok: float = 0.1,
) -> dict[str, Bool[t.Tensor, "instance"]]:
    """Classifies the loss trajectory into grokked and ungrokked per instance."""
    grok_dict = dict()
    for k, acc in trajectory.items():
        if 'acc' not in k:
            continue
        grok_dict[k.replace('acc', 'grokked')] = acc[:, -1] >= thresh_grok
        grok_dict[k.replace('acc', 'ungrokked')] = (acc.max(dim=1).values - acc[:, -1] > thresh_ungrok)
    return grok_dict

@jaxtyped(typechecker=beartype)
def is_grokked_summary(
    trajectory: dict[str, Float[t.Tensor, "instance epoch"]],
    instances: int,
    thresh_grok: float = 1 - 5e-3,
    thresh_ungrok: float = 0.1,
) -> None:
    grokked = is_grokked(trajectory, thresh_grok=thresh_grok, thresh_ungrok=thresh_ungrok)
    for k in grokked:
        print(
            f"{k}: {frac_str(grokked[k].sum().item(), instances)}"
        )


def make_fourier_basis(group_order):
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

    train_data = GroupData(params=params)
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

def autocast(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    if ';' in x:
        return tuple(map(autocast, x.split(';')))
    try:
        return int(x)
    except:
        pass
    try:
        return float(x)
    except:
        return x

@jaxtyped(typechecker=beartype)
@t.no_grad()
def model_table(model: InstancedModule) -> Int[t.Tensor, 'instance N N']:
    """Returns the model's current multiplication table"""
    inputs = t.tensor(list(product(range(model.N), repeat=2)), dtype=int).to(device)
    model.eval()
    logits = model(inputs)  # shape N^2 x instance x N
    max_prob_entry = t.argmax(logits, dim=-1)  # shape N^2 x instance
    return einops.rearrange(max_prob_entry, " (n m) instance -> instance n m", n=model.N)  # shape instance x N x N

def get_number_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def load_model_paths(path, sel=None):
    from train import Parameters
    model_paths = []

    with open(path + "/params.json", "r") as f:
        json_str = f.read()
        params = Parameters(**json.loads(json_str))

    for root, dirs, files in os.walk(path + "/ckpts"):
        for filename in sorted(files, key=get_number_from_filename)[1:]:
            model_paths.append(os.path.join(root, filename))

    if (isinstance(sel, str) and sel.lower() == 'final') or len(model_paths) == 0:
        model_paths = [os.path.join(root, 'final.pt')]
    elif sel is not None:
        model_paths = model_paths[sel]

    return model_paths, params

def load_models(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    models = []
    N = len(string_to_groups(params.group_string)[0])
    itr = model_paths if len(model_paths) < 5 else tqdm(model_paths)
    for model_path in itr:
        model = MODEL_DICT[params.model](N=N, params=params)
        model.load_state_dict(t.load(model_path))
        models.append(model)
    return models, params

def load_models_itr(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    N = len(string_to_groups(params.group_string)[0])
    for model_path in model_paths:
        model = MODEL_DICT[params.model](N=N, params=params)
        model.load_state_dict(t.load(model_path))
        yield model

# Use this to quickly create a params object for testing.
# Please don't use for anything else; very hacky.
class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __deepcopy__ = None