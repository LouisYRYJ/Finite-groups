import sys
import torch as t
import numpy as np
from matplotlib import pyplot as plt
import json
from itertools import product
from model import InstancedModule
from utils import *
from group_data import *
from jaxtyping import Float
from typing import Union
from einops import repeat
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import copy
import math
from itertools import product

def sgld_trace(
    model: InstancedModule,
    dataset: GroupData,
    eps: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    beta: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    gamma: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    epochs: int=2000,
    instances: int=1,
    tq: bool=True,
) -> Float[t.Tensor, 'instance epoch']:
    hyparams = {
        'eps': eps, 'beta': beta, 'gamma': gamma
    }
    del eps, beta, gamma   # to avoid confusion
    for k, v in hyparams.items():
        if isinstance(v, float):
            hyparams[k] = t.tensor([v], device=device)
            
    instances = max(model.num_instances(), *(v.shape[0] for v in hyparams.values()), instances)
    model = copy.deepcopy(model)   # don't mutate original model
    if model.num_instances() != instances:
        assert model.num_instances() == 1, f'Expected either {1} or {instances} model instances, but got {model.num_instances()}!'
        model = model.repeat(instances)
    for k, v in hyparams.items():
        if v.shape[0] != instances:
            assert v.shape[0] == 1, f'Expected either {1} or {instances} {k} instances, but got {v.shape[0]}!'
            hyparams[k] = einops.repeat(v, '1 -> (n 1)', n=instances)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False
    )

    eps, beta, gamma = hyparams['eps'], hyparams['beta'], hyparams['gamma']

    orig_params = {}
    for name, param in model.named_parameters():
        orig_params[name] = param.data.clone().detach()
        
    trace = []
    itr = tqdm(range(epochs), desc='SGLD') if tq else range(epochs)
    model.train()
    for epoch in itr:
        epoch_loss = 0
        for x, z in loader:
            x = x.to(device)
            z = z.to(device)
            output = model(x)
            loss = get_cross_entropy(output, z)
            epoch_loss += loss
            loss.sum().backward()
            for name, param in model.named_parameters():
                grad_step = einops.einsum(
                    -(eps / 2) * beta,
                    param.grad.data,
                    'instance, instance ... -> instance ...'
                )
                localization = einops.einsum(
                    (eps / 2) * gamma,
                    orig_params[name] - param,
                    'instance, instance ... -> instance ...'
                )
                noise = einops.einsum(
                    t.sqrt(eps),
                    t.randn_like(param),
                    'instance, instance ... -> instance ...'
                )
                param.data.add_(grad_step + localization + noise)
        trace.append(epoch_loss.detach() / len(loader))
    return einops.rearrange(trace, 'epoch instance -> instance epoch')

def llc_from_trace(
    trace: Float[t.Tensor, 'instance epoch'],
    orig_loss: Float[t.Tensor, 'instance'],
    beta:  Float[t.Tensor, 'instance'],
    burnin: float=0.6,
) -> Float[t.Tensor, 'instance']:
    start = int(burnin * trace.shape[1])
    return beta * (trace[:,start:].mean(dim=1) - orig_loss)

def get_llc(
    model: InstancedModule,
    dataset: GroupData,
    eps: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    beta: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    gamma: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    burnin: float=0.6,
    epochs: int=2000,
    tq: bool=True,
) -> Float[t.Tensor, 'instance']:
    trace = sgld_trace(model, dataset, eps, beta, gamma, epochs=epochs, tq=tq)
    orig_loss = full_train_loss(model, dataset)
    return llc_from_trace(trace, orig_loss, beta, burnin=burnin), trace

def plot_trace(trace: Float[t.Tensor, 'instance epoch']) -> go.Figure:
    fig = go.Figure()

    for i in range(trace.shape[0]):
        fig.add_trace(go.Scatter(y=trace[i].detach().cpu().numpy(), mode='lines'))

    return fig

def sweep_llc(
    model: InstancedModule,
    dataset: GroupData,
    eps_l: list[float],
    beta_l: list[float],
    gamma_l: list[float],
    burnin: float=0.6,
    epochs: int=2000,
    chains: int=5,
) -> Float[t.Tensor, 'eps beta gamma']:
    hyparams = list(product(eps_l, beta_l, gamma_l)) * chains
    eps, beta, gamma = zip(*hyparams)
    eps = t.tensor(eps).to(device)
    beta = t.tensor(beta).to(device)
    gamma = t.tensor(gamma).to(device)
    llc, trace = get_llc(
        model,
        dataset,
        eps,
        beta,
        gamma,
        burnin=burnin,
        epochs=epochs,
    )
    llc = einops.rearrange(
        llc, 
        '(chain eps beta gamma) -> eps beta gamma chain',  # chain is outermost ordering. so goes first on LHS
        eps=len(eps_l), beta=len(beta_l), gamma=len(gamma_l), chain=chains
    ).mean(dim=-1)
    trace = einops.rearrange(
        trace,
        '(chain eps beta gamma) epoch -> eps beta gamma chain epoch',
        eps=len(eps_l), beta=len(beta_l), gamma=len(gamma_l), chain=chains
    )
    return llc, trace

def plot_llc_sweep(
    llc: Float[t.Tensor, 'eps beta gamma'],
    eps_l: list[float],
    beta_l: list[float],
    gamma_l: list[float],
) -> go.Figure:
    n_eps, n_beta, n_gamma = llc.shape
    fig = go.Figure()
    for (i, beta), (j, gamma) in product(enumerate(beta_l), enumerate(gamma_l)):
        y = llc[:, i, j]
        fig.add_trace(go.Scatter(
            x=eps_l,
            y=y.detach().cpu().numpy(),
            mode='lines',
            name=f'β={beta:.1e}, γ={gamma:.1e}'
        ))

    fig.update_layout(
        title='LLC sweep',
        xaxis_title='eps',
        yaxis_title='LLC',
        legend_title='β,γ'
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig

def plot_trace_sweep(
    trace: Float[t.Tensor, 'eps beta gamma chain epoch'],
    eps_l: list[float],
    beta_l: list[float],
    gamma_l: list[float],
) -> go.Figure:
    n_eps, n_beta, n_gamma, _, _ = trace.shape
    fig = go.Figure()
    for (i, eps), (j, beta), (k, gamma) in product(enumerate(eps_l), enumerate(beta_l), enumerate(gamma_l)):
        y = trace[i, j, k, 0, :]  # only plot first chain
        if y.isnan().any() or y.median() < 0.001 or y.max() > 100:
            continue
        fig.add_trace(go.Scatter(
            y=y.cpu().numpy(),
            mode='lines',
            name=f'e{eps:.0e}b{beta:.0e}g{gamma:.0e}'
        ))

    fig.update_layout(
        title='LLC sweep trajectories',
        xaxis_title='epochs',
        yaxis_title='loss',
        legend_title='eps,beta,gamma'
    )
    return fig