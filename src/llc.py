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
from typing import Union, Callable
from einops import repeat
from torch.utils.data import DataLoader, RandomSampler
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import copy
import math
from itertools import product
import plotly.subplots as sb

def sgld_trace(
    model: InstancedModule,
    dataset: GroupData,
    eps: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    beta: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    gamma: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    epochs: int=2000,
    instances: int=1,
    floor: Union[Float[t.Tensor, 'instance'], float]=0.,
    pos_func: Callable=lambda x: x.abs(),
    tq: bool=True,
    batch_size=-1,
    ibatch_size: int=-1,
    replacement: bool=False,
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

    if batch_size < 0:
        batch_size = len(dataset)
    
    sampler = RandomSampler(dataset, replacement=replacement)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
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
            output = model(x, ibatch_size=ibatch_size)
            loss = get_cross_entropy(output, z)
            epoch_loss += loss
            # (loss - floor).abs().sum().backward()
            pos_func(loss - floor).sum().backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    import pdb; pdb.set_trace()
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
    positive: bool=False,
    pos_func: Callable=lambda x: x.abs(),
) -> Float[t.Tensor, 'instance']:
    start = int(burnin * trace.shape[1])
    if positive:
        orig_loss = einops.repeat(orig_loss, 'instance -> instance n', n=trace.shape[1])
        # return beta * (trace - orig_loss).abs().mean(dim=1)
        return beta * pos_func(trace - orig_loss).mean(dim=1)
    else:
        return beta * (trace[:,start:].mean(dim=1) - orig_loss)

def get_llc(
    model: InstancedModule,
    dataset: GroupData,
    eps: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    beta: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    gamma: Union[Float[t.Tensor, 'instance'], Float[t.Tensor, '1'], float],
    chains: int=5,
    burnin: float=0.6,
    epochs: int=2000,
    positive: bool=False,
    parallel_chain: bool=False,
    tq: bool=True,
    ibatch_size: int=-1,
    batch_size: int=-1,
    replacement: bool=False,
    pos_func: Callable=lambda x: x.abs(),
) -> Float[t.Tensor, 'instance']:
    orig_loss = full_train_loss(model, dataset)
    floor = orig_loss.detach() if positive else 0.
    if parallel_chain:
        model = model.stack([model for _ in range(chains)])
        trace = sgld_trace(
            model, dataset, eps, beta, gamma, floor=floor,
            epochs=epochs, tq=tq, ibatch_size=ibatch_size, replacement=replacement,
            batch_size=batch_size, pos_func=pos_func,
        )
        trace = einops.rearrange(trace, '(chain instance) ... -> chain instance ...', chain=chains)
        trace = trace.mean(dim=0)
    else:
        traces = []
        for _ in range(chains):
            traces.append(
                sgld_trace(
                    model, dataset, eps, beta, gamma, floor=floor,
                    epochs=epochs, tq=tq, ibatch_size=ibatch_size, replacement=replacement,
                    batch_size=batch_size, pos_func=pos_func,
                )
            )
        # mean over chains
        trace = sum(traces) / len(traces)
    return llc_from_trace(trace, orig_loss, beta, burnin=burnin, positive=positive, pos_func=pos_func), trace

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
    replacement: bool=False,
    positive: bool=False,
    batch_size: int=-1,
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
        replacement=replacement,
        positive=positive,
        chains=1,   # model is already *chains, so just 1 chain here.
        batch_size=batch_size,
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
    skip_bad: bool=False,
) -> go.Figure:
    n_eps, n_beta, n_gamma, _, _ = trace.shape
    fig = go.Figure()
    ncols = len(eps_l)
    nrows = len(beta_l) * len(gamma_l)
    fig = sb.make_subplots(rows=nrows, cols=ncols)

    for (i, eps), (j, beta), (k, gamma) in product(enumerate(eps_l), enumerate(beta_l), enumerate(gamma_l)):
        for c in range(trace.shape[3]): # iterate over chains
            y = trace[i, j, k, c, :] 
            if skip_bad and (y.isnan().any() or y.median() < 0.001 or y.max() > 100):
                continue
            row = k*len(beta_l)+j+1
            col = i+1
            fig.add_trace(
                go.Scatter(
                    y=y.cpu().numpy(),
                    mode='lines',
                    name='loss',
                    legendgroup='loss',
                    showlegend= (row==1 and col==1),
                    # name=f'e{eps:.0e}b{beta:.0e}g{gamma:.0e}'
                ),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=f'e{eps:.0e} b{beta:.0e} g{gamma:.0e}', row=row, col=col)

    fig.update_layout(
        title='LLC sweep trajectories',
        height=300*nrows, 
        width=300*ncols,
        # xaxis_title='epochs',
        # yaxis_title='loss',
        # legend_title='eps,beta,gamma'
    )
    return fig