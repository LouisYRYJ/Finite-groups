import torch as t
import numpy as np
from matplotlib import pyplot as plt
import json
from itertools import product
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import copy
import math
from itertools import product
import pandas as pd
from typing import Union
from einops import repeat
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from collections import defaultdict


from model import MLP3, MLP4, InstancedModule
from utils import *
from group_data import *
from model_utils import *
from group_utils import *
from bound_utils import *

import time

import sys, os, re
import argparse

def part_var(x, part, dim):
    '''
    Computes E[var(x | part)] / var(x), where part is a partition of x along dim.
    By the law of total variance, this quantity is in [0, 1].
    Stander et al instead compute (\sum_part var(x | part)) / var(x), which ranges in [0, len(part)].
    '''
    part = list(map(list, part))
    assert sorted(sum(part, [])) == list(range(x.shape[dim])), "part must be a partition of x along dim"
    denom = x.var(dim=dim, correction=0)
    num = sum(
        x.index_select(dim, t.Tensor(p).int().to(x.device)).var(dim=dim, correction=0) * (len(p) / x.shape[dim])
        for p in part
    )
    return (num / denom).nan_to_num(nan=0.)

def get_neuron_subgroups(model, group, thresh=0.01):
    '''
    Return [subgroup H st that neuron is approx constant on G/H for neuron in model]
    '''
    subgroups = group.get_subgroups_idx()
    left_cosets = {
        name: group.get_cosets_idx(subgroup, left=True)
        for name, subgroup in subgroups.items()
    }
    right_cosets = {
        name: group.get_cosets_idx(subgroup, left=False)
        for name, subgroup in subgroups.items()
    }

    assert len(model) == 1, "model must be a single instance"
    left_neurons, right_neurons, _ = model.get_neurons(squeeze=True)

    # hypothesis is that left neurons are constant on *right* cosets
    left_coset_vars = {
        name: part_var(left_neurons, right_cosets[name], dim=0)
        for name in right_cosets.keys()
    }

    # and right neurons are constant on *left* cosets
    right_coset_vars = {
        name: part_var(right_neurons, left_cosets[name], dim=0)
        for name in left_cosets.keys()
    }

    left_neuron_subgroups = [None] * left_neurons.shape[1]
    for i in range(left_neurons.shape[1]):
        left_neuron_subgroups[i] = None
        for name, subgroup in subgroups.items():
            if left_coset_vars[name][i] < thresh:
                if left_neuron_subgroups[i] is None or len(subgroups[left_neuron_subgroups[i]]) < len(subgroup):
                    left_neuron_subgroups[i] = name

    right_neuron_subgroups = [None] * right_neurons.shape[1]
    for i in range(right_neurons.shape[1]):
        right_neuron_subgroups[i] = None
        for name, subgroup in subgroups.items():
            if right_coset_vars[name][i] < thresh:
                if right_neuron_subgroups[i] is None or len(subgroups[right_neuron_subgroups[i]]) < len(subgroup):
                    right_neuron_subgroups[i] = name

    conjugators = []
    for i in range(len(left_neuron_subgroups)):
        left_subgroup = subgroups[left_neuron_subgroups[i]]
        right_subgroup = subgroups[right_neuron_subgroups[i]]
        found_conjugate = False
        for g in range(len(group)):
            if set(right_subgroup) == group.get_conj_subgroup_idx(left_subgroup, g):
                conjugators.append(g)
                found_conjugate = True
                break
        # try:
        assert found_conjugate, f"left and right subgroups {left_neuron_subgroups[i]} and {right_neuron_subgroups[i]} not conjugate for neuron {i}"
        # except:
            # import pdb; pdb.set_trace()
    
    return left_neuron_subgroups, right_neuron_subgroups, conjugators, subgroups, left_cosets, right_cosets

def coset_ideal_model(model, group, left_neuron_subgroups, right_neuron_subgroups, conjugators, subgroups, left_cosets, right_cosets):
    '''
    Return a new model where each neuron is replaced by its average over the cosets of its subgroup
    '''
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    if model.unembed_bias is not None:
        print('Warning: bias present. Will be zeroed in ideal model.')
    ln, rn, un = model.get_neurons(True)

    ideal_ln, ideal_rn, ideal_un = t.zeros_like(ln), t.zeros_like(rn), t.zeros_like(un)
    for i in range(ln.shape[-1]):
        for coset in right_cosets[left_neuron_subgroups[i]]:
            ideal_ln[list(coset), i] = ln[list(coset), i].mean()
        for coset in left_cosets[right_neuron_subgroups[i]]:
            ideal_rn[list(coset), i] = rn[list(coset), i].mean()

    for i in range(un.shape[-1]):
        Hginv = list(set(group.mult_idx(a, group.inv_idx(conjugators[i])) for a in subgroups[left_neuron_subgroups[i]]))
        not_Hginv = list(set(range(len(group))) - set(Hginv))
        if not Hginv or not not_Hginv:
            continue
        not_Hginv_mean = un[not_Hginv, i].mean()
        ideal_un[Hginv, i] = t.minimum(not_Hginv_mean, un[Hginv, i])
        ideal_un[not_Hginv, i] = not_Hginv_mean


    # print('ln 1-r2', ((ln - ideal_ln).norm()**2 / ln.norm()**2).item())
    # print('rn 1-r2', ((rn - ideal_rn).norm()**2 / rn.norm()**2).item())
    # print('un 1-r2', ((un - ideal_un).norm()**2 / un.norm()**2).item())
    
    new_model = copy.deepcopy(model)
    new_model.embedding_left = nn.Parameter(ideal_ln.unsqueeze(0))
    new_model.embedding_right = nn.Parameter(ideal_rn.unsqueeze(0))
    new_model.unembedding = nn.Parameter(ideal_un.unsqueeze(0).mT)
    new_model.unembed_bias = None
    return new_model

def coset_bound(model, group, left_neuron_subgroups, right_neuron_subgroups, conjugators, subgroups, left_cosets, right_cosets):
    '''
    neuron_subgroups = {neuron: subgroup_name}
    subgroups = {subgroup_name: subgroup_idxs}
    '''
    t0 = time.time()
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()

    for subgroup in subgroups.values():
        assert group.is_subgroup_idx(subgroup), f"{subgroup} is not a subgroup of {group}"

    ideal_model = coset_ideal_model(model, group, left_neuron_subgroups, right_neuron_subgroups, conjugators, subgroups, left_cosets, right_cosets)
    ln, rn, un = ideal_model.get_neurons(True)

    # compute s(H, g) and H*g^-1 for all H, g
    Hg_to_neurons = defaultdict(list)
    Hginv = dict()
    for i in range(ln.shape[-1]):
        H = left_neuron_subgroups[i]
        g = conjugators[i]
        Hg_to_neurons[(H, g)].append(i)
        Hginv[H, g] = set(group.mult_idx(a, group.inv_idx(g)) for a in subgroups[H])
        
    s = dict()
    for (H, g), neurons in Hg_to_neurons.items():
        s[(H, g)] = t.inf
        # TODO: this can be |G/H|^2 instead of |G|^2 because of coset concentration
        for x_coset, y_coset in product(right_cosets[H], left_cosets[H]):
            x = set(x_coset).pop()
            y = set(y_coset).pop()
            if group.mult_idx(x, y) in Hginv[(H, g)]:
                continue
            s[(H, g)] = min(
                s[(H, g)],
                sum(F.relu(ln[x, i] + rn[y, i]) for i in neurons)
            )

    # print(s)

    # compute margins
    # print('hi')
    margins = t.zeros(len(group))
    for z in range(len(group)):
        margins[z] = t.inf
        # print('z', z)
        for zp in range(len(group)):
            if z == zp:
                continue
            zp_margin = 0.
            for (H, g), neurons in Hg_to_neurons.items():
                if z in Hginv[(H, g)]:
                    continue
                zp_margin += s[(H, g)] * sum(un[z, i] - un[zp, i] for i in neurons)
                assert zp_margin.item() >= -1e-8, f"zp_margin = {zp_margin.item()} < 0"
            # print('   zp_margin', zp_margin)
            margins[z] = min(margins[z], zp_margin)

    # print(margins)

    # inf dist from orig to ideal model
    dist = model_dist_xy(model, ideal_model, 'inf', group=group, collapse_xy=True)
    return (dist < margins).float().mean(), dist, margins, time.time() - t0
    
    
    
    

# def coset_bound(model, group, left_neuron_subgroups, right_neuron_subgroups, subgroups):
#     '''
#     neuron_subgroups = {neuron: subgroup_name}
#     subgroups = {subgroup_name: subgroup_idxs}
#     '''
#     assert len(model) == 1, "model must be a single instance"
#     if not isinstance(model, MLP4):
#         model = model.fold_linear()

#     for subgroup in subgroups.values():
#         assert group.is_subgroup_idx(subgroup), f"{subgroup} is not a subgroup of {group}"
#     left_cosets = {
#         name: list(group.get_cosets_idx(subgroup, left=True))
#         for name, subgroup in subgroups.items()
#     }
#     right_cosets = {
#         name: list(group.get_cosets_idx(subgroup, left=False))
#         for name, subgroup in subgroups.items()
#     }
    
#     avg_model = coset_avg_model(model, group, left_neuron_subgroups, right_neuron_subgroups, left_cosets, right_cosets)
#     ln, rn, un = avg_model.get_neurons()
#     ln, rn, un = ln.squeeze(0), rn.squeeze(0), un.squeeze(0)
#     # sum_phi = 0
#     sum_loss = 0
#     m = ln.shape[-1]
#     for i in tqdm(range(m)):
#         left_subgroup = left_neuron_subgroups[i]
#         right_subgroup = right_neuron_subgroups[i]
#         loss = 0
#         for right, left in product(right_cosets[right_subgroup], left_cosets[left_subgroup]):
#             x, y = set(right).pop(), set(left).pop()   # need to cast from frozenset to set to pop
#             output = m * (un[:, i] * max(0, (ln[x, i] + rn[y, i]).item()))
#             loss += F.cross_entropy(output.unsqueeze(0), t.tensor([group.mult_idx(x, y)]))
#         loss /= len(right_cosets[right_subgroup]) * len(left_cosets[left_subgroup])
#         sum_loss += loss
            
        
#         # double_cosets = group.get_double_cosets_idx(subgroups[left_subgroup], subgroups[right_subgroup])
#         # to_double_coset = [
#         #     [j for j, double in enumerate(double_cosets) if i in double][0]
#         #     for i in range(len(group))
#         # ]

#         # phi = t.zeros(len(group))
#         # for a in range(len(group)):
#         #     if a == group.identity_idx():  # take min over x, y, z st a=x^-1 z y^-1
#         #         phi[a] = t.inf
#         #     else:  # take max over x, y, z st a=x^-1 z y^-1
#         #         phi[a] = -t.inf
#         #     for right, left in product(right_cosets[right_subgroup], left_cosets[left_subgroup]):
#         #         x, y = set(right).pop(), set(left).pop()
#         #         z = group.mult_idx(x, group.mult_idx(a, y))
#         #         output = max(0, ln[x, i] + rn[y, i]) * un[z, i]
#         #         if a == group.identity_idx():
#         #             phi[a] = min(phi[a], output)
#         #         else:
#         #             phi[a] = max(phi[a], output)
#         # sum_phi += phi
#         # min_phi = defaultdict(lambda: t.inf)
#         # max_phi = defaultdict(lambda: -t.inf)
#         # print('un', un[:, i])
#         # for right, left in product(right_cosets[right_subgroup], left_cosets[left_subgroup]):
#         #     x, y = set(right).pop(), set(left).pop()   # need to cast from frozenset to set to pop
#         #     for z in range(len(group)):
#         #         z_inv = group.inv_idx(z)
#         #         double_idx = to_double_coset[group.mult_idx(x, group.mult_idx(z_inv, y))] # double coset containing x * z^-1 * y
#         #         output = max(0, ln[x, i] + rn[y, i]) * un[z, i]
#         #         min_phi[double_idx] = min(min_phi[double_idx], output)
#         #         max_phi[double_idx] = max(max_phi[double_idx], output)
#         # phi = t.tensor([max_phi[to_double_coset[i]] for i in range(len(group))])
#         # id = group.identity_idx()
#         # id_double = [i for i, double in enumerate(double_cosets) if id in double][0] # double coset containing the identity
#         # phi[id] = min_phi[id_double]
#         # sum_phi += phi
#         # print(left_subgroup, right_subgroup)
#         # print(phi)
#         # print()
#     sum_loss /= m
#     return sum_loss, model_dist(model, avg_model)
#     # print(sum_phi)
#     # loss = -t.log(t.exp(sum_phi[0]) / t.exp(sum_phi).sum())
#     # loss_grad_norm = t.sqrt((t.exp(sum_phi[1:]).sum()**2 + t.exp(2*sum_phi[1:]).sum()) / (t.exp(sum_phi).sum()**2))
#     # M = model_dist(model, avg_model)
#     # print('phi loss', loss)
#     # print('M', M)
#     # print('loss grad norm', loss_grad_norm)
#     # return loss + M * loss_grad_norm + M**2/2
                
                
        
        
    