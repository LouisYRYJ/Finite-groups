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

# def model_dist_res(model1, model2, proj):
#     assert len(model1) == 1 and len(model2) == 1, "must be single instances"
#     ln1, rn1, un1 = model1.get_neurons(squeeze=True)
#     ln2, rn2, un2 = model2.get_neurons(squeeze=True)
#     norm21 = lambda A: A.norm(dim=1).max()  # max 2-norm along neuron dimension
#     norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
#     print('l diff', norm21(ln1 - ln2))
#     print('r diff', norm21(rn1 - rn2))
#     print('u diff', norm22(un1 - un2))
#     print('u diff res', norm22(proj @ (un1 - un2)))
#     print('l norm', norm21(ln1))
#     print('r norm', norm21(rn1))
#     print('u norm', norm22(un1))
#     return norm22(un1) * (norm21(ln1 - ln2) + norm21(rn1 - rn2)) + norm22(proj @ (un2 - un1)) * (norm21(ln2) + norm21(rn2))

@t.no_grad()
def model_dist(model1, model2, p):
    '''
    Upper bound on max_{x, y}||model1(x, y) - model2(x, y)||_p
    '''
    assert p in [2, 'inf']
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons(squeeze=True)
    ln2, rn2, un2 = model2.get_neurons(squeeze=True)
    bias1 = model1.unembed_bias.detach().T
    bias2 = model2.unembed_bias.detach().T
    # max 2-norm along neuron dimension
    # note this is the 1->2 norm for embeddings and 2->inf norm for unembeddings
    # bc of how the transposes are
    norm2inf = lambda A: A.norm(dim=1).max()
    norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
    norm_e = norm2inf
    norm_u = norm22 if p == 2 else norm2inf
    norm_b = (lambda b: b.norm()) if p == 2 else (lambda b: b.abs().max())
    # print('l diff', norm_e(ln1 - ln2))
    # print('r diff', norm_e(rn1 - rn2))
    # print('u diff', norm_u(un1 - un2))
    # print('l norm', norm_e(ln1))
    # print('r norm', norm_e(rn1))
    # print('u norm', norm_u(un1))
    u_diff = norm_u(un2 - un1) * (norm_e(ln2) + norm_e(rn2))
    e_diff = norm_u(un1) * (norm_e(ln1 - ln2) + norm_e(rn1 - rn2))
    bias_diff = norm_b(bias1 - bias2)
    # print('u diff term', u_diff)
    # print('e diff term', e_diff)
    # print('bias diff term', bias_diff)
    return (u_diff + e_diff + bias_diff).item()

@t.no_grad()
def model_dist_xy(model1, model2, p, group=None, collapse_xy=False):
    '''
    [Upper bound on ||model1(x, y) - model2(x, y)||_p for (x, y) in inputs]
    '''
    assert p in [2, 'inf']
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons(squeeze=True)
    ln2, rn2, un2 = model2.get_neurons(squeeze=True)
    if model1.unembed_bias is not None:
        bias1 = model1.unembed_bias.detach().T
    else:
        bias1 = t.zeros(un1.shape[0])
    if model2.unembed_bias is not None:
        bias2 = model2.unembed_bias.detach().T
    else:
        bias2 = t.zeros(un2.shape[0])
    
    # max 2-norm along neuron dimension
    # note this is the 1->2 norm for embeddings and 2->inf norm for unembeddings
    # bc of how the transposes are
    norm2inf = lambda A: A.norm(dim=1).max()
    norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
    norm_u = norm22 if p == 2 else norm2inf
    norm_b = (lambda b: b.norm()) if p == 2 else (lambda b: b.abs().max())
    u1_norm = norm_u(un1).item()
    ures_norm = norm_u(un1 - un2).item()
    bias_diff = norm_b(bias1 - bias2)
    ret = []
    if collapse_xy:
        ret = t.zeros(len(group))
        for i, j in product(range(len(group)), repeat=2):
            ij = group.mult_idx(i, j)
            ret[ij] = max(ret[ij], (u1_norm * (F.relu(ln1[i] + rn1[j]) - F.relu(ln2[i] + rn2[j])).norm()
                + ures_norm * F.relu(ln2[i] + rn2[j]).norm().item() + bias_diff))
    else:
        for i, j in product(range(ln1.shape[0]), repeat=2):
            ret.append(
                (u1_norm * (F.relu(ln1[i] + rn1[j]) - F.relu(ln2[i] + rn2[j])).norm()
                + ures_norm * F.relu(ln2[i] + rn2[j]).norm().item() + bias_diff)
            )
        ret = t.tensor(ret)
    return ret

# def model_dist2(model1, model2):
#     assert len(model1) == 1 and len(model2) == 1, "must be single instances"
#     ln1, rn1, un1 = model1.get_neurons()
#     ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
#     ln2, rn2, un2 = model2.get_neurons()
#     ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
#     return ((un1 - un2).norm(dim=0) * (ln1.max(dim=0).values + rn1.max(dim=0).values)).sum() \
#         + (un2.norm(dim=0) * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).sum()
#         # + un2 * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).abs().sum(dim=1).max()

# def model_dist_inf(model1, model2):
#     assert len(model1) == 1 and len(model2) == 1, "must be single instances"
#     ln1, rn1, un1 = model1.get_neurons()
#     ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
#     ln2, rn2, un2 = model2.get_neurons()
#     ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
#     return ((un1 - un2) * (ln1.max(dim=0).values + rn1.max(dim=0).values) 
#         + un2 * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).abs().sum(dim=1).max()
