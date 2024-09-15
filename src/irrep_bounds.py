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

import sys, os, re

from coset_bounds import model_dist

def get_neuron_irreps(model, group, r2_thresh=0.95, norm_thresh=1e-2):
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    lneurons, rneurons, uneurons= model.get_neurons(squeeze=True)
    irreps = group.get_real_irreps()
    irrep_bases = dict()
    for name, irrep in irreps.items():
        irrep = einops.rearrange(irrep, 'N d1 d2 -> N (d1 d2)')
        U, S, V = t.svd(irrep)
        nonzero = S > 1e-5
        irrep_bases[name] = U[:,nonzero]

    
    # Proportion of variance explained by each irrep, for each neuron
    lexpl, rexpl, uexpl = dict(), dict(), dict()

    for name, basis in irrep_bases.items():
        lexpl[name] = (basis.H @ lneurons).norm(dim=0)**2 / (lneurons.norm(dim=0)**2 + 1e-8)
        rexpl[name] = (basis.H @ rneurons).norm(dim=0)**2 / (rneurons.norm(dim=0)**2 + 1e-8)
        uexpl[name] = (basis.H @ uneurons).norm(dim=0)**2 / (uneurons.norm(dim=0)**2 + 1e-8)

    irrep_idx_dict = {
        name: [
            i for i in range(lneurons.shape[-1]) 
            if lexpl[name][i].item() > r2_thresh and lneurons[:,i].norm() > norm_thresh
        ]
        for name in irreps
    }
    return irreps, irrep_idx_dict

def get_neuron_vecs(model, group, irreps, irrep_idx_dict):
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    lneurons, rneurons, uneurons= model.get_neurons(squeeze=True)

    vecs = dict()
    for name, irrep in irreps.items():
        print(name)
        if not irrep_idx_dict[name]:
            continue
        irrep_lneurons, irrep_rneurons, irrep_uneurons = lneurons[:,irrep_idx_dict[name]], rneurons[:,irrep_idx_dict[name]], uneurons[:,irrep_idx_dict[name]]
        assert np.sign(group.get_frobenius_schur(irrep)).item() == 1, 'Only real irreps supported'
        irrep_d = irrep.shape[-1]
        
        flat_irrep = einops.rearrange(irrep, 'n d1 d2 -> n (d1 d2)')
        # Project neurons onto subspace spanned by flat_irrep
        A_flat = t.linalg.lstsq(flat_irrep, irrep_lneurons, driver='gelsd').solution
        B_flat = t.linalg.lstsq(flat_irrep, irrep_rneurons, driver='gelsd').solution
        C_flat = t.linalg.lstsq(flat_irrep, irrep_uneurons, driver='gelsd').solution

        A = einops.rearrange(A_flat, '(d1 d2) m -> m d1 d2', d1=irrep_d, d2=irrep_d).mH
        B = einops.rearrange(B_flat, '(d1 d2) m -> m d1 d2', d1=irrep_d, d2=irrep_d).mH
        C = einops.rearrange(C_flat, '(d1 d2) m -> m d1 d2', d1=irrep_d, d2=irrep_d).mH

        A_norm = t.linalg.matrix_norm(A)
        B_norm = t.linalg.matrix_norm(B)

        # Normalize A and B to have unit Frobenius norm
        A = A / A_norm.unsqueeze(1).unsqueeze(1)
        B = B / B_norm.unsqueeze(1).unsqueeze(1)
        C = C * ( (A_norm + B_norm) / 2).unsqueeze(1).unsqueeze(1)

        x = einops.rearrange(B @ A, 'm d1 d2 -> m (d1 d2)')
        y = einops.rearrange(C, 'm d1 d2 -> m (d1 d2)')
        coef = (x.conj() * y).sum(dim=-1) / (x.conj() * x).sum(dim=-1)
        yhat = coef.unsqueeze(1) * x
        r2 = (yhat - y).norm(dim=-1).pow(2) / y.norm(dim=-1).pow(2)
        print('1-r2 90th percentile', t.quantile(r2, 0.9).item())

        a, b, c, d = [], [], [], []
        for i in range(A.shape[0]):
            lU, lS, lV = t.linalg.svd(A[i])
            rU, rS, rV = t.linalg.svd(B[i])
            a.append(lU[:,0])
            b.append(lV[0])
            c.append(rU[:,0])
            d.append(rV[0])
        a, b, c, d = t.stack(a, dim=0), t.stack(b, dim=0), t.stack(c, dim=0), t.stack(d, dim=0)
        # Correct sign ambiguity from svd
        a_sign = t.sgn(a[:,0])
        a = t.diag(a_sign) @ a
        b = t.diag(a_sign) @ b
        d_sign = t.sgn(d[:,0])
        c = t.diag(d_sign) @ c
        d = t.diag(d_sign) @ d

        for name, v in zip(['a', 'b', 'c', 'd'], [a, b, c, d]):
            print(f'{name} variance:', ((v - v.mean(dim=0)).norm()**2 / v.norm()**2).item())

        print('a vs d', (a - d).norm()**2 / a.norm()**2)

        full_b = einops.einsum(b, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        full_c = einops.einsum(c, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        b_kmeans, b_clusters, b_losses = cluster(full_b)
        c_kmeans, c_clusters, c_losses = cluster(full_c)
        b_labels, c_labels = t.tensor(b_kmeans.predict(b.numpy())), t.tensor(c_kmeans.predict(c.numpy()))
        b_mean, c_mean = t.tensor(b_kmeans.cluster_centers_), t.tensor(c_kmeans.cluster_centers_)
        print(f'b has {b_clusters} clusters with total loss {b_losses[-1]}')
        print(f'c has {c_clusters} clusters with total loss {c_losses[-1]}')

        # partition clusters into rho-closed sets
        b_remain = set(range(b_clusters))
        b_parts = []
        c_parts = []
        while b_remain:
            i = b_remain.pop()
            b_part = {i}
            b_orbit = einops.einsum(b_mean[[i]], irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
            orbit_labels = t.tensor(b_kmeans.predict(b_orbit.numpy()))
            b_part |= set(orbit_labels.tolist())
            b_parts.append(b_part)
            b_remain -= b_part
            c_parts.append(set(c_labels[t.isin(b_labels, t.tensor(list(b_part)))].tolist()))
        b_parts = list(map(list, b_parts))
        c_parts = list(map(list, c_parts))
        print('b_parts', b_parts)
        print('c_parts', c_parts)
        print('a_mean', a.mean(dim=0))
        print('b_mean', b_mean)
        print('c_mean', c_mean)

        # Check that irrep is G-action on each partition of b's clusters
        for i, b_part in enumerate(b_parts):
            T = einops.einsum(b_mean[b_part], irrep, b_mean[b_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            T = (T > 1 - 1e-2).float()
            if not (T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all():
                print(f'Rho is not permutation on partition {i} of b!!!!')

        # Check that irrep is G-action on each partition of c's clusters
        for i, c_part in enumerate(c_parts):
            T = einops.einsum(c_mean[c_part], irrep, c_mean[c_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            T = (T > 1 - 1e-2).float()
            if not (T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all():
                print(f'Rho is not permutation on partition {i} of c!!!!')

        # Check that {b_i} = {-c_i} within each partition
        for b_part, c_part in zip(b_parts, c_parts):
            S = b_mean[b_part] @ -c_mean[c_part].T
            S = (S > 1 - 1e-2).float()
            if not (S.sum(axis=0) == 1).all() and (S.sum(axis=1) == 1).all():
                print(f'(b_i) != (-c_i) within partition {b_part},{c_part}!!!!')

        # check that coefs are uniform over {b_i}x{c_i}
        for b_part, c_part in zip(b_parts, c_parts):
            coef_sum = t.tensor([
                coef[(b_labels == i) & (c_labels == j)].sum().item()
                for i, j in product(b_part, c_part)
            ])
            print(f'part{b_part} coefs: norm={coef_sum.norm()}, var={(coef_sum - coef_sum.mean()).norm()**2/coef_sum.norm()**2}')
            print(coef_sum)
            
        
        vecs[name] = (coef, a.mean(dim=0), b_mean, c_mean, b_labels, c_labels, b_parts, c_parts)
        print()
    return vecs
            
def get_idealized_model(model, group, irreps, irrep_idx_dict, vecs):
    pass

def irrep_bound(model, group, irreps, irrep_idx_dict, vecs):
    pass