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

from coset_bounds import model_dist

import sys, os, re

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
    for irrep_name, irrep in irreps.items():
        print(irrep_name)
        if not irrep_idx_dict[irrep_name]:
            continue
        irrep_lneurons, irrep_rneurons, irrep_uneurons = lneurons[:,irrep_idx_dict[irrep_name]], rneurons[:,irrep_idx_dict[irrep_name]], uneurons[:,irrep_idx_dict[irrep_name]]
        assert np.sign(group.get_frobenius_schur(irrep)).item() == 1, 'Only real irreps supported'
        d_irrep = irrep.shape[-1]
        
        flat_irrep = einops.rearrange(irrep, 'n d1 d2 -> n (d1 d2)')
        # Project neurons onto subspace spanned by flat_irrep
        A_flat = t.linalg.lstsq(flat_irrep, irrep_lneurons, driver='gelsd').solution
        B_flat = t.linalg.lstsq(flat_irrep, irrep_rneurons, driver='gelsd').solution
        C_flat = t.linalg.lstsq(flat_irrep, irrep_uneurons, driver='gelsd').solution

        A = einops.rearrange(A_flat, '(d1 d2) m -> m d1 d2', d1=d_irrep, d2=d_irrep).mH
        B = einops.rearrange(B_flat, '(d1 d2) m -> m d1 d2', d1=d_irrep, d2=d_irrep).mH
        C = einops.rearrange(C_flat, '(d1 d2) m -> m d1 d2', d1=d_irrep, d2=d_irrep).mH

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
            uU, uS, uV = t.linalg.svd(C[i])
            a.append(lU[:,0])
            # b.append(lV[0])
            bi = uV[0] * t.dot(lV[0], uV[0])
            b.append(bi / bi.norm())
            ci = uU[:,0] * t.dot(rU[:,0], uU[:,0])
            c.append(ci / ci.norm())
            # c.append(rU[:,0])
            d.append(rV[0])
        a, b, c, d = t.stack(a, dim=0), t.stack(b, dim=0), t.stack(c, dim=0), t.stack(d, dim=0)
        # Correct sign ambiguity from svd
        a_sign = t.sgn(a[:,0])
        a = t.diag(a_sign) @ a
        b = t.diag(a_sign) @ b
        d_sign = t.sgn(d[:,0])
        c = t.diag(d_sign) @ c
        d = t.diag(d_sign) @ d

        for vec_name, v in zip(['a', 'b', 'c', 'd'], [a, b, c, d]):
            print(f'{vec_name} variance:', ((v - v.mean(dim=0)).norm()**2 / v.norm()**2).item())

        print('a vs d', (a - d).norm()**2 / a.norm()**2)

        full_b = einops.einsum(b, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        full_c = einops.einsum(c, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        b_kmeans, b_clusters, b_losses = cluster(full_b, max=d_irrep*2+2)
        c_kmeans, c_clusters, c_losses = cluster(full_c, max=d_irrep*2+2)
        b_labels, c_labels = t.tensor(b_kmeans.predict(b.numpy())), t.tensor(c_kmeans.predict(c.numpy()))
        full_b_labels, full_c_labels = t.tensor(b_kmeans.predict(full_b.numpy())), t.tensor(c_kmeans.predict(full_c.numpy()))
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
            c_parts.append(set(full_c_labels[t.isin(full_b_labels, t.tensor(list(b_part)))].tolist()))
        b_parts = list(map(list, b_parts))
        c_parts = list(map(list, c_parts))
        print('b_parts', b_parts)
        print('c_parts', c_parts)
        print('a_mean', a.mean(dim=0))

        # Check that irrep is G-action on each partition of b's clusters
        for i, b_part in enumerate(b_parts):
            T = einops.einsum(b_mean[b_part], irrep, b_mean[b_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            T = (T > 1 - 1e-2).int()
            assert (T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all(), f'Rho is not permutation on partition {i} of b!!!!'

        # Check that irrep is G-action on each partition of c's clusters
        for i, c_part in enumerate(c_parts):
            T = einops.einsum(c_mean[c_part], irrep, c_mean[c_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            T = (T > 1 - 1e-2).int()
            assert (T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all(), f'Rho is not permutation on partition {i} of c!!!!'

        # Check that {b_i} = {-c_i} within each partition
        for b_part, c_part in zip(b_parts, c_parts):
            S = b_mean[b_part] @ -c_mean[c_part].T
            S = (S > 1 - 1e-2).int()
            assert (S.sum(axis=0) == 1).all() and (S.sum(axis=1) == 1).all(), print(f'(b_i) != (-c_i) within partition {b_part},{c_part}!!!!')
            # replace c_j with corresponding -b_i
            for j in range(len(c_part)):
                i = S[:,j].tolist().index(1)
                c_mean[c_part[j]] = -b_mean[b_part[i]]
                print(f'replacing c_{c_part[j]} with -b_{b_part[i]}')

        print('b_mean', b_mean)
        print('c_mean', c_mean)

        unif_coef = t.zeros_like(coef)
        if d_irrep == 1:
            # Hardcoded for sign irrep. TODO: Support for general complex 1d irreps
            # Normalize st sum over (0,0) and (1,1) is the same
            # and st sum over (0,1) and (1,0) is the same
            mask00 = (b_labels == 0) & (c_labels == 0)
            mask01 = (b_labels == 0) & (c_labels == 1)
            mask10 = (b_labels == 1) & (c_labels == 0)
            mask11 = (b_labels == 1) & (c_labels == 1)
            mean1 = (coef[mask00].sum() + coef[mask11].sum()) / 2
            mean2 = (coef[mask01].sum() + coef[mask10].sum()) / 2
            unif_coef[mask00] = coef[mask00] * mean1 / coef[mask00].sum()
            unif_coef[mask11] = coef[mask11] * mean1 / coef[mask11].sum()
            unif_coef[mask01] = coef[mask01] * mean2 / coef[mask01].sum()
            unif_coef[mask10] = coef[mask10] * mean2 / coef[mask10].sum()
        else:
            for b_part, c_part in zip(b_parts, c_parts):
                # Rescale such that sum of coefs over each (i, j) pair is the same
                coef_sum = t.tensor([
                    coef[(b_labels == i) & (c_labels == j)].sum().item()
                    for i, j in product(b_part, c_part)
                ])
                coef_mean = coef_sum.mean()
                for i, j in product(b_part, c_part):
                    ij_mask = (b_labels == i) & (c_labels == j)
                    if not ij_mask.any():
                        print(f'no neurons corresponding to ({i},{j}) pair! zeroing partition!')
                        coef_mean = 0.
                print(b_part, 'coef_sum', coef_sum)
                print(b_part, 'coef_mean', coef_mean)
                for i, j in product(b_part, c_part):
                    ij_mask = (b_labels == i) & (c_labels == j)
                    ij_sum = coef[ij_mask].sum()
                    # print('ij', i, j)
                    # print('coef', coef[ij_mask])
                    # print('coef_mean', coef_mean)
                    # print('ij_sum', ij_sum)
                    unif_coef[ij_mask] = coef[ij_mask] * coef_mean / ij_sum
        print('b_labels', b_labels)
        print('c_labels', c_labels)
        # print('coef', coef)
        # print('unif_coef', unif_coef)
        print('coef diff', (coef - unif_coef).norm()**2 / coef.norm()**2)
        
        vecs[irrep_name] = (unif_coef, (A_norm + B_norm) / 2, a.mean(dim=0), b_mean, c_mean, b_labels, c_labels, b_parts, c_parts)
        print(vecs.keys())
        print()
    return vecs
            
def get_idealized_model(model, irreps, irrep_idx_dict, vecs):
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    lneurons, rneurons, uneurons = model.get_neurons(squeeze=True)
    new_ln, new_rn, new_un = t.zeros_like(lneurons), t.zeros_like(rneurons), t.zeros_like(uneurons)
    for irrep_name, (coef, A_norm, a_mean, b_mean, c_mean, b_labels, c_labels, b_parts, c_parts) in vecs.items():
        print(irrep_name)
        irrep_idxs = irrep_idx_dict[irrep_name]
        b = b_mean[b_labels]
        c = c_mean[c_labels]
        irrep_ln = einops.einsum(b, irreps[irrep_name], a_mean, 'm d1, G d1 d2, d2 -> G m')# * A_norm
        irrep_rn = einops.einsum(a_mean, irreps[irrep_name], c, 'd1, G d1 d2, m d2 -> G m')# * A_norm
        irrep_un = coef * einops.einsum(b, irreps[irrep_name], c, 'm d1, G d1 d2, m d2 -> G m')# / A_norm
        for i in range(irrep_ln.shape[1]):
            if irrep_un[:, i].norm() > 1e-8:
                # degree of freedom in scaling u and l/r proportionally
                # use this to match u to original norm
                ucoef = t.dot(irrep_un[:,i], uneurons[:,irrep_idxs[i]]) / t.dot(irrep_un[:,i], irrep_un[:,i])
                irrep_un[:, i] *= ucoef
                irrep_ln[:, i] /= ucoef
                irrep_rn[:, i] /= ucoef
            else:
                #un has been zeroed out
                irrep_un[:, i] = 0#uneurons[:, irrep_idxs[i]]
                irrep_ln[:, i] = lneurons[:, irrep_idxs[i]]
                irrep_rn[:, i] = rneurons[:, irrep_idxs[i]]
        print('l diff', (irrep_ln - lneurons[:,irrep_idxs]).norm()**2 / lneurons[:,irrep_idxs].norm()**2)
        print('r diff', (irrep_rn - rneurons[:,irrep_idxs]).norm()**2 / rneurons[:,irrep_idxs].norm()**2)
        print('u diff', (irrep_un - uneurons[:,irrep_idxs]).norm()**2 / uneurons[:,irrep_idxs].norm()**2)
        new_ln[:,irrep_idxs] = irrep_ln
        new_rn[:,irrep_idxs] = irrep_rn
        new_un[:,irrep_idxs] = irrep_un

    print('total')
    print('l diff', (new_ln - lneurons).norm()**2 / lneurons.norm()**2)
    print('r diff', (new_rn - rneurons).norm()**2 / rneurons.norm()**2)
    print('u diff', (new_un - uneurons).norm()**2 / uneurons.norm()**2)
    ret = copy.deepcopy(model)
    ret.embedding_left = nn.Parameter(new_ln.unsqueeze(0))
    ret.embedding_right = nn.Parameter(new_rn.unsqueeze(0))
    ret.unembedding = nn.Parameter(new_un.unsqueeze(0).mT)
    return ret

def model_dist_parted(model1, model2, irrep_idx_dict, vecs):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons()
    ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
    ln2, rn2, un2 = model2.get_neurons()
    ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
    M = 0
    norm21 = lambda A: A.norm(dim=0).max()
    norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
    for irrep_name, (coef, A_norm, a_mean, b_mean, c_mean, b_labels, c_labels, b_parts, c_parts) in vecs.items():
        irrep_idxs = irrep_idx_dict[irrep_name]
        for b_part in b_parts:
            part_idxs = t.tensor(irrep_idxs)[t.isin(b_labels, t.tensor(b_part))]
            part_ln1, part_rn1, part_un1 = ln1[:,part_idxs], rn1[:,part_idxs], un1[:,part_idxs]
            part_ln2, part_rn2, part_un2 = ln2[:,part_idxs], rn2[:,part_idxs], un2[:,part_idxs]
            part_M = norm22(part_un1) * (norm21(part_ln1 - part_ln2) + norm21(part_rn1 - part_rn2)) \
                        + norm22(part_un2 - part_un1) * (norm21(part_ln2) + norm21(part_rn2))
            M += part_M.item()
            print(irrep_name)
            print('l diff', norm21(part_ln1 - part_ln2))
            print('r diff', norm21(part_rn1 - part_rn2))
            print('u diff', norm22(part_un1 - part_un2))
            print('l norm', norm21(part_ln1))
            print('r norm', norm21(part_rn1))
            print('u norm', norm22(part_un1))
            print(part_M.item())
    return M

def model_dist_inf(model1, model2):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons()
    ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
    ln2, rn2, un2 = model2.get_neurons()
    ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
    ret = 0
    for i in range(ln1.shape[1]):
        ret += (un1[:,i] - un2[:,i]) * (ln1[:,i])
    return ((un1 - un2) * (ln1.max(dim=0).values + rn1.max(dim=0).values) 
        + un2 * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).abs().sum(dim=1).max()

def irrep_bound(model, group, irreps, irrep_idx_dict, vecs):
    pass