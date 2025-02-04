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
import einx


from model import MLP3, MLP4, InstancedModule
from utils import *
from group_data import *
from model_utils import *
from group_utils import *
from bound_utils import *

import time

# from coset_bounds import model_dist

import sys, os, re

CLUSTER_THRESH = 5e-3
MAX_CLUSTERS = 2

def check_rho_set(irrep, X):
    irrep_X = einops.einsum(irrep, X, 'group d1 d2, k d2 -> group k d1')
    X = X.unsqueeze(0)
    dist = (irrep_X - X).norm(dim=-1)
    assert (dist < 1e-3).any(dim=0).all(), 'Not a rho-set!'


def get_Xhat(irrep, X_mean, rho_labels, k_labels):
    full_means = einops.einsum(irrep, X_mean, 'group d1 d2, k d2 -> group k d1')
    labels = t.concat([rho_labels.unsqueeze(-1), k_labels.unsqueeze(-1)], dim=-1)
    Xhat = einx.get_at('[group k] d, n [2] -> n d', full_means, labels)
    return Xhat

@jaxtyped(typechecker=beartype)
def irrep_kmeans(
    irrep: Float[t.Tensor, 'group d d'], 
    X: Float[t.Tensor, 'n d'],
    n_clusters: int,
    means: Any = None,
    outlier_zscore: int= 2,
) -> Any:
    if means is None:
        means = t.randn(n_clusters, irrep.shape[-1])
    prev_err = 10000
    while True:
        # assign labels
        full_means = einops.einsum(irrep, means, 'group d1 d2, k d2 -> group k d1')
        full_means = full_means.unsqueeze(0)   # 1 group k d
        full_X = X.unsqueeze(1).unsqueeze(1)   # n 1 1 d
        dists = (full_means - full_X).norm(dim=-1)   # n group k
        dists_shape = dists.shape
        dists_flat = dists.flatten(1, 2)  # n group * k
        min_dist, labels = dists_flat.min(dim=-1)
        rho_labels, k_labels = t.unravel_index(labels, dists_shape[1:])

        # update means
        # TODO: check irrep is unitary
        rho_inv_X = einops.einsum(irrep.mH, X, 'group d1 d2, n d2 -> group n d1')  # assume irrep is unitary so irrep.mH is inverse
        rho_inv_X = einx.get_at('[group] n d, n [1] -> n d', rho_inv_X, rho_labels.unsqueeze(-1))
        for k in range(n_clusters):
            means[k] = rho_inv_X[k_labels == k].mean(dim=0)
            # remove outliers
            if outlier_zscore is not None:
                dists = (rho_inv_X[k_labels == k] - means[k]).norm(dim=-1)**2
                std = dists.mean().sqrt()
                means[k] = rho_inv_X[k_labels == k][dists <= outlier_zscore * std].mean(dim=0)
            if t.isnan(means[k]).any():
                means[k] = t.randn(irrep.shape[-1])

        err = min_dist.mean()
        if err > prev_err - 1e-4:
            break
        prev_err = err
        # print(err)
        if t.isnan(err).any():
            import pdb; pdb.set_trace()

    return means, rho_labels, k_labels, err
        

@t.no_grad()
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

def get_neuron_vecs(model, group, irreps, irrep_idx_dict, verbose=False, num_clusters=None, stab_thresh=0.3):
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    lneurons, rneurons, uneurons= model.get_neurons(squeeze=True)

    vecs = dict()
    max_avar = 0
    for irrep_name, irrep in irreps.items():
        if verbose:
            print(irrep_name)
        if not irrep_idx_dict[irrep_name]:
            continue
        if np.sign(group.get_frobenius_schur(irrep)).item() <= 0: 
            print(f'{irrep_name} has Frobenius-Schur indicator <= 0, not currently supported. Skipping!')
            continue
        irrep_lneurons, irrep_rneurons, irrep_uneurons = lneurons[:,irrep_idx_dict[irrep_name]], rneurons[:,irrep_idx_dict[irrep_name]], uneurons[:,irrep_idx_dict[irrep_name]]
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
        if verbose and ((A_norm - B_norm).abs() / A_norm).max() > 1e-1:
            print('A and B have different norm!')

        # Normalize A and B to have unit Frobenius norm
        A = A / A_norm.unsqueeze(1).unsqueeze(1)
        B = B / B_norm.unsqueeze(1).unsqueeze(1)
        C = C * ( (A_norm + B_norm) / 2).unsqueeze(1).unsqueeze(1)

        x = einops.rearrange(B @ A, 'm d1 d2 -> m (d1 d2)')
        y = einops.rearrange(C, 'm d1 d2 -> m (d1 d2)')
        coef = (x.conj() * y).sum(dim=-1) / (x.conj() * x).sum(dim=-1)
        yhat = coef.unsqueeze(1) * x
        r2 = (yhat - y).norm(dim=-1).pow(2) / y.norm(dim=-1).pow(2)
        if verbose:
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

        if verbose:
            for vec_name, v in zip(['a', 'b', 'c', 'd'], [a, b, c, d]):
                print(f'{vec_name} variance:', ((v - v.mean(dim=0)).norm()**2 / v.norm()**2).item())

            print('a vs d', (a - d).norm()**2 / a.norm()**2)

        if a.shape[0] > 10:
            max_avar = max(max_avar, ((a - a.mean(dim=0)).norm()**2 / a.norm()**2).item())

        # full_b = einops.einsum(b, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        # full_c = einops.einsum(c, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        # max_clusters = 12 if d_irrep == 3 else d_irrep*2+2    # TODO: figure out something better
        # b_kmeans, b_clusters, b_losses = cluster(full_b, verbose=verbose, max=max_clusters)
        # c_kmeans, c_clusters, c_losses = cluster(full_c, verbose=verbose, max=max_clusters)
        # b_labels, c_labels = t.tensor(b_kmeans.predict(b.numpy())), t.tensor(c_kmeans.predict(c.numpy()))
        # full_b_labels, full_c_labels = t.tensor(b_kmeans.predict(full_b.numpy())), t.tensor(c_kmeans.predict(full_c.numpy()))
        # b_mean, c_mean = t.tensor(b_kmeans.cluster_centers_), t.tensor(c_kmeans.cluster_centers_)
        # if verbose:
        #     print(f'b has {b_clusters} clusters with total loss {b_losses[-1]}')
        #     print(f'c has {c_clusters} clusters with total loss {c_losses[-1]}')

        # # partition clusters into rho-closed sets
        # b_remain = set(range(b_clusters))
        # b_parts = []
        # c_parts = []
        # while b_remain:
        #     i = b_remain.pop()
        #     b_part = {i}
        #     b_orbit = einops.einsum(b_mean[[i]], irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
        #     orbit_labels = t.tensor(b_kmeans.predict(b_orbit.numpy()))
        #     b_part |= set(orbit_labels.tolist())
        #     b_parts.append(b_part)
        #     b_remain -= b_part
        #     c_parts.append(set(full_c_labels[t.isin(full_b_labels, t.tensor(list(b_part)))].tolist()))
        # b_parts = list(map(list, b_parts))
        # c_parts = list(map(list, c_parts))
        # if verbose:
        #     print('b_parts', b_parts)
        #     print('c_parts', c_parts)
        #     print('a_mean', a.mean(dim=0))

        err = 100
        b_mean = None
        b_rho_labels = None
        b_k_labels = None
        best_err = 10000
        for tries in range(200):
            # for num_clusters in ([num_clusters] if num_clusters is not None else range(1, MAX_CLUSTERS)):
            cur_b_mean, cur_b_rho_labels, cur_b_k_labels,  err = irrep_kmeans(irrep, b, n_clusters=num_clusters if d_irrep > 1 else 1)
            if err < best_err:
                best_err = err
                b_mean = cur_b_mean
                b_rho_labels = cur_b_rho_labels
                b_k_labels = cur_b_k_labels



        c_mean, c_rho_labels, c_k_labels,  err = irrep_kmeans(irrep, c, n_clusters=num_clusters if d_irrep > 1 else 1, means=-b_mean)
        # c_mean = None
        # c_rho_labels = None
        # c_k_labels = None
        # best_err = 10000
        # for tries in range(200):
        #     # for num_clusters in ([num_clusters] if num_clusters is not None else range(1, MAX_CLUSTERS)):
        #     cur_c_mean, cur_c_rho_labels, cur_c_k_labels,  err = irrep_kmeans(irrep, c, n_clusters=num_clusters, means=-b_mean)
        #     if err < best_err:
        #         best_err = err
        #         b_mean = cur_b_mean
        #         b_rho_labels = cur_b_rho_labels
        #         b_k_labels = cur_b_k_labels

        for k in range(b_mean.shape[0]):
            full_b_mean = einops.einsum(irrep, b_mean[k], 'G d1 d2, d2 -> G d1')
            id_dist = (full_b_mean - full_b_mean[group.identity_idx()]).norm(dim=-1)
            stab = (id_dist < stab_thresh).nonzero().flatten().tolist()
            b_mean[k] = full_b_mean[stab].mean(dim=0)
        for k in range(c_mean.shape[0]):
            full_c_mean = einops.einsum(irrep, c_mean[k], 'G d1 d2, d2 -> G d1')
            id_dist = (full_c_mean - full_c_mean[group.identity_idx()]).norm(dim=-1)
            stab = (id_dist < stab_thresh).nonzero().flatten().tolist()
            c_mean[k] = full_c_mean[stab].mean(dim=0)
        b_hat = get_Xhat(irrep, b_mean, b_rho_labels, b_k_labels)
        c_hat = get_Xhat(irrep, c_mean, c_rho_labels, c_k_labels)
        if verbose:
            print('b_hat diff', (b_hat - b).norm()**2 / b.norm()**2)
            print('c_hat diff', (b_hat - b).norm()**2 / b.norm()**2)

        # bc_diff = ((b_mean + c_mean).norm()**2 / b_mean.norm()**2).item()
        # assert bc_diff < 1e-2, f'b and -c differ by {bc_diff},\n {b_mean},\n {-c_mean}'
            
        # if strict:
            # # Check that irrep is G-action on each partition of b's clusters
            # for i, b_part in enumerate(b_parts):
            #     T = einops.einsum(b_mean[b_part], irrep, b_mean[b_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            #     sT = (T > 1 - 1e-2).int()
            #     assert ((sT.sum(axis=1) == 1).all() and (sT.sum(axis=2) == 1).all()), f'Rho is not permutation on partition {i} of b!!!!'

            # # Check that irrep is G-action on each partition of c's clusters
            # for i, c_part in enumerate(c_parts):
            #     T = einops.einsum(c_mean[c_part], irrep, c_mean[c_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
            #     T = (T > 1 - 1e-2).int()
            #     assert ((T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all()), f'Rho is not permutation on partition {i} of c!!!!'

            # # Check that {b_i} = {-c_i} within each partition
            # for b_part, c_part in zip(b_parts, c_parts):
            #     S = b_mean[b_part] @ -c_mean[c_part].T
            #     S = (S > 1 - 1e-2).int()
            #     assert ((S.sum(axis=0) == 1).all() and (S.sum(axis=1) == 1).all()), print(f'(b_i) != (-c_i) within partition {b_part},{c_part}!!!!')
            #     # replace c_j with corresponding -b_i
            #     for j in range(len(c_part)):
            #         i = S[:,j].tolist().index(1)
            #         c_mean[c_part[j]] = -b_mean[b_part[i]]
            #         # print(f'replacing c_{c_part[j]} with -b_{b_part[i]}')
      
        # vecs[irrep_name] = (unif_coef, (A_norm + B_norm) / 2, a.mean(dim=0), b_mean, c_mean, b_labels, c_labels, b_parts, c_parts, bias_coef)
        vecs[irrep_name] = (coef, a.mean(dim=0), b_mean, c_mean, b_rho_labels, c_rho_labels, b_k_labels, c_k_labels)

        if verbose:
            print('b_mean', b_mean)
            print('c_mean', c_mean)
            print()

    return vecs, max_avar

def get_unif_vecs(group, irreps, vecs, irrep_idx_dict, total_neurons, verbose=False, stab_thresh=1e-2, bad_irrep_thresh=0.2, add_neurons=True):
    '''
    Main idea: we're provided coefs and vecs (from get_neuron_vecs) comprising rho-set circuits
    \sum_{b, c} coef_{b, c} b^T rho(z) c ReLU[b^T rho(x) a + a^T rho(y) b]
    for this to be equivariant, we need coef_{b, c} to be constant across the summation.
    We do this by setting each coef_{b, c} to the average over the summation.
    Subtleties:
        - multiple rho-sets corresponding to each rho; thus, we do this for each b_k_label and c_k_label
        - possibly multiple neurons per (b, c) vec pair. we scale all neurons corresponding to each (b, c) by the same amount so that they sum to the average
        - possibly some (b, c) vec pairs are missing from the double summation.
            - If missing pairs are more than half of total, zero out entire circuit
            - If less, then add the missing pairs as new neurons. This requires modifying irrep_idx_dict to include these new neurons.
        - two vectors are the same in the rho-set if their rho_labels are in the same coset of the stabilizer
    '''
    # unif_vecs should count towards the timing of the verifier
    # (if provided by the interpretations string, it would all need to be checked anyways.)
    unif_vecs = dict()
    bad_irreps = set()    # irreps for which a substantial subset of the double summation is missing
    for name, (coef, a_mean, b_mean, c_mean, b_rho_labels, c_rho_labels, b_k_labels, c_k_labels) in vecs.items():
        if verbose:
            print('UNIF VECS', name)
        irrep = irreps[name]
        d_irrep = irreps[name].shape[-1]
        unif_coef = t.zeros_like(coef)
        bias_coef = 0.
        if d_irrep == 1:
            # Hardcoded for sign irrep. TODO: Support for general complex 1d irreps
            # Normalize st sum over (0,0) and (1,1) is the same
            # and st sum over (0,1) and (1,0) is the same
            mask00 = (irrep[b_rho_labels].flatten() < 0) & (irrep[c_rho_labels].flatten() < 0)
            mask01 = (irrep[b_rho_labels].flatten() < 0) & (irrep[c_rho_labels].flatten() > 0)
            mask10 = (irrep[b_rho_labels].flatten() > 0) & (irrep[c_rho_labels].flatten() < 0)
            mask11 = (irrep[b_rho_labels].flatten() > 0) & (irrep[c_rho_labels].flatten() > 0)
            if mask00.any() and mask11.any():
                mean1 = ((coef[mask00].sum() + coef[mask11].sum()) / 2).item()
            else:
                mean1 = 0.
            if mask01.any() and mask10.any():
                mean2 = ((coef[mask01].sum() + coef[mask10].sum()) / 2).item()
            else:
                mean2 = 0.
            sign = 1. if b_mean[0] == c_mean[0] else -1.
            bias_coef -= sign * (mean1 - mean2)    # use bias to subtract out the extra rho(z) from single summation
            unif_coef[mask00] = coef[mask00] * mean1 / coef[mask00].sum()
            unif_coef[mask11] = coef[mask11] * mean1 / coef[mask11].sum()
            unif_coef[mask01] = coef[mask01] * mean2 / coef[mask01].sum()
            unif_coef[mask10] = coef[mask10] * mean2 / coef[mask10].sum()
            if verbose:
                print('coef', coef)
                print('unif_coef', unif_coef)
        else:
            bad_irreps |= {name}
            for k, l in product(range(b_mean.shape[0]), range(c_mean.shape[0])):
                # use b_mean to determine stabilizer
                kl_mask = (b_k_labels == k) & (c_k_labels == l)
                full_b_mean = einops.einsum(irrep, b_mean[k], 'G d1 d2, d2 -> G d1')
                id_dist = (full_b_mean - full_b_mean[group.identity_idx()]).norm(dim=-1)
                stab = frozenset((id_dist < stab_thresh).nonzero().flatten().tolist())
                if len(stab) < 5:  # rho-set too big; just zero the whole circuit
                    coef[kl_mask] = 0.
                    continue
                left_cosets = group.get_cosets_idx(stab)
                coset_prod = [(t.tensor(list(A)), t.tensor(list(B))) for A, B in product(left_cosets, repeat=2)]
                ij_mask = lambda A, B: t.isin(b_rho_labels, A) & t.isin(c_rho_labels, B)
                coef_sum = t.tensor([
                    coef[ij_mask(A, B) & kl_mask].sum().item()
                    for A, B in coset_prod
                ])
                coef_mean = coef_sum.mean()
                missing = set()
                for A, B in coset_prod:
                    if not (ij_mask(A, B) & kl_mask).any():
                        missing.add((A, B))
                        # print('ZEROING!', A, B)
                rhoset_size = len(group) // len(stab)
                if verbose:
                    print(name, f'missing {len(missing)}/{rhoset_size**2}=size^2. neuron count: {coef.shape[0]}')
                if not(bad_irrep_thresh * rhoset_size**2 <= len(missing) <= (1 - bad_irrep_thresh) * rhoset_size**2):
                    bad_irreps -= {name}
                if len(missing) > 0.5 * rhoset_size**2 or (len(missing) > 0 and not add_neurons):
                    coef_mean = 0.
                else:
                    # add missing vec pairs to the sum
                    if verbose:
                        print(f'ADDING {len(missing)} NEURONS')
                    for A, B in missing:
                        irrep_idx_dict[name].append(total_neurons)
                        coef = t.cat([coef, t.ones(1)])
                        unif_coef = t.cat([unif_coef, t.zeros(1)])
                        b_rho_labels = t.cat([b_rho_labels, t.tensor([A[0]])])
                        c_rho_labels = t.cat([c_rho_labels, t.tensor([B[0]])])
                        b_k_labels = t.cat([b_k_labels, t.tensor([k])])
                        c_k_labels = t.cat([c_k_labels, t.tensor([l])])
                        kl_mask = (b_k_labels == k) & (c_k_labels == l)
                        total_neurons += 1
                for A, B in coset_prod:
                    mask = ij_mask(A, B) & kl_mask
                    ij_sum = coef[mask].sum()
                    unif_coef[mask] = coef[mask] * coef_mean / ij_sum
            if verbose:
                print('coef', coef)
                print('unif_coef', unif_coef)
        unif_vecs[name] = (unif_coef, a_mean, b_mean, c_mean, b_rho_labels, c_rho_labels, b_k_labels, c_k_labels, bias_coef)
    return unif_vecs, bad_irreps, irrep_idx_dict, total_neurons

def get_scale(ln, ln_hat, rn, rn_hat, un, un_hat, ucoef=1):
    '''
    Finds argmax_a |a^{-1} ln_hat - ln|^2 + |a^{-1} rn_hat - rn|^2 + ucoef * |a un_hat - un|^2
    Setting derivative to zero and multipling by a^3 gives a quartic polynomial.
    '''
    
    poly_coefs = [
        ucoef * un_hat.T @ un_hat,
        ucoef * -un_hat.T @ un,
        ln_hat.T @ ln + rn_hat.T @ rn,
        -ln_hat.T @ ln_hat - rn_hat.T @ rn_hat,
    ]
    roots = np.roots(poly_coefs)
    roots = roots[np.isreal(roots) & (np.abs(roots) > 1e-8)].real
    
    obj = lambda a: ((a * un_hat - un).norm()**2 + ((1/a) * ln_hat - ln).norm()**2 + ((1/a) * rn_hat - rn).norm()**2).item()
    return min(roots, key=obj).real
            
@t.no_grad()
def get_idealized_model(model, irreps, irrep_idx_dict, unif_vecs, total_neurons, verbose=False, ucoef=1):
    assert len(model) == 1, "model must be a single instance"
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    lneurons, rneurons, uneurons = model.get_neurons(squeeze=True)
    # unif_vecs may add new neurons
    if verbose:
        print('TOTAL NEURONS', total_neurons)
    assert total_neurons >= lneurons.shape[-1]
    orig_neurons = lneurons.shape[-1]
    new_neurons = total_neurons - orig_neurons
    if total_neurons > lneurons.shape[-1]:
        lneurons = t.cat([lneurons, t.zeros(lneurons.shape[:-1] + (new_neurons,))], dim=-1)
        rneurons = t.cat([rneurons, t.zeros(rneurons.shape[:-1] + (new_neurons,))], dim=-1)
        uneurons = t.cat([uneurons, t.zeros(uneurons.shape[:-1] + (new_neurons,))], dim=-1)
        model.embedding_left = nn.Parameter(lneurons.unsqueeze(0))
        model.embedding_right = nn.Parameter(rneurons.unsqueeze(0))
        model.unembedding = nn.Parameter(uneurons.mT.unsqueeze(0))
    new_ln, new_rn, new_un = t.zeros_like(lneurons), t.zeros_like(rneurons), t.zeros_like(uneurons)
    new_bias = t.zeros_like(model.unembed_bias.squeeze())
    dead_neurons = set()
    for irrep_name, (coef, a_mean, b_mean, c_mean, b_rho_labels, c_rho_labels, b_k_labels, c_k_labels, bias_coef) in unif_vecs.items():
        if verbose:
            print('GET_IDEALIZED_MODEL', irrep_name)
        irrep_idxs = irrep_idx_dict[irrep_name]
        irrep = irreps[irrep_name]
        b = get_Xhat(irrep, b_mean, b_rho_labels, b_k_labels)
        c = get_Xhat(irrep, c_mean, c_rho_labels, c_k_labels)
        check_rho_set(irrep, b)
        check_rho_set(irrep, c)
        irrep_ln = einops.einsum(b, irrep, a_mean, 'm d1, G d1 d2, d2 -> G m')
        irrep_rn = einops.einsum(a_mean, irrep, c, 'd1, G d1 d2, m d2 -> G m')
        irrep_un = coef * einops.einsum(b, irrep, c, 'm d1, G d1 d2, m d2 -> G m')
        for i in range(irrep_ln.shape[1]):
            if irrep_un[:, i].abs().max() > 1e-8:
                # degree of freedom in scaling u and l/r proportionally
                # use this to match u to original norm
                # ucoef = t.dot(irrep_un[:,i], uneurons[:,irrep_idxs[i]]) / t.dot(irrep_un[:,i], irrep_un[:,i])
                alpha = get_scale(lneurons[:,irrep_idxs[i]], irrep_ln[:,i], rneurons[:,irrep_idxs[i]], irrep_rn[:,i], uneurons[:,irrep_idxs[i]], irrep_un[:,i], ucoef=ucoef)
                irrep_un[:, i] *= alpha
                irrep_ln[:, i] /= alpha
                irrep_rn[:, i] /= alpha
                # if irrep_name=='5d-0' and (irrep_ln[:, i] - lneurons[:,irrep_idxs[i]]).norm()**2 > (lneurons[:,irrep_idxs[i]]).norm()**2:
                #     import pdb; pdb.set_trace()
            else:
                # un has been zeroed out in the preceding step
                # more efficient to keep it the same and zero out l/r
                irrep_un[:, i] = uneurons[:, irrep_idxs[i]]
                irrep_ln[:, i] = 0. # lneurons[:, irrep_idxs[i]]
                irrep_rn[:, i] = 0. # rneurons[:, irrep_idxs[i]]
        if verbose:
            print('l diff', (irrep_ln - lneurons[:,irrep_idxs]).norm()**2 / lneurons[:,irrep_idxs].norm()**2)
            print('r diff', (irrep_rn - rneurons[:,irrep_idxs]).norm()**2 / rneurons[:,irrep_idxs].norm()**2)
            print('u diff', (irrep_un - uneurons[:,irrep_idxs]).norm()**2 / uneurons[:,irrep_idxs].norm()**2)
        new_ln[:,irrep_idxs] = irrep_ln
        new_rn[:,irrep_idxs] = irrep_rn
        new_un[:,irrep_idxs] = irrep_un
        # Hardcoded ideal bias for sign rep
        if abs(bias_coef) > 1e-8:
            assert irreps[irrep_name].shape[-1] == 1, 'Only 1d irrep supported for bias'
            new_bias += irreps[irrep_name].squeeze() * bias_coef

    for i in range(lneurons.shape[-1]):
        if max(new_un[:, i].abs().max(),  new_ln[:, i].abs().max(), new_rn[:, i].abs().max()) < 1e-8:
            # dead neuron. might as well set un to origial
            dead_neurons.add(i)
            new_un[:, i] = uneurons[:, i]
            new_ln[:, i] = 0. # lneurons[:, irrep_idxs[i]]
            new_rn[:, i] = 0. # rneurons[:, irrep_idxs[i]]

    if verbose:
        print('total')
        print('l 1-r2', (new_ln - lneurons).norm()**2 / lneurons.norm()**2)
        print('r 1-r2', (new_rn - rneurons).norm()**2 / rneurons.norm()**2)
        print('u 1-r2', (new_un - uneurons).norm()**2 / uneurons.norm()**2)
        print('bias 1-r2', (new_bias - model.unembed_bias.squeeze()).norm()**2 / model.unembed_bias.squeeze().norm()**2)
    # full representation for bounding the error
    ideal = copy.deepcopy(model)
    ideal.embedding_left = nn.Parameter(new_ln.unsqueeze(0))
    ideal.embedding_right = nn.Parameter(new_rn.unsqueeze(0))
    ideal.unembedding = nn.Parameter(new_un.unsqueeze(0).mT)
    ideal.unembed_bias = nn.Parameter(new_bias.unsqueeze(0))

    live_neurons = list(set(range(lneurons.shape[-1])) - dead_neurons)
    # compact representation for doing forward pass
    cpct = copy.deepcopy(model)
    cpct.embedding_left = nn.Parameter(new_ln[:,live_neurons].unsqueeze(0))
    cpct.embedding_right = nn.Parameter(new_rn[:,live_neurons].unsqueeze(0))
    cpct.unembedding = nn.Parameter(new_un[:,live_neurons].unsqueeze(0).mT)
    cpct.unembed_bias = nn.Parameter(new_bias.unsqueeze(0))
    return ideal, cpct, model  # original model might have some zeros appended to have same shape as ideal

# def model_dist_parted(model1, model2, irrep_idx_dict, vecs):
#     assert len(model1) == 1 and len(model2) == 1, "must be single instances"
#     ln1, rn1, un1 = model1.get_neurons()
#     ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
#     ln2, rn2, un2 = model2.get_neurons()
#     ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
#     M = 0
#     norm21 = lambda A: A.norm(dim=0).max()
#     norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
#     for irrep_name, (coef, A_norm, a_mean, b_mean, c_mean, b_labels, c_labels, b_parts, c_parts) in vecs.items():
#         irrep_idxs = irrep_idx_dict[irrep_name]
#         for b_part in b_parts:
#             part_idxs = t.tensor(irrep_idxs)[t.isin(b_labels, t.tensor(b_part))]
#             part_ln1, part_rn1, part_un1 = ln1[:,part_idxs], rn1[:,part_idxs], un1[:,part_idxs]
#             part_ln2, part_rn2, part_un2 = ln2[:,part_idxs], rn2[:,part_idxs], un2[:,part_idxs]
#             part_M = norm22(part_un1) * (norm21(part_ln1 - part_ln2) + norm21(part_rn1 - part_rn2)) \
#                         + norm22(part_un2 - part_un1) * (norm21(part_ln2) + norm21(part_rn2))
#             M += part_M.item()
#             print(irrep_name)
#             print('l diff', norm21(part_ln1 - part_ln2))
#             print('r diff', norm21(part_rn1 - part_rn2))
#             print('u diff', norm22(part_un1 - part_un2))
#             print('l norm', norm21(part_ln1))
#             print('r norm', norm21(part_rn1))
#             print('u norm', norm22(part_un1))
#             print(part_M.item())
#     return M

@t.no_grad()
def irrep_acc_bound(model, group, irreps, irrep_idx_dict, vecs, ucoef=1, verbose=False, do_ce=False, add_neurons=True):
    assert len(model) == 1, "model must be a single instance"
    t0 = time.time()
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    ln, rn, un = model.get_neurons(squeeze=True)
    if model.unembed_bias is None:
        model.unembed_bias = nn.Parameter(t.zeros(1, len(group)))
    ubias = model.unembed_bias.squeeze()
    try:
        unif_vecs, bad_irreps, irrep_idx_dict, total_neurons = get_unif_vecs(group, irreps, vecs, irrep_idx_dict, total_neurons=ln.shape[-1], verbose=verbose, add_neurons=add_neurons)
        ideal, cpct, model = get_idealized_model(model, irreps, irrep_idx_dict, unif_vecs, total_neurons, ucoef=ucoef, verbose=verbose)
    except AssertionError as e:
        print(e)
        return 0., time.time() - t0, True
    t1 = time.time()

    # Check that ideal_model depends only on x^-1zy^-1. Theoretically this should always hold, so don't time  this.
    # margins = []
    # for i, j in product(range(len(group)), repeat=2):
    #     out = ideal(t.tensor([[i, j]])).flatten()
    #     top2 = out.topk(k=2).values
    #     margins.append((top2[0] - top2[1]).item())
    # assert np.std(margins) < 1e-5, 'ideal model not equivariant'

    t2 = time.time()
    # ideal is equivariant, so we can just check the margin of the identity
    id = group.identity_idx()
    out = cpct(t.tensor([[id, id]])).flatten()
    ideal_correct_logit = out[id].item()
    out[id] = -t.inf
    margin = ideal_correct_logit - out.max().item()
    orig_correct_logits = []
    for x, y in product(range(len(group)), repeat=2): # It's important that this iteration is done in the same order as in model_dist_xy
        x_embed = ln[x]
        y_embed = rn[y]
        act = F.relu(x_embed + y_embed)
        z = group.mult_idx(x, y)
        orig_correct_logits.append(un[z].dot(act) + ubias[z])
    errs = model_dist_xy(model, ideal, 'inf')
    # import pdb; pdb.set_trace()
    if do_ce:
        # this is a bound on cross-entropy
        # but too lazy to rename the variable
        acc = margin + t.tensor(orig_correct_logits) - ideal_correct_logit - t.tensor(errs)
        acc = -t.log(t.exp(acc) / (t.exp(acc) + len(group) - 1)).mean().item()
    else:
        acc = (t.tensor(errs) < margin + t.tensor(orig_correct_logits) - ideal_correct_logit).float().mean().item()
    t3 = time.time()
    # check irreps
    # this is |G|^2d^3 time complexity, which is unnecessarily wasteful.
    # All we really need is that each rho-set is permuted by rho (checked in get_idealized_model)
    # for name, irrep in irreps.items():
    #     if not irrep_idx_dict[name]:
    #         continue
    #     if not group.is_rep(irrep):
    #         print('Not rep!!')
    #         return 0., time.time() - t2 + t1 - t0

    t4 = time.time()
    # print('idealization time', t1 - t0)
    # print('bound time', t3 - t2)
    # print('irrep checks time', t4 - t3)
    # print('total time', t4 - t2 + t1 - t0)
    return acc, t4 - t2 + t1 - t0, ideal, bad_irreps

@t.no_grad()
def naive_acc_bound(model, group, do_ce=False):
    # Do this untensorized for a fair comparison with irrep_acc_bound
    # TODO: tensorize irrep_acc_bound
    t0 = time.time()
    if not isinstance(model, MLP4):
        model = model.fold_linear()
    corrects = []
    for i, j in product(range(len(group)), repeat=2):
        out = model(t.tensor([[i, j]])).flatten()
        label = group.mult_idx(i, j)
        if do_ce:
            corrects.append(-F.log_softmax(out)[label].item())
        else:
            corrects.append((out.argmax() == label).int().item())
    acc = np.mean(corrects).item()
    t1 = time.time()
    return acc, t1 - t0
    
    