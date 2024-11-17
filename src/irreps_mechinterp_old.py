#!/usr/bin/env python

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


from model import MLP3, MLP4, InstancedModule
from utils import *
from group_data import *
from model_utils import *
from group_utils import *

import sys, os, re
import argparse

# This is just a copy of irreps_mechinterp.ipynb but with a for loop
# TODO: Rewrite to be more modular

A_VAR = []
LOSS = []
NORM = []

def irrep_report(irrep, group, irrep_lneurons, irrep_rneurons, irrep_uneurons, clusters=1000):
    irrep_frobschur = np.sign(group.get_frobenius_schur(irrep)).item()
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
    print('A and B norm diff', ((A_norm - B_norm).norm()**2 / B_norm.norm()**2).item())

    # Normalize A and B to have unit Frobenius norm
    A = A / A_norm.unsqueeze(1).unsqueeze(1)
    B = B / A_norm.unsqueeze(1).unsqueeze(1)
    C = C * ( (A_norm + B_norm) / 2).unsqueeze(1).unsqueeze(1)

    x = einops.rearrange(B @ A, 'm d1 d2 -> m (d1 d2)')
    y = einops.rearrange(C, 'm d1 d2 -> m (d1 d2)')
    coef = (x.conj() * y).sum(dim=-1) / (x.conj() * x).sum(dim=-1)
    yhat = coef.unsqueeze(1) * x
    r2 = (yhat - y).norm(dim=-1).pow(2) / y.norm(dim=-1).pow(2)
    good = r2 < 0.1

    # Restrict to good neurons
    A, B, C, coef = A[good], B[good], C[good], coef[good]

    print('1-r2 90th percentile', t.quantile(r2, 0.9).item())

    rank = {
        1: 1,
        0: 2,
        -1: 4
    }[irrep_frobschur]

    if irrep_frobschur == 1:
        a, b, c, d, Ss = [], [], [], [], []
        for i in range(A.shape[0]):
            lU, lS, lV = t.linalg.svd(A[i])
            rU, rS, rV = t.linalg.svd(B[i])
            Ss.append(lS[0])
            a.append(lU[:,0])
            b.append(lV[0])
            c.append(rU[:,0])
            d.append(rV[0])
            if not lS[0]**2 / lS.norm()**2 > 1-1e-2:
                print(f'A is incorrect rank! Variance explained by first rank: {lS[0]**2 / lS.norm()**2}')
            if not rS[0]**2 / rS.norm()**2 > 1-1e-2:
                print(f'B is incorrect rank! Variance explained by first rank: {rS[0]**2 / rS.norm()**2}')
        a, b, c, d = t.stack(a, dim=0), t.stack(b, dim=0), t.stack(c, dim=0), t.stack(d, dim=0)
        a_sign = t.sgn(a[:,0])
        a = t.diag(a_sign) @ a
        b = t.diag(a_sign) @ b
        d_sign = t.sgn(d[:,0])
        c = t.diag(d_sign) @ c
        d = t.diag(d_sign) @ d
        Ss = t.stack(Ss, dim=0)
    else:
        # Our guess is that left singular vectors of A are const across neurons
        # and same for right singular vectors of B (up to sign, etc)
        # So we use the first neuron to get the relevant singular vectors
        # and share them across all neurons
        lU0 = t.linalg.svd(A[0])[0]
        # rV0 = t.linalg.svd(B[0])[2]
        rV0 = lU0.T
        lUs, lVs, rUs, rVs, Ss = [], [], [], [], []
        for i in range(A.shape[0]):
            lS = t.linalg.svd(A[i])[1]
            rS = t.linalg.svd(B[i])[1]
            assert (lS[0] - rS[0])**2 / lS[0]**2 < 1e-2, 'A and B have different singular values!'
            Ss.append(lS[0])
            assert lS[:rank].norm()**2 / lS.norm()**2 > 1-1e-2, 'A is incorrect rank!'
            assert rS[:rank].norm()**2 / rS.norm()**2 > 1-1e-2, 'B is incorrect rank!'
            lS_inv, rS_inv = t.zeros_like(lS), t.zeros_like(rS)
            lS_inv[:rank] = 1 / lS[:rank]
            rS_inv[:rank] = 1 / rS[:rank]
            lV = t.diag(lS_inv) @ lU0.T @ A[i]
            rU = B[i] @ rV0.T @ t.diag(rS_inv)
            # check that lV and rU are orthogonal,
            # i.e. that lU0 and rV0 are valid left/right singular vectors
            assert ((lV @ lV.T)[:rank,:rank] - t.eye(rank)).norm() < 1e-2,  'Shared left singular vectors for A failed!'
            assert ((rU.T @ rU)[:rank,:rank] - t.eye(rank)).norm() < 1e-2,  'Shared right singular vectors for B failed!'
            lUs.append(lU0)
            rUs.append(rU)
            lVs.append(lV)
            rVs.append(rV0)

        rUs = t.stack(rUs, dim=0)
        rVs = t.stack(rVs, dim=0)
        lUs = t.stack(lUs, dim=0)
        lVs = t.stack(lVs, dim=0)
        Ss = t.stack(Ss, dim=0)
        a, b, c, d = lUs[:,:,0], lVs[:,0], rUs[:,:,0], rVs[:,0]

    for name, v in zip(['a', 'b', 'c', 'd'], [a, b, c, d]):
        print(f'{name} variance:', ((v - v.mean(dim=0)).norm()**2 / v.norm()**2).item())
    A_VAR.append(((a - a.mean(dim=0)).norm()**2 / a.norm()**2).item())
    
    

    print('a vs d', (a - d).norm()**2 / a.norm()**2)
    
    full_b = einops.einsum(b, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
    full_c = einops.einsum(c, irrep, 'neuron d2, G d1 d2 -> neuron G d1').flatten(0, 1)
    b_kmeans, b_clusters, b_losses = cluster(full_b, max=clusters)
    c_kmeans, c_clusters, c_losses = cluster(full_c, max=clusters)
    b_labels, c_labels = b_kmeans.predict(b.numpy()), c_kmeans.predict(c.numpy())
    b_mean, c_mean = b_kmeans.cluster_centers_, c_kmeans.cluster_centers_
    print(f'b has {b_clusters} clusters with total loss {b_losses[-1]}')
    print(f'c has {c_clusters} clusters with total loss {c_losses[-1]}')
    # print('CLUSTERS')
    b_parts = []
    c_parts = []
    for i in range(b_clusters):
        c_set = set(c_labels[b_labels == i].tolist())
        if not c_set:
            continue
        done = False
        for j in range(len(c_parts)):
            if len(c_set & c_parts[j]) > 0:
                c_parts[j] = c_set.union(c_parts[j])
                b_parts[j].add(i)
                done = True
                break
        if not done:
            c_parts.append(c_set)
            b_parts.append({i})
        # print(f'b={i}, c={sorted(c_set)}')

    # for b_part1, b_part2 in product(b_parts, repeat=2):
    #     assert b_part1==b_part2 or not b_part1 & b_part2
    # for c_part1, c_part2 in product(c_parts, repeat=2):
    #     assert c_part1==c_part2 or not c_part1 & c_part2
    b_parts = list(map(sorted, map(list, b_parts)))
    c_parts = list(map(sorted, map(list, c_parts)))
    # print(b_parts)
    # print(c_parts)

    # Check that irrep is G-action on each partition of b's clusters
    for i, b_part in enumerate(b_parts):
        T = einops.einsum(b_mean[b_part], irrep, b_mean[b_part], 'm1 d1, G d1 d2, m2 d2 -> G m1 m2')
        T = (T > 1 - 1e-2).astype(float)
        if (T.sum(axis=1) == 1).all() and (T.sum(axis=2) == 1).all():
            print(f'Rho is permutation on partition {i} of b!')

    # Check that {b_i} = {-c_i} over all points
    S = b_mean @ -c_mean.T
    S = (S > 1 - 1e-2).astype(float)
    if (S.sum(axis=0) == 1).all() and (S.sum(axis=1) == 1).all():
        print('\{b_i\} = \{-c_i\}!')
        
    # Check that {b_i} = {-c_i} within each partition
    for b_part, c_part in zip(b_parts, c_parts):
        S = b_mean[b_part] @ -c_mean[c_part].T
        S = (S > 1 - 1e-2).astype(float)
        if (S.sum(axis=0) == 1).all() and (S.sum(axis=1) == 1).all():
            print(f'(b_i) = (-c_i) within partition {b_part},{c_part}!')

    # Check that there are two partitions, and that they are antipodal
    if len(b_parts) == 2:
        T = b_mean[b_parts[0]] @ -b_mean[b_parts[1]].T
        T = (T > 1 - 1e-2).astype(float)
        if (T.sum(axis=0) == 1).all() and (T.sum(axis=1) == 1).all():
            print('Two antipodal partitions!')
            
    # check that coefs are uniform over {b_i}x{c_i}
    for b_part, c_part in zip(b_parts, c_parts):
        coef_sum = t.tensor([
            coef[(b_labels == i) & (c_labels == j)].sum().item()
            for i, j in product(b_part, c_part)
        ])
        print(f'part{b_part} coefs: size={coef_sum.norm()}, variance={(coef_sum - coef_sum.mean()).norm()**2/coef_sum.norm()**2}')

    N = len(group)
    inputs = t.tensor(list(product(range(N), repeat=2)), device=device)
    l_in, r_in = inputs[..., 0], inputs[..., 1]
    l_onehot = F.one_hot(l_in, num_classes=N).float()
    r_onehot = F.one_hot(r_in, num_classes=N).float()
    l_embed = einops.einsum(l_onehot, irrep_lneurons.to(device), 'batch group, group embed -> batch embed')
    r_embed = einops.einsum(r_onehot, irrep_rneurons.to(device), 'batch group, group embed -> batch embed')
    pre_act = l_embed + r_embed
    act = t.maximum(t.zeros_like(pre_act), pre_act)
    lin_logits = einops.einsum(pre_act, irrep_uneurons.to(device), 'batch hid, group hid-> batch group')
    logits = einops.einsum(act, irrep_uneurons.to(device), 'batch hid, group hid-> batch group')
    print('output norm ratio', (lin_logits.norm() / logits.norm()).item())
            

def irreps_report(model, params, irrep_filter=None, acc_thresh=0.999, loss_thresh=0.1, clusters=1000):
    data = GroupData(params)
    group = data.groups[0]
    loss_dict = test_loss(model.to(device), data)
    all_models = model
    for i in range(len(all_models)):
        if loss_dict['G0_loss'][i] > loss_thresh or loss_dict['G0_acc'][i] < acc_thresh:
            continue
        model = all_models[i].to(device)
        print('MODEL INSTANCE', i)
        print('Loss:', loss_dict['G0_loss'][i].item())
        print('Norm:', (model.embedding_left.norm() + model.embedding_right.norm() + model.unembedding.norm() + model.linear.norm()).item())
        
        lneurons, rneurons, uneurons = model.get_neurons(squeeze=True)
        irreps = group.get_real_irreps(verbose=True)

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
                if lexpl[name][i].item() > 0.95 and lneurons[:,i].norm() > 1e-2
            ]
            for name in irreps
        }
        irrep_idx_dict['none'] = [
            i for i in range(lneurons.shape[-1])
            if all(lexpl[name][i].item() <= 0.95 for name in irreps)
        ]

        for k, irrep_idxs in irrep_idx_dict.items():
            if irrep_filter is not None and not re.compile(irrep_filter).match(k):
                A_VAR.append(100)
                continue
            if len(irrep_idxs) < 20:
                A_VAR.append(100)
                continue

            print(f'Irrep {k} has {len(irrep_idxs)} neurons')
            LOSS.append(loss_dict['G0_loss'][i].item())
            NORM.append((model.embedding_left.norm() + model.embedding_right.norm() + model.unembedding.norm() + model.linear.norm()).item())

            irrep_lneurons = lneurons[:, irrep_idxs]
            irrep_rneurons = rneurons[:, irrep_idxs]
            irrep_uneurons = uneurons[:, irrep_idxs]
            irrep_report(irreps[k], group, irrep_lneurons, irrep_rneurons, irrep_uneurons, clusters=clusters)
        print('A_VAR', len(A_VAR))
        print('LOSS', len(LOSS))
        print('--------------------\n\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of model")
    parser.add_argument("--filter", type=str, help="Irrep regex filter")
    parser.add_argument("--acc_thresh", type=float, help="Accuracy threshold", default=0.999)
    parser.add_argument("--loss_thresh", type=float, help="Loss threshold", default=0.5)
    parser.add_argument("--clusters", type=float, help="Max num of clusters", default=1000)
    args = parser.parse_args()
    models, params = dl_model(args.model_name, os.getenv('HOME') + '/models')
    irreps_report(models[-1], params, irrep_filter=args.filter, acc_thresh=args.acc_thresh, loss_thresh=args.loss_thresh, clusters=args.clusters)
    t.save((A_VAR, LOSS, NORM), 'avar_loss_norm_temp.pt')
    plt.scatter(A_VAR, LOSS)
    plt.xscale('log')
    plt.xlabel('a variance')
    plt.ylabel('loss')
    plt.show()
    plt.scatter(A_VAR, NORM)
    plt.xscale('log')
    plt.xlabel('a variance')
    plt.ylabel('weight norm')
    plt.show()