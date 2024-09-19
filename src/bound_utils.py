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

def model_dist_res(model1, model2, proj):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons(squeeze=True)
    ln2, rn2, un2 = model2.get_neurons(squeeze=True)
    norm21 = lambda A: A.norm(dim=1).max()  # max 2-norm along neuron dimension
    norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
    print('l diff', norm21(ln1 - ln2))
    print('r diff', norm21(rn1 - rn2))
    print('u diff', norm22(un1 - un2))
    print('u diff res', norm22(proj @ (un1 - un2)))
    print('l norm', norm21(ln1))
    print('r norm', norm21(rn1))
    print('u norm', norm22(un1))
    return norm22(un1) * (norm21(ln1 - ln2) + norm21(rn1 - rn2)) + norm22(proj @ (un2 - un1)) * (norm21(ln2) + norm21(rn2))

def model_dist(model1, model2):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons(squeeze=True)
    ln2, rn2, un2 = model2.get_neurons(squeeze=True)
    norm21 = lambda A: A.norm(dim=1).max()  # max 2-norm along neuron dimension
    norm22 = lambda A: t.linalg.matrix_norm(A, ord=2)
    print('l diff', norm21(ln1 - ln2))
    print('r diff', norm21(rn1 - rn2))
    print('u diff', norm22(un1 - un2))
    print('l norm', norm21(ln1))
    print('r norm', norm21(rn1))
    print('u norm', norm22(un1))
    print('u diff term', norm22(un2 - un1) * (norm21(ln2) + norm21(rn2)))
    print('lr diff term', norm22(un1) * (norm21(ln1 - ln2) + norm21(rn1 - rn2)))
    return norm22(un1) * (norm21(ln1 - ln2) + norm21(rn1 - rn2)) + norm22(un2 - un1) * (norm21(ln2) + norm21(rn2))

def model_dist2(model1, model2):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons()
    ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
    ln2, rn2, un2 = model2.get_neurons()
    ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
    return ((un1 - un2).norm(dim=0) * (ln1.max(dim=0).values + rn1.max(dim=0).values)).sum() \
        + (un2.norm(dim=0) * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).sum()
        # + un2 * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).abs().sum(dim=1).max()

def model_dist_inf(model1, model2):
    assert len(model1) == 1 and len(model2) == 1, "must be single instances"
    ln1, rn1, un1 = model1.get_neurons()
    ln1, rn1, un1 = ln1.squeeze(0), rn1.squeeze(0), un1.squeeze(0)
    ln2, rn2, un2 = model2.get_neurons()
    ln2, rn2, un2 = ln2.squeeze(0), rn2.squeeze(0), un2.squeeze(0)
    return ((un1 - un2) * (ln1.max(dim=0).values + rn1.max(dim=0).values) 
        + un2 * ((ln1 - ln2).abs().max(dim=0).values + (rn1 - rn2).abs().max(dim=0).values)).abs().sum(dim=1).max()

def logit_bound(model1, model2):
    '''
    Returns a (|G|, |G|) tensor bound such that, for each x, y, z in G,
    model2(xy, x, y) - model2(z, x, y) >= model1(xy, x, y) - model(z, x, y) - bound[x, y]
    '''