from torch.func import jvp, grad, vjp, vmap, functional_call
from typing import Dict, Tuple
import torch as t
import numpy as np
from utils import *

# Adapted from https://github.com/akshayka/hessian_trace_estimation/blob/main/hessian_trace_estimation.ipynb

class LinearOperator:
    def __init__(self, matvec):
        self._matvec = matvec

    def matvec(self, vecs):
        return self._matvec(vecs)

def hutchpp(A, d, m):
    """https://arxiv.org/abs/2010.09649

    A is the LinearOperator whose trace to estimate
    d is the input dimension
    m is the number of queries (larger m yields better estimates)
    """
    S = t.randn(d, m // 3)
    G = t.randn(d, m // 3)
    AS = A.matvec(S)
    Q, _ = t.qr(AS)
    proj = G - Q @ (Q.T @ G)
    return t.trace(Q.T @ A.matvec(Q)) + (3./m)*t.trace(proj.T @ A.matvec(proj))

def make_hvp(f, x):
    def hvp(f, primals, tangents):
        return vmap(lambda t: jvp(grad(f), (primals, ), (t, ))[1], in_dims=1, out_dims=1)(tangents)

    return LinearOperator(lambda v: hvp(f, x, v))

def flatten(
    params: Dict[str, t.Tensor]
) -> Tuple[t.Tensor, Dict[str, t.Size]]:
    return t.cat([p.flatten() for p in params.values()]), {k: p.size() for k, p in params.items()}

def unflatten(
    flat_params: t.Tensor, 
    shapes: Dict[str, t.Size],
) -> Dict[str, t.Tensor]:
    params = dict()
    for name, shape in shapes.items():
        size = np.prod(shape)
        params[name] = flat_params[:size].reshape(shape)
        flat_params = flat_params[size:]
    return params

def h_tr(models, data, m=100):
    ret = []
    train_dataset = t.tensor(data.train_data, device=device)

    for i in tqdm(range(len(models))):
        model = models[i]
        flat_init_params, shapes = flatten(dict(model.named_parameters()))
        def loss_func(flat_params, model=model, shapes=shapes, dataset=train_dataset):
            x, z = dataset[:, :-1], dataset[:, -1]
            output = t.func.functional_call(model, unflatten(flat_params, shapes), x)
            return get_cross_entropy(output, z).squeeze()
        
        model_hvp = make_hvp(loss_func, flat_init_params)
        ret.append(hutchpp(model_hvp, flat_init_params.size(0), m))
    return t.tensor(ret)