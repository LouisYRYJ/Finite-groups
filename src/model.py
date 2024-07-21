import torch as t
import torch.nn.functional as F
from torch import nn
from jaxtyping import Float, Int, jaxtyped
from typing import Optional, Callable, Union, List, Tuple
import einops
import numpy as np
from beartype import beartype
from copy import deepcopy


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = nn.Embedding(params.N, params.embed_dim)
        self.Embedding_right = nn.Embedding(params.N, params.embed_dim)
        self.linear = nn.Linear(params.embed_dim * 2, params.hidden_size, bias=True)
        if params.activation == "gelu":
            self.activation = nn.GELU()
        if params.activation == "relu":
            self.activation = nn.ReLU()
        self.Umbedding = nn.Linear(params.hidden_size, params.N, bias=True)

    def forward(self, a):
        x1 = self.Embedding_left(a[0])
        x2 = self.Embedding_right(a[1])
        x12 = t.cat([x1, x2], -1)
        hidden = self.linear(x12)
        hidden = self.activation(hidden)
        out = self.Umbedding(hidden)
        return out


class MLP2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = nn.Embedding(params.N, params.embed_dim)
        self.Embedding_right = nn.Embedding(params.N, params.embed_dim)
        self.linear_left = nn.Linear(params.embed_dim, params.hidden_size, bias=False)
        self.linear_right = nn.Linear(
            params.embed_dim, params.hidden_size, bias=False
        )
        if params.activation == "gelu":
            self.activation = nn.GELU()
        if params.activation == "relu":
            self.activation = nn.ReLU()
        self.Umbedding = nn.Linear(params.hidden_size, params.N, bias=False)

    def forward(self, a):

        a_1, a_2 = a[:, 0], a[:, 1]
        x1 = self.Embedding_left(a_1)
        x2 = self.Embedding_right(a_2)
        hidden_x1 = self.linear_left(x1)
        hidden_x2 = self.linear_right(x2)
        hidden_sum = hidden_x1 + hidden_x2
        hidden = self.activation(hidden_sum)
        out = self.Umbedding(hidden)
        return out


def custom_xavier(dims):
    # Assumes shape [..., fan_in, fan_out]
    return nn.Parameter(t.randn(dims) * np.sqrt(2.0 / float(dims[-2] + dims[-1])))


def custom_kaiming(dims):
    # Assumes shape [..., fan_in, fan_out]
    return nn.Parameter(t.randn(dims) * np.sqrt(2.0 / float(dims[-2])))


class MLP3(nn.Module):
    # TODO: It's probably better practice to explicitly list the args (instances, embed_dim, etc)
    # instead of using the params object
    def __init__(self, N, params):
        super().__init__()
        self.params = params
        self.N = N

        self.embedding_left = custom_kaiming(
            [
                params.instances,
                self.N,
                params.embed_dim,
            ]
        )

        self.embedding_right = custom_kaiming(
            [params.instances, self.N, params.embed_dim]
        )

        self.linear = custom_kaiming(
            [params.instances, params.embed_dim, params.hidden_size]
        )

        self.unembedding = custom_kaiming(
            [params.instances, params.hidden_size, self.N]
        )

        if params.activation == "gelu":
            self.activation = nn.GELU()
        elif params.activation == "relu":
            self.activation = nn.ReLU()
        elif params.activation == "linear":
            self.activation = lambda x: x
        else:
            raise ValueError("Activation not recognized")

    def __getitem__(self, slice):
        '''
        Returns a new model with parameters sliced along the instances dimension.
        '''
        ret = deepcopy(self)
        for name, param in self.named_parameters():
            sliced_param = param[slice].clone()
            if isinstance(slice, int):
                sliced_param = sliced_param.unsqueeze(0)
            setattr(ret, name, nn.Parameter(sliced_param))
        return ret

    @jaxtyped(typechecker=beartype)
    def forward(
        self, a: Int[t.Tensor, "batch_size entries"]
    ) -> Float[t.Tensor, "batch_size instances d_vocab"]:

        a_instances = einops.repeat(
            a, " batch_size entries -> batch_size n entries", n=self.params.instances
        )  # batch_size instances entries

        a_1, a_2 = a_instances[..., 0], a_instances[..., 1]

        a_1_onehot = F.one_hot(a_1, num_classes=self.N).float()
        a_2_onehot = F.one_hot(a_2, num_classes=self.N).float()

        x_1 = einops.einsum(
            a_1_onehot,
            self.embedding_left,
            "batch_size instances d_vocab, instances d_vocab embed_dim -> batch_size instances embed_dim",
        )
        x_2 = einops.einsum(
            a_2_onehot,
            self.embedding_right,
            "batch_size instances d_vocab, instances d_vocab embed_dim -> batch_size instances embed_dim",
        )

        hidden = einops.einsum(
            x_1 + x_2,
            self.linear,
            "batch_size instances embed_dim, instances embed_dim hidden -> batch_size instances hidden",
        )

        out = einops.einsum(
            self.activation(hidden),
            self.unembedding,
            "batch_size instances hidden, instances hidden d_vocab-> batch_size instances d_vocab ",
        )
        return out
