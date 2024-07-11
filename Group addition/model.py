import torch as t
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
import einops


class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = t.nn.Embedding(params.N, params.embed_dim)
        self.Embedding_right = t.nn.Embedding(params.N, params.embed_dim)
        self.linear = t.nn.Linear(params.embed_dim * 2, params.hidden_size, bias=True)
        if params.activation == "gelu":
            self.activation = t.nn.GELU()
        if params.activation == "relu":
            self.activation = t.nn.ReLU()
        self.Umbedding = t.nn.Linear(params.hidden_size, params.N, bias=True)

    def forward(self, a):
        x1 = self.Embedding_left(a[0])
        x2 = self.Embedding_right(a[1])
        x12 = t.cat([x1, x2], -1)
        hidden = self.linear(x12)
        hidden = self.activation(hidden)
        out = self.Umbedding(hidden)
        return out


class MLP2(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = t.nn.Embedding(params.N, params.embed_dim)
        self.Embedding_right = t.nn.Embedding(params.N, params.embed_dim)
        self.linear_left = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear_right = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        if params.activation == "gelu":
            self.activation = t.nn.GELU()
        if params.activation == "relu":
            self.activation = t.nn.ReLU()
        self.Umbedding = t.nn.Linear(params.hidden_size, params.N, bias=True)

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


class MLP3(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = t.nn.Linear(params.instances, params.N, params.embed_dim)

        self.Embedding_right = t.nn.Linear(params.instances, params.N, params.embed_dim)
        self.linear = t.nn.Linear(
            params.instances, params.embed_dim, params.hidden_size, bias=True
        )

        self.params = params

        if params.activation == "gelu":
            self.activation = t.nn.GELU()
        if params.activation == "relu":
            self.activation = t.nn.ReLU()
        self.Umbedding = t.nn.Linear(
            params.instances, params.hidden_size, params.N, bias=True
        )

    # entries =2
    def forward(self, a: Int[t.Tensor, "batch_size entries"]):

        a_instances = einops.repeat(
            a, " batch_size entries -> batch_size n entries", n=self.params.instances
        )  # batch_size instances entries

        a_1, a_2 = a_instances[:, :, 0], a_instances[:, :, 1]

        a_1_onehot = F.one_hot(a_1, num_classes=self.params.N)
        a_2_onehot = F.one_hot(a_2, num_classes=self.params.N)

        x1 = self.Embedding_left(a_1_onehot)
        x2 = self.Embedding_right(a_2_onehot)

        hidden = self.linear(x1 + x2)

        hidden_post_activation = self.activation(hidden)
        out = self.Umbedding(hidden_post_activation)
        return out
