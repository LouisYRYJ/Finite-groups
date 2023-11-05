import torch as t
import einops


class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding_left = t.nn.Embedding(params.N, params.embed_dim, bias=True)
        self.Embedding_right = t.nn.Embedding(params.N, params.embed_dim, bias=True)
        self.linear = t.nn.Linear(params.embed_dim * 2, params.hidden_size, bias=True)
        self.activation = t.nn.GELU()
        self.Umbedding = t.nn.Linear(params.hidden_size, params.N, bias=True)

    def forward(self, a, b):
        x1 = self.Embedding_left(a)
        x2 = self.Embedding_right(b)
        x12 = einops.rearrange([x1, x2], axis=-1)
        hidden = self.linear(x12)
        hidden = self.activation(hidden)
        out = self.Umbedding(hidden)
        return out
