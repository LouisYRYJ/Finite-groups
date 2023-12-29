import torch as t


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
        x1 = self.Embedding_left(a[0])
        x2 = self.Embedding_right(a[1])
        hidden_x1 = self.linear_left(x1)
        hidden_x2 = self.linear_right(x2)
        hidden_sum = hidden_x1 + hidden_x2
        hidden = self.activation(hidden_sum)
        out = self.Umbedding(hidden)
        return out
