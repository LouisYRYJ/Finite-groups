import torch as t
import torch.nn.functional as F
from torch import nn
from jaxtyping import Float, Int, jaxtyped
from typing import Optional, Callable, Union, List, Tuple
import einops
import numpy as np
from beartype import beartype
from copy import deepcopy
from abc import ABC, abstractmethod


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding_left = nn.Embedding(params.N, params.embed_dim)
        self.embedding_right = nn.Embedding(params.N, params.embed_dim)
        self.linear = nn.Linear(params.embed_dim * 2, params.hidden_size, bias=True)
        if params.activation == "gelu":
            self.activation = nn.GELU()
        if params.activation == "relu":
            self.activation = nn.ReLU()
        self.unembedding = nn.Linear(params.hidden_size, params.N, bias=True)

    def forward(self, a):
        x1 = self.embedding_left(a[0])
        x2 = self.embedding_right(a[1])
        x12 = t.cat([x1, x2], -1)
        hidden = self.linear(x12)
        hidden = self.activation(hidden)
        out = self.unembedding(hidden)
        return out


class MLP2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding_left = nn.Embedding(params.N, params.embed_dim)
        self.embedding_right = nn.Embedding(params.N, params.embed_dim)
        self.linear_left = nn.Linear(params.embed_dim, params.hidden_size, bias=False)
        self.linear_right = nn.Linear(params.embed_dim, params.hidden_size, bias=False)
        if params.activation == "gelu":
            self.activation = nn.GELU()
        if params.activation == "relu":
            self.activation = nn.ReLU()
        self.unembedding = nn.Linear(params.hidden_size, params.N, bias=False)

    def forward(self, a):

        a_1, a_2 = a[:, 0], a[:, 1]
        x1 = self.embedding_left(a_1)
        x2 = self.embedding_right(a_2)
        hidden_x1 = self.linear_left(x1)
        hidden_x2 = self.linear_right(x2)
        hidden_sum = hidden_x1 + hidden_x2
        hidden = self.activation(hidden_sum)
        out = self.unembedding(hidden)
        return out


def custom_xavier(dims):
    # Assumes shape [..., fan_in, fan_out]
    return nn.Parameter(t.randn(dims) * np.sqrt(2.0 / float(dims[-2] + dims[-1])))


def custom_kaiming(dims):
    # Assumes shape [..., fan_in, fan_out]
    return nn.Parameter(t.randn(dims) * np.sqrt(2.0 / float(dims[-2])))

    
class InstancedModule(ABC, nn.Module):
    '''
    Module with instance dimension to allow for parallel runs
    '''
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, slice):
        """
        Returns a new model with parameters sliced along the instances dimension.
        """
        ret = deepcopy(self)
        for name, param in self.named_parameters():
            sliced_param = param[slice]
            if isinstance(slice, int):
                sliced_param = sliced_param.unsqueeze(0)
            setattr(ret, name, nn.Parameter(sliced_param.clone()))
        return ret

    def repeat(self, n):
        """
        Returns a new model repeated along instances dimension.
        """
        ret = deepcopy(self)
        for name, param in self.named_parameters():
            rep_param = einops.repeat(param, 'instance ... -> (n instance) ...', n=n)
            setattr(ret, name, nn.Parameter(rep_param.clone()))
        return ret

    def num_instances(self):
        # this is not great. maybe explicitly store num_instances as int?
        for k, v in self.named_parameters():
            return v.shape[0]

    def __len__(self):
        return self.num_instances()

    @staticmethod
    def stack(models):
        ret = deepcopy(models[0])
        for name, param in models[0].named_parameters():
            all_params = [model.get_parameter(name) for model in models]
            stacked_param = einops.rearrange(all_params, 'model instance ... -> (model instance) ...')
            setattr(ret, name, nn.Parameter(stacked_param.clone()))
        return ret

    @abstractmethod
    def _forward(
        self, a: Int[t.Tensor, "batch entry"]
    ) -> Float[t.Tensor, "batch instance vocab"]:
        pass

    # @jaxtyped(typechecker=beartype)
    def forward(
        self, 
        a: Int[t.Tensor, "batch entry"], 
        ibatch_size: int=-1,
    ) -> Float[t.Tensor, "batch instance vocab"]:
        '''
        Forward pass batched along instances.
        '''
        if ibatch_size < 0:
            return self._forward(a)
        
        batched_models = [
            self[i: i + ibatch_size] for i in range(0, self.num_instances(), ibatch_size)
        ]
        out = [model._forward(a) for model in batched_models]
        # can't use einops rearrange here bc variable number of instances per ibatch
        return t.concat(out, dim=1)

class MLP3(InstancedModule):
    # TODO: It's maybe better practice to explicitly list the args (instances, embed_dim, etc)
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

    @jaxtyped(typechecker=beartype)
    def _forward(
        self, a: Int[t.Tensor, "batch_size entries"]
    ) -> Float[t.Tensor, "batch_size instances d_vocab"]:

        a_instances = einops.repeat(
            a, " batch_size entries -> batch_size n entries", n=self.num_instances(),
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


class MLP4(InstancedModule):
    """
    Architecture studied in Morwani et al. "Feature emergence via margin maximization"
    """
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

        self.unembedding = custom_kaiming([params.instances, params.embed_dim, self.N])

        if params.activation == "gelu":
            self.activation = nn.GELU()
        elif params.activation == "relu":
            self.activation = nn.ReLU()
        elif params.activation == "linear":
            self.activation = lambda x: x
        elif params.activation == "square":
            self.activation = lambda x: x**2
        else:
            raise ValueError("Activation not recognized")

    @jaxtyped(typechecker=beartype)
    def _forward(
        self, a: Int[t.Tensor, "batch_size entries"]
    ) -> Float[t.Tensor, "batch_size instances d_vocab"]:

        a_instances = einops.repeat(
            a, " batch_size entries -> batch_size n entries", n=self.num_instances(),
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

        hidden = x_1 + x_2

        out = einops.einsum(
            self.activation(hidden),
            self.unembedding,
            "batch_size instances embed_dim, instances embed_dim d_vocab-> batch_size instances d_vocab ",
        )
        return out


class Normal(InstancedModule):
    '''Just for LLC testing'''
    def __init__(self, N, instances):
        super().__init__()
        self.w = nn.Parameter(t.zeros((instances, N)))
        
    def _forward(self, x):
        return t.prod(self.w**2, dim=1)


MODEL_DICT = {
    "MLP": MLP,
    "MLP2": MLP2,
    "MLP3": MLP3,
    "MLP4": MLP4,
}
