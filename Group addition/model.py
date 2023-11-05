import torch as t
from transformer_lens import (
    utils,
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Size of the set endowed with the group structure

device = t.device("cuda" if t.cuda.is_available() else "cpu")
