import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import devinterp
import random
from model import MLP
import wandb
from dataclasses import dataclass
from groups_data import IntersectionData


@dataclass
class ExperimentsParameters:
    N_1: int = 7
    N_2: int = 2
    N: int = N_1 * N_1 * N_2
    embed_dim: int = 16
    hidden_size: int = 64
    num_epochs: int = 20000
    checkpoint_every: int = 500
    batch_size: int = 256


ExperimentsParameter = ExperimentsParameters()


# Creating DataLoader object
dataset = IntersectionData()
train_loader = DataLoader(
    dataset=dataset,
    batch_size=ExperimentsParameter.batch_size,
    shuffle=True,
    drop_last=True,
)
model = MLP(ExperimentsParameter)

for x, y, z in train_loader:
    print(model(x, y).shape)
