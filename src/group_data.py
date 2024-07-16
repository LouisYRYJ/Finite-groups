from torch.utils.data import Dataset
import torch as t
import random
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from typing import Callable, Union
from utils import *

# TODO: Find a better place to put this.
# Can't put in utils b/c utils imports GroupData
@jaxtyped(typechecker=beartype)
def frac_str(a: Union[int, float], b: Union[int, float]) -> str:
    return f'{a}/{b} ({a/b:.2f})'


def random_frac(full_dataset, frac):
    num_indices = int(len(full_dataset) * frac)
    return random.sample(list(full_dataset), num_indices)


@jaxtyped(typechecker=beartype)
def twisted(
    group: Int[t.Tensor, "N N"], 
    automorphism: Callable[int, int] =lambda x: x
) -> Int[t.Tensor, "2*N 2*N"]:
    """Constructs semidirect product of groups with Z/2Z using the given automorphism"""
    group_cardinality = group.size(dim=0)
    new_cardinality = group_cardinality * 2
    new_group = t.zeros((new_cardinality, new_cardinality), dtype=t.int64)

    for i in range(new_cardinality):
        for j in range(new_cardinality):
            if i < group_cardinality and j < group_cardinality:
                new_group[i, j] = group[i, j]

            if i < group_cardinality and j >= group_cardinality:
                new_group[i, j] = group[i, j - group_cardinality] + group_cardinality

            if i >= group_cardinality and j < group_cardinality:
                new_group[i, j] = (
                    group[i - group_cardinality, automorphism(j) % group_cardinality]
                    + group_cardinality
                )

            if i >= group_cardinality and j >= group_cardinality:
                new_group[i, j] = group[
                    i - group_cardinality,
                    automorphism(j - group_cardinality) % group_cardinality,
                ]

    return new_group


@jaxtyped(typechecker=beartype)
def cyclic(N: int) -> Int[t.Tensor, "N N"]:
    cyclic_group = t.zeros((N, N), dtype=t.int64)
    for i in range(N):
        for j in range(N):
            cyclic_group[i, j] = (i + j) % N
    return cyclic_group


@jaxtyped(typechecker=beartype)
def random_magma(N: int) -> Int[t.Tensor, "N N"]:
    rand_magma = t.zeros((N, N), dtype=t.int64)
    for i in range(N):
        for j in range(N):
            rand_magma[i, j] = random.randint(0, N-1)
    return rand_magma

    
@jaxtyped(typechecker=beartype)
def string_to_groups(
    s: Union[str, tuple[str, ...]]
) -> tuple[Int[t.Tensor, "N N"], ...]:
    '''
    Input string s should be calls to above functions (returning NxN multiplication tables), delimited by ';'
    '''
    if isinstance(s, str):
        s = s.split(';')
    return tuple(map(eval, s))


class GroupData(Dataset):
    def __init__(self, params):
        self.groups = string_to_groups(params.group_string)
        self.num_groups = len(self.groups)
        for i in range(1, self.num_groups):
            assert self.groups[i].size(0) == self.groups[i].size(1) == self.groups[0].size(0), 'All groups must be of equal size.'
        self.N = self.groups[0].size(0)
        self.group_sets = [
            {
                (i, j, group[i, j].item())
                for i in range(self.N)
                for j in range(self.N)
            }
            for group in self.groups
        ]

        if not isinstance(params.delta_frac, list):
            params.delta_frac = [params.delta_frac] * self.num_groups
        self.group_deltas = [
            self.group_sets[i] - set.union(*[self.group_sets[j] for j in range(self.num_groups) if j != i])
            for i in range(len(self.groups))
        ]

        intersect = set.intersection(*self.group_sets)
        print(f'Intersection size: {frac_str(len(intersect), len(self.group_sets[0]))}')

        train_set = set()
        train_set |= set(random_frac(intersect, params.intersect_frac))
        print(f'Added {len(train_set)} elements from intersection')
        for i in range(self.num_groups):
            to_add = set(random_frac(self.group_deltas[i], params.delta_frac[i]))
            train_set |= to_add
            print(f"Added {len(to_add)} elements from group {i}: {params.group_string[i]}")

        self.train_data = random_frac(list(train_set), params.train_frac)
        print(f'Train set size: {len(train_set)}')

    def __getitem__(self, idx):
        return (
            t.tensor([self.train_data[idx][0], self.train_data[idx][1]]),
            self.train_data[idx][2],
        )

    def __len__(self):
        return len(self.train_data)
