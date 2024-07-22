from torch.utils.data import Dataset
import torch as t
import random
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from typing import Callable, Union, Any
from utils import *
from itertools import product
from collections import defaultdict
import warnings


# TODO: Find a better place to put this.
# Can't put in utils b/c utils imports GroupData
@jaxtyped(typechecker=beartype)
def frac_str(a: Union[int, float], b: Union[int, float]) -> str:
    return f"{a}/{b} ({a/b:.2f})"


def random_frac(full_dataset, frac):
    num_indices = int(len(full_dataset) * frac)
    return random.sample(list(full_dataset), num_indices)


class Group:
    """
    Implements a group (technically, any magma) as a idx -> element lookup table and a Cayley table on idxs.
    """

    def __init__(self, elements, cayley_table, name="group"):
        self.elements = elements
        self.cayley_table = cayley_table
        self.identity = None
        self.name = name
        for a in self.elements:
            if all(self.mult(a, b) == b == self.mult(b, a) for b in self.elements):
                self.identity = a
                break

    def elem_to_idx(self, elem):
        return self.elements.index(elem)

    def idx_to_elem(self, idx):
        return self.elements[idx]

    def __len__(self):
        return len(self.elements)

    def mult(self, a, b):
        return self.idx_to_elem(
            self.cayley_table[self.elem_to_idx(a), self.elem_to_idx(b)]
        )

    def __repr__(self):
        return f"{self.name}({self.elements}, {self.cayley_table})"

    # TODO: more efficient if inv, order, etc are cached.
    def inv(self, a):
        for b in self.elements:
            if self.mult(a, b) == self.identity:
                return b

    def order(self, a):
        x = a
        n = 1
        while x != self.identity:
            x = self.mult(x, a)
            n += 1
        return n


@jaxtyped(typechecker=beartype)
def cyclic(N: int) -> Group:
    elements = list(range(N))
    cayley_table = t.zeros((N, N), dtype=t.int64)
    for i in range(N):
        for j in range(N):
            cayley_table[i, j] = (i + j) % N
    return Group(elements, cayley_table)


@jaxtyped(typechecker=beartype)
def clamped(N: int) -> Group:
    elements = list(range(N))
    cayley_table = t.zeros((N, N), dtype=t.int64)
    for i in range(N):
        for j in range(N):
            cayley_table[i, j] = min(i + j, N-1)
    return Group(elements, cayley_table)


@jaxtyped(typechecker=beartype)
def randm(N: int) -> Group:
    elements = list(range(N))
    cayley_table = t.randint(0, N, (N, N), dtype=t.int64)
    return Group(elements, cayley_table)


@jaxtyped(typechecker=beartype)
def semidirect_product(
    group1: Group, group2: Group, phi: Callable[..., Callable[..., Any]]
) -> Group:
    N1 = len(group1)
    N2 = len(group2)
    elements = [
        (a, b) for b in group2.elements for a in group1.elements
    ]  # right-to-left lexicographic order
    cayley_table = t.zeros((N1 * N2, N1 * N2), dtype=t.int64)
    for i in range(N1 * N2):
        for j in range(N1 * N2):
            a1, b1 = elements[i]
            a2, b2 = elements[j]
            cayley_table[i, j] = elements.index(
                (group1.mult(a1, phi(b1)(a2)), group2.mult(b1, b2))
            )
    return Group(elements, cayley_table)


@jaxtyped(typechecker=beartype)
def direct_product(group1: Group, group2: Group) -> Group:
    return semidirect_product(group1, group2, lambda x: lambda y: y)


times = direct_product


@jaxtyped(typechecker=beartype)
def Z(*args: int) -> Group:
    """
    Convenience function for products of cyclic groups.
    """
    group = cyclic(args[0])
    for arg in args[1:]:
        group = direct_product(group, cyclic(arg))
    return group


@jaxtyped(typechecker=beartype)
def twisted(group: Group, automorphism: Callable[..., Any]) -> Group:
    phi = lambda x: automorphism if x else lambda y: y
    return semidirect_product(group, cyclic(2), phi)


@jaxtyped(typechecker=beartype)
def D(N: int) -> Group:
    """Dihedral group"""
    return twisted(cyclic(N), lambda x: (N - x) % N)


@jaxtyped(typechecker=beartype)
def holoconj(group: Group, a: Any) -> Group:
    """
    Given group G, returns semidirect G x Z(m) where Z(m) acts on G by conjugation by elem
    and m is the order of elem. (This is a subgroup of hol(G))
    """
    m = group.order(a)

    def phi(x):
        def aut(y):
            ret = y
            for _ in range(x):
                ret = group.mult(group.mult(a, ret), group.inv(a))
            return ret

        return aut

    return semidirect_product(group, cyclic(m), phi)


@jaxtyped(typechecker=beartype)
def twZ(N: int) -> Group:
    return twisted(cyclic(N), lambda x: ((N // 2 + 1) * x) % N)


@jaxtyped(typechecker=beartype)
def XFam(N: int) -> list[Group]:
    """
    List of all possible twisted(Z(N, N), automorphism) groups such that
    automorphism is order 2 and preserves the diagonal of Z(N, N)
    """
    ret = []
    for a, b, c, d in product(range(N), repeat=4):
        # Automorphisms on Z(N, N) can be written as 2x2 matrices [[a, b], [c, d]] over Z(N)
        conds = [
            # Automorphism is order 2
            (a**2 + b * c) % N == 1,
            (d**2 + b * c) % N == 1,
            (a * b + b * d) % N == 0,
            (a * c + c * d) % N == 0,
            # Automorphism preserves the diagonal
            (a + b) % N == 1,
            (c + d) % N == 1,
        ]
        if all(conds):
            # a=a etc needed to capture values of a, b, c, d
            aut = lambda x, a=a, b=b, c=c, d=d: (
                (a * x[0] + b * x[1]) % N,
                (c * x[0] + d * x[1]) % N,
            )
            group = twisted(Z(N, N), aut)
            group.name = f"XFam({N},[[{a},{b}],[{c},{d}]])"
            ret.append(group)
    return ret


@jaxtyped(typechecker=beartype)
def string_to_groups(strings: Union[str, tuple[str, ...]]) -> list[Group]:
    """
    Input string s should be calls to above functions (returning NxN multiplication tables), delimited by ';'
    """
    if isinstance(strings, str):
        strings = strings.split(";")
    ret = []
    for s in strings:
        group = eval(s)
        if isinstance(group, Group):
            if group.name == "group":
                group.name = s
            ret.append(group)
        elif isinstance(group, tuple) or isinstance(group, list):
            for i, g in enumerate(group):
                if g.name == "group":
                    g.name = f"{s}_{i}"
            ret.extend(group)
        else:
            raise ValueError(f"Invalid group: {s}")
    return ret


class GroupData(Dataset):
    def __init__(self, params):
        self.groups = string_to_groups(params.group_string)
        self.num_groups = len(self.groups)
        self.N = len(self.groups[0])
        for group in self.groups:
            assert len(group) == self.N, "All groups must be of equal size."

        # Deduplicate group names
        names_dict = defaultdict(int)
        for group in self.groups:
            if names_dict[group.name] > 0:
                new_name = f"{group.name}_{names_dict[group.name]+1}"
                warnings.warn(
                    f"Duplicate group name: {group.name}. Deduplicating to {new_name}"
                )
                group.name = new_name
            names_dict[group.name] += 1

        self.group_sets = [
            {
                (i, j, group.cayley_table[i, j].item())
                for i in range(self.N)
                for j in range(self.N)
            }
            for group in self.groups
        ]

        if not isinstance(params.delta_frac, list):
            params.delta_frac = [params.delta_frac] * self.num_groups
        if len(self.groups) == 1:
            self.group_deltas = [set()]
        else:
            self.group_deltas = [
                self.group_sets[i]
                - set.union(
                    *[self.group_sets[j] for j in range(self.num_groups) if j != i]
                )
                for i in range(len(self.groups))
            ]

        intersect = set.intersection(*self.group_sets)
        print(f"Intersection size: {frac_str(len(intersect), len(self.group_sets[0]))}")

        train_set = set()
        train_set |= set(random_frac(intersect, params.intersect_frac))
        print(f"Added {len(train_set)} elements from intersection")
        for i in range(self.num_groups):
            to_add = set(random_frac(self.group_deltas[i], params.delta_frac[i]))
            train_set |= to_add
            print(f"Added {len(to_add)} elements from group {i}: {self.groups[i].name}")

        self.train_data = random_frac(list(train_set), params.train_frac)
        print(f"Taking random subset:", frac_str(len(self.train_data), len(train_set)))
        print(f"Train set size: {len(self.train_data)}")

    def __getitem__(self, idx):
        return (
            t.tensor([self.train_data[idx][0], self.train_data[idx][1]]),
            self.train_data[idx][2],
        )

    def __len__(self):
        return len(self.train_data)
