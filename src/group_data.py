from __future__ import annotations
from torch.utils.data import Dataset
import torch as t
from torch import nn
import random
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from typing import Callable, Union, Any, Optional
from utils import *
import utils
from itertools import product
from collections import defaultdict
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import math
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.combinatorics.named_groups import AlternatingGroup
from tqdm import tqdm
from methodtools import lru_cache
import os
GAP_ROOT = '/usr/share/gap'
if os.path.isdir(GAP_ROOT):
    os.environ['GAP_ROOT'] = GAP_ROOT
    from gappy import gap
    from gappy.gapobj import GapObj
else:
    print('WARNING: GAP is not installed!')


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
        self.name = name

    def elem_to_idx(self, elem):
        return self.elements.index(elem)

    def idx_to_elem(self, idx):
        return self.elements[idx]

    @lru_cache(maxsize=None)
    def identity(self):
        for a in self.elements:
            if all(self.mult(a, b) == b == self.mult(b, a) for b in self.elements):
                return a
        return None

    @lru_cache(maxsize=None)
    def is_unital(self):
        return self.identity() is not None

    def __len__(self):
        return len(self.elements)

    def cayley_set(self):
        return {
            (i, j, self.cayley_table[i, j].item())
            for i in range(len(self))
            for j in range(len(self))
        }

    def mult(self, a, b):
        return self.idx_to_elem(
            self.cayley_table[self.elem_to_idx(a), self.elem_to_idx(b)]
        )

    def __repr__(self):
        return f"{self.name}({self.elements}, {self.cayley_table})"

    @lru_cache(maxsize=None)
    def inv(self, a):
        for b in self.elements:
            if self.mult(a, b) == self.identity() == self.mult(b, a):
                return b
        return None

    @lru_cache(maxsize=None)
    def has_inverses(self):
        return all(self.inv(a) is not None for a in self.elements)

    @lru_cache(maxsize=None)
    def order(self, a):
        x = a
        n = 1
        while x != self.identity:
            x = self.mult(x, a)
            n += 1
        return n

    @staticmethod
    def from_sympy(
        pgroup: PermutationGroup
    ) -> Group:
        elements = pgroup.elements
        N = len(elements)
        table = t.zeros((N, N), dtype=t.int64)
        for (i, a), (j, b) in product(enumerate(elements), repeat=2):
            table[i, j] = elements.index(a * b)
        return Group(elements, table)

    @staticmethod
    def from_gap(
        group: GapObj
    ) -> Group:
        elements = [str(elem) for elem in group.Elements()]
        N = len(elements)
        gap_table = group.MultiplicationTable()
        table = t.zeros((N, N), dtype=t.int64)
        for i, j in product(range(N), repeat=2):
            table[i, j] = int(gap_table[i, j]) - 1   # gap_table is 1-indexed
        return Group(elements, table)

    # def to_gap(self) -> GapObj:
    #     return gap.GroupByMultiplicationTable((self.cayley_table + 1).tolist())  # gap table is 1-indexed

    def to_gap(self) -> GapObj:
        N = len(self.elements)
        f = gap.FreeGroup(gap(N))
        gens = gap.GeneratorsOfGroup(f)
        rels = []
        for i, j in product(range(N), repeat=2):
            rels.append(gens[i] * gens[j] / gens[self.cayley_table[i, j]])
        return f / rels

    @staticmethod
    def from_model(
        model: nn.Module,
        instance: int,
        elements: None,
    ) -> Group:
        table = utils.model_table(model[instance]).squeeze(0)
        N = table.shape[0]
        elements = list(range(N)) if not elements else elements
        return Group(elements, table)

    @lru_cache(maxsize=None)
    def is_abelian(self) -> bool:
        return (self.cayley_table == self.cayley_table.T).all()

    @lru_cache(maxsize=None)
    def is_associative(self, verbose: bool=False) -> bool:
        # faster to operate directly on table instead of using mult
        itr = product(range(len(self)), repeat=3)
        if verbose:
            itr = tqdm(itr, desc='associativity', total=len(self)**3)
        table = self.cayley_table
        for i, j, k in itr:
            if table[i, table[j, k]] != table[table[i, j], k]:
                if verbose:
                    print(f'Associativity failed on {i}, {j}, {k}')
                return False
        return True

    @lru_cache(maxsize=None)
    def is_group(self, verbose: bool=False) -> bool:
        return self.is_unital() and self.has_inverses() and self.is_associative(verbose=verbose)

    @lru_cache(maxsize=None)
    def get_classes(self):
        '''Returns list of conjugacy classes'''
        elems_remain = set(self.elements)
        # make sure identity goes first
        ret = [{self.identity()}]
        elems_remain.remove(self.identity())
        while elems_remain:
            a = elems_remain.pop()
            conj_class = {a}
            for b in self.elements:
                c = self.mult(self.inv(b), self.mult(a, b))
                elems_remain.discard(c)
                conj_class.add(c)
            ret.append(conj_class)
        return ret

    @lru_cache(maxsize=None)
    def get_class_of(self, a):
        '''Returns a\'s conjugacy class'''
        return [cl for cl in self.get_classes() if a in cl][0]

    @lru_cache(maxsize=None)
    def get_char_table(self, uniq_thresh=1e-2, zero_thresh=0):
        '''
        Returns the (num_classes x num_classes) character table over \C using Burnside-Dixon.
        See Eick et al. "Handbook of Computational Group Theory" p. 257
        '''
        classes = self.get_classes()
        r = len(classes)
        M = t.zeros((r, r, r))
        for j, k, l in product(range(r), repeat=3):
            l_elem = list(classes[l])[0]
            M[j, k, l] = sum(1 for x, y in product(classes[j], classes[k]) if self.mult(x, y) == l_elem)
        # Need shared row eigenvectors of M[0], ..., M[r]
        # Do this by getting eig of \sum_i a_i M[i] for random a_i
        # Will recover uniquely as long as eigenvalues are unique.
        chars = None
        while chars is None:
            a = t.randn(r)
            aM = einops.einsum(a, M, 'j, j k l -> k l')
            L, V = t.linalg.eig(aM.T)
            # check that all eigvalues in L are unique
            L = L.unsqueeze(1)
            if (L - L.T + t.eye(r)).abs().min() > uniq_thresh:
                chars = V.T
            else:
                print('failed', (L - L.T + t.eye(r)).abs().min())
                
        class_sizes = t.Tensor([len(c) for c in classes]).unsqueeze(0)
        char_norms = (chars * chars.conj() * class_sizes / len(self)).sum(1).unsqueeze(1).sqrt()
        chars /= char_norms
        # Rescale by sgn so that first column is all reals
        # Assumes that identity is first in self.get_classes()
        chars /= (chars[:,0].sgn().unsqueeze(1))
        # Snap small real/complex parts to zero
        snap = lambda x: x * (x.abs() > zero_thresh)
        return t.complex(snap(chars.real), snap(chars.imag))

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
            cayley_table[i, j] = min(i + j, N - 1)
    return Group(elements, cayley_table)


@jaxtyped(typechecker=beartype)
def randm(N: int) -> Group:
    elements = list(range(N))
    cayley_table = t.randint(0, N, (N, N), dtype=t.int64)
    return Group(elements, cayley_table)

def permute(
    group: Group,
    pi: Callable,
    pi_inv: Callable,
) -> Group:
    table = t.zeros_like(group.cayley_table)
    for (i, x), (j, y) in product(enumerate(group.elements), repeat=2):
        table[i, j] = group.elem_to_idx(pi_inv(group.mult(pi(x), pi(y))))
    return Group(group.elements, table)

def swapZ(N: int) -> Group:
    def pi(x):
        nonlocal N
        a = 1
        b = N-1 # N-2
        if x == a:
            return b
        elif x == b:
            return a
        else:
            return x
    return permute(Z(N), pi, pi)

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
# TODO: this is a stupid name
def twisted(group: Group, automorphism: Callable[..., Any], m: int=2) -> Group:
    def phi(x):
        def aut(y):
            ret = y
            for _ in range(x):
                ret = automorphism(ret)
            return ret
        return aut
        
    return semidirect_product(group, cyclic(m), phi)


@jaxtyped(typechecker=beartype)
def D(N: int) -> Group:
    """Dihedral group"""
    return twisted(cyclic(N), lambda x: (N - x) % N)


@jaxtyped(typechecker=beartype)
# TODO: this is a stupid name
def holoconj(group: Group, a: Any) -> Group:
    """
    Given group G, returns semidirect G x Z(m) where Z(m) acts on G by conjugation by elem
    and m is the order of elem.
    """
    m = group.order(a)
    phi = lambda x, a=a: group.mult(group.mult(a, x), group.inv(a))
    return twisted(group, phi, m=m)


@jaxtyped(typechecker=beartype)
def twZ(N: int) -> Group:
    return twisted(cyclic(N), lambda x, N=N: ((N // 2 + 1) * x) % N)


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

    

def abFam(a: int, b:int, r=None) -> Optional[list[Group]]:
    """
    Returns two semidirect products Z/ab x Z/m where the automorphisms are
    multiplication by either r or a-r, and r is chosen such that
    ord(r)=ord(m-r)=m.
    """
    base_group = Z(a * b)

    # multiplicative order
    def order(c):
        assert c in base_group.elements
        assert math.gcd(c, a * b) == 1, 'c must be unit of Z(a*b)'
        x = c
        n = 1
        while x != 1:
            x = (x * c) % (a * b)
            n += 1
        return n

    min_r = None
    min_s = None
    min_lcm = None
    r_range = [r] if r else range(2, a * b)
    for r in r_range:
        s = (a - r) % (a * b)
        if s == 1 or math.gcd(r, a * b) != 1 or math.gcd(s, a * b) != 1:
            continue
        lcm = math.lcm(order(r), order(s))
        # print('abFam:', r, s, lcm)
        if min_r is None or lcm < min_lcm:
            min_lcm = lcm
            min_r = r
            min_s = s

    r, s, m = min_r, min_s, min_lcm
    print(f'Found r={r} and s={s} of order {m}')
    # Make sure to capture r in the nested lambdas
    phi_r = lambda x, r=r: lambda y, r=r: (y * r ** x) % (a * b)
    phi_s = lambda x, s=s: lambda y, s=s: (y * s ** x) % (a * b)
    return [semidirect_product(base_group, Z(m), phi=phi) for phi in [phi_r, phi_s]]

def A(n: int) -> Group:
    return Group.from_sympy(AlternatingGroup(n))

def S(n: int) -> Group:
    # Construct this as semidirect prod of A(n) and Z/2
    # To have consistent labeling with A(n) x Z/2
    Sn = twisted(
        A(n),
        lambda p: Permutation(0, 1) * p * Permutation(0, 1),
        m=2,
    )
    Sn.elements = [n * Permutation(0, 1) if h else n for n, h in Sn.elements]
    return Sn

@jaxtyped(typechecker=beartype)
def string_to_groups(strings: Union[str, tuple[str, ...], list[str]]) -> list[Group]:
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

        self.group_sets = [group.cayley_set() for group in self.groups]

        if isinstance(params.delta_frac, Union[int, float]):
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
        if params.train_frac < 1.:
            print(f"Taking random subset:", frac_str(len(self.train_data), len(train_set)))
        print(f"Train set size: {frac_str(len(self.train_data), self.N**2)}")

    def __getitem__(self, idx):
        return (
            t.tensor([self.train_data[idx][0], self.train_data[idx][1]]),
            self.train_data[idx][2],
        )

    def __len__(self):
        return len(self.train_data)

    def frequency_histogram(self):
        self.distribution = [x[2] for x in self.train_data]
        frequency_count = Counter(self.distribution)
        x = list(frequency_count.keys())
        y = list(frequency_count.values())

        sorted_pairs = sorted(zip(x, y))
        x_sorted, y_sorted = zip(*sorted_pairs)
        plt.figure(figsize=(10, 6))
        plt.bar(x_sorted, y_sorted)

        plt.xlabel("Integers")
        plt.ylabel("Frequency")
        plt.title("Frequency Plot of Integers")
