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
import pathlib
import hashlib
import os

ROOT = pathlib.Path(__file__).parent.parent.resolve()
GAP_ROOT = "/usr/share/gap"
if os.path.isdir(GAP_ROOT):
    os.environ["GAP_ROOT"] = GAP_ROOT
    from gappy import gap
    from gappy.gapobj import GapObj
else:
    print("WARNING: GAP is not installed!")


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
    Implements a group as a idx -> element lookup table and a Cayley table on idxs.
    Generally, methods ending in _idx operate on idxs, while those without operate on elements.
    """

    def __init__(self, elements, cayley_table, name="group"):
        self.elements = elements
        self.cayley_table = cayley_table
        self.name = name

    @lru_cache(maxsize=None)  # lru_cache is probably faster than index?
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
    def identity_idx(self):
        return self.elem_to_idx(self.identity())

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
            self.cayley_table[self.elem_to_idx(a), self.elem_to_idx(b)].item()
        )

    def mult_idx(self, a, b):
        return self.cayley_table[a, b].item()

    def __repr__(self):
        return f"{self.name}({self.elements}, {self.cayley_table})"

    @lru_cache(maxsize=None)
    def inv(self, a):
        for b in self.elements:
            if self.mult(a, b) == self.identity() == self.mult(b, a):
                return b
        return None

    @lru_cache(maxsize=None)
    def inv_idx(self, idx):
        return self.elem_to_idx(self.inv(self.idx_to_elem(idx)))

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
    def from_sympy(pgroup: PermutationGroup) -> Group:
        elements = pgroup.elements
        N = len(elements)
        table = t.zeros((N, N), dtype=t.int64)
        for (i, a), (j, b) in product(enumerate(elements), repeat=2):
            table[i, j] = elements.index(a * b)
        return Group(elements, table)

    @staticmethod
    def from_gap(group: GapObj) -> Group:
        elements = [str(elem) for elem in group.Elements()]
        N = len(elements)
        gap_table = group.MultiplicationTable()
        table = t.zeros((N, N), dtype=t.int64)
        for i, j in product(range(N), repeat=2):
            table[i, j] = int(gap_table[i, j]) - 1  # gap_table is 1-indexed
        return Group(elements, table)

    def to_gap(self) -> GapObj:
        return gap.GroupByMultiplicationTable(
            (self.cayley_table + 1).tolist()
        )  # gap table is 1-indexed

    def to_gap_fp(self) -> GapObj:
        N = len(self.elements)
        f = gap.FreeGroup(gap(N))
        gens = gap.GeneratorsOfGroup(f)
        rels = []
        for i, j in product(range(N), repeat=2):
            rels.append(gens[i] * gens[j] / gens[self.mult_idx(i, j)])
        return f / rels

    @lru_cache(maxsize=None)
    def exponent(self, a, power):
        if power == 0:
            return self.identity()
        elif power < 0:
            return self.exponent(self.inv(a), -power)
        else:
            return self.mult(a, self.exponent(a, power - 1))

    @lru_cache(maxsize=None)
    def exponent_idx(self, idx, power):
        return self.elem_to_idx(self.exponent(self.idx_to_elem(idx), power))

    def fp_elem_to_idx(self, fp_elem):
        '''
        GAP fp groups have elements like f1^2*f2^3*f3^-1. Parse to idx.
        '''
        fp_elem = str(fp_elem).replace('*', '')  # * not necessary for parsing
        if 'identity' in fp_elem:
            return self.identity_idx()

        def get_power(fp_elem):
            # returns power, remaining string
            if not fp_elem or fp_elem[0] != '^':
                return 1, fp_elem
            else:
                power = ''
                fp_elem = fp_elem[1:]
                while fp_elem and (fp_elem[0].isdigit() or fp_elem[0] == '-'):
                    power += fp_elem[0]
                    fp_elem = fp_elem[1:]
                return int(power), fp_elem

        def next_token(fp_elem):
            # returns token, power, remaining string
            if fp_elem[0] == '(':
                return '(', None, fp_elem[1:]
            elif fp_elem[0] == ')':
                fp_elem = fp_elem[1:]
                power, fp_elem = get_power(fp_elem)
                return ')', power, fp_elem
            elif fp_elem[0] == 'f':
                token = ''
                fp_elem = fp_elem[1:]
                while fp_elem and fp_elem[0].isdigit():
                    token += fp_elem[0]
                    fp_elem = fp_elem[1:]
                power, fp_elem = get_power(fp_elem)
                return int(token) - 1, power, fp_elem  # GAP is 1-indexed
            else:
                import pdb; pdb.set_trace()

        stack = [self.identity_idx()]
        while fp_elem:
            token, power, fp_elem = next_token(fp_elem)
            if token == '(':
                stack.append(self.identity_idx())
            elif token == ')':
                cur = stack.pop()
                stack[-1] = self.mult_idx(stack[-1], self.exponent_idx(cur, power))
            else:
                assert isinstance(token, int)
                stack[-1] = self.mult_idx(stack[-1], self.exponent_idx(token, power))

        assert len(stack) == 1, 'Mismatched parentheses!'
        ret = stack[0]
        if isinstance(ret, t.Tensor):
            ret = ret.item()
        return ret

    def fp_elem_to_elem(self, fp_elem):
        return self.idx_to_elem(self.fp_elem_to_idx(fp_elem))

    @lru_cache(maxsize=None)
    def get_subgroups_idx(self, cache_dir=f'{ROOT}/subgroups/') -> list[set]:
        '''Return set of all subgroups of the group'''
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f'{cache_dir}/{self.hash()}'
        if os.path.exists(cache_path):
            return t.load(cache_path)

        print('Computing subgroups')
        gap_subgroups = self.to_gap_fp().LowIndexSubgroupsFpGroup(len(self))
        print('Computing orders')
        gap_subgroups = [g for g in gap_subgroups if g.Order() > 1 and g.Order() < len(self)] # do trivial and full group separately for efficiency
        print('Computing elements')
        subgroups= {frozenset([self.identity_idx()]), frozenset(range(len(self)))}
        subgroups |= {
            frozenset([self.fp_elem_to_idx(elem) for elem in subgroup.Elements()]) for subgroup in tqdm(gap_subgroups, desc='Computing subgroups')
        }
        # print(f"Found {len(subgroups)} subgroups up to conjugates with orders", [len(s) for s in subgroups])

        print('Computing conjugates')
        for h in list(subgroups):
            subgroups |= {frozenset(self.get_conj_subgroup_idx(h, g)) for g in range(len(self))}

        print('Saving to', cache_path)
        t.save(subgroups, cache_path)
        return subgroups

    @lru_cache(maxsize=None)
    def get_subgroups(self, cache_dir=f'{ROOT}/subgroups/') -> list[set]:
        return {frozenset(map(self.idx_to_elem, h)) for h in self.get_subgroups_idx(cache_dir=cache_dir)}

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
    def is_latin(self) -> bool:
        """
        Checks if multiplication table is a latin square (no repeated elements in rows or columns)
        """
        for i in range(len(self)):
            if len(set(self.cayley_table[i].tolist())) != len(self):
                return False
            if len(set(self.cayley_table[:, i].tolist())) != len(self):
                return False
        return True

    @lru_cache(maxsize=None)
    def is_subgroup(self, h: set) -> bool:
        """
        Checks if h (given as set of elements) is a subgroup of self.
        """
        for a, b in product(h, repeat=2):
            # One-step subgroup test
            if self.mult(a, self.inv(b)) not in h:
                return False
        return True

    @lru_cache(maxsize=None)
    def is_abelian(self) -> bool:
        return (self.cayley_table == self.cayley_table.T).all()

    @lru_cache(maxsize=None)
    def is_associative(self, verbose: bool = False) -> bool:
        # faster to operate directly on table instead of using mult
        itr = product(range(len(self)), repeat=3)
        if verbose:
            itr = tqdm(itr, desc="associativity", total=len(self) ** 3)
        table = self.cayley_table
        for i, j, k in itr:
            if table[i, table[j, k]] != table[table[i, j], k]:
                if verbose:
                    print(f"Associativity failed on {i}, {j}, {k}")
                return False
        return True

    @lru_cache(maxsize=None)
    def is_group(self, verbose: bool = False) -> bool:
        return (
            self.is_unital()
            and self.has_inverses()
            and self.is_associative(verbose=verbose)
        )

    @lru_cache(maxsize=None)
    def get_conj(self, a, b):
        '''Conjugate of a by b'''
        return self.mult(self.inv(b), self.mult(a, b))

    @lru_cache(maxsize=None)
    def get_conj_idx(self, a, b):
        return self.mult_idx(self.inv_idx(b), self.mult_idx(a, b))

    @lru_cache(maxsize=1024)
    def get_conj_subgroup(self, subgroup, b):
        '''Conjugate of a subgroup by b'''
        return set([self.get_conj(a, b) for a in subgroup])

    @lru_cache(maxsize=None)
    def get_conj_subgroup_idx(self, subgroup, b):
        return set([self.get_conj_idx(a, b) for a in subgroup])

    @lru_cache(maxsize=None)
    def get_cosets_idx(self, subgroup, left=True):
        '''Returns left/right cosets of subgroup in self'''
        cosets = set()
        for i in range(len(self)):
            if left:
                action = lambda x: self.mult_idx(i, x)
            else:
                action = lambda x: self.mult_idx(x, i)
            cosets.add(frozenset(map(action, subgroup)))
        return cosets

    @lru_cache(maxsize=None)
    def get_classes(self):
        """Returns list of conjugacy classes"""
        elems_remain = set(self.elements)
        # make sure identity goes first
        ret = [{self.identity()}]
        elems_remain.remove(self.identity())
        while elems_remain:
            a = elems_remain.pop()
            conj_class = {a}
            for b in self.elements:
                c = self.get_conj(a, b)
                elems_remain.discard(c)
                conj_class.add(c)
            ret.append(conj_class)
        return ret

    @lru_cache(maxsize=None)
    def get_class_of(self, a):
        """Returns a\'s conjugacy class"""
        return [cl for cl in self.get_classes() if a in cl][0]

    @lru_cache(maxsize=None)
    def get_char_table(self, uniq_thresh=1e-2, zero_thresh=0):
        """
        Returns the (num_classes x num_classes) character table over C using Burnside-Dixon.
        See Eick et al. "Handbook of Computational Group Theory" p. 257
        """
        classes = self.get_classes()
        r = len(classes)
        M = t.zeros((r, r, r))
        for j, k, l in product(range(r), repeat=3):
            l_elem = list(classes[l])[0]
            M[j, k, l] = sum(
                1
                for x, y in product(classes[j], classes[k])
                if self.mult(x, y) == l_elem
            )
        # Need shared row eigenvectors of M[0], ..., M[r]
        # Do this by getting eig of \sum_i a_i M[i] for random a_i
        # Will recover uniquely as long as eigenvalues are unique.
        chars = None
        while chars is None:
            a = t.randn(r)
            aM = einops.einsum(a, M, "j, j k l -> k l")
            L, V = t.linalg.eig(aM.T)
            # check that all eigvalues in L are unique
            L = L.unsqueeze(1)
            if (L - L.T + t.eye(r)).abs().min() > uniq_thresh:
                chars = V.T
            else:
                print("failed", (L - L.T + t.eye(r)).abs().min())

        class_sizes = t.Tensor([len(c) for c in classes]).unsqueeze(0)
        char_norms = (
            (chars * chars.conj() * class_sizes / len(self)).sum(1).unsqueeze(1).sqrt()
        )
        chars /= char_norms
        # Rescale by sgn so that first column is all reals
        # Assumes that identity is first in self.get_classes()
        chars /= chars[:, 0].sgn().unsqueeze(1)
        # Snap small real/complex parts to zero
        snap = lambda x: x * (x.abs() > zero_thresh)
        return t.complex(snap(chars.real), snap(chars.imag))

    def hash(self):
        m = hashlib.sha256()
        m.update(str(self.cayley_table.int().tolist()).encode())
        return m.hexdigest()
