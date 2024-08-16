from __future__ import annotations
import torch as t
from torch import nn
import random
from jaxtyping import Bool, Int, Float, Inexact, jaxtyped
from beartype import beartype
from typing import Callable, Union, Any, Optional
from itertools import product
from collections import defaultdict
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from methodtools import lru_cache
import pathlib
import hashlib
import os
from sympy.combinatorics import PermutationGroup, Permutation
import einops
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.resolve()
GAP_ROOT = "/usr/share/gap"
if os.path.isdir(GAP_ROOT):
    os.environ["GAP_ROOT"] = GAP_ROOT
    from gappy import gap
    from gappy.gapobj import GapObj
    gap.eval('LoadPackage("SmallGrp");')
else:
    print("WARNING: GAP is not installed!")


# TODO: Move to utils or smth
def is_complex(M, thresh=1e-10):
    ret = t.is_complex(M) and M.imag.abs().max() > thresh
    if isinstance(ret, t.Tensor):
        ret = ret.item()
    return ret

class Group:
    """
    Implements a group as a idx -> element lookup table and a Cayley table on idxs.
    Generally, methods ending in _idx operate on idxs, while those without operate on elements.
    """

    def __init__(self, elements, cayley_table, name="group"):
        self.elements = elements
        self.cayley_table = cayley_table
        self.name = name
        self.gap_repr = None   # group as GapObj

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
    def from_func(elements, mult: Callable) -> Group:
        N = len(elements)
        table = t.zeros((N, N), dtype=t.int64)
        for (i, a), (j, b) in product(enumerate(elements), repeat=2):
            table[i, j] = elements.index(mult(a, b))
        return Group(elements, table)

    @staticmethod
    def from_sympy(pgroup: PermutationGroup) -> Group:
        elements = pgroup.elements
        N = len(elements)
        table = t.zeros((N, N), dtype=t.int64)
        for (i, a), (j, b) in product(enumerate(elements), repeat=2):
            table[i, j] = elements.index(a * b)
        return Group(elements, table)

    @staticmethod
    def from_gap(gap_group: GapObj) -> Group:
        elements = [str(elem) for elem in gap_group.Elements()]
        N = len(elements)
        gap_table = gap_group.MultiplicationTable()
        table = t.zeros((N, N), dtype=t.int64)
        for i, j in product(range(N), repeat=2):
            table[i, j] = int(gap_table[i, j]) - 1  # gap_table is 1-indexed
        ret = Group(elements, table)
        ret.gap_repr = gap_group
        return ret

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

    def pow(self, a, power):
        # for convenience
        return self.exponent(a, power)

    @lru_cache(maxsize=None)
    def exponent_idx(self, idx, power):
        return self.elem_to_idx(self.exponent(self.idx_to_elem(idx), power))

    def pow_idx(self, a, power):
        # for convenience
        return self.exponent_idx(a, power)

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
        '''
        Return set of all subgroups of the group
        '''
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = f'{cache_dir}/{self.hash()}'
            if os.path.exists(cache_path):
                return t.load(cache_path)
        else:
            cache_path = None

        if self.gap_repr is None:
            print('Computing subgroups')
            gap_subgroups = self.to_gap_fp().LowIndexSubgroupsFpGroup(len(self))
            # do trivial and full group separately for efficiency
            gap_subgroups = [g for g in tqdm(gap_subgroups, desc='Computing orders') if g.Order() > 1 and g.Order() < len(self)]
            subgroups= {frozenset([self.identity_idx()]), frozenset(range(len(self)))}
            subgroups |= {
                frozenset([self.fp_elem_to_idx(elem) for elem in subgroup.Elements()]) for subgroup in tqdm(gap_subgroups, desc='Computing elements')
            }
        else:
            print('Computing subgroups from gap_repr')
            assert set(self.elements) == set(map(str, self.gap_repr.Elements())), "self.elements and self.gap_repr.Elements() don't match!"
            subgroups = {frozenset(map(lambda g: self.elem_to_idx(str(g)), s.Elements())) for s in self.gap_repr.AllSubgroups()}
        
        for h in tqdm(list(subgroups), desc='Computing conjugates'):
            subgroups |= {frozenset(self.get_conj_subgroup_idx(h, g)) for g in range(len(self))}

        if cache_path is not None:
            print('Saving to', cache_path)
            t.save(subgroups, cache_path)
        return subgroups

    @lru_cache(maxsize=None)
    def get_complex_irreps(self):
        '''
        Returns dict {irrep_name: irrep_basis}, where irrep_basis is a [len(self), d, d] matrix for irreps of degree d.
        '''
        if self.gap_repr is None:
            gap_group = self.to_gap_fp()
            to_idx = self.fp_elem_to_idx
        else:
            gap_group = self.gap_repr
            to_idx = self.elem_to_idx
            
        def to_complex(z):
            try:
                ret = float(z)
            except TypeError:
                # Gap cyclotomic numbers look like -E(5)^3
                z = str(z)
                neg = z[0] == '-'
                if neg:
                    z = z[1:]
                if '^' in z:
                    root, exp = str(z).split('^')
                else:
                    root = z
                    exp = 1
                root = float(root.strip('E()'))
                exp = float(exp)
                real = np.cos(2 * exp * np.pi / root)
                imag = np.sin(2 * exp * np.pi / root)
                real = real if np.abs(real) > 1e-10 else 0.
                imag = imag if np.abs(imag) > 1e-10 else 0.
                ret = t.complex(t.tensor(real).double(), t.tensor(imag).double()).item()
                if neg:
                    ret *= -1
            return ret
                
        irreps = gap.IrreducibleRepresentations(gap_group)
        d_count = defaultdict(lambda: 0)
        ret = dict()
        for irrep in irreps:
            dim = len(irrep.Image(gap_group.Identity()))
            name = f'{dim}d-{d_count[dim]}'
            d_count[dim] += 1
            M = [None] * len(self)
            for gap_elem in gap_group.Elements():
                M[to_idx(str(gap_elem))] = t.tensor(
                    [
                        [to_complex(irrep.Image(gap_elem)[j][i]) for i in range(dim)]
                         for j in range(dim)
                    ]
                )
            ret[name] = t.stack(M, dim=0)
        return ret
    
    def get_frobenius_schur(
        self, irrep: Inexact[t.tensor, 'n d d'], power: int=2,
    ) -> Int:
        '''
        Returns Frobenius-Schur indicator of irrep.
        irrep[i] is a dxd matrix for each element idx i
        Indexing should be the same as for self.elements
        '''
        assert irrep.shape[0] == len(self)
        ret = (sum(t.trace(irrep[self.pow_idx(g, power)]) for g in range(len(self))) / len(self))
        if t.is_complex(ret):
            assert ret.imag.abs().item() < 1e-6
            return t.round(ret.real).int().item()
        else:
            return t.round(ret).int().item()

    @lru_cache(maxsize=None)
    def get_real_irreps(self, max_tries=100):
        real_irreps = dict()
        d_count = defaultdict(lambda: 0)
        for irrep in self.get_complex_irreps().values():
            if not irrep.is_complex() or irrep.imag.abs().max() < 1e-10:
                real_irrep = irrep.real
            elif int(self.get_frobenius_schur(irrep)) == 1:   # real irrep
                # In this case, let \rho be the irrep.
                # We are guaranteed a symmetric S st S\rho(g)S^{-1}=\rho^*(g) for all g
                # 1) Find this S by averaging over G (https://en.wikipedia.org/wiki/Schur%27s_lemma#Corollary_of_Schur's_Lemma)
                # 2) Transform by sqrt(S) (Lemma 2.12.6 in https://sites.ualberta.ca/~vbouchar/MAPH464/section-real-complex.html)
                d = irrep.shape[-1]
                S = t.zeros((d, d), dtype=irrep.dtype)
                tries = 0
                while S.abs().max() < 1e-5 or (S - S.T).abs().max() > 1e-5:
                    H = t.randn((d, d), dtype=irrep.dtype)
                    H /= t.trace(H)
                    S = sum(
                        t.linalg.inv(irrep[i]) @ H @ t.conj(irrep[i])
                        for i in range(len(self))
                    )
                    if tries > max_tries:
                        assert False, f"Exceeded {max_tries} tries without finding nonzero symmetric S"
                    tries += 1
                S = (S + S.T) / 2
                L, V = t.linalg.eig(S)
                W = V @ t.diag(L.sqrt()) @ t.linalg.inv(V)
                real_irrep = einops.einsum(
                    W, irrep, t.linalg.inv(W),
                    'd0 d1, n d1 d2, d2 d3 -> n d0 d3'
                )
                assert real_irrep.imag.abs().max() < 1e-5, 'Real irrep transformation failed!'
                real_irrep = real_irrep.real
            else:  # complex or quaternionic irrep
                real_irrep = t.concat(
                    [
                        t.concat([irrep.real, -irrep.imag], dim=2), 
                        t.concat([irrep.imag, irrep.real], dim=2)
                    ],
                    dim=1
                )
            d = real_irrep.shape[-1]
            real_irreps[f'{d}d-{d_count[d]}'] = real_irrep
            d_count[d] += 1
        return real_irreps
        
    def is_irrep(self, irrep, thresh=1e-4):
        for i, j in product(range(len(self)), repeat=2):
            if (irrep[i] @ irrep[j] - irrep[self.mult_idx(i, j)]).abs().max() > thresh:
                import pdb; pdb.set_trace()
                return False
        return True

    # for convenience
    def get_irreps(self, real=False):
        return self.get_real_irreps() if real else self.get_complex_irreps()
            
    @lru_cache(maxsize=None)
    def get_subgroups(self, cache_dir=f'{ROOT}/subgroups/') -> list[set]:
        return {frozenset(map(self.idx_to_elem, h)) for h in self.get_subgroups_idx(cache_dir=cache_dir)}

    # @staticmethod
    # def from_model(
    #     model: nn.Module,
    #     instance: int,
    #     elements: None,
    # ) -> Group:
    #     table = utils.model_table(model[instance]).squeeze(0)
    #     N = table.shape[0]
    #     elements = list(range(N)) if not elements else elements
    #     return Group(elements, table)

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
