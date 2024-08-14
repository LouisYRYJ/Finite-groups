import torch as t
from torch import nn
from group import *
from jaxtyping import Bool, Int, Float, jaxtyped
from beartype import beartype
from typing import Callable, Union, Any, Optional
import pathlib
import os
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.combinatorics.named_groups import AlternatingGroup
import math

ROOT = pathlib.Path(__file__).parent.parent.resolve()
GAP_ROOT = "/usr/share/gap"
if os.path.isdir(GAP_ROOT):
    os.environ["GAP_ROOT"] = GAP_ROOT
    from gappy import gap
    from gappy.gapobj import GapObj
else:
    print("WARNING: GAP is not installed!")

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
        b = N - 1  # N-2
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
    ret = semidirect_product(group1, group2, lambda x: lambda y: y)
    # TODO: compute gap_repr for general semidirect_product
    if None not in [group1.gap_repr, group2.gap_repr]:
        ret.gap_repr = gap.DirectProduct(group1.gap_repr, group2.gap_repr)
    return ret

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
def B(n: int) -> Group:
    """
    Convenience function for Boolean hypercube.
    """
    return Z(*(2,)*n)


@jaxtyped(typechecker=beartype)
# TODO: this is a stupid name
def twisted(group: Group, automorphism: Callable[..., Any], m: int = 2) -> Group:
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


def abFam(a: int, b: int, r=None) -> Optional[list[Group]]:
    """
    Returns two semidirect products Z/ab x Z/m where the automorphisms are
    multiplication by either r or a-r, and r is chosen such that
    ord(r)=ord(m-r)=m.
    """
    base_group = Z(a * b)

    # multiplicative order
    def order(c):
        assert c in base_group.elements
        assert math.gcd(c, a * b) == 1, "c must be unit of Z(a*b)"
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
    print(f"Found r={r} and s={s} of order {m}")
    # Make sure to capture r in the nested lambdas
    phi_r = lambda x, r=r: lambda y, r=r: (y * r**x) % (a * b)
    phi_s = lambda x, s=s: lambda y, s=s: (y * s**x) % (a * b)
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

def gapS(n: int) -> Group:
    return Group.from_gap(gap.SymmetricGroup(n))

def Q(p: int) -> Group:
    # Extra special group of order p^3 with exponent p
    return Group.from_gap(gap.ExtraspecialGroup(p**3, p))
