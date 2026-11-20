"""Support canonicalization, products and the feasible-domain identity reducer."""

import numpy as np
import pytest

from helpers_core import all_states, direct_products, onehot_identities, permutation_states
from nimbus.compiler.table_fit import domain_identities
from nimbus.core.algebra import (
    IdentityReducer,
    bits_to_support,
    canonical_support,
    product,
    products,
    support_to_bits,
)
from nimbus.core.errors import ConfigError


def test_canonical_support_parity():
    assert canonical_support([3, 1, 3, 2, 2, 2]) == (1, 2)
    assert canonical_support([]) == ()
    assert canonical_support([4, 4]) == ()
    assert canonical_support((5,)) == (5,)
    with pytest.raises(ValueError):
        canonical_support([-1])


def test_bits_roundtrip():
    for sup in [(), (0,), (1, 5, 63, 64, 200), tuple(range(0, 300, 7))]:
        assert bits_to_support(support_to_bits(sup)) == sup
    assert support_to_bits([2, 2, 3]) == 1 << 3


def test_products_match_numpy():
    S = all_states(6)
    sups = [(), (0,), (1, 3), (0, 2, 4, 5), tuple(range(6))]
    ref = direct_products(S, sups)
    assert np.array_equal(products(S, sups), ref)
    for k, sup in enumerate(sups):
        assert np.array_equal(product(S, sup), ref[:, k])
        assert product(S[5], sup) == ref[5, k]
    assert np.array_equal(products(S[3], sups), ref[3])


def test_reducer_assignment_example():
    """3x3 one-hot: s1 s4 -> s7 using the column identity {1,4,7} (0-based {0,3,6})."""
    red = IdentityReducer(9, onehot_identities(3))
    assert red.rank == 5  # 6 identities, one dependency (sum of rows = sum of columns)
    assert red.best_representative((0, 3)) == ((6,), 1)
    assert red.relation((0, 3), (6,)) == 1
    assert red.reduce((0, 3)) == red.reduce((6,))
    assert red.is_constant((0, 1, 2)) == 1
    assert red.is_constant((0, 1, 2, 3, 4, 5)) == 1
    assert red.is_constant((0, 3)) is None
    assert red.relation((3, 4, 6, 7), (2,)) == 1  # used for T3 under the second group
    assert red.relation((0,), (8,)) is None
    S, _ = permutation_states(3)
    assert red.holds_on(S)


def test_reducer_signs_even_P():
    """P = 2: every row/column product is -1, so s_00 = -s_01 on the feasible set."""
    red = IdentityReducer(4, onehot_identities(2))
    assert red.is_constant((0, 1)) == -1
    assert red.relation((0,), (1,)) == -1
    assert red.relation((0,), (3,)) == 1  # s_00 = -s_01 = s_11
    rep, sg = red.best_representative((0, 1, 2))
    assert len(rep) == 1 and sg * 1 == red.relation((0, 1, 2), rep)


def test_inconsistent_identities_raise():
    with pytest.raises(ConfigError):
        IdentityReducer(3, [((0, 1), 1), ((0, 1), -1)])
    with pytest.raises(ConfigError):
        IdentityReducer(3, [((0, 1), 1), ((1, 2), 1), ((0, 2), -1)])
    with pytest.raises(ConfigError):
        IdentityReducer(3, [((), -1)])
    with pytest.raises(ConfigError):
        IdentityReducer(3, [((0, 5), 1)])
    IdentityReducer(3, [((0, 1), 1), ((1, 2), 1), ((0, 2), 1)])  # consistent dependency is fine


@pytest.mark.parametrize("seed", range(12))
def test_reducer_vs_bruteforce_on_random_domains(seed):
    """Keys equal <=> T_A T_B constant on Omega (identities complete); relation sign is correct;
    best representatives are equivalent and never heavier."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    S = all_states(n)
    keep = rng.random(S.shape[0]) < 0.35
    keep[int(rng.integers(S.shape[0]))] = True
    Om = S[keep]
    ids = domain_identities(Om)
    red = IdentityReducer(n, ids)
    assert red.holds_on(Om)
    sups = [tuple(np.flatnonzero(rng.random(n) < 0.5)) for _ in range(30)]
    T = direct_products(Om, sups)
    for a in range(len(sups)):
        ka, sa = red.reduce(sups[a])
        c = red.is_constant(sups[a])
        assert (c is not None) == (len(set(T[:, a])) == 1)
        if c is not None:
            assert np.all(T[:, a] == c)
        rep, sg = red.best_representative(sups[a])
        assert len(rep) <= len(sups[a])
        assert np.all(T[:, a] == sg * direct_products(Om, [rep])[:, 0])
        for b in range(len(sups)):
            rel = red.relation(sups[a], sups[b])
            ratio = T[:, a] * T[:, b]
            if rel is None:
                assert len(set(ratio)) == 2
            else:
                assert np.all(ratio == rel)
