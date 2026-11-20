"""Spin-action descriptors, transition libraries, template builders and library verification."""

import numpy as np
import pytest

from helpers_core import (
    all_states,
    amo_states,
    chain_states,
    codewords,
    exact_cardinality_states,
    or_states,
    permutation_states,
    random_parity_code,
    section29_problem,
)
from nimbus.core.errors import InfeasibleState, UnsupportedTransition
from nimbus.core.model import KIND_CONDMASK as M_COND, KIND_MASK as M_MASK, KIND_PERM as M_PERM, KIND_READOUT as M_RO
from nimbus.transitions import builders as B
from nimbus.transitions.descriptors import (
    KIND_CONDMASK,
    KIND_MASK,
    KIND_PERM,
    KIND_READOUT,
    ConditionalMask,
    FixedMask,
    SignedPermutation,
    TableTransition,
    descriptor_from_dict,
)
from nimbus.transitions.library import TransitionLibrary
from nimbus.transitions.verify import verify_library


def test_kind_codes_are_stable():
    assert (KIND_MASK, KIND_CONDMASK, KIND_PERM, KIND_READOUT) == (0, 1, 2, 3)
    assert (M_MASK, M_COND, M_PERM, M_RO) == (0, 1, 2, 3)
    assert TableTransition({}, [(1, 1)]).kind == KIND_READOUT


def test_fixed_mask():
    g = FixedMask([3, 1])
    s = np.array([1, 1, -1, -1, 1], dtype=np.int8)
    assert g.apply(s).tolist() == [1, -1, -1, 1, 1]
    assert s.tolist() == [1, 1, -1, -1, 1]  # pure
    S = all_states(5)
    assert np.array_equal(g.apply(S)[:, [1, 3]], -S[:, [1, 3]])
    assert not g.is_selfloop(s) and not g.is_selfloop(S).any()
    assert g.touched() == (1, 3) and g.is_involution and g.check_involution(S)
    with pytest.raises(ValueError):
        FixedMask([])
    with pytest.raises(ValueError):
        FixedMask([1, 1])


def test_conditional_mask():
    with pytest.raises(UnsupportedTransition):
        ConditionalMask([0, 1], [1, 2], -1)  # |cond ∩ mask| odd
    g = ConditionalMask([0, 1], [1, 2, 0], -1)  # even overlap {0,1}
    S = all_states(4)
    act = (-1 * S[:, 0] * S[:, 1] * S[:, 2]) == 1
    mu = g.apply(S)
    assert np.array_equal(np.all(mu == S, axis=1), ~act)
    assert np.array_equal(g.is_selfloop(S), ~act)
    assert g.check_involution(S) and g.is_involution
    always = ConditionalMask([2], [], 1)
    never = ConditionalMask([2], [], -1)
    assert not always.is_selfloop(S).any() and never.is_selfloop(S).all()


def test_signed_permutation():
    g = SignedPermutation({0: (2, 1), 2: (0, 1), 1: (1, -1)})
    s = np.array([1, 1, -1], dtype=np.int8)
    assert g.apply(s).tolist() == [-1, -1, 1]
    assert g.is_involution and g.check_involution(all_states(3))
    assert g.touched() == (0, 1, 2)
    cyc = SignedPermutation([(0, 1, 1), (1, 2, 1), (2, 0, 1)])
    assert not cyc.is_involution and not cyc.check_involution(all_states(3))
    bad_sign = SignedPermutation({0: (1, -1), 1: (0, 1)})
    assert not bad_sign.is_involution
    with pytest.raises(UnsupportedTransition):
        SignedPermutation({0: (1, 1)})  # not a bijection on moved coordinates
    with pytest.raises(ValueError):
        SignedPermutation({0: (0, 1)})  # identity
    sw = B.pair_exchange(1, 4)
    S = all_states(5)
    sl = sw.is_selfloop(S)
    assert np.array_equal(sl, S[:, 1] == S[:, 4])


def test_table_transition():
    dom = [(1, -1), (-1, 1), (1, 1)]
    t = TableTransition({(1, -1): (-1, 1), (-1, 1): (1, -1)}, dom)
    assert t.apply(np.array([1, -1])).tolist() == [-1, 1]
    assert t.apply(np.array([1, 1])).tolist() == [1, 1]
    assert t.is_involution and t.touched() == (0, 1)
    with pytest.raises(InfeasibleState):
        t.apply(np.array([-1, -1]))
    with pytest.raises(UnsupportedTransition):
        TableTransition({(1, -1): (-1, -1)}, dom)
    nonin = TableTransition({(1, -1): (-1, 1), (-1, 1): (1, 1), (1, 1): (1, -1)}, dom)
    assert not nonin.is_involution


def test_descriptor_roundtrip_and_hash():
    ds = [FixedMask([0, 2]), ConditionalMask([1], [0, 2], -1), B.pair_exchange(0, 3),
          TableTransition({(1, -1): (-1, 1), (-1, 1): (1, -1)}, [(1, -1), (-1, 1)])]
    for d in ds:
        e = descriptor_from_dict(d.to_dict())
        assert e == d and e.stable_hash() == d.stable_hash()
    assert FixedMask([0, 2]).stable_hash() != FixedMask([0, 1]).stable_hash()


def test_library_concat_hash_and_incidence():
    a = B.single_flips(4, [0, 1])
    b = B.pair_exchanges([2, 3], n_spins=4)
    u = TransitionLibrary.concat(a, b, require_disjoint=True)
    assert u.n_groups == 3 and u.labels[-1] == "swap(2,3)"
    assert u.touched_incidence() == [[0], [1], [2], [2]]
    assert u.metadata["components"][0]["template"] == "single_flips"
    assert (a + b).hash == u.hash
    assert TransitionLibrary.concat(b, a).hash != u.hash
    with pytest.raises(ValueError):
        TransitionLibrary.concat(a, B.single_flips(4, [1]), require_disjoint=True)
    rt = TransitionLibrary.from_dict(u.to_dict())
    assert rt.hash == u.hash
    with pytest.raises(ValueError):
        TransitionLibrary(3, [FixedMask([3])])


def test_tree_edges_and_random_trees():
    assert B.tree_edges(4, "path") == [(0, 1), (1, 2), (2, 3)]
    assert B.tree_edges(4, "star") == [(0, 1), (0, 2), (0, 3)]
    assert len(B.tree_edges(5, "complete")) == 10
    for seed in range(20):
        for n in (2, 3, 6, 9):
            E = B.tree_edges(n, "random_tree", seed=seed)
            assert len(E) == n - 1
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x

            for u, v in E:
                parent[find(u)] = find(v)
            assert len({find(x) for x in range(n)}) == 1
    assert B.tree_edges(9, "random_tree", seed=3) == B.tree_edges(9, "random_tree", seed=3)
    assert len({tuple(B.tree_edges(9, "random_tree", seed=s)) for s in range(10)}) > 1


def test_row_exchanges_and_segment_reversals():
    P = 5
    comp = B.row_exchanges(P)
    tree = B.row_exchanges(P, "random_tree", seed=11)
    assert comp.n_groups == 10 and tree.n_groups == 4
    assert comp.template == "complete_row_exchange" and tree.template == "tree_row_exchange"
    S, tours = permutation_states(P)
    for lib in (comp, tree, B.segment_reversals(P, min_len=2)):
        rep = verify_library(lib, S)
        assert rep.ok and rep.connected and rep.mode == "exhaustive"
    # in a valid permutation a row swap always changes the state
    assert all(f == 0.0 for f in verify_library(comp, S).selfloop_fraction)
    # segment reversal semantics on tours; length-2/3 reversals equal one column swap
    rev = B.segment_reversal(P, 1, 4)
    for s, tour in zip(S, tours):
        new = rev.apply(s)
        new_tour = [int(np.flatnonzero(new.reshape(P, P)[:, p] == 1)[0]) for p in range(P)]
        assert new_tour == list(tour[:1]) + list(tour[1:5][::-1])
    swap13 = B.column_permutation(P, [0, 3, 2, 1, 4])
    assert B.segment_reversal(P, 1, 3) == swap13
    assert B.segment_reversal(P, 2, 3) == B.column_permutation(P, [0, 1, 3, 2, 4])


def test_random_addition_deterministic():
    cand = B.segment_reversals(8, min_len=4)
    a = B.random_addition(cand, 5, seed=42)
    b = B.random_addition(cand, 5, seed=42)
    c = B.random_addition(cand, 5, seed=43)
    assert a.hash == b.hash and a.n_groups == 5
    assert a.params["indices"] != c.params["indices"] or a.hash != c.hash
    r = B.random_segment_reversals(8, 3, seed=1, exclude=[(0, 3)])
    assert (0, 3) not in [tuple(x) for x in [cand.params["segments"][i] for i in r.params["indices"]]]
    m = B.random_fixed_masks(10, 6, 3, seed=5)
    assert m.n_groups == 6 and all(len(g.mask) == 3 for g in m.groups)
    assert m.hash == B.random_fixed_masks(10, 6, 3, seed=5).hash


def test_masks_and_nullspace_library():
    C = random_parity_code(10, 4, seed=3)
    lib = B.nullspace_basis(C)
    Om = codewords(C)
    assert lib.n_groups == 10 - lib.params["rank"]
    assert 2 ** lib.n_groups == Om.shape[0]
    rep = verify_library(lib, Om)
    assert rep.ok and rep.connected
    lib2 = B.fixed_masks(np.array([[1, 0, 1], [0, 1, 1]]))
    assert [g.mask for g in lib2.groups] == [(0, 2), (1, 2)]


def _script_transform(family, s, i, j):
    """Original feasibility conditions of the reference verification script."""
    n = len(s)
    if family == "chain_precedence":
        active = (i == 0 or s[i - 1] == 1) and (i == n - 1 or s[i + 1] == -1)
    elif family == "disjoint_or_block":
        active = s[j] == 1
    else:
        active = all(s[k] == -1 for k in range(n) if k != i)
    out = list(s)
    if active:
        out[i] *= -1
    return tuple(out)


@pytest.mark.parametrize("m", range(1, 8))
def test_section29_templates_match_reference_semantics(m):
    """Builders reproduce the reference script's transformations on the feasible domain (sizes 1..7)."""
    fam = {"chain_precedence": (B.chain_precedence(list(range(m))), chain_states(m),
                                [(i, None) for i in range(m)]),
           "at_most_one_block": (B.amo_block(list(range(m))), amo_states(m), [(i, None) for i in range(m)])}
    if m > 1:
        groups = []
        for i in range(m - 1):
            groups.extend(((i, i + 1), (i + 1, i)))
        fam["disjoint_or_block"] = (B.or_block([(i, 1) for i in range(m)]), or_states([1] * m), groups)
    else:
        assert B.or_block([(0, 1)]).n_groups == 0  # one-literal requirement fixes the variable
    for family, (lib, Om, ref_groups) in fam.items():
        assert lib.n_groups == len(ref_groups)
        for g, (i, j) in zip(lib.groups, ref_groups):
            for s in Om:
                assert tuple(g.apply(s).tolist()) == _script_transform(family, tuple(s.tolist()), i, j)
        rep = verify_library(lib, Om)
        assert rep.ok and rep.connected, (family, rep.summary())


def test_or_block_signed_literals_and_trees():
    signs = [1, -1, -1, 1, -1]
    lits = [(i, sg) for i, sg in enumerate(signs)]
    Om = or_states(signs)
    for tree in ("path", "star", "random_tree"):
        lib = B.or_block(lits, tree=tree, seed=4)
        assert lib.n_groups == 2 * (len(signs) - 1)
        rep = verify_library(lib, Om)
        assert rep.ok and rep.connected


def test_verify_detects_failures_and_sampled_mode():
    S, _ = permutation_states(3)
    rep = verify_library(B.single_flips(9), S)
    assert not rep.feasibility_preserved and rep.infeasible_groups == list(range(9)) and not rep.ok
    one = B.row_exchanges(3, edges=[(0, 1)])
    rep = verify_library(one, S)
    assert rep.feasibility_preserved and rep.involution and rep.connected is False and rep.n_components == 3
    cyc = TransitionLibrary(3, [SignedPermutation([(0, 1, 1), (1, 2, 1), (2, 0, 1)])])
    rep = verify_library(cyc, all_states(3))
    assert not rep.involution
    with pytest.raises(UnsupportedTransition):
        verify_library(cyc, all_states(3), require=True)
    samp = verify_library(B.pair_exchanges(range(6)), sample_states=exact_cardinality_states(6, 3)[:5],
                          is_feasible=lambda s: int((s == 1).sum()) == 3)
    assert samp.mode == "sampled, not a proof" and samp.connected is None and samp.ok


def test_section29_problem_helper():
    n, lib, Om, feas, _ = section29_problem([("chain", 3), ("or", [1, -1]), ("amo", 3)])
    assert n == 8 and Om.shape[0] == 4 * 3 * 4
    assert all(feas(s) for s in Om)
    assert sum(feas(s) for s in all_states(n)) == Om.shape[0]
    assert verify_library(lib, Om, is_feasible=feas).ok
