"""Finding moves by grouping spin transitions (``nimbus.transitions.grouping``): every move found is
feasible, its own inverse and compiles with one clause per switch; the search recovers the known
moves of standard constraint families."""

import itertools
import time

import numpy as np
import pytest

from helpers_core import (
    amo_states,
    chain_states,
    check_compiled_exhaustive,
    codewords,
    direct_energy,
    exact_cardinality_states,
    or_states,
    permutation_states,
    product_states,
    random_parity_code,
    random_polynomial,
)
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.transitions.descriptors import FixedMask, SignedPermutation, TableTransition
from nimbus.transitions.grouping import _Domain, _pairs_by_masks, _pairs_pairwise, candidate_transitions, group_transitions
from nimbus.transitions.verify import verify_library


def bead_states():
    """Three beads in three positions; spin 3 * colour + position (red, green, blue = 0, 1, 2)."""
    col = {"R": 0, "G": 1, "B": 2}
    S = -np.ones((6, 9), dtype=np.int8)
    for t, o in enumerate(["RGB", "RBG", "GRB", "BRG", "GBR", "BGR"]):
        for a, ch in enumerate(o):
            S[t, 3 * col[ch] + a] = 1
    return S


def check_grouping(S, g, *, seed=0, n_terms=10, max_degree=3, compile_all=True):
    """Each move: feasible, involutive, only candidate transitions; library compiles and 2 v = dE."""
    m = S.shape[0]
    keys = {tuple(s) for s in S.tolist()}
    pair_set = {tuple(p) for p in g.transitions.tolist()}
    for mv in g.moves:
        d = mv.dest
        assert np.array_equal(d[d], np.arange(m)), "not its own inverse"
        moved = np.flatnonzero(d != np.arange(m))
        assert moved.size > 0
        for t in moved.tolist():
            assert (min(t, d[t]), max(t, d[t])) in pair_set
        img = mv.descriptor.apply(S)
        assert np.array_equal(img, S[d]), "descriptor differs from the grouped transitions"
        assert all(tuple(r) in keys for r in img.tolist())
    if not compile_all or not g.moves:
        return
    lib = g.library("all")
    assert verify_library(lib, S, check_connectivity=False).feasibility_preserved
    structure, w, c = random_polynomial(S.shape[1], n_terms, max_degree, seed=seed)
    model = compile_model(structure, lib, identities=g.identities)
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))


def test_beads_all_moves_and_backbone():
    S = bead_states()
    g = group_transitions(S, mode="all")
    assert len(g.transitions) == 9 and set(g.distances.tolist()) == {4}
    # three colour exchanges and three position exchanges, each joining three pairs of states
    assert len(g.moves) == 6
    assert all(isinstance(mv.descriptor, SignedPermutation) and mv.n_transitions == 3 for mv in g.moves)
    swaps = {frozenset(zip(mv.descriptor.dests, mv.descriptor.srcs)) for mv in g.moves}
    red_green = frozenset({(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)})
    green_blue = frozenset({(3, 6), (6, 3), (4, 7), (7, 4), (5, 8), (8, 5)})
    assert red_green in swaps and green_blue in swaps
    assert g.connected and len(g.backbone) == 2                 # two exchanges connect all six states
    check_grouping(S, g)


def test_beads_cover_mode_connects():
    S = bead_states()
    g = group_transitions(S)
    assert g.connected and len(g.ungrouped) == 0
    check_grouping(S, g)


@pytest.mark.parametrize("n,k", [(5, 2), (6, 3), (7, 3)])
def test_cardinality_recovers_pair_exchanges(n, k):
    S = exact_cardinality_states(n, k)
    g = group_transitions(S, mode="all")
    pairs = set()
    for mv in g.moves:
        assert isinstance(mv.descriptor, SignedPermutation)
        assert len(mv.descriptor.dests) == 2
        pairs.add(mv.descriptor.dests)
    assert pairs == set(itertools.combinations(range(n), 2))
    assert g.connected and len(g.backbone) == n - 1
    check_grouping(S, g, seed=n)


@pytest.mark.parametrize("P", [3, 4])
def test_permutations_rows_and_columns(P):
    S, _ = permutation_states(P)
    g = group_transitions(S, mode="all")
    assert len(g.moves) == 2 * (P * (P - 1) // 2)            # row exchanges and column exchanges
    assert all(isinstance(mv.descriptor, SignedPermutation) for mv in g.moves)
    assert g.connected
    check_grouping(S, g, seed=P)


def test_parity_code_gives_fixed_masks():
    C = random_parity_code(8, 3, seed=2, row_weight=4)
    S = codewords(C)
    g = group_transitions(S, mode="all")
    assert g.moves and all(isinstance(mv.descriptor, FixedMask) for mv in g.moves)
    check_grouping(S, g, seed=3)


@pytest.mark.parametrize("states", [chain_states(5), amo_states(4), or_states([1, -1, 1, 1])],
                         ids=["chain", "packing", "covering"])
def test_block_families_are_connected_and_compile(states):
    g = group_transitions(states, mode="all")
    assert g.connected and len(g.ungrouped) == 0
    check_grouping(states, g, seed=7)


def test_mixed_blocks_with_max_distance():
    S = product_states([exact_cardinality_states(4, 2), chain_states(3), amo_states(3)])
    g_min = group_transitions(S)
    assert not g_min.connected                                  # distance-1 transitions miss the cardinality block
    g = group_transitions(S, max_distance=2)
    assert g.connected
    # moves acting on two blocks at once come from compositions of commuting moves
    assert any(mv.found_by == "composition" for mv in g.moves)
    check_grouping(S, g, seed=11)


def test_tables_compile_without_structured_descriptors():
    S = bead_states()
    g = group_transitions(S, mode="all", structured=False)
    assert all(isinstance(mv.descriptor, TableTransition) for mv in g.moves)
    assert g.moves[0].descriptor.domain is g.moves[1].descriptor.domain   # one shared domain
    check_grouping(S, g)


def test_kempe_three_vertices_only_affine_moves():
    """Hand-written two-colour interchanges on 3-colourings of a path of 3 vertices do not compile
    (test_compiler_table); grouping returns only moves that do."""
    cols = [c for c in itertools.product(range(3), repeat=3) if c[0] != c[1] and c[1] != c[2]]
    S = -np.ones((len(cols), 9), dtype=np.int8)
    for t, c in enumerate(cols):
        for j, x in enumerate(c):
            S[t, 3 * j + x] = 1
    g = group_transitions(S, mode="all")
    assert g.moves
    check_grouping(S, g, seed=5)
    # transitions that no affine move contains are reported, never forced into a move
    in_moves = set()
    for mv in g.moves:
        t = np.flatnonzero(mv.dest > np.arange(len(S)))
        in_moves |= {(int(a), int(mv.dest[a])) for a in t}
    rest = {tuple(p) for p in g.transitions[g.ungrouped].tolist()}
    assert in_moves | rest == {tuple(p) for p in g.transitions.tolist()} and not (in_moves & rest)


def test_candidate_strategies_agree():
    for S in (exact_cardinality_states(7, 3), permutation_states(4)[0], codewords(random_parity_code(9, 3, 4))):
        dom = _Domain(S)
        a, b, d = _pairs_pairwise(dom, None)
        k = int(d.min())
        a2, b2 = _pairs_by_masks(dom, k, budget=1 << 40)
        assert set(zip(a.tolist(), b.tolist())) == set(zip(a2.tolist(), b2.tolist()))
        a3, b3, _ = _pairs_pairwise(dom, k + 2)
        pairs, dist = candidate_transitions(S, max_distance=k + 2)
        assert set(zip(a3.tolist(), b3.tolist())) == {tuple(p) for p in pairs.tolist()}


def test_from_destinations_matches_dict_constructor():
    S = bead_states()
    dest = np.array([1, 0, 3, 2, 5, 4])
    t1 = TableTransition.from_destinations(S, dest)
    t2 = TableTransition({tuple(S[i]): tuple(S[dest[i]]) for i in range(6)}, [tuple(s) for s in S])
    assert t1.to_dict() == t2.to_dict() and t1.touched() == t2.touched()
    assert np.array_equal(t1.apply(S), t2.apply(S))


def test_rejects_duplicates_and_bad_spins():
    S = bead_states()
    with pytest.raises(ValueError):
        group_transitions(np.vstack([S, S[:1]]))
    with pytest.raises(ValueError):
        group_transitions(np.zeros((2, 3)))


def test_speed_cardinality_and_permutations():
    t0 = time.perf_counter()
    g = group_transitions(exact_cardinality_states(14, 7))          # 3,432 states, 168,168 transitions
    t1 = time.perf_counter()
    assert len(g.moves) == 91 and g.connected
    g2 = group_transitions(permutation_states(6)[0])               # 720 states, 36 spins
    t2 = time.perf_counter()
    assert g2.connected and len(g2.ungrouped) == 0
    assert t1 - t0 < 20 and t2 - t1 < 20
