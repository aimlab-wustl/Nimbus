"""General affine moves (every new spin a signed product of old spins) and moves that are not their own
inverse (cyclic exchanges), compiled with their reverse moves."""

import itertools
from math import comb

import numpy as np
import pytest

from helpers_core import (
    all_states,
    check_compiled_exhaustive,
    direct_energy,
    onehot_identities,
    permutation_states,
    random_polynomial,
)
from nimbus.compiler import compile_model
from nimbus.compiler.compile import CompileOptions
from nimbus.compiler.table_fit import AffineFitter
from nimbus.core.errors import UnsupportedTransition
from nimbus.core.model import KIND_READOUT, ROLE_GATE, ROLE_OBJECTIVE, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.descriptors import AffineMove, SignedPermutation, TableTransition, descriptor_from_dict
from nimbus.transitions.grouping import _structured
from nimbus.transitions.library import TransitionLibrary
from nimbus.transitions.verify import verify_library


def two_flips():
    """Flip s0 where s2 = -1 and s1 where s2 = +1: new s0 = s0 s2, new s1 = -s1 s2 (neither a
    permutation nor one conditional flip)."""
    return AffineMove([(0, [0, 2], 1), (1, [1, 2], -1)])


def test_affine_apply_inverse_and_roundtrip():
    S = all_states(3)
    g = two_flips()
    out = g.apply(S)
    assert np.array_equal(out[:, 0], S[:, 0] * S[:, 2])
    assert np.array_equal(out[:, 1], -S[:, 1] * S[:, 2])
    assert np.array_equal(out[:, 2], S[:, 2])
    assert g.is_involution and g.inverse() is g
    assert descriptor_from_dict(g.to_dict()) == g
    rot = AffineMove([(0, [1], 1), (1, [2], -1), (2, [0, 1], 1)])       # not its own inverse
    S5 = all_states(3)
    assert not rot.is_involution
    assert np.array_equal(rot.inverse().apply(rot.apply(S5)), S5)
    with pytest.raises(UnsupportedTransition):
        AffineMove([(0, [0, 1], 1), (1, [0, 1], 1)])                  # two states map to one


def test_affine_move_compiles_and_matches_energy_changes():
    S = all_states(4)
    lib = TransitionLibrary(4, [two_flips(), AffineMove([(3, [3, 0, 1], -1)])], ["two flips", "s3 <- -s3 s0 s1"])
    assert verify_library(lib, S, check_connectivity=False).ok
    structure, w, c = random_polynomial(4, 10, 3, seed=3)
    model = compile_model(structure, lib)
    assert np.all(model.act_kind == KIND_READOUT)                       # executed by the decoder
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X), require_live_routes=True)


def test_grouping_returns_affine_moves():
    S = all_states(3)
    g = two_flips()
    idx = {tuple(s): i for i, s in enumerate(S.tolist())}
    dest = np.array([idx[tuple(r)] for r in g.apply(S).tolist()])
    found = _structured(S, dest, AffineFitter(S))
    assert isinstance(found, AffineMove) and np.array_equal(found.apply(S), S[dest])


def test_row_cycles_compile_with_their_reverses():
    P = 4
    S, _ = permutation_states(P)
    lib = B.row_cycles(P)
    assert lib.n_groups == 2 * comb(P, 3) and not lib.all_involutions and lib.missing_reverses() == []
    rep = verify_library(lib, S)
    assert rep.feasibility_preserved and not rep.involution and rep.missing_reverse_groups == []
    # three-row cycles are even permutations: alone they connect only half of the states
    assert rep.connected is False and rep.n_components == 2
    assert verify_library(TransitionLibrary.concat(B.row_exchanges(P, 'path'), lib), S).ok
    structure, w, c = random_polynomial(P * P, 14, 3, seed=2)
    model = compile_model(structure, lib, identities=onehot_identities(P))
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))
    # a cycle and its reverse: the same connections, different switches
    a, b = model.group_routes(0), model.group_routes(1)
    assert sorted(a["clause"].tolist()) == sorted(b["clause"].tolist())
    sw_a = dict(zip(a["clause"].tolist(), zip(a["ctrl"].tolist(), a["pol"].tolist())))
    sw_b = dict(zip(b["clause"].tolist(), zip(b["ctrl"].tolist(), b["pol"].tolist())))
    assert any(sw_a[k] != sw_b[k] for k in sw_a)


def test_missing_reverse_is_refused_and_added():
    P = 4
    S, _ = permutation_states(P)
    one = TransitionLibrary(P * P, [B.row_cycle(0, 1, 2, P)], ["cycle"])
    assert one.missing_reverses() == [0]
    rep = verify_library(one, S, check_connectivity=False)
    assert not rep.ok and rep.missing_reverse_groups == [0]
    structure, w, c = random_polynomial(P * P, 8, 2, seed=4)
    with pytest.raises(UnsupportedTransition, match="with_reverses"):
        compile_model(structure, one)
    compile_model(structure, one, options=CompileOptions(allow_non_involutive=True))   # explicit opt-out
    both = one.with_reverses()
    assert both.n_groups == 2 and both.labels[1] == "cycle (reverse)" and both.missing_reverses() == []
    assert both.groups[1] == B.row_cycle(0, 2, 1, P)
    assert verify_library(both, S, check_connectivity=False).ok
    assert both.with_reverses().n_groups == 2                          # nothing added twice


def test_reverses_of_affine_moves_and_tables():
    rot = AffineMove([(0, [1], 1), (1, [2], 1), (2, [0], 1)])
    lib = TransitionLibrary(3, [rot], ["rotate"]).with_reverses()
    assert lib.n_groups == 2 and verify_library(lib, all_states(3), check_connectivity=False).ok
    S = all_states(3)
    dest = np.array([{tuple(s): i for i, s in enumerate(S.tolist())}[tuple(r)] for r in rot.apply(S).tolist()])
    t = TableTransition.from_destinations(S, dest)
    assert np.array_equal(t.inverse().apply(t.apply(S)), S)
    assert isinstance(SignedPermutation([(0, 1, 1), (1, 2, 1), (2, 0, 1)]).inverse(), SignedPermutation)


def _lam(L, l2):
    s = sum(comb(L, k) for k in range(1, min(l2, (L - 1) // 2) + 1))
    return s + (comb(L, L // 2) // 2 if L % 2 == 0 and l2 >= L // 2 else 0)


@pytest.mark.parametrize("L", [5, 6])
def test_row_cycles_gate_clause_count(L):
    """All 2 C(L,3) cyclic row exchanges under a quadratic-assignment objective add exactly
    3 L (L-1) C(L,3) + C(L,2) Lambda(L,2) gate clauses."""
    idx = B.onehot_index
    terms = [(idx(i, a, L), idx(j, b, L)) for i, j in itertools.combinations(range(L), 2)
             for a in range(L) for b in range(L) if a != b]
    model = compile_model(ObjectiveStructure(n_spins=L * L, terms=terms), B.row_cycles(L),
                          identities=onehot_identities(L))
    gates = sum(1 for k in range(model.n_clauses)
                if model.clause_role[k] & ROLE_GATE and not model.clause_role[k] & ROLE_OBJECTIVE)
    assert gates == model.resources["materialized"]["M_add"] == 3 * L * (L - 1) * comb(L, 3) + comb(L, 2) * _lam(L, 2)


def _run(name, model, binding, spins0, n_attempts, seed, **prep):
    from nimbus.annealing import schedule as Sched
    from nimbus.backends.base import get_backend

    be = get_backend(name)
    plan = be.prepare(model, **prep)
    state = be.init_state(plan, [binding], spins0)
    sched = Sched.geometric(2.0, 0.05, n_attempts)
    be.run(plan, state, sched, n_attempts_total=n_attempts, seed=seed)
    return be.export_state(state)


@pytest.mark.parametrize("case", ["cycles", "affine"])
def test_runs_on_emulation_and_cpu(case):
    """Cycles with their reverses and affine moves run on the event-driven emulation and on every CPU
    strategy with identical trajectories; every state stays feasible and the energies are exact."""
    if case == "cycles":
        P = 4
        S, _ = permutation_states(P)
        lib = TransitionLibrary.concat(B.row_exchanges(P, "path"), B.row_cycles(P))
        ids = onehot_identities(P)
    else:
        S = all_states(4)
        lib = TransitionLibrary(4, [two_flips(), AffineMove([(3, [3, 0, 1], -1)]), AffineMove([(2, [2, 3], 1)])],
                                ["two flips", "s3", "s2"])
        ids = ()
    n = S.shape[1]
    structure, w, c = random_polynomial(n, 12, 3, seed=9)
    model = compile_model(structure, lib, identities=ids)
    binding = bind_weights(model, w, c)
    spins0 = S[np.random.default_rng(1).choice(len(S), size=4)]
    ref = _run("emulation", model, binding, spins0, 1200, seed=3)
    keys = {tuple(s) for s in S.tolist()}
    assert all(tuple(s) in keys for s in ref.spins.tolist())
    np.testing.assert_allclose(ref.energies[:, 0], direct_energy(structure, w, c, ref.spins), rtol=1e-9, atol=1e-9)
    assert int(ref.counters["accepted_changed"].sum()) > 0
    for strategy in ("recompute", "incremental", "lazy"):
        out = _run("cpu", model, binding, spins0, 1200, seed=3, strategy=strategy)
        np.testing.assert_array_equal(out.spins, ref.spins, err_msg=strategy)
        np.testing.assert_array_equal(out.T, ref.T, err_msg=strategy)
        np.testing.assert_allclose(out.energies, ref.energies, rtol=1e-9, atol=1e-9)
