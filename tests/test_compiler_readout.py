"""Singleton readout clauses: compiled through the same decoder, seeded before closure, counted (gate+readout counts as gate; M~ = M + M_add + M_read with full readout)."""

import numpy as np
import pytest

from helpers_core import (
    check_compiled_exhaustive,
    direct_energy,
    exact_cardinality_states,
    cardinality_identities,
    onehot_identities,
    permutation_states,
    random_distances,
    random_polynomial,
    cyclic_assignment_objective,
)
from nimbus.compiler import compile_model, preflight
from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE, ROLE_READOUT, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary


def _assignment(readout):
    structure = ObjectiveStructure(n_spins=9, terms=[(0,), (8,), (3, 7), (2,)])
    lib = TransitionLibrary(9, [B.row_exchange(0, 1, 3), B.row_exchange(1, 2, 3)])
    return structure, lib, compile_model(structure, lib, identities=onehot_identities(3), readout=readout)


def test_readout_seeded_before_closure_and_counted():
    structure, lib, model = _assignment("all")
    sup = model.clause_supports()
    # objective clauses first, then readout singletons in spin order; the gate s7 is the readout
    # clause (6,) seeded before closure, so closure creates no new clause
    assert sup[:4] == [(0,), (8,), (3, 7), (2,)]
    assert sup[4:] == [(1,), (3,), (4,), (5,), (6,), (7,)]
    res = model.resources["materialized"]
    assert res["gate_clauses_created"] == 0
    k7 = sup.index((6,))
    assert model.clause_role[k7] == ROLE_READOUT | ROLE_GATE
    # The readout clauses need switches of their own: e.g. s2 under rows A<->B gives s2 s5 = s8 and
    # s5 under rows B<->C gives s5 s8 = s2 (column identities). So s2, s7 and s8 also control
    # switches and count as gates; s4, s5, s6 stay readout-only.
    gate_only = sorted(sup[k] for k in range(model.n_clauses)
                       if model.clause_role[k] & ROLE_GATE and not model.clause_role[k] & ROLE_OBJECTIVE)
    assert gate_only == [(1,), (6,), (7,)]
    assert (res["M"], res["M_add"], res["M_read"]) == (4, 3, 3)
    assert res["M"] + res["M_add"] + res["M_read"] == model.n_clauses == 10
    ov = res["role_overlaps"]
    assert ov["objective+readout"] == 3  # s1, s9, s3
    assert ov["gate+readout"] == sum(1 for k in range(model.n_clauses)
                                     if model.clause_role[k] & ROLE_GATE and model.clause_role[k] & ROLE_READOUT)
    # every spin has its singleton readout clause
    assert all(sup[model.readout_clause_of_spin[i]] == (i,) for i in range(9))
    S, _ = permutation_states(3)
    rep = check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, [1, 2, 1, 2]),
                                    energy=lambda X: direct_energy(structure, [1, 2, 1, 2], 0.0, X))
    assert rep["clause_group_state_checks"] == 6 * 2 * 10
    # readout-only clauses never enter the weighted encoder
    for r in range(2):
        g = model.group_routes(r)
        assert np.all(model.clause_role[g["clause"][g["weighted"]]] & ROLE_OBJECTIVE)


def test_readout_decoder_tracks_spins():
    """sigma of the readout clause {i} = 1 exactly when the accepted group changes s_i."""
    P = 4
    d = random_distances(P, 3)
    structure, w, c = cyclic_assignment_objective(d)
    lib = B.row_exchanges(P, "path")
    model = compile_model(structure, lib, identities=onehot_identities(P), readout="all")
    S, _ = permutation_states(P)
    T = model.clause_values(S)
    for r in range(lib.n_groups):
        sigma = model.decoder_toggles(r, T)
        changed = (lib.groups[r].apply(S) != S)
        assert np.array_equal(sigma[:, model.readout_clause_of_spin], changed.astype(np.uint8))
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))
    pf = preflight(structure, lib, "all", identities=onehot_identities(P))
    res = model.resources["materialized"]
    assert pf["M_read_upper"] >= res["M_read"] and pf["M_add_upper"] >= res["M_add"]
    assert pf["M_total_upper"] >= res["M_total"] and pf["routes_decoder_upper"] >= res["routes_decoder"]


def test_every_spin_singleton_and_reuse_of_objective_singletons():
    """Every spin gets a singleton clause; objective singletons are reused (gain READOUT)."""
    n, k = 6, 3
    structure, w, c = random_polynomial(n, 10, 3, 4)
    lib = B.pair_exchanges(range(n), "path")
    model = compile_model(structure, lib, identities=cardinality_identities(n, k))
    sup = model.clause_supports()
    obj_singletons = {t[0] for t in structure.terms if len(set(t)) == 1}
    for i in range(n):
        kk = int(model.readout_clause_of_spin[i])
        assert sup[kk] == (i,) and model.clause_role[kk] & ROLE_READOUT
        assert sup.count((i,)) == 1  # stored once: reused, never duplicated
        if i in obj_singletons:
            assert model.clause_role[kk] & 1  # OBJECTIVE role kept
    assert model.metadata["readout"] == "all"
    check_compiled_exhaustive(model, lib, exact_cardinality_states(n, k), binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))


def test_readout_index_validation():
    structure = ObjectiveStructure(n_spins=3, terms=[(0,)])
    from nimbus.core.errors import ConfigError

    with pytest.raises(ConfigError):
        compile_model(structure, B.single_flips(3), readout=[3])
    with pytest.raises(ConfigError):
        compile_model(structure, B.single_flips(3), readout="some")
    with pytest.raises(ConfigError):  # a partial readout request cannot be honoured
        compile_model(structure, B.single_flips(3), readout=[0, 1])
