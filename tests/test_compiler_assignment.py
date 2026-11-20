"""Manuscript worked example 1 (Methods, An example toy problem), compiled by the common compiler.

Paper spins s_1..s_9 are 0-based here (s_1 -> 0). Objective T1 = s1, T2 = s9, T3 = s4 s8, T4 = s3 with
J = (1, 2, 1, 2); groups mu1 = swap rows A,B and mu2 = swap rows B,C; identities = row and column
products (+1 for P = 3). Expected (Methods, An example toy problem): one gate clause T5 = s7,
J* = (1,2,1,2,0), Pi* = [[1,0],[0,1],[1,1],[1,0],[0,1]], the alpha table of the manuscript and the
neuron inputs of Methods, An example toy problem at all six feasible states. This is a compiler unit test, not
a benchmark.
"""

import numpy as np

from helpers_core import check_compiled_exhaustive, direct_energy, onehot_identities, permutation_states
from nimbus.compiler import compile_model
from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary


def _example():
    structure = ObjectiveStructure(n_spins=9, terms=[(0,), (8,), (3, 7), (2,)],
                                   labels=["T1=s1", "T2=s9", "T3=s4s8", "T4=s3"])
    J = np.array([1.0, 2.0, 1.0, 2.0])
    lib = TransitionLibrary(9, [B.row_exchange(0, 1, 3), B.row_exchange(1, 2, 3)],
                            labels=["{s}_1->{mu}_1 (rows A,B)", "{s}_2->{mu}_2 (rows B,C)"], name="assignment")
    return structure, J, lib


# the manuscript's clauses, by support (paper T1..T5); T5 = s7 is also the spin singleton of s7
_PAPER = {"T1": (0,), "T2": (8,), "T3": (3, 7), "T4": (2,), "T5": (6,)}


def test_assignment_architecture_exact():
    """The manuscript's H*, J*, Pi* and alpha table are reproduced exactly; in addition every spin is a
    stored singleton clause. T1, T2, T4 and the gate T5 = s7 already are singletons and
    are reused; singletons for s2, s4, s5, s6, s8 are added, and their routes execute the row exchanges."""
    structure, J, lib = _example()
    model = compile_model(structure, lib, identities=onehot_identities(3))
    sup = model.clause_supports()
    k_of = {s: k for k, s in enumerate(sup)}
    # H*: the four objective clauses first, then the spin singletons; T5 = s7 is a singleton
    assert sup[:4] == [(0,), (8,), (3, 7), (2,)]
    assert sorted(sup) == sorted([(0,), (8,), (3, 7), (2,)] + [(i,) for i in (1, 3, 4, 5, 6, 7)])
    assert all(k_of[(i,)] == int(model.readout_clause_of_spin[i]) for i in range(9))
    assert all(model.clause_role[k] & ROLE_OBJECTIVE for k in range(4))
    assert model.clause_role[k_of[(6,)]] & ROLE_GATE  # the manuscript's gate T5 = s7, reused as s7's singleton
    # J* restricted to the manuscript's clauses = (1, 2, 1, 2, 0); every added singleton has weight 0
    binding = bind_weights(model, J, const=0.0)
    assert [binding.J[0, k_of[_PAPER[n]]] for n in ("T1", "T2", "T3", "T4", "T5")] == [1.0, 2.0, 1.0, 2.0, 0.0]
    assert np.count_nonzero(binding.J[0]) == 4
    # Pi* rows of the manuscript's clauses
    pi = model.pi_matrix()
    assert [pi[k_of[_PAPER[n]]].tolist() for n in ("T1", "T2", "T3", "T4", "T5")] == [[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]]
    # alpha table: (clause, group) -> (controller clause, polarity), by support
    table = {}
    for r in range(2):
        g = model.group_routes(r)
        for k, c, p in zip(g["clause"].tolist(), g["ctrl"].tolist(), g["pol"].tolist()):
            table[(sup[k], r)] = (sup[c] if c >= 0 else None, p)
    T1, T2, T3, T4, T5 = (_PAPER[n] for n in ("T1", "T2", "T3", "T4", "T5"))
    paper = {(T1, 0): (T5, -1), (T3, 0): (T5, -1), (T4, 0): (T2, -1),
             (T2, 1): (T4, -1), (T3, 1): (T4, -1), (T5, 1): (T1, -1)}
    assert {key: table[key] for key in paper} == paper
    # every other route is a spin-singleton route; each is controlled by a stored singleton
    for (s_k, r), (s_c, p) in table.items():
        if (s_k, r) not in paper:
            assert len(s_k) == 1 and s_c is not None and len(s_c) == 1 and p == -1
    # weighted routes first: the manuscript's 3 + 2 weighted routes
    assert (model.route_wend - model.route_ptr[:-1]).tolist() == [3, 2]
    assert model.feasible_simplification  # s1 s4 -> s7 etc. use the column identities


def test_assignment_fields_at_all_six_states():
    structure, J, lib = _example()
    model = compile_model(structure, lib, identities=onehot_identities(3))
    binding = bind_weights(model, J)
    S, _ = permutation_states(3)
    T = model.clause_values(S).astype(np.float64)
    v = model.inputs(T.astype(np.int8), binding)
    k_of = {s: k for k, s in enumerate(model.clause_supports())}
    T1, T2, T3, T4, T5 = (T[:, k_of[_PAPER[n]]] for n in ("T1", "T2", "T3", "T4", "T5"))
    v1 = (1 - T5) / 2 * (T1 + T3) + (1 - T2) * T4  # Methods, An example toy problem
    v2 = (1 - T4) / 2 * (2 * T2 + T3)
    assert np.allclose(v[:, 0], v1) and np.allclose(v[:, 1], v2)
    energy = lambda X: direct_energy(structure, J, 0.0, X)  # noqa: E731
    rep = check_compiled_exhaustive(model, lib, S, energy=energy, binding=binding, require_live_routes=True)
    assert rep["states"] == 6 and rep["clause_group_state_checks"] == 6 * 2 * model.n_clauses
    # the second group exchanges s^(5) and s^(6) without changing a weighted clause, but toggles T5
    s5 = np.array([-1, -1, 1, 1, -1, -1, -1, 1, -1], dtype=np.int8)
    s6 = np.array([-1, -1, 1, -1, 1, -1, 1, -1, -1], dtype=np.int8)
    assert np.array_equal(model.apply_action(1, s5), s6)
    sigma = model.decoder_toggles(1, model.clause_values(s5))
    # T5 = s7 toggles, together with the singletons of the exchanged spins s4, s5 (row B) and s7, s8 (row C);
    # no weighted clause toggles
    toggled = sorted(model.clause_support(k) for k in np.flatnonzero(sigma))
    assert toggled == [(3,), (4,), (6,), (7,)]
    s_new = model.spin_readout((1 - 2 * sigma.astype(np.int8)) * model.clause_values(s5))
    assert np.array_equal(s_new, s6)
    assert model.inputs(model.clause_values(s5), binding)[1] == 0.0
