"""Compiler performance sanity on realistic sizes (one-hot assignment row exchanges, a 1008 x 504
parity-check mask library).

Thresholds are generous wall-time bounds for a typical workstation; measured times are recorded in the
resource report (``compile_seconds``). The counts also pin the manuscript gate bound
M_add <= 2 P |E|, which is attained for P >= 5 without identity reuse.
"""

import numpy as np
import pytest

from helpers_core import onehot_identities, random_distances, cyclic_assignment_objective
from nimbus.compiler import compile_model, preflight
from nimbus.core.model import ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B


@pytest.mark.parametrize("P,pattern,limit_s", [(10, "complete", 10.0), (20, "complete", 60.0),
                                               (20, "random_tree", 10.0), (30, "random_tree", 20.0)])
def test_onehot_assignment_compile_time(P, pattern, limit_s):
    s, w, c = cyclic_assignment_objective(random_distances(P, 1))
    lib = B.row_exchanges(P, pattern, seed=3)
    model = compile_model(s, lib, identities=onehot_identities(P))
    res = model.resources["materialized"]
    assert res["compile_seconds"] < limit_s, res["compile_seconds"]
    assert res["M"] == P * P + P * P * (P - 1)
    assert res["M_add"] == 2 * P * lib.n_groups  # bound attained (no reuse for P >= 5)
    assert res["closure_rounds"] == 2
    pf = model.resources["estimated"]
    assert pf["M_add_upper"] == res["M_add"] and pf["routes_decoder_upper"] >= res["routes_decoder"]
    b = bind_weights(model, w, c)
    assert b.J.shape == (1, model.n_clauses)


@pytest.mark.slow
def test_onehot_assignment_P30_complete():
    P = 30
    s, w, c = cyclic_assignment_objective(random_distances(P, 1))
    model = compile_model(s, B.row_exchanges(P), identities=onehot_identities(P))
    res = model.resources["materialized"]
    assert res["compile_seconds"] < 120.0
    assert res["M_add"] == 2 * P * (P * (P - 1) // 2)


def test_large_parity_mask_library():
    rng = np.random.default_rng(0)
    N, Mc = 1008, 504
    H = np.zeros((Mc, N), np.uint8)
    rows = []
    for a in range(Mc):
        r = sorted(rng.choice(N, 6, replace=False).tolist())
        rows.append(r)
        H[a, r] = 1
    lib = B.nullspace_basis(H)
    ids = [(tuple(r), 1) for r in rows]
    triples = [tuple(sorted(rng.choice(N, 3, replace=False).tolist())) for _ in range(300)]
    s = ObjectiveStructure(n_spins=N, terms=[(i,) for i in range(N)] + triples)
    model = compile_model(s, lib, identities=ids)
    res = model.resources["materialized"]
    assert res["M_add"] == 0 and res["routes_gate_controlled"] == 0
    assert res["compile_seconds"] < 30.0
    pf = preflight(s, lib, identities=ids)
    # singletons route to every touching mask; triples only where the overlap is odd -> upper bound
    assert pf["routes_decoder_upper"] >= res["routes_decoder"]
    assert pf["M_add_upper"] == 0
