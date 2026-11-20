"""One-hot permutation (assignment) objectives with row exchanges (complete and tree) and segment
reversals, exhaustive over all P! permutations (gate bound M_add <= 2 P |E|)."""

import numpy as np
import pytest

from helpers_core import (
    check_compiled_exhaustive,
    direct_energy,
    onehot_identities,
    permutation_states,
    random_distances,
    tour_length,
    cyclic_assignment_objective,
)
from nimbus.compiler import compile_model, preflight, row_exchange_B
from nimbus.compiler.table_fit import domain_identities
from nimbus.core.algebra import IdentityReducer
from nimbus.core.model import bind_weights
from nimbus.transitions import builders as B


@pytest.mark.parametrize("P,symmetric", [(4, True), (4, False), (5, True), (5, False)])
@pytest.mark.parametrize("pattern", ["complete", "path", "random_tree"])
def test_row_exchanges_exhaustive(P, symmetric, pattern):
    d = random_distances(P, seed=10 * P + symmetric, symmetric=symmetric)
    structure, w, c = cyclic_assignment_objective(d)
    lib = B.row_exchanges(P, pattern, seed=P)
    ids = onehot_identities(P)
    model = compile_model(structure, lib, identities=ids)
    S, tours = permutation_states(P)
    binding = bind_weights(model, w, c)
    native = lambda X: np.array([tour_length(d, [int(np.flatnonzero(x.reshape(P, P)[:, p] == 1)[0])  # noqa: E731
                                                 for p in range(P)]) for x in X])
    rep = check_compiled_exhaustive(model, lib, S, binding=binding, energy=native)
    assert rep["states"] == len(tours)
    res = model.resources["materialized"]
    assert res["M_add"] <= 2 * P * lib.n_groups  # gate bound M_add <= 2 P |E|
    if P == 4 and pattern == "complete":
        # s_ap s_bp = s_cp s_dp by the column identity: the swap (c,d) reuses the gate of swap (a,b)
        assert model.feasible_simplification
    pf = preflight(structure, lib, identities=ids)
    assert pf["gate_bound_detail"]["S0"] <= 2 * P
    assert res["M_add"] <= pf["M_add_upper"] <= 2 * P * lib.n_groups
    assert pf["routes_decoder_upper"] >= res["routes_decoder"]
    assert pf["memory_bytes_upper"] >= res["memory_bytes"]
    assert pf["gate_bound_detail"]["closed_form_bound"] == lib.n_groups * row_exchange_B(P, 2)


@pytest.mark.parametrize("P", [4, 5])
def test_row_exchanges_plus_segment_reversals(P):
    d = random_distances(P, seed=77 + P, symmetric=False)
    structure, w, c = cyclic_assignment_objective(d)
    lib = B.row_exchanges(P) + B.segment_reversals(P, min_len=2)
    model = compile_model(structure, lib, identities=onehot_identities(P))
    S, _ = permutation_states(P)
    binding = bind_weights(model, w, c)
    check_compiled_exhaustive(model, lib, S, binding=binding, energy=lambda X: direct_energy(structure, w, c, X))
    assert preflight(structure, lib, identities=onehot_identities(P))["M_add_upper"] is None  # no structural bound


def test_declared_identities_complete_for_permutations():
    """Row/column products span every parity identity of the permutation states for P = 4, 5, so
    gate reuse modulo the declared identities is as strong as reuse modulo all identities of Omega."""
    for P in (3, 4, 5):
        S, _ = permutation_states(P)
        derived = domain_identities(S)
        red = IdentityReducer(P * P, onehot_identities(P))
        assert red.rank == 2 * P - 1
        assert len(derived) == red.rank
        for sup, sign in derived:
            assert red.is_constant(sup) == sign


def test_rebind_new_distances_without_recompiling():
    P = 5
    structure, w1, c1 = cyclic_assignment_objective(random_distances(P, 1))
    lib = B.row_exchanges(P)
    model = compile_model(structure, lib, identities=onehot_identities(P))
    S, tours = permutation_states(P)
    hashes = set()
    for seed in (1, 2, 3):
        d = random_distances(P, seed)
        s2, w, c = cyclic_assignment_objective(d)
        assert s2.terms == structure.terms  # same structure, different weights
        b = bind_weights(model, w, c)
        assert b.topology_hash == model.topology_hash
        hashes.add(b.weight_hash)
        assert np.allclose(model.energy(S, b), [tour_length(d, t) for t in tours])
        check_compiled_exhaustive(model, lib, S, binding=b, energy=lambda X: direct_energy(s2, w, c, X))
    assert len(hashes) == 3
