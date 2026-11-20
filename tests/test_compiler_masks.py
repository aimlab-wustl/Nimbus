"""Fixed spin masks (Supplementary Information, Number of auxiliary clauses added by compilation (parity constraints: fixed reversals)): parity-code nullspace bases with higher-order
objectives; M_add = 0 and every retained switch is constant."""

import numpy as np
import pytest

from helpers_core import (
    all_states,
    check_compiled_exhaustive,
    code_identities,
    codewords,
    direct_energy,
    random_parity_code,
    random_polynomial,
)
from nimbus.compiler import compile_model, preflight
from nimbus.core import gf2
from nimbus.core.model import bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.verify import verify_library


@pytest.mark.parametrize("n,r,seed", [(8, 3, 0), (10, 4, 1), (12, 5, 2), (11, 6, 3), (9, 5, 4)])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_nullspace_masks_random_code(n, r, seed, nonzero_syndrome):
    C = random_parity_code(n, r, seed)
    rows = gf2.rows_from_dense(C)
    rank = gf2.rank(rows, n)
    rng = np.random.default_rng(seed)
    if nonzero_syndrome:
        y0 = rng.integers(0, 2, size=n)
        b = (C.astype(np.int64) @ y0) % 2
    else:
        b = None
    Om = codewords(C, b)
    assert Om.shape[0] == 2 ** (n - rank)
    lib = B.nullspace_basis(C)
    assert lib.n_groups == n - rank
    rep = verify_library(lib, Om)
    assert rep.ok and rep.connected
    structure, w, c = random_polynomial(n, 18, 4, seed + 100)
    ids = code_identities(C, b)
    model = compile_model(structure, lib, identities=ids)
    res = model.resources["materialized"]
    assert res["M_add"] == 0
    assert res["routes_gate_controlled"] == 0 and res["routes_constant"] == res["routes_decoder"]
    assert not model.feasible_simplification
    # retained connections are exactly the odd-overlap (clause, mask) pairs
    sup = model.clause_supports()
    for rr, g in enumerate(lib.groups):
        expect = sorted(k for k, A in enumerate(sup) if len(set(A) & set(g.mask)) % 2)
        assert model.group_routes(rr)["clause"].tolist() == expect
    binding = bind_weights(model, w, c)
    out = check_compiled_exhaustive(model, lib, Om, binding=binding, energy=lambda X: direct_energy(structure, w, c, X),
                                    require_live_routes=True)
    assert out["dead_routes"] == 0
    pf = preflight(structure, lib, identities=ids)
    assert pf["M_add_upper"] == 0 and pf["routes_decoder_upper"] >= res["routes_decoder"]


def test_single_flips_recover_nhoim():
    """Single-spin flips: retained connectivity = columns of H, all switches 1 (Methods, Derivation of the transition-programmable architecture)."""
    structure, w, c = random_polynomial(7, 12, 3, 9)
    lib = B.single_flips(7)
    model = compile_model(structure, lib)
    pi = model.pi_matrix()
    sup = model.clause_supports()
    H = np.zeros((model.n_clauses, 7), dtype=np.uint8)
    for k, A in enumerate(sup):
        H[k, list(A)] = 1
    assert np.array_equal(pi, H)
    assert np.all(model.route_ctrl == -1)
    check_compiled_exhaustive(model, lib, all_states(7), binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))
