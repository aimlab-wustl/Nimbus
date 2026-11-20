"""Compile limits (ResourceLimitExceeded with partial reports), preflight upper bounds, score
structures for baselines, and compiler input validation."""

import numpy as np
import pytest

from helpers_core import (
    all_states,
    cardinality_identities,
    check_compiled_exhaustive,
    code_identities,
    direct_energy,
    onehot_identities,
    random_distances,
    random_parity_code,
    random_polynomial,
    section29_problem,
    cyclic_assignment_objective,
)
from nimbus.compiler import CompileLimits, CompileOptions, compile_model, preflight
from nimbus.core.errors import ConfigError, ResourceLimitExceeded, UnsupportedTransition
from nimbus.core.model import ROLE_SCORE, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.descriptors import SignedPermutation
from nimbus.transitions.library import TransitionLibrary


def _onehot_assignment(P=6):
    structure, w, c = cyclic_assignment_objective(random_distances(P, 0))
    return structure, B.row_exchanges(P), onehot_identities(P)


@pytest.mark.parametrize("limits,field", [
    (CompileLimits(max_clauses=50), "stored clauses"),
    (CompileLimits(max_routes=100), "routes"),
    (CompileLimits(max_closure_rounds=1), "closure rounds"),
    (CompileLimits(max_memory_bytes=10_000), "memory"),
    (CompileLimits(time_limit_s=0.0), "time"),
])
def test_limits_raise_with_partial_report(limits, field):
    structure, lib, ids = _onehot_assignment()
    with pytest.raises(ResourceLimitExceeded) as ei:
        compile_model(structure, lib, identities=ids, limits=limits)
    rep = ei.value.report
    assert rep["partial"] is True and field in rep["reason"]
    for key in ("n_clauses", "n_routes", "closure_rounds", "M", "M_add_so_far", "memory_bytes_estimate", "limits"):
        assert key in rep


def test_limits_not_hit_compile_matches_unlimited():
    structure, lib, ids = _onehot_assignment(5)
    a = compile_model(structure, lib, identities=ids)
    res = a.resources["materialized"]
    b = compile_model(structure, lib, identities=ids,
                      limits=CompileLimits(max_clauses=res["M_total"], max_routes=res["routes_decoder"],
                                           max_closure_rounds=res["closure_rounds"], time_limit_s=60.0))
    assert a.topology_hash == b.topology_hash


def test_preflight_bounds_cover_materialized_counts():
    cases = []
    s, w, c = random_polynomial(8, 16, 3, 1)
    cases.append((s, B.pair_exchanges(range(8)), cardinality_identities(8, 4), None))
    cases.append((s, B.single_flips(8), (), "all"))
    C = random_parity_code(10, 4, 2)
    s2, _, _ = random_polynomial(10, 20, 3, 2)
    cases.append((s2, B.nullspace_basis(C), code_identities(C), None))
    for P, pat in ((5, "complete"), (6, "path"), (7, "random_tree")):
        st, _, _ = cyclic_assignment_objective(random_distances(P, P))
        cases.append((st, B.row_exchanges(P, pat, seed=P), onehot_identities(P), None))
        cases.append((st, B.row_exchanges(P, pat, seed=P), onehot_identities(P), "all"))
    n, lib29, _, _, _ = section29_problem([("chain", 4), ("or", [1, -1, 1]), ("amo", 4)])
    s3, _, _ = random_polynomial(n, 25, 3, 3)
    cases.append((s3, lib29, (), None))
    for st, lib, ids, ro in cases:
        model = compile_model(st, lib, identities=ids, readout=ro)
        res = model.resources["materialized"]
        pf = preflight(st, lib, ro, identities=ids)
        est = model.resources["estimated"]
        assert est == pf
        assert pf["M"] == res["M"] and pf["R"] == res["R"]
        assert pf["M_add_upper"] >= res["M_add"], (lib.name, pf, res)
        assert pf["M_total_upper"] >= res["M_total"]
        assert pf["routes_weighted_upper"] >= res["routes_weighted"]
        assert pf["routes_decoder_upper"] >= res["routes_decoder"]
        assert pf["support_storage_upper"] >= res["support_storage"]
        assert pf["memory_bytes_upper"] >= res["memory_bytes"]


def test_preflight_unknown_class_is_explicit():
    st, _, _ = cyclic_assignment_objective(random_distances(5, 1))
    pf = preflight(st, B.segment_reversals(5, min_len=4))
    assert pf["M_add_upper"] is None and pf["memory_bytes_upper"] is None
    assert "no structural" in pf["gate_bound_source"]


def test_score_structures_for_baselines():
    """Run = objective + penalty; score slots 'objective' and 'violation' plus an extra SCORE-only clause."""
    n = 6
    obj, w, c = random_polynomial(n, 8, 3, 7, duplicates=False, constant=False)
    pen_terms = [(i, j) for i in range(n) for j in range(i + 1, n)]  # (n_+ - k)^2 expands into pairs
    run = ObjectiveStructure(n_spins=n, terms=list(obj.terms) + pen_terms)
    extra = ObjectiveStructure(n_spins=n, terms=[(0, 1, 2, 3, 4, 5)] + list(obj.terms[:2]))
    lib = B.single_flips(n)
    model = compile_model(run, lib, score_structures={"objective": obj, "violation": ObjectiveStructure(
        n_spins=n, terms=pen_terms), "extra": extra})
    full = model.clause_supports().index((0, 1, 2, 3, 4, 5))
    assert model.clause_role[full] == ROLE_SCORE
    assert model.resources["materialized"]["M_score"] == 1
    lam = 2.0
    run_w = np.concatenate([w, -lam * np.ones(len(pen_terms))])
    xw = np.array([0.5, 1.0, -1.0])
    b = bind_weights(model, run_w, 1.0, extra_slots={"objective": (w, 0.0), "violation": (-lam * np.ones(len(pen_terms)), 0.0),
                                                     "extra": (xw, 3.0)})
    assert b.slot_names == ("run", "objective", "violation", "extra")
    assert b.J[0, full] == 0.0 and b.J[3, full] == 0.5
    S = all_states(n)
    assert np.allclose(model.energy(S, b, "objective"), direct_energy(obj, w, 0.0, S))
    assert np.allclose(model.energy(S, b, "extra"), direct_energy(extra, xw, 3.0, S))
    assert np.allclose(model.energy(S, b, "run"),
                       model.energy(S, b, "objective") + model.energy(S, b, "violation") + 1.0)
    # all slots' energy changes follow from all routes
    T = model.clause_values(S)
    for slot in ("objective", "violation", "extra"):
        sums = model.route_sums(T, b, slot, weighted_only=False)
        for r, g in enumerate(lib.groups):
            dE = model.energy(g.apply(S), b, slot) - model.energy(S, b, slot)
            assert np.allclose(2 * sums[:, r], dE)
    check_compiled_exhaustive(model, lib, S, binding=b, energy=lambda X: model.energy(X, b, 0))
    with pytest.raises(ConfigError):
        compile_model(run, lib, score_structures={"run": obj})


def test_input_validation():
    s = ObjectiveStructure(n_spins=4, terms=[(0, 1)])
    with pytest.raises(ConfigError):
        compile_model(s, B.single_flips(5))
    with pytest.raises(ConfigError):
        compile_model(s, TransitionLibrary(4, []))
    cyc = TransitionLibrary(4, [SignedPermutation([(0, 1, 1), (1, 2, 1), (2, 0, 1)])])
    with pytest.raises(UnsupportedTransition):
        compile_model(s, cyc)
    m = compile_model(s, cyc, options=CompileOptions(allow_non_involutive=True))
    # a non-involutive group still compiles exactly (direction r as listed)
    check_compiled_exhaustive(m, cyc, all_states(4))
    with pytest.raises(ConfigError):
        compile_model(s, B.single_flips(4), identities=[((0, 1), 1), ((0, 1), -1)])


def test_gate_representative_option():
    """'requested' keeps the requested product; 'best' stores the low-weight representative."""
    structure = ObjectiveStructure(n_spins=9, terms=[(0,), (8,), (3, 7), (2,)])
    lib = TransitionLibrary(9, [B.row_exchange(0, 1, 3), B.row_exchange(1, 2, 3)])
    ids = onehot_identities(3)
    best = compile_model(structure, lib, identities=ids)
    req = compile_model(structure, lib, identities=ids, options=CompileOptions(gate_representative="requested"))
    # every spin is a stored singleton, so the requested product s1 s4 = s7 (column
    # identity) is found as the stored singleton of s7 under either option: no new gate is created
    for m in (best, req):
        k7 = int(m.readout_clause_of_spin[6])
        assert all(len(s) == 1 for s in m.clause_supports()[4:])
        assert k7 in m.group_routes(0)["ctrl"].tolist()
    plain = compile_model(structure, lib, identities=ids, options=CompileOptions(use_identities=False))
    assert plain.n_clauses > best.n_clauses and plain.identities == () and not plain.feasible_simplification
    from helpers_core import permutation_states

    S, _ = permutation_states(3)
    for m in (best, req, plain):
        check_compiled_exhaustive(m, lib, S)
