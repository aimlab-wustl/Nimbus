"""Exact-cardinality pair exchanges (Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges)) and the
simultaneous double exchange compiled from the net product (Methods, An example toy problem)."""

import pytest

from helpers_core import (
    cardinality_identities,
    check_compiled_exhaustive,
    direct_energy,
    exact_cardinality_states,
    gate_only_supports,
    random_polynomial,
    toy_structure,
)
from nimbus.compiler import compile_model, preflight
from nimbus.core.model import ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary


@pytest.mark.parametrize("n,k,seed,pattern", [(6, 3, 0, "complete"), (7, 3, 1, "complete"), (8, 4, 2, "complete"),
                                              (7, 2, 3, "path"), (8, 3, 4, "random_tree"), (6, 2, 5, "star")])
@pytest.mark.parametrize("use_identities", [True, False])
def test_pair_exchanges_random_polynomials(n, k, seed, pattern, use_identities):
    structure, w, c = random_polynomial(n, 14, 3, seed)
    lib = B.pair_exchanges(range(n), pattern=pattern, seed=seed)
    ids = cardinality_identities(n, k) if use_identities else ()
    model = compile_model(structure, lib, identities=ids)
    Om = exact_cardinality_states(n, k)
    binding = bind_weights(model, w, c)
    rep = check_compiled_exhaustive(model, lib, Om, binding=binding,
                                    energy=lambda X: direct_energy(structure, w, c, X))
    assert rep["clause_group_state_checks"] == Om.shape[0] * lib.n_groups * model.n_clauses
    res = model.resources["materialized"]
    assert res["M_add"] <= lib.n_groups  # Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges)
    # every added gate is a requested pair product s_i s_j of an exchanged pair (or its complement
    # product, which the total-product identity makes equivalent)
    pairs = {tuple(sorted(g.dests)) for g in lib.groups}
    for sup in gate_only_supports(model):
        if len(sup) == 2:
            assert sup in pairs
        else:
            assert use_identities and tuple(sorted(set(range(n)) - set(sup))) in pairs
    # alpha switch: (1 - s_i s_j)/2 for clauses containing exactly one of i, j; never a constant route
    assert res["routes_constant"] == 0
    pf = preflight(structure, lib, identities=ids)
    assert pf["M_add_upper"] == lib.n_groups and pf["M_add_upper"] >= res["M_add"]
    assert pf["routes_decoder_upper"] >= res["routes_decoder"]
    assert pf["memory_bytes_upper"] >= res["memory_bytes"]


def test_six_spin_toy_and_double_exchange():
    """Paper toy: 13 nonconstant clauses, const -11/8, J(s3) = 11/8; 15 pair exchanges + (1,4)+(2,5)."""
    structure, w, c = toy_structure()
    assert len(structure.unique_supports()) == 13
    assert c == pytest.approx(-11 / 8)
    backbone = B.pair_exchanges(range(6))
    shortcut = TransitionLibrary(6, [B.simultaneous_exchanges([(0, 3), (1, 4)])], ["swap(0,3)+swap(1,4)"],
                                 name="toy_shortcut", template="simultaneous_exchange")
    lib = backbone + shortcut
    assert lib.n_groups == 16
    model = compile_model(structure, lib, identities=cardinality_identities(6, 3))
    binding = bind_weights(model, w, c)
    j3 = binding.J[0, model.term_to_clause[structure.terms.index((2,))]]
    assert j3 == pytest.approx(11 / 8)
    Om = exact_cardinality_states(6, 3)
    assert Om.shape[0] == 20
    check_compiled_exhaustive(model, lib, Om, binding=binding, energy=lambda X: direct_energy(structure, w, c, X))
    # the double exchange uses one stored product per switch: (s1 s4)^d14 (s2 s5)^d25
    g = model.group_routes(15)
    sup = model.clause_supports()
    for k, ctrl in zip(g["clause"].tolist(), g["ctrl"].tolist()):
        A = set(sup[k])
        d14 = (0 in A) != (3 in A)
        d25 = (1 in A) != (4 in A)
        want = set()
        if d14:
            want ^= {0, 3}
        if d25:
            want ^= {1, 4}
        assert want, "a route of the shortcut must change the clause"
        got = set(sup[ctrl])
        comp = set(range(6)) - want
        assert got == want or got == comp  # equal, or equivalent via prod_all s = -1
    # at every feasible state nine pair exchanges change the state and six are self-loops
    for s in Om:
        changed = sum(not g_.is_selfloop(s) for g_ in backbone.groups)
        assert changed == 9


def test_double_exchange_without_identities_stores_net_product():
    structure = ObjectiveStructure(n_spins=6, terms=[(0,), (1,), (0, 1), (2, 3)])
    lib = TransitionLibrary(6, [B.simultaneous_exchanges([(0, 3), (1, 4)])])
    model = compile_model(structure, lib)
    sup = model.clause_supports()
    assert (0, 3) in sup and (1, 4) in sup and (0, 1, 3, 4) in sup
    check_compiled_exhaustive(model, lib, exact_cardinality_states(6, 3))
