"""Toy exact-cardinality problem adapter: expansion, feasibility, libraries, penalty, barriers."""

from __future__ import annotations

import numpy as np
import pytest

from helpers_core import all_states
from nimbus.problems import toy_cardinality as toy
from nimbus.problems.formats.spinpoly import read_spinpoly
from nimbus.transitions.verify import verify_library

pytestmark = pytest.mark.quick

FIXTURE = "tests/fixtures/spinpoly/six_spin_toy.yaml"


def test_analytic_expansion_matches_schema_fixture():
    doc = read_spinpoly(FIXTURE)
    poly = toy._toy_polynomial()
    assert len(poly.supports) == 13
    assert poly.const == pytest.approx(-11 / 8)
    assert set(poly.supports) == set(doc.supports)
    j_analytic = dict(zip(poly.supports, poly.clause_weights()))
    j_schema = dict(zip(doc.supports, doc.weights))
    for sup in poly.supports:
        assert j_analytic[sup] == pytest.approx(j_schema[sup]), sup
    assert float(doc.const) == pytest.approx(poly.const)
    assert j_analytic[(2,)] == pytest.approx(11 / 8)  # shared singleton (0-based spin 2 = vertex 3)


def test_from_config_builds_the_fixed_instance():
    prob = toy.from_config({"kind": "toy_cardinality"}, base_dir=".")
    assert isinstance(prob, toy.ToyCardinalityProblem)
    assert prob.n_spins == 6 and prob.family == "exact_cardinality"
    with pytest.raises(Exception):
        toy.from_config({"kind": "toy_cardinality", "bogus": 1}, base_dir=".")
    with pytest.raises(Exception):
        toy.from_config({"kind": "wrong"}, base_dir=".")


def test_feasible_states_and_identity():
    prob = toy.ToyCardinalityProblem()
    feas = prob.enumerate_feasible()
    assert feas.shape == (20, 6)
    assert all(prob.is_feasible(s) for s in feas)
    sup, sign = prob.identities()[0]
    T = np.prod(feas[:, list(sup)], axis=1)
    assert np.all(T == sign)


def test_start_and_optimum_energies():
    prob = toy.ToyCardinalityProblem()
    start = toy.spins_from_selection(toy.START_SELECTION)
    opt = toy.spins_from_selection(toy.OPTIMAL_SELECTION)
    assert prob.is_feasible(start) and prob.is_feasible(opt)
    assert prob.energy(start) == pytest.approx(-4.0)
    assert prob.energy(opt) == pytest.approx(-7.0)
    assert prob.native_objective(start) == pytest.approx(-4.0)
    assert prob.native_objective(opt) == pytest.approx(-7.0)
    assert prob.decode(start)["selected"] == (1, 2, 3)
    assert prob.decode(opt)["selected"] == (3, 4, 5)
    # energy() (own polynomial) and native_objective() (independent reward sum) agree everywhere feasible
    for s in prob.enumerate_feasible():
        assert prob.energy(s) == pytest.approx(prob.native_objective(s))


def test_random_feasible_is_deterministic_and_feasible():
    prob = toy.ToyCardinalityProblem()
    a = prob.random_feasible(seed=123, n=50)
    b = prob.random_feasible(seed=123, n=50)
    assert np.array_equal(a, b)
    assert all(prob.is_feasible(s) for s in a)
    c = prob.random_feasible(seed=124, n=50)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("name,n_groups", [("pair_exchanges", 15), ("augmented", 16)])
def test_libraries_preserve_feasibility_and_involution(name, n_groups):
    prob = toy.ToyCardinalityProblem()
    lib = prob.transition_library(name)
    assert lib.n_groups == n_groups
    feas = prob.enumerate_feasible()
    rep = verify_library(lib, feas, is_feasible=prob.is_feasible, require=True)
    assert rep.ok
    assert rep.feasibility_preserved and rep.involution


def test_complete_pair_exchange_selfloop_counts():
    """At every feasible state: 9 changing + 6 self-loop pair exchanges (3*3 cross, C(3,2)+C(3,2))."""
    prob = toy.ToyCardinalityProblem()
    lib = prob.transition_library("pair_exchanges")
    feas = prob.enumerate_feasible()
    for s in feas:
        changed = sum(1 for r in range(lib.n_groups) if not np.array_equal(lib.apply(r, s), s))
        assert changed == 9
        assert lib.n_groups - changed == 6


def test_penalty_formulation_exhaustive_on_all_64_states():
    prob = toy.ToyCardinalityProblem()
    pf = prob.penalty_formulation(lam=6.0)
    S = all_states(6)
    n_plus = (S.astype(np.float64) + 1).sum(axis=1) / 2.0
    viol_direct = (n_plus - 3.0) ** 2
    run_direct = pf.objective.evaluate(S) + 6.0 * viol_direct
    assert np.allclose(pf.violation.evaluate(S), viol_direct)
    assert np.allclose(pf.run.evaluate(S), run_direct)
    for s in prob.enumerate_feasible():
        assert pf.violation.evaluate(s) == pytest.approx(0.0, abs=1e-9)


def test_emulator_reaches_optimum():
    """Emulator run on the toy (pair_exchanges) reaches the optimum under a simple fixed-seed schedule."""
    from nimbus.annealing import schedule as S
    from nimbus.backends import get_backend
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights

    prob = toy.ToyCardinalityProblem()
    lib = prob.transition_library("pair_exchanges")
    model = compile_model(prob.structure(), lib, identities=prob.identities())
    binding = bind_weights(model, prob.term_weights(), prob.const())
    emu = get_backend("emulation")
    plan = emu.prepare(model, library=lib)
    start = toy.spins_from_selection(toy.START_SELECTION)
    state = emu.init_state(plan, [binding], start[None, :], [0], [0], [0], is_feasible=prob.is_feasible)
    sched = S.geometric(2.0, 0.05, 3000)
    res = emu.run(plan, state, sched, seed=42)
    assert prob.is_feasible(res.state.spins[0])
    assert float(res.state.best_energy[0]) == pytest.approx(-7.0, abs=1e-9)
