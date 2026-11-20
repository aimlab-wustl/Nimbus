"""constrained_blocks: generic block-constrained adapter.

Exhaustive checks on tiny instances of each block kind and mixtures: feasibility/involution/
connectivity, 2v = dE for every feasible state and group, OPB and spinpoly fixtures through the
public parser -> compiler -> emulator path, unsupported inputs rejected, penalty formulation zero
iff feasible (exhaustive).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.compiler import compile_model
from nimbus.core.errors import ConfigError, UnsupportedProblem
from nimbus.core.model import bind_weights
from nimbus.problems.constrained_blocks import ConstrainedBlocksProblem, from_config

pytestmark = pytest.mark.quick

_OPB_DIR = Path(__file__).parent / "fixtures" / "opb"
_SPINPOLY_DIR = Path(__file__).parent / "fixtures" / "spinpoly"


def _all_states(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    return (1 - 2 * bits).astype(np.int8)


def _brute_feasible(problem, n):
    S = _all_states(n)
    mask = np.array([problem.is_feasible(s) for s in S])
    return S[mask]


def _check_exhaustive(problem, n, *, library_name="default"):
    brute = _brute_feasible(problem, n)
    enum_ = problem.enumerate_feasible()
    assert enum_ is not None
    assert set(map(tuple, enum_.tolist())) == set(map(tuple, brute.tolist()))

    lib = problem.transition_library(library_name) if library_name else problem.transition_library()
    model = compile_model(problem.structure(), lib)
    binding = bind_weights(model, problem.term_weights(), problem.const())
    T = model.clause_values(brute)
    v = model.inputs(T, binding, 0)
    poly = problem.polynomial(0)
    for r in range(model.n_groups):
        mu = lib.groups[r].apply(brute)
        dE = poly.evaluate(mu) - poly.evaluate(brute)
        assert np.allclose(2.0 * v[:, r], dE, atol=1e-9), f"group {r} ({lib.labels[r]})"
        # involution + feasibility preservation: mu(s) must also be feasible
        mu_feasible = np.array([problem.is_feasible(m) for m in mu])
        assert mu_feasible.all(), f"group {r} leaves the feasible set"
    return model, lib, brute


# ---------------------------------------------------------------------------------------------
# single block kinds
# ---------------------------------------------------------------------------------------------
def test_exact_cardinality_block():
    p = ConstrainedBlocksProblem(5, supports=((0, 1),), coeffs=[1.0], const=0.0,
                                 exact_cardinality=[((0, 1, 2, 3), 2)], free=[4])
    assert p.family == "exact_cardinality"
    _check_exhaustive(p, 5)


def test_chain_precedence_block():
    p = ConstrainedBlocksProblem(5, supports=((0, 4),), coeffs=[1.0], const=0.0, chains=[(0, 1, 2)], free=[3, 4])
    assert p.family == "chain_precedence"
    _check_exhaustive(p, 5)


def test_disjoint_or_cover_block():
    p = ConstrainedBlocksProblem(5, supports=((2, 3),), coeffs=[1.0], const=0.0,
                                 or_blocks=[((0, 1, 2), (1, -1, 1))], free=[3, 4])
    assert p.family == "disjoint_or_cover"
    _check_exhaustive(p, 5)


def test_unit_capacity_packing_block():
    p = ConstrainedBlocksProblem(5, supports=((3, 4),), coeffs=[1.0], const=0.0,
                                 amo_blocks=[(0, 1, 2)], free=[3, 4])
    assert p.family == "unit_capacity_packing"
    _check_exhaustive(p, 5)


@pytest.mark.parametrize("cyclic", [False, True])
def test_contiguity_block(cyclic):
    n = 6
    spins = (0, 1, 2, 3, 4) if cyclic else (0, 1, 2, 3)
    p = ConstrainedBlocksProblem(n, supports=((len(spins) - 1, n - 1),), coeffs=[1.0], const=0.0,
                                 contiguity=[(spins, cyclic)], free=[i for i in range(n) if i not in spins])
    assert p.family == "contiguity"
    _check_exhaustive(p, n)


def test_mixed_blocks_with_coupled_objective():
    """Objective couples variables across two different block kinds plus a free spin."""
    n = 9
    p = ConstrainedBlocksProblem(
        n, supports=((0, 4), (1, 6, 8), (8,)), coeffs=[1.0, -0.5, 2.0], const=0.3,
        exact_cardinality=[((0, 1, 2), 1)], chains=[(3, 4)], amo_blocks=[(5, 6)],
        contiguity=[((7,), False)], free=[8])
    assert p.family == "mixed_blocks"
    _check_exhaustive(p, n)


# ---------------------------------------------------------------------------------------------
# disjointness / validation
# ---------------------------------------------------------------------------------------------
def test_overlapping_blocks_rejected():
    with pytest.raises(UnsupportedProblem):
        ConstrainedBlocksProblem(4, supports=(), coeffs=[], const=0.0,
                                 exact_cardinality=[((0, 1), 1)], amo_blocks=[(1, 2)])


def test_unlisted_spin_rejected():
    with pytest.raises(UnsupportedProblem):
        ConstrainedBlocksProblem(4, supports=(), coeffs=[], const=0.0, exact_cardinality=[((0, 1), 1)], free=[2])


# ---------------------------------------------------------------------------------------------
# penalty formulation: zero iff feasible, exhaustive
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cyclic", [False, True])
def test_penalty_zero_iff_feasible(cyclic):
    n = 7
    spins = (0, 1, 2, 3, 4) if cyclic else (0, 1, 2, 3)
    free = [i for i in range(n) if i not in spins and i != 5 and i != 6]
    p = ConstrainedBlocksProblem(
        n, supports=((5, 6),), coeffs=[1.0], const=0.0, contiguity=[(spins, cyclic)],
        amo_blocks=[(5, 6)], free=free)
    pf = p.penalty_formulation(lam=6.0)
    S = _all_states(n)
    feasible = np.array([p.is_feasible(s) for s in S])
    viol = pf.violation.evaluate(S)
    assert np.allclose(viol[feasible], 0.0, atol=1e-9)
    assert np.all(np.abs(viol[~feasible]) > 1e-9)
    run = pf.run.evaluate(S)
    obj = pf.objective.evaluate(S)
    assert np.allclose(run[feasible], obj[feasible], atol=1e-9)


# ---------------------------------------------------------------------------------------------
# OPB / spinpoly end-to-end: public parser -> compiler -> emulator
# ---------------------------------------------------------------------------------------------
def test_spinpoly_end_to_end_through_emulator():
    p = from_config({"kind": "spinpoly", "path": "mixed_blocks.yaml"}, _SPINPOLY_DIR)
    lib = p.transition_library()
    model = compile_model(p.structure(), lib)
    binding = bind_weights(model, p.term_weights(), p.const())
    emu = get_backend("emulation")
    plan = emu.prepare(model, library=lib)
    s0 = p.random_feasible(seed=1, n=1)[0]
    assert p.is_feasible(s0)
    state = emu.init_state(plan, [binding], s0[None, :], [0], [0], [0], is_feasible=p.is_feasible)
    sched = S.geometric(2.0, 0.2, 50)
    for n in range(50):
        emu.step(state, n, sched, 123)
    assert p.is_feasible(state.rs.spins[0])


def test_opb_end_to_end_through_emulator():
    p = from_config({"kind": "opb", "path": "chain_precedence.opb"}, _OPB_DIR)
    lib = p.transition_library()
    model = compile_model(p.structure(), lib)
    binding = bind_weights(model, p.term_weights(), p.const())
    emu = get_backend("emulation")
    plan = emu.prepare(model, library=lib)
    s0 = p.random_feasible(seed=2, n=1)[0]
    assert p.is_feasible(s0)
    state = emu.init_state(plan, [binding], s0[None, :], [0], [0], [0], is_feasible=p.is_feasible)
    sched = S.geometric(2.0, 0.2, 50)
    for n in range(50):
        emu.step(state, n, sched, 321)
    final = state.rs.spins[0]
    assert p.is_feasible(final)
    assert model.energy(final, binding, 0) == pytest.approx(p.energy(final), abs=1e-6)


def test_opb_contiguity_sidecar():
    p = from_config({"kind": "opb", "path": "chain_precedence.opb", "contiguity": [{"spins": [6], "cyclic": False}]},
                    _OPB_DIR)
    assert p.family == "mixed_blocks"
    assert p.contiguity == (((5,), False),)


def test_opb_unsupported_rejected():
    with pytest.raises(UnsupportedProblem):
        from_config({"kind": "opb", "path": "unsupported_overlap.opb"}, _OPB_DIR)


def test_unknown_config_fields_rejected():
    with pytest.raises(ConfigError):
        from_config({"kind": "spinpoly", "path": "mixed_blocks.yaml", "bogus": 1}, _SPINPOLY_DIR)
    with pytest.raises(ConfigError):
        from_config({"kind": "opb", "path": "chain_precedence.opb", "bogus": 1}, _OPB_DIR)
