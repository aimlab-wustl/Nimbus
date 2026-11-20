"""Surface-code adapter: code structure, exact -log P energies, syndrome-preserving moves, logical-class
reachability (the shortcut), compile resources (no gates), registry."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nimbus.compiler import compile_model
from nimbus.core import gf2
from nimbus.core.model import bind_weights, structural_resources
from nimbus.problems import surface_code as SC
from nimbus.problems.factory import make_problem
from nimbus.transitions.verify import verify_library


@pytest.mark.parametrize("d", [3, 5, 7])
def test_code_structure(d):
    c = SC.rotated_surface_code(d)
    HX, HZ, n = c["HX"], c["HZ"], c["n"]
    assert HX.shape[0] + HZ.shape[0] == n - 1
    assert not (HX.astype(int) @ HZ.T.astype(int) % 2).any()
    rows = [gf2.indices_to_bits(np.flatnonzero(r).tolist()) for r in np.vstack([HX, HZ])]
    assert len(gf2.nullspace(rows, n)) == n - (n - 1)  # sum of ranks = n - 1  (k = 1)
    assert len(gf2.nullspace([gf2.indices_to_bits(np.flatnonzero(r).tolist()) for r in HZ], n)) == n - HZ.shape[0]
    for ln in c["lx_all"]:
        v = np.zeros(n, int); v[list(ln)] = 1
        assert len(ln) == d and not (HZ.astype(int) @ v % 2).any()
    assert set(map(len, c["lz_all"])) == {d}
    lx = np.zeros(n, int); lx[list(c["lx_all"][0])] = 1
    lz = np.zeros(n, int); lz[list(c["lz_all"][0])] = 1
    assert lx @ lz % 2 == 1


@pytest.mark.parametrize("noise", ["independent", "depolarizing"])
def test_energy_equals_direct_neg_log_likelihood(noise):
    prob = SC.SurfaceCodeProblem(5, 0.07, noise=noise)
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = rng.choice([-1, 1], size=prob.n_spins).astype(np.int8)
        e = prob.energy(s)
        assert prob.energy_to_native(e) == pytest.approx(prob.native_objective(s), abs=1e-9)
    # explicit check of the depolarizing clause: per-qubit cost log(3(1-p)/p) iff x or z
    if noise == "depolarizing":
        n = prob.nq
        s = np.ones(prob.n_spins, np.int8)
        s[2] = -1; s[n + 2] = -1; s[4] = -1          # qubit 2: Y, qubit 4: X
        assert prob.energy(s) == pytest.approx(2 * math.log(3 * 0.93 / 0.07))


@pytest.mark.parametrize("noise", ["independent", "depolarizing"])
@pytest.mark.parametrize("lib", ["stabilizers", "stabilizers_logical", "stabilizers_logical_all"])
def test_moves_preserve_syndrome(noise, lib):
    prob = SC.SurfaceCodeProblem(5, 0.05, noise=noise, library=lib)
    L = prob.transition_library()
    rng = np.random.default_rng(1)
    e = (rng.random(prob.n_spins) < 0.1).astype(np.uint8)
    syn = prob.syndrome_of(e)
    starts = prob.particular_spins(np.stack([syn, syn]))
    pr = SC.SurfaceCodeProblem(5, 0.05, noise=noise, syndrome=syn)
    assert all(pr.is_feasible(s) for s in starts)
    for g in L.groups:
        assert all(pr.is_feasible(x) for x in g.apply(starts))


@pytest.mark.parametrize("noise,d", [("independent", 3), ("independent", 5), ("depolarizing", 3)])
def test_reachability_verifier_and_shortcut(noise, d):
    for lib, conn in [("stabilizers", False), ("stabilizers_logical", True), ("stabilizers_logical_all", True)]:
        prob = SC.SurfaceCodeProblem(d, 0.05, noise=noise, library=lib)
        states = prob.enumerate_feasible()
        assert states is not None
        rep = verify_library(prob.transition_library(), states, is_feasible=prob.is_feasible)
        assert rep.feasibility_preserved and rep.involution
        assert rep.connected is conn, (lib, rep.n_components)
        if not conn:
            assert rep.n_components == 2 ** (2 if noise == "depolarizing" else 1)  # logical classes


def test_nonzero_syndrome_reachability():
    prob0 = SC.SurfaceCodeProblem(3, 0.05)
    e = np.zeros(9, np.uint8); e[[1, 4]] = 1
    prob = SC.SurfaceCodeProblem(3, 0.05, syndrome=prob0.syndrome_of(e))
    states = prob.enumerate_feasible()
    assert states.shape[0] == 32 and all(prob.is_feasible(s) for s in states)
    assert verify_library(prob.transition_library(), states, is_feasible=prob.is_feasible).connected


def test_logical_failure_metric():
    prob = SC.SurfaceCodeProblem(5, 0.05)
    c = prob.code
    e = np.zeros(25, np.uint8); e[[0, 1]] = 1
    stab = c["HX"][3].copy()                      # error equivalent by an X stabilizer: not a failure
    assert not prob.logical_failure((e + stab) % 2, e)[0]
    lx = np.zeros(25, np.uint8); lx[list(c["lx_all"][0])] = 1
    assert prob.logical_failure((e + lx) % 2, e)[0]
    assert prob.logical_failure(e, np.zeros(25, np.uint8))[0]  # residual with syndrome != 0 counts as failure


@pytest.mark.parametrize("noise", ["independent", "depolarizing"])
def test_compile_no_gates_and_compiled_dE(noise):
    prob = SC.SurfaceCodeProblem(5, 0.06, noise=noise)
    lib = prob.transition_library()
    model = compile_model(prob.structure(), lib, identities=())
    r = structural_resources(model)
    assert r["M_add"] == 0 and r["routes_gate_controlled"] == 0
    assert r["R"] == lib.n_groups
    binding = bind_weights(model, prob.term_weights(0), prob.const(0))
    assert binding is not None


def test_registry_and_config():
    p = make_problem({"kind": "surface_code", "d": 3, "p": 0.1, "noise": "depolarizing"})
    assert p.n_spins == 18 and p.family == "affine_parity"
    with pytest.raises(Exception):
        make_problem({"kind": "surface_code", "d": 3, "p": 0.1, "bogus": 1})


@pytest.mark.parametrize("noise", ["independent", "depolarizing"])
def test_cpu_backend_run_keeps_syndrome_and_energy(noise):
    from nimbus.annealing import schedule as S
    from nimbus.backends import get_backend

    prob0 = SC.SurfaceCodeProblem(5, 0.08, noise=noise)
    rng = np.random.default_rng(3)
    e = (rng.random((6, prob0.n_spins)) < 0.08).astype(np.uint8)
    syn = prob0.syndrome_of(e)
    spins0 = prob0.particular_spins(syn)
    model = compile_model(prob0.structure(), prob0.transition_library(), identities=())
    binding = bind_weights(model, prob0.term_weights(0), prob0.const(0))
    be = get_backend("cpu")
    plan = be.prepare(model, threads=1)
    st = be.init_state(plan, binding, spins0, replica_id=np.arange(6), instance_id=np.zeros(6, dtype=np.int64))
    s = be.run(plan, st, S.fn_endpoints(8.0, 1.0, 0.1, 4096), seed=5).state
    for k in range(6):
        pk = SC.SurfaceCodeProblem(5, 0.08, noise=noise, syndrome=syn[k])
        assert pk.is_feasible(s.spins[k]) and pk.is_feasible(s.best_spins[k])
        assert prob0.energy(s.best_spins[k]) == pytest.approx(float(s.best_energy[k]), abs=1e-9)
        assert prob0.energy(s.spins[k]) == pytest.approx(float(s.energies[k, 0]), abs=1e-9)
        assert s.best_energy[k] <= prob0.energy(spins0[k]) + 1e-9
