"""Spins are stored singleton clauses updated by the decoder.

The manuscript's explicit runtime spin update s_new = mu^(r)(s_old) is not an implementation
requirement: every spin is a stored singleton clause (existing singletons reused, missing ones added),
updated through the same compiled decoder as every other clause, sigma_k = sum_r Pi_kr alpha_kr q_r and
T_k <- (1 - 2 sigma_k) T_k. The clause state is authoritative; spin readout, state-change detection,
best assignments and checkpoints come from the singleton clauses.

Checks:
1. compiler: every spin has a singleton clause, for every family; partial readout requests are refused;
2. exactness: on every feasible state of every replay family, the decoder applied to T(s) gives
   T(mu_r(s)) for every group r and every stored clause -- singletons included -- where mu_r is the
   descriptor transformation (independent reference);
3. backends: exported spins are the singleton readout of T (read-only), best assignments too, and
   the architecture backends (emulation, cpu in every strategy/layout) match the independent oracle,
   which applies mu_r to its own private spin array;
4. checkpoints: snapshots store the clause state and the singleton map; a snapshot whose clause state
   is inconsistent with its singleton readout is refused.
"""

import numpy as np
import pytest

from test_replay_core import CASE_BUILDERS, mixing_schedule
from nimbus.backends import get_backend
from nimbus.backends.base import ReplicaStates, clause_signs
from nimbus.backends.cpu import LAYOUTS, STRATEGIES
from nimbus.backends.gpu import cuda_available
from nimbus.compiler import compile_model
from nimbus.core.errors import ConfigError, IncompatibleCheckpoint
from nimbus.core.model import ROLE_READOUT
from nimbus.transitions import builders as B

N_ATT, SEED, N_REP = 400, 5, 3
ARCH = [("emulation", {})] + [("cpu", dict(strategy=s, layout=l)) for s in STRATEGIES for l in LAYOUTS]
ARCH_IDS = [f"{b}-{'-'.join(o.values())}" if o else b for b, o in ARCH]


# -- 1. compiler -----------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CASE_BUILDERS))
def test_every_spin_is_a_singleton_clause(name):
    m = CASE_BUILDERS[name]().model
    rcs = m.readout_clause_of_spin
    assert rcs.shape == (m.n_spins,) and np.all(rcs >= 0)
    for i, k in enumerate(rcs.tolist()):
        assert m.clause_support(k) == (i,)
        assert m.clause_role[k] & ROLE_READOUT


def test_partial_readout_is_refused():
    from nimbus.core.model import ObjectiveStructure

    st = ObjectiveStructure(3, [(0, 1)])
    with pytest.raises(ConfigError, match="singleton"):
        compile_model(st, B.single_flips(3), readout=[0, 1])
    m = compile_model(st, B.single_flips(3), readout=None)
    assert np.all(m.readout_clause_of_spin >= 0)


# -- 2. exactness of the singleton decoder on every feasible state ----------------------------
@pytest.mark.parametrize("name", sorted(CASE_BUILDERS))
def test_decoder_executes_mu_on_all_clauses(name):
    case = CASE_BUILDERS[name]()
    m = case.model
    S = case.states
    T = m.clause_values(S)
    for r in range(m.n_groups):
        sigma = m.decoder_toggles(r, T).astype(np.int8)
        T_dec = (1 - 2 * sigma) * T                      # T_k <- (1 - 2 sigma_k) T_k
        mu = np.stack([np.asarray(case.library.groups[r].apply(s), dtype=np.int8) for s in S])
        np.testing.assert_array_equal(T_dec, m.clause_values(mu), err_msg=f"{name}: group {r}")
        np.testing.assert_array_equal(m.spin_readout(T_dec), mu, err_msg=f"{name}: group {r} readout")


# -- 3. backends ----------------------------------------------------------------------------------
def _run(case, backend, opts, spins0, rb):
    b = get_backend(backend)
    if backend == "oracle":
        opts = dict(opts, library=case.library)
    plan = b.prepare(case.model, **opts)
    st = b.init_state(plan, [case.bindings[i] for i in rb], spins0, np.arange(len(rb)), np.arange(len(rb)), rb)
    return b.run(plan, st, mixing_schedule(case, N_ATT), n_attempts_total=N_ATT, seed=SEED).state


@pytest.mark.parametrize("backend,opts", ARCH, ids=ARCH_IDS)
@pytest.mark.parametrize("name", sorted(CASE_BUILDERS))
def test_backend_state_is_clause_state(name, backend, opts):
    case = CASE_BUILDERS[name]()
    m = case.model
    spins0 = case.initial(N_REP, 3)
    rb = np.arange(N_REP) % len(case.bindings)
    ref = _run(case, "oracle", {}, spins0, rb)
    out = _run(case, backend, opts, spins0, rb)
    assert int(out.counters["accepted_changed"].sum()) > 0
    # the readout is a pure function of the clause state, and it cannot be written
    np.testing.assert_array_equal(out.spins, m.spin_readout(out.T))
    with pytest.raises(ValueError):
        out.spins[0, 0] = 1
    np.testing.assert_array_equal(clause_signs(m, out.spins), out.T)
    # identical trajectory outcome to the oracle, which applies mu to its private spin array
    np.testing.assert_array_equal(out.T, ref.T)
    np.testing.assert_array_equal(out.spins, ref.spins)
    np.testing.assert_array_equal(out.best_spins, ref.best_spins)
    np.testing.assert_array_equal(out.best_attempt, ref.best_attempt)
    for k in ("accepted", "accepted_changed", "self_loops", "rejected"):
        np.testing.assert_array_equal(out.counters[k], ref.counters[k], err_msg=k)
    # best assignments are feasible readouts whose energy is the recorded best energy
    for i in range(N_REP):
        assert case.is_feasible(out.best_spins[i])
        assert case.energy(int(rb[i]), out.best_spins[i][None, :])[0] == pytest.approx(out.best_energy[i], abs=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="not executed: no CUDA device")
@pytest.mark.parametrize("strategy", ["recompute", "incremental"])
@pytest.mark.parametrize("variant", ["thread", "warp", "block"])
@pytest.mark.parametrize("name", sorted(CASE_BUILDERS))
def test_gpu_state_is_clause_state(name, variant, strategy):
    """The GPU kernel has no spin array: its exported clause state, singleton readout, best assignments
    and counters match the oracle, which applies mu to its own spin array (every kernel variant)."""
    test_backend_state_is_clause_state(name, "gpu", {"kernel_variant": variant, "strategy": strategy})


@pytest.mark.parametrize("backend,opts", ARCH + [("oracle", {})], ids=ARCH_IDS + ["oracle"])
def test_checkpoint_round_trip_and_consistency(tmp_path, backend, opts):
    case = CASE_BUILDERS["onehot_P5"]()
    spins0 = case.initial(N_REP, 3)
    rb = np.arange(N_REP) % len(case.bindings)
    snap = _run(case, backend, opts, spins0, rb)
    path = snap.save(tmp_path / "s")
    back = ReplicaStates.load(path)
    assert "state.spins" not in np.load(path).files  # the clause state is what is persisted
    np.testing.assert_array_equal(back.T, snap.T)
    np.testing.assert_array_equal(back.spins, snap.spins)
    b = get_backend(backend)
    plan = b.prepare(case.model, **(dict(opts, library=case.library) if backend == "oracle" else opts))
    b.load_state(plan, back, bindings=[case.bindings[i] for i in rb])
    # flip one non-singleton clause only: the clause state no longer agrees with its singleton readout
    bad = back.copy()
    k = int(np.flatnonzero(np.diff(case.model.clause_ptr) > 1)[0])
    bad.T[0, k] = -bad.T[0, k]
    with pytest.raises(IncompatibleCheckpoint):
        b.load_state(plan, bad, bindings=[case.bindings[i] for i in rb])
