"""Cross-family integration: every required problem adapter, compiled through the public path, runs on
the event-driven emulation and on the optimized CPU backend (all strategies) with shared RNG streams.

Checks per family: identical trajectories (spins, clause signs, energies, counters) between emulation
and cpu; T equals the spin products; feasibility preserved; the maintained energy equals the problem's own
polynomial and its independent native objective (where defined)."""

from pathlib import Path

import numpy as np
import pytest

from nimbus.annealing import schedule as S
from nimbus.backends.base import get_backend
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.problems import constrained_blocks, matching, surface_code, toy_cardinality

FIX = Path(__file__).parent / "fixtures"
EX = Path(__file__).parent.parent / "examples" / "problems"


def _cases():
    return [
        ("toy_pairs", lambda: toy_cardinality.from_config({"kind": "toy_cardinality"}, FIX), "pair_exchanges"),
        ("toy_augmented", lambda: toy_cardinality.from_config({"kind": "toy_cardinality"}, FIX), "augmented"),
        ("matching_complete", lambda: matching.from_config({"kind": "matching", "tensor": "matching/five.npz"}, FIX),
         "complete"),
        ("matching_path", lambda: matching.from_config({"kind": "matching", "tensor": "matching/five.npz"}, FIX),
         "path"),
        ("surface_code_d3", lambda: surface_code.from_config({"kind": "surface_code", "d": 3, "p": 0.1}, FIX),
         "stabilizers_logical"),
        ("blocks_precedence", lambda: constrained_blocks.from_config({"kind": "opb", "path": "precedence_small.opb"}, EX), "default"),
        ("blocks_covering", lambda: constrained_blocks.from_config({"kind": "opb", "path": "covering_small.opb"}, EX), "default"),
        ("blocks_packing", lambda: constrained_blocks.from_config({"kind": "opb", "path": "packing_small.opb"}, EX), "default"),
        ("blocks_mixed", lambda: constrained_blocks.from_config({"kind": "spinpoly", "path": "blocks_small.spinpoly.yaml"}, EX), "default"),
    ]


def _compile(problem, lib_name):
    lib = problem.transition_library(lib_name)
    model = compile_model(problem.structure(), lib, identities=problem.identities(),
                          readout=problem.readout_default())
    binding = bind_weights(model, problem.term_weights(0), problem.const(0))
    return model, binding


def _run(name, model, binding, spins0, n_attempts, seed, **prep):
    be = get_backend(name)
    plan = be.prepare(model, **prep)
    state = be.init_state(plan, [binding], spins0)
    scale = float(np.abs(binding.J[0]).sum()) + 1.0
    sched = S.geometric(0.5 * scale / max(1, model.n_spins) + 1.0, 0.02 * scale / max(1, model.n_spins) + 0.01,
                        n_attempts)
    be.run(plan, state, sched, n_attempts_total=n_attempts, seed=seed)
    return be.export_state(state)


@pytest.mark.parametrize("case,factory,lib_name", _cases(), ids=[c[0] for c in _cases()])
def test_family_on_emulation_and_cpu(case, factory, lib_name):
    problem = factory()
    model, binding = _compile(problem, lib_name)
    n_rep = 4
    spins0 = problem.random_feasible(11, n_rep)
    ref = _run("emulation", model, binding, spins0, 1500, seed=5)

    # independent checks on the emulator's final states
    for r in range(n_rep):
        s = ref.spins[r]
        assert problem.is_feasible(s), f"{case}: infeasible final state"
        e_poly = problem.energy(s)
        assert ref.energies[r, 0] == pytest.approx(e_poly, rel=1e-9, abs=1e-9)
        native = problem.native_objective(s)
        assert problem.energy_to_native(e_poly) == pytest.approx(native, rel=1e-9, abs=1e-9)
        for k in range(model.n_clauses):
            sup = model.clause_spins[model.clause_ptr[k]:model.clause_ptr[k + 1]]
            assert ref.T[r, k] == int(np.prod(s[sup]))
    assert int(ref.counters["accepted_changed"].sum()) > 0, f"{case}: no state change happened"

    for strategy in ("recompute", "incremental", "lazy"):
        out = _run("cpu", model, binding, spins0, 1500, seed=5, strategy=strategy)
        np.testing.assert_array_equal(out.spins, ref.spins, err_msg=f"{case}/{strategy}")
        np.testing.assert_array_equal(out.T, ref.T, err_msg=f"{case}/{strategy}")
        np.testing.assert_allclose(out.energies, ref.energies, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(out.best_energy, ref.best_energy, rtol=1e-9, atol=1e-9)
        for key in ("attempts", "accepted", "accepted_changed", "self_loops", "rejected"):
            np.testing.assert_array_equal(out.counters[key], ref.counters[key], err_msg=f"{case}/{strategy}/{key}")
