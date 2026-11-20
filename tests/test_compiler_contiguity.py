"""Contiguity (consecutive-ones) blocks on paths and cycles: a constraint family verified exhaustively here.
Compiled through the same encoder/decoder; exhaustive alpha / v / decoder checks with random coupled
objectives of degree <= 3, plus feasibility, involution and connectivity."""

import itertools

import numpy as np
import pytest

from helpers_core import check_compiled_exhaustive, direct_energy, gate_only_supports, random_polynomial
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary
from nimbus.transitions.verify import verify_library


def _feasible(s, blocks):
    for spins, cyclic in blocks:
        x = [s[i] for i in spins]
        sel = [k for k, v in enumerate(x) if v == 1]
        if not sel:
            return False
        if cyclic:
            if len(sel) == len(x) or sum(1 for k in range(len(x)) if x[k] == 1 and x[(k + 1) % len(x)] == -1) != 1:
                return False
        elif sel != list(range(sel[0], sel[-1] + 1)):
            return False
    return True


def _problem(blocks):
    n = sum(len(sp) for sp, _ in blocks)
    Om = np.array([s for s in itertools.product((-1, 1), repeat=n) if _feasible(s, blocks)], dtype=np.int8)
    libs = [B.contiguity_block(sp, cyclic=cy, n_spins=n) for sp, cy in blocks]
    lib = libs[0] if len(libs) == 1 else TransitionLibrary.concat(*libs, name="contiguity_mix")
    return n, lib, Om


@pytest.mark.parametrize("m,cyclic", [(m, False) for m in range(1, 8)] + [(m, True) for m in range(3, 8)])
def test_single_block(m, cyclic):
    blocks = [(list(range(m)), cyclic)]
    n, lib, Om = _problem(blocks)
    if m == 1:  # the single variable is fixed to +1: one feasible state, no transition neuron
        assert lib.n_groups == 0 and len(Om) == 1
        return
    rep = verify_library(lib, Om, is_feasible=lambda s: _feasible(s, blocks))
    assert rep.feasibility_preserved and rep.involution and rep.connected, rep.summary()
    structure, w, c = random_polynomial(n, 2 * n + 4, 3, 100 + m + 10 * cyclic)
    model = compile_model(structure, lib)
    check_compiled_exhaustive(model, lib, Om, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))
    # closure: only neighbour products (or singletons at path ends) are requested, at most m of them
    allowed = {tuple(sorted(g.cond)) for g in lib.groups if g.cond}
    gates = gate_only_supports(model)
    assert all(tuple(sup) in allowed for sup in gates) and len(gates) <= m


def test_mixed_blocks_coupled_objective():
    blocks = [([0, 1, 2, 3], False), ([4, 5, 6], True), ([7, 8], False)]
    n, lib, Om = _problem(blocks)
    rep = verify_library(lib, Om, is_feasible=lambda s: _feasible(s, blocks))
    assert rep.feasibility_preserved and rep.involution and rep.connected
    structure, w, c = random_polynomial(n, 30, 3, 7)
    model = compile_model(structure, lib)
    check_compiled_exhaustive(model, lib, Om, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))
