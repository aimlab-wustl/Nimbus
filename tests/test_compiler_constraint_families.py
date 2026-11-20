"""Additional constraint families: disjoint precedence chains, logical covering
(OR) blocks with signed literals, unit-capacity packing (AMO) blocks; random coupled objectives of
degree <= 3; exhaustive alpha / v / decoder checks plus feasibility, involution, connectivity and
the closure-size statements."""

import numpy as np
import pytest

from helpers_core import check_compiled_exhaustive, direct_energy, gate_only_supports, random_polynomial, section29_problem
from nimbus.compiler import compile_model, preflight
from nimbus.core.model import bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.verify import verify_library


def _compile_and_check(blocks, seed, tree="path", n_terms=None):
    n, lib, Om, feas, spins_of = section29_problem(blocks, tree=tree)
    rep = verify_library(lib, Om, is_feasible=feas)
    assert rep.feasibility_preserved and rep.involution and rep.connected, rep.summary()
    structure, w, c = random_polynomial(n, n_terms or (2 * n + 4), 3, seed)
    model = compile_model(structure, lib)
    binding = bind_weights(model, w, c)
    out = check_compiled_exhaustive(model, lib, Om, binding=binding,
                                    energy=lambda X: direct_energy(structure, w, c, X))
    pf = preflight(structure, lib)
    res = model.resources["materialized"]
    assert res["M_add"] <= pf["M_add_upper"]
    assert res["routes_decoder"] <= pf["routes_decoder_upper"]
    assert res["memory_bytes"] <= pf["memory_bytes_upper"]
    return model, lib, spins_of, out


@pytest.mark.parametrize("m", range(1, 7))
def test_single_chain(m):
    model, lib, spins_of, _ = _compile_and_check([("chain", m)], seed=m)
    gates = gate_only_supports(model)
    # at most one requested product per chain position: the neighbour product of position i
    allowed = {g.cond for g in lib.groups if g.cond}
    assert all(sup in allowed for sup in gates) and len(gates) <= m


@pytest.mark.parametrize("m", range(2, 7))
@pytest.mark.parametrize("tree", ["path", "star"])
def test_single_or_block_signed(m, tree):
    signs = [(-1) ** (i * 7 % 3) for i in range(m)]
    model, lib, spins_of, _ = _compile_and_check([("or", signs)], seed=10 + m, tree=tree)
    gates = gate_only_supports(model)
    # only singleton gate clauses of the block's spins are ever needed
    assert all(len(sup) == 1 and sup[0] < m for sup in gates) and len(gates) <= m
    assert lib.n_groups == 2 * (m - 1)


@pytest.mark.parametrize("m", range(1, 7))
def test_single_amo_block(m):
    model, lib, spins_of, _ = _compile_and_check([("amo", m)], seed=20 + m)
    gates = gate_only_supports(model)
    allowed = {g.cond for g in lib.groups if g.cond}
    assert all(sup in allowed for sup in gates) and len(gates) <= m


def test_or_block_of_one_literal_fixes_the_variable():
    n, lib, Om, feas, _ = section29_problem([("or", [-1]), ("chain", 2)])
    assert Om.shape[0] == 3 and np.all(Om[:, 0] == -1)
    assert lib.n_groups == 2  # only the chain's groups


@pytest.mark.parametrize("seed", range(6))
def test_coupled_disjoint_blocks(seed):
    rng = np.random.default_rng(seed)
    blocks = [("chain", int(rng.integers(1, 6))), ("or", list(rng.choice([-1, 1], size=int(rng.integers(2, 5))))),
              ("amo", int(rng.integers(1, 6))), ("chain", int(rng.integers(1, 4)))]
    model, lib, spins_of, out = _compile_and_check(blocks, seed=100 + seed, n_terms=30)
    gates = gate_only_supports(model)
    conds = {g.cond for g in lib.groups if g.cond}
    assert all(sup in conds for sup in gates)
    # per-block closure sizes
    for kind, sp_, spec in spins_of:
        own = [sup for sup in gates if set(sup) <= set(sp_)]
        if kind == "or":
            assert all(len(s) == 1 for s in own)
        assert len(own) <= len(sp_)
    assert out["clause_group_state_checks"] > 0


def test_reproduces_reference_verification_counts():
    """Compile EVERY nonempty spin-product support as a clause for block sizes 1..7 and check all of
    them exhaustively through the compiled arrays. Adding the empty support (one trivially unchanged
    product per state and group) reproduces the reference clause-transition check counts: 11,260 / 245,016 / 11,260."""
    from nimbus.core.model import ObjectiveStructure

    expected = {"chain": 11260, "or": 245016, "amo": 11260}
    for family, want in expected.items():
        total = 0
        for m in range(1, 8):
            if family == "or":
                if m == 1:
                    continue  # no groups: the script also has zero groups for m = 1
                n, lib, Om, feas, _ = section29_problem([("or", [1] * m)])
            else:
                n, lib, Om, feas, _ = section29_problem([(family, m)])
            assert verify_library(lib, Om, is_feasible=feas).ok
            supports = [tuple(i for i in range(m) if (mask >> i) & 1) for mask in range(1, 1 << m)]
            model = compile_model(ObjectiveStructure(n_spins=m, terms=supports), lib)
            assert model.n_clauses == len(supports)  # every requested switch product is already stored
            out = check_compiled_exhaustive(model, lib, Om)
            total += out["clause_group_state_checks"] + Om.shape[0] * lib.n_groups  # + empty support
        assert total == want, (family, total)


def test_chain_switch_formula():
    """alpha = (1 - s_{i-1} s_{i+1})/2 for clauses containing position i (boundaries folded)."""
    m = 5
    lib = B.chain_precedence(list(range(m)))
    n, _, Om, _, _ = section29_problem([("chain", m)])
    from nimbus.core.model import ObjectiveStructure

    model = compile_model(ObjectiveStructure(n_spins=m, terms=[(i,) for i in range(m)]), lib)
    T = model.clause_values(Om)
    alpha = model.route_alpha(T)
    for r in range(m):
        g = model.group_routes(r)
        assert g["clause"][g["weighted"]].tolist() == [r]  # unweighted routes: gate clauses containing r
        left = Om[:, r - 1] if r > 0 else np.ones(Om.shape[0], dtype=np.int8)
        right = Om[:, r + 1] if r < m - 1 else -np.ones(Om.shape[0], dtype=np.int8)
        expect = (1 - left.astype(int) * right) // 2
        assert np.array_equal(alpha[:, g["positions"][0]], expect)
