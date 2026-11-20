"""Lock-step replay: event-driven emulation vs the direct recomputation oracle (Methods, Derivation of the transition-programmable architecture and Methods, Derivation of the transition-programmable architecture).

Both backends receive the same compiled model, weights, initial states, seed and schedule. After
every attempt we require identical (r, u, xi, accepted, changed), identical spins and stored clause
signs, |v_emulation - v_oracle| <= 1e-9 * scale for every group (whenever a state changed), and slot
energies equal to direct recomputation. The emulator also runs in validation mode, so every settled
transaction is checked against Methods, Derivation of the transition-programmable architecture independently of the oracle. Each family is
required to actually exercise rejections, self-loops, gate-controlled switch changes and
simultaneous sign + switch changes (counted, not assumed).

The case builders below are reused by the other backend tests (import the functions, not tests).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pytest

from helpers_core import (
    cardinality_identities,
    chain_states,
    code_identities,
    codewords,
    direct_energy,
    exact_cardinality_states,
    onehot_identities,
    permutation_states,
    random_distances,
    random_parity_code,
    random_polynomial,
    section29_problem,
    toy_structure,
    cyclic_assignment_objective,
)
from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.compiler import compile_model
from nimbus.core.model import ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.descriptors import TableTransition
from nimbus.transitions.library import TransitionLibrary


# ============================================================================================
# case builders
# ============================================================================================
@dataclass
class Case:
    name: str
    structure: ObjectiveStructure
    library: TransitionLibrary
    model: object
    states: np.ndarray
    weights: list
    consts: list
    bindings: list = field(default_factory=list)
    has_gates: bool = True
    has_selfloops: bool = True

    def energy(self, b, X):
        return direct_energy(self.structure, self.weights[b], self.consts[b], X)

    def is_feasible(self, s) -> bool:
        return bool(np.any(np.all(self.states == np.asarray(s)[None, :], axis=1)))

    def initial(self, n_rep: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.states[rng.integers(0, self.states.shape[0], size=n_rep)].copy()


def _finish(case: Case) -> Case:
    case.bindings = [bind_weights(case.model, w, c) for w, c in zip(case.weights, case.consts)]
    return case


def _dyadic(rng, n, scale=8):
    return rng.integers(-12, 13, size=n).astype(np.float64) / scale


def case_cardinality_toy() -> Case:
    """Paper toy (6 spins, exactly 3 selected): complete pair exchanges + double exchange (0,3)+(1,4)."""
    structure, w, c = toy_structure()
    lib = B.pair_exchanges(range(6)) + TransitionLibrary(6, [B.simultaneous_exchanges([(0, 3), (1, 4)])],
                                                         ["swap(0,3)+swap(1,4)"], name="toy_shortcut")
    model = compile_model(structure, lib, identities=cardinality_identities(6, 3), readout="all")
    w2 = _dyadic(np.random.default_rng(11), len(structure.terms))
    return _finish(Case("cardinality_toy", structure, lib, model, exact_cardinality_states(6, 3),
                        [w, w2], [c, 0.25]))


def case_cardinality_random() -> Case:
    """Random real-weight polynomial on 7 spins, exactly 3 selected, path exchanges + a double exchange."""
    structure, w, c = random_polynomial(7, 16, 3, seed=21)
    lib = B.pair_exchanges(range(7), "complete") + TransitionLibrary(
        7, [B.simultaneous_exchanges([(0, 3), (1, 4)]), B.simultaneous_exchanges([(2, 6), (3, 5)])])
    model = compile_model(structure, lib, identities=cardinality_identities(7, 3), readout="all")
    w2 = np.round(np.random.default_rng(5).normal(size=len(structure.terms)), 4)
    return _finish(Case("cardinality_random", structure, lib, model, exact_cardinality_states(7, 3),
                        [w, w2], [c, -0.3]))


def case_parity_masks() -> Case:
    """Parity constraints C y = 0: nullspace-basis masks plus two extra codeword masks; constant switches."""
    C = random_parity_code(9, 3, seed=4, row_weight=4)
    lib0 = B.nullspace_basis(C)
    extra = [sorted(set(lib0.groups[0].mask) ^ set(lib0.groups[1].mask)),
             sorted(set(lib0.groups[2].mask) ^ set(lib0.groups[3].mask))]
    lib = lib0 + B.fixed_masks(extra, 9, name="extra_masks")
    structure, w, c = random_polynomial(9, 18, 3, seed=8)
    model = compile_model(structure, lib, identities=code_identities(C), readout="all")
    w2 = _dyadic(np.random.default_rng(3), len(structure.terms), 4)
    case = Case("parity_masks", structure, lib, model, codewords(C), [w, w2], [c, 1.0])
    case.has_gates = False
    case.has_selfloops = False
    return _finish(case)


def case_onehot(P: int, pattern: str, min_len: int, symmetric: bool) -> Case:
    """One-hot P x P assignment (cyclic quadratic assignment objective): row exchanges + segment reversals (column permutations)."""
    d1 = random_distances(P, seed=P, symmetric=symmetric)
    d2 = random_distances(P, seed=P + 100, symmetric=symmetric)
    structure, w, c = cyclic_assignment_objective(d1)
    s2, w2, c2 = cyclic_assignment_objective(d2)
    assert s2.terms == structure.terms
    lib = B.row_exchanges(P, pattern) + B.segment_reversals(P, min_len=min_len)
    model = compile_model(structure, lib, identities=onehot_identities(P))
    S_, _ = permutation_states(P)
    case = Case(f"onehot_P{P}_{pattern}", structure, lib, model, S_, [w, w2], [c, c2])
    case.has_selfloops = False  # distinct rows/columns of a permutation matrix: every move changes the state
    return _finish(case)


def case_section29() -> Case:
    """Disjoint section-29 blocks: chain precedence, OR covering and unit-capacity packing (CONDMASK)."""
    n, lib, states, _, _ = section29_problem([("chain", 4), ("or", [1, -1, 1]), ("amo", 3)])
    structure, w, c = random_polynomial(n, 18, 3, seed=29)
    model = compile_model(structure, lib)
    w2 = _dyadic(np.random.default_rng(29), len(structure.terms))
    return _finish(Case("section29", structure, lib, model, states, [w, w2], [c, 0.5]))


def kempe_path2():
    """Endpoint-seeded two-colour interchanges on proper 3-colourings of a 2-vertex path, as explicit
    tables (Methods, An example toy problem). Spin 3*j + colour."""
    nv = 2
    cols = [cc for cc in itertools.product(range(3), repeat=nv) if all(cc[j] != cc[j + 1] for j in range(nv - 1))]

    def spins(cc):
        s = [-1] * (3 * nv)
        for j, x in enumerate(cc):
            s[3 * j + x] = 1
        return tuple(s)

    dom = [spins(cc) for cc in cols]
    groups = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        tab = {}
        for cc in cols:
            if cc[0] not in (a, b):
                continue
            new = list(cc)
            for j in range(nv):
                if cc[j] in (a, b):
                    new[j] = b if cc[j] == a else a
                else:
                    break
            tab[spins(cc)] = spins(tuple(new))
        groups.append(TableTransition(tab, dom))
    return np.array(dom, dtype=np.int8), TransitionLibrary(3 * nv, groups, ["RG", "RB", "GB"], name="kempe2",
                                                           template="table")


def case_table_kempe() -> Case:
    """Finite-table route with full singleton readout (READOUT spin actions)."""
    states, lib = kempe_path2()
    structure = ObjectiveStructure(n_spins=6, terms=[(0, 3), (1, 4), (2, 5), (0,), (4,), (1, 5)])
    model = compile_model(structure, lib, readout="all")
    w = np.array([1.0, 2.5, -0.75, 0.5, -1.25, 0.375])
    w2 = np.array([-0.5, 1.0, 1.5, -2.0, 0.25, 1.0])
    return _finish(Case("table_kempe", structure, lib, model, states, [w, w2], [0.0, 1.0]))


def case_table_chain() -> Case:
    """Chain precedence written as explicit tables (finite-table route, full readout)."""
    m = 5
    Om = chain_states(m)
    sym = B.chain_precedence(list(range(m)))
    dom = [tuple(s) for s in Om.tolist()]
    tables = [TableTransition({tuple(s): tuple(g.apply(s)) for s in Om.tolist()}, dom) for g in sym.groups]
    lib = TransitionLibrary(m, tables, sym.labels, name="chain_table", template="table")
    structure, w, c = random_polynomial(m, 10, 3, seed=55)
    model = compile_model(structure, lib, readout="all")
    w2 = _dyadic(np.random.default_rng(55), len(structure.terms))
    return _finish(Case("table_chain", structure, lib, model, Om, [w, w2], [c, 0.0]))


CASE_BUILDERS = {
    "cardinality_toy": case_cardinality_toy,
    "cardinality_random": case_cardinality_random,
    "parity_masks": case_parity_masks,
    "onehot_P4": lambda: case_onehot(4, "path", 3, True),
    "onehot_P5": lambda: case_onehot(5, "complete", 4, False),
    "section29": case_section29,
    "table_kempe": case_table_kempe,
    "table_chain": case_table_chain,
}


def mixing_schedule(case: Case, n_attempts: int) -> S.Schedule:
    """Geometric schedule spanning the actual positive |dE| of the proposed moves (from the oracle at
    the feasible states), so that both acceptances and rejections occur."""
    o = get_backend("oracle")
    plan = o.prepare(case.model, library=case.library)
    st = o.init_state(plan, case.bindings[0], case.states[: min(40, case.states.shape[0])])
    dE = 2.0 * np.abs(o.evaluate_inputs(st)).ravel()
    pos = dE[dE > 1e-12]
    hi, lo = float(np.quantile(pos, 0.75)), 0.2 * float(np.median(pos))
    return S.geometric(max(hi, lo), lo, n_attempts)


def make_pair(case: Case, n_rep_per_binding: int = 3, seed: int = 0, validate="full", **emu_opts):
    """(oracle, oracle state, emulator, emulator plan, emulator state) on the same replicas."""
    nb = len(case.bindings)
    n = nb * n_rep_per_binding
    spins0 = case.initial(n, seed)
    rb = np.repeat(np.arange(nb), n_rep_per_binding)
    rid = np.tile(np.arange(n_rep_per_binding), nb)
    o = get_backend("oracle")
    e = get_backend("emulation")
    po = o.prepare(case.model, library=case.library)
    pe = e.prepare(case.model, validate=validate, library=case.library, **emu_opts)
    so = o.init_state(po, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    se = e.init_state(pe, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    return o, so, e, pe, se


def assert_states_agree(case: Case, so, se, rows=None):
    """Spins and T of every replica; for ``rows`` (default all): slot-0 energies of both backends vs
    direct recomputation from the term list, and |v_emulation - v_oracle| <= 1e-9 * scale for all groups."""
    assert np.array_equal(so.rs.spins, se.rs.spins)
    assert np.array_equal(so.rs.T, se.rs.T)
    rows = np.arange(se.rs.n_replicas) if rows is None else np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return
    rb = se.rs.replica_binding.astype(np.int64)
    for b in np.unique(rb[rows]).tolist():
        rr = rows[rb[rows] == b]
        scale = float(se.layout.scale[b, 0])
        E_direct = case.energy(b, se.rs.spins[rr])
        assert np.all(np.abs(se.rs.energies[rr, 0] - E_direct) <= 1e-9 * scale)
        assert np.all(np.abs(so.rs.energies[rr, 0] - E_direct) <= 1e-9 * scale)
    vo = get_backend("oracle").evaluate_inputs(so)[rows]
    ve = get_backend("emulation").evaluate_inputs(se)[rows]
    scale = se.layout.scale[rb[rows], 0][:, None]
    assert np.all(np.abs(vo - ve) <= 1e-9 * scale), float(np.max(np.abs(vo - ve)))


# ============================================================================================
# the lock-step test
# ============================================================================================
N_ATTEMPTS = 2000


@pytest.mark.parametrize("name", list(CASE_BUILDERS))
def test_lockstep_emulation_vs_oracle(name):
    case = CASE_BUILDERS[name]()
    sched = mixing_schedule(case, N_ATTEMPTS)
    o, so, e, pe, se = make_pair(case, n_rep_per_binding=3, seed=7)
    seed = 12345
    assert_states_agree(case, so, se)
    tot = {"rejected": 0, "self_loops": 0, "changed": 0, "switch_changes_ctrl_gate": 0,
           "switch_changes_ctrl_objective": 0, "sign_and_switch_routes": 0, "toggles_gate": 0,
           "toggles_readout": 0, "readout_toggles": 0}
    for n in range(N_ATTEMPTS):
        ro = o.step(so, n, sched, seed)
        re = e.step(se, n, sched, seed, collect_events=True)
        assert np.array_equal(ro.group, re.group), n
        assert np.array_equal(ro.u, re.u) and np.array_equal(ro.xi, re.xi), n
        assert np.array_equal(ro.accepted, re.accepted), (n, ro.v, re.v, ro.xi)
        assert np.array_equal(ro.changed, re.changed), n
        # acceptance law checked independently of both backends
        tau = sched.tau(n)
        xi = -(tau / 2.0) * np.log(ro.u)
        assert np.allclose(ro.xi, xi, rtol=1e-15, atol=0)
        assert np.array_equal(ro.accepted, ro.v <= ro.xi)
        changed_rows = np.flatnonzero(ro.changed)
        assert_states_agree(case, so, se, rows=changed_rows)
        tot["rejected"] += int(np.count_nonzero(~ro.accepted))
        tot["self_loops"] += int(np.count_nonzero(ro.accepted & ~ro.changed))
        tot["changed"] += int(changed_rows.size)
        for k in ("switch_changes_ctrl_gate", "switch_changes_ctrl_objective", "sign_and_switch_routes",
                  "toggles_gate", "toggles_readout", "readout_toggles"):
            tot[k] += int(re.events[k].sum())
    assert_states_agree(case, so, se)  # all replicas at the end
    # identical counters and best bookkeeping
    for k in so.rs.counters:
        assert np.array_equal(so.rs.counters[k], se.rs.counters[k]), k
    assert np.allclose(so.rs.best_energy, se.rs.best_energy, rtol=0, atol=1e-9 * float(se.layout.scale.max()))
    assert np.array_equal(so.rs.best_attempt, se.rs.best_attempt)
    assert np.array_equal(so.rs.best_spins, se.rs.best_spins)
    # every run exercised the required situations (Methods, Derivation of the transition-programmable architecture validation list)
    assert tot["rejected"] > 0 and tot["changed"] > 0, tot
    if case.has_selfloops:
        assert tot["self_loops"] > 0, tot
    if case.has_gates:
        assert tot["switch_changes_ctrl_gate"] + tot["switch_changes_ctrl_objective"] > 0, tot
        assert tot["sign_and_switch_routes"] > 0, tot
    if name in ("cardinality_toy", "cardinality_random", "onehot_P4", "onehot_P5", "section29"):
        assert tot["switch_changes_ctrl_gate"] > 0 and tot["toggles_gate"] > 0, tot  # gate-only clauses at work
    res = case.model.resources["materialized"]
    if res["readout_routes"]:
        assert tot["readout_toggles"] > 0, tot  # singleton readout signs updated by the same decoder
    if res["M_read"]:
        assert tot["toggles_readout"] > 0, tot  # readout-only (zero-weight, non-controlling) singletons
    # emulator states stay feasible (the spin array is the readout of the architecture)
    assert all(case.is_feasible(s) for s in se.rs.spins)


def test_lockstep_run_equals_step():
    """``run`` (blocked vectorized draws) and ``step`` (scalar draws) give the same trajectory."""
    case = case_onehot(5, "complete", 4, False)
    sched = mixing_schedule(case, 600)
    o, so, e, pe, se = make_pair(case, n_rep_per_binding=2, seed=3, validate="off")
    res = e.run(pe, se, sched, seed=99)
    for n in range(600):
        o.step(so, n, sched, 99)
    assert np.array_equal(res.state.spins, so.rs.spins)
    for k in so.rs.counters:
        assert np.array_equal(res.state.counters[k], so.rs.counters[k]), k
    po = o.prepare(case.model, library=case.library)
    so2 = o.init_state(po, case.bindings, case.initial(4, 3), [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    ro = o.run(po, so2, sched, seed=99)
    assert np.array_equal(ro.state.spins, res.state.spins)
    assert np.array_equal(ro.state.best_attempt, res.state.best_attempt)
