"""Finite-table route: GF(2) fit of the clause-change pattern over an enumerated
domain, automatic parity identities of Omega, READOUT spin actions, and honest failure."""

import itertools

import numpy as np
import pytest

from helpers_core import (
    chain_states,
    check_compiled_exhaustive,
    direct_energy,
    onehot_identities,
    permutation_states,
    random_polynomial,
)
from nimbus.compiler import compile_model
from nimbus.compiler.table_fit import AffineFitter, domain_identities
from nimbus.core.errors import ConfigError, UnsupportedTransition
from nimbus.core.model import KIND_READOUT, ROLE_GATE, ROLE_READOUT, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B
from nimbus.transitions.descriptors import TableTransition
from nimbus.transitions.library import TransitionLibrary
from nimbus.transitions.verify import verify_library


def kempe_path(nv: int):
    """Endpoint-seeded two-colour interchanges on proper 3-colourings of a path (Supplementary
    Methods, An example toy problem), as explicit tables. Spin index 3*j + colour (R=0, G=1, B=2)."""
    cols = [c for c in itertools.product(range(3), repeat=nv) if all(c[j] != c[j + 1] for j in range(nv - 1))]

    def spins(c):
        s = [-1] * (3 * nv)
        for j, x in enumerate(c):
            s[3 * j + x] = 1
        return tuple(s)

    dom = [spins(c) for c in cols]
    groups = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        tab = {}
        for c in cols:
            if c[0] not in (a, b):
                continue  # self-loop when the first vertex carries the third colour
            new = list(c)
            for j in range(nv):
                if c[j] in (a, b):
                    new[j] = b if c[j] == a else a
                else:
                    break
            tab[spins(c)] = spins(tuple(new))
        groups.append(TableTransition(tab, dom))
    return np.array(dom, dtype=np.int8), TransitionLibrary(3 * nv, groups, ["RG", "RB", "GB"], name=f"kempe{nv}",
                                                            template="table")


def test_kempe_two_vertices_table_route():
    Om, lib = kempe_path(2)
    assert verify_library(lib, Om).ok
    structure = ObjectiveStructure(n_spins=6, terms=[(0, 3), (1, 4), (2, 5)])  # T1,T2,T3 of Methods, An example toy problem
    w = np.array([1.0, 2.5, -0.7])
    model = compile_model(structure, lib, readout="all")
    assert np.all(model.act_kind == KIND_READOUT)
    binding = bind_weights(model, w, 0.0)
    check_compiled_exhaustive(model, lib, Om, binding=binding, energy=lambda X: direct_energy(structure, w, 0.0, X),
                              require_live_routes=True)
    sup = model.clause_supports()
    # Methods, An example toy problem: T1 under RG is switched by the middle-vertex blue sign s_2B = clause (5,)
    g = model.group_routes(0)
    k1 = sup.index((0, 3))
    pos = g["clause"].tolist().index(k1)
    assert sup[g["ctrl"][pos]] == (5,) and g["pol"][pos] == 1
    # every singleton is a readout clause; those used as controllers count as gates
    assert all(model.clause_role[sup.index((i,))] & ROLE_READOUT for i in range(6))
    res = model.resources["materialized"]
    assert res["M_add"] == sum(1 for k in range(model.n_clauses) if model.clause_role[k] & ROLE_GATE
                               and not model.clause_role[k] & 1)
    assert res["M"] + res["M_add"] + res["M_read"] == model.n_clauses


def test_kempe_three_vertices_full_readout_is_unsupported():
    """With readout='all', the third-vertex readout sign changes iff all three vertices carry the two
    exchanged colours: an AND that no single stored product can switch. Reported, not approximated."""
    Om, lib = kempe_path(3)
    assert verify_library(lib, Om).ok
    structure = ObjectiveStructure(n_spins=9, terms=[(0, 3), (1, 4), (2, 5)])
    with pytest.raises(UnsupportedTransition, match="single signed spin product"):
        compile_model(structure, lib, readout="all")


def test_table_uses_the_spin_singletons():
    """Finite-table groups are executed by the decoder on the spin singletons, which every model has
; no explicit readout request is needed and a partial one is refused."""
    Om, lib = kempe_path(2)
    structure = ObjectiveStructure(n_spins=6, terms=[(0, 3)])
    model = compile_model(structure, lib)
    assert np.all(model.readout_clause_of_spin >= 0)
    with pytest.raises(ConfigError):
        compile_model(structure, lib, readout=[0, 1, 2])


def test_chain_as_table_matches_symbolic_condmask():
    m = 4
    Om = chain_states(m)
    sym = B.chain_precedence(list(range(m)))
    tables = [TableTransition({tuple(s): tuple(g.apply(s)) for s in Om.tolist()}, [tuple(s) for s in Om.tolist()])
              for g in sym.groups]
    tab = TransitionLibrary(m, tables, sym.labels, name="chain_table", template="table")
    structure, w, c = random_polynomial(m, 9, 3, seed=5)
    m_tab = compile_model(structure, tab, readout="all")
    m_sym = compile_model(structure, sym, readout="all")
    for model, lib in ((m_tab, tab), (m_sym, sym)):
        check_compiled_exhaustive(model, lib, Om, binding=bind_weights(model, w, c),
                                  energy=lambda X: direct_energy(structure, w, c, X))
    # same encoder: identical inputs at every feasible state
    v_tab = m_tab.inputs(m_tab.clause_values(Om), bind_weights(m_tab, w, c))
    v_sym = m_sym.inputs(m_sym.clause_values(Om), bind_weights(m_sym, w, c))
    assert np.allclose(v_tab, v_sym)


def test_table_on_permutations_derives_onehot_identities():
    P = 3
    S, _ = permutation_states(P)
    dom = [tuple(s) for s in S.tolist()]
    rx = B.row_exchanges(P)
    tables = [TableTransition({s: tuple(g.apply(np.array(s))) for s in dom}, dom) for g in rx.groups]
    lib = TransitionLibrary(P * P, tables, rx.labels)
    structure = ObjectiveStructure(n_spins=9, terms=[(0,), (8,), (3, 7), (2,)])
    model = compile_model(structure, lib, readout="all")
    assert model.metadata["n_derived_identities"] == 2 * P - 1
    assert model.feasible_simplification
    from nimbus.core.algebra import IdentityReducer

    red = IdentityReducer(9, model.identities)
    for sup, sign in onehot_identities(P):
        assert red.is_constant(sup) == sign
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, [1, 2, 1, 2]),
                              energy=lambda X: direct_energy(structure, [1, 2, 1, 2], 0.0, X))
    # declared identities must hold on the table domain
    with pytest.raises(ConfigError):
        compile_model(structure, lib, readout="all", identities=[((0, 1), 1)])


def test_affine_fitter_bruteforce():
    rng = np.random.default_rng(3)
    S = np.array(list(itertools.product((-1, 1), repeat=5)), dtype=np.int8)
    Om = S[rng.random(S.shape[0]) < 0.4]
    fit = AffineFitter(Om)
    Y = (Om < 0).astype(int)
    for _ in range(40):
        g = rng.integers(0, 2, size=5)
        c = int(rng.integers(0, 2))
        b = (Y @ g + c) % 2
        gb, cb = fit.fit(b)
        gi = [i for i in range(5) if (gb >> i) & 1]
        assert np.array_equal((Y[:, gi].sum(1) + cb) % 2, b)
    ids = domain_identities(Om)
    for sup, sign in ids:
        assert np.all(np.prod(Om[:, list(sup)].astype(int), axis=1) == sign)
    # random patterns: the fitter succeeds iff some (g, c) reproduces b (brute force over all 2^6)
    n_rejected = 0
    for _ in range(40):
        b = rng.integers(0, 2, size=Om.shape[0])
        exists = any(np.array_equal((Y @ np.array(gc[:5]) + gc[5]) % 2, b)
                     for gc in itertools.product((0, 1), repeat=6))
        if exists:
            fit.fit(b)
        else:
            n_rejected += 1
            with pytest.raises(UnsupportedTransition):
                fit.fit(b)
    assert n_rejected > 0
