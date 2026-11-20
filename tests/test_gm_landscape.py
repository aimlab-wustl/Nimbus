"""Landscape properties of the matching move libraries (experiments/graph_matching/landscape.py): relaxation time at
infinite temperature against its closed form, and the union-find critical depth and Catoni difficulty against an
exhaustive search of escape routes on a small instance."""
from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "graph_matching"))
import landscape as LS  # noqa: E402

from nimbus.problems.formats.matching_tensor import load_tensor  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "matching"


@pytest.mark.parametrize("L", [5, 10, 20])
def test_path_relaxation_time_closed_form(L):
    edges, w = LS.library_edges(None, L, "path", 0, 0)
    nu2 = 2 * (1 - math.cos(math.pi / L)) / (L - 1)
    assert LS.t_rel(L, edges, w) == pytest.approx(1 / nu2, rel=1e-10)


def test_complete_uniform_relaxation_time():
    L = 8
    edges, w = LS.library_edges(None, L, "complete_uniform", 0, 0)
    assert LS.t_rel(L, edges, w) == pytest.approx((L - 1) / 2, rel=1e-10)    # nu2 = L / C(L, 2)


def _brute(E, nbr):
    """Depth of every nonoptimal local minimum by a minimax (bottleneck) search to any strictly lower state."""
    V = len(E)
    emin = E.min()
    nmin, dstar, diff = 0, 0.0, 0.0
    for x in range(V):
        if E[x] <= emin or any(E[u] < E[x] for u in nbr[x]):
            continue
        nmin += 1
        best = np.full(V, np.inf)
        best[x] = E[x]
        heap = [(E[x], x)]
        level = None
        while heap:
            b, v = heapq.heappop(heap)
            if b > best[v]:
                continue
            if E[v] < E[x]:
                level = b
                break
            for u in nbr[v]:
                nb = max(b, E[u])
                if nb < best[u]:
                    best[u] = nb
                    heapq.heappush(heap, (nb, u))
        d = level - E[x]
        dstar = max(dstar, d)
        diff = max(diff, d / (E[x] - emin))
    return nmin, dstar, diff


@pytest.mark.parametrize("kind,k,seed", [("path", 0, 0), ("informed", 2, 0), ("random", 2, 3), ("complete", 0, 0)])
def test_union_find_equals_exhaustive_search(kind, k, seed):
    t = load_tensor(FIX / "five.npz")
    n = int(t.n)
    P, col = LS.neighbor_tables(n)
    E = LS.energies(t, P)
    edges, _ = LS.library_edges(t, n, kind, k, seed)
    nbr = np.stack([col[e] for e in edges], axis=1)
    got = LS.landscape(E, nbr)
    exp = _brute(E, nbr)
    assert got[0] == exp[0]
    assert got[1] == pytest.approx(exp[1], abs=1e-12)
    assert got[2] == pytest.approx(exp[2], rel=1e-12)


def test_energies_match_the_objective():
    from nimbus.problems.formats.matching_tensor import evaluate_F

    t = load_tensor(FIX / "five.npz")
    P = LS.all_perms(5)
    E = LS.energies(t, P)
    for i in (0, 17, 119):
        assert E[i] == pytest.approx(-evaluate_F(t, P[i].astype(np.int64)))
    assert np.array_equal(LS.perm_rank(P), np.arange(len(P)))
