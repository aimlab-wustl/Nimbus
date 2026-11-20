"""Batched move libraries of the graph-matching experiments (experiments/graph_matching/batching.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nimbus.problems.formats.matching_tensor import load_tensor
from nimbus.problems.matching import MatchingProblem, backbone_edges, shortcut_library

ROOT = Path(__file__).resolve().parents[1]
GM = ROOT / "experiments" / "graph_matching"
FIX = Path(__file__).parent / "fixtures" / "matching"


@pytest.fixture(scope="module")
def batching():
    sys.path.insert(0, str(GM))
    try:
        import batching as mod
        yield mod
    finally:
        sys.path.remove(str(GM))


@pytest.mark.parametrize("k", [2, 3, 5])
def test_first_batch_is_the_static_library(batching, k):
    t = load_tensor(FIX / "seven.npz")
    prob = MatchingProblem(t)
    batches = batching.batch_libraries(prob, t, k, share=0.5)
    static, info = shortcut_library(prob, {"base": "path", "extra": "informed", "k": k, "share": 0.5}, t)
    lib0, pairs0, copies0 = batches[0]
    assert list(lib0.labels) == list(static.labels)
    assert pairs0 == info["pairs"] and copies0 == info["copies"]


def test_batches_cover_every_shortcut_once(batching):
    t = load_tensor(FIX / "seven.npz")
    prob = MatchingProblem(t)
    n = prob.P
    batches = batching.batch_libraries(prob, t, 4, share=0.5)
    pairs = [p for _, ps, _ in batches for p in ps]
    expected = {(i, j) for i in range(n) for j in range(i + 1, n)} - set(backbone_edges(n))
    assert len(pairs) == len(expected) and set(pairs) == expected
    assert len(batches) == -(-len(expected) // 4)
