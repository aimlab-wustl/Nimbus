"""A complete new problem adapter: densest k-subgraph (select exactly k vertices, maximize the edge weight inside).

This file is the worked example of docs/guides/new_problem/adapter.md. It defines the adapter, checks its move library and its
compiled machine on the full feasible set, registers it as the problem kind ``densest_k``, and solves an instance
through the configuration runner.

    python examples/new_problem.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from nimbus.core.errors import ConfigError
from nimbus.core.rng import INIT, stream_integers
from nimbus.problems.base import Problem, SpinPolynomial
from nimbus.transitions import builders as B

PROBLEM_KIND = "densest_k"


class DensestKProblem(Problem):
    """Select exactly ``k`` of ``n`` vertices (s_i = +1: selected) to maximize the total weight of the edges with
    both ends selected. Energy E = -sum_(i,j) w_ij x_i x_j with x = (1 + s) / 2."""

    family = "exact_cardinality"

    def __init__(self, n: int, edges, k: int, name: str = "densest_k"):
        self.n_spins, self.k, self.name = int(n), int(k), name
        self.edges = [(int(i), int(j), float(w)) for i, j, w in edges]
        if not 0 < self.k < self.n_spins:
            raise ValueError("need 0 < k < n")

    # ---- objective: E = const + sum_t c_t prod_{i in support_t} s_i ---------------------------------------------
    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        terms, const = {}, 0.0
        for i, j, w in self.edges:            # -w x_i x_j = -(w / 4) (1 + s_i + s_j + s_i s_j)
            const -= w / 4
            for sup in ((min(i, j),), (max(i, j),), (min(i, j), max(i, j))):
                terms[sup] = terms.get(sup, 0.0) - w / 4
        supports = tuple(sorted(terms))
        return SpinPolynomial(self.n_spins, supports, np.array([terms[s] for s in supports]), const)

    # ---- constraints, independent objective and decoding --------------------------------------------------------
    def is_feasible(self, spins) -> bool:
        return int(np.count_nonzero(np.asarray(spins) == 1)) == self.k

    def native_objective(self, spins, instance: int = 0) -> float:
        sel = np.asarray(spins) == 1
        return float(sum(w for i, j, w in self.edges if sel[i] and sel[j]))

    def energy_to_native(self, energy: float, instance: int = 0) -> float:
        return -float(energy)

    def decode(self, spins) -> dict:
        return {"selected": [int(i) for i in np.flatnonzero(np.asarray(spins) == 1)]}

    # ---- feasible starts from the INIT random stream ------------------------------------------------------------
    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        out = -np.ones((int(n), self.n_spins), dtype=np.int8)
        for m in range(int(n)):               # Fisher-Yates on INIT draws (counter: instance, sample index)
            perm = list(range(self.n_spins))
            for t in range(self.n_spins - 1):
                j = t + int(stream_integers(seed, INIT, 1, self.n_spins - t, c1=t, c2=instance, c3=m)[0])
                perm[t], perm[j] = perm[j], perm[t]
            out[m, perm[: self.k]] = 1
        return out

    def enumerate_feasible(self) -> np.ndarray:
        combos = list(itertools.combinations(range(self.n_spins), self.k))
        out = -np.ones((len(combos), self.n_spins), dtype=np.int8)
        for r, c in enumerate(combos):
            out[r, list(c)] = 1
        return out

    # ---- move library: pair exchanges keep the cardinality --------------------------------------------------------
    def library_names(self) -> list:
        return ["complete", "path"]

    def transition_library(self, name: str = "complete", **options):
        if options:
            raise ValueError(f"unknown library options {sorted(options)}")
        if name not in self.library_names():
            raise ValueError(f"unknown library {name!r}; known: {self.library_names()}")
        return B.pair_exchanges(range(self.n_spins), pattern=name, name=f"densest_k_{name}")


def from_config(cfg: dict, base_dir) -> DensestKProblem:
    """``problem: {kind: densest_k, graph: <edge-list file: lines "i j w">, k: <int>}``."""
    cfg = dict(cfg)
    cfg.pop("kind", None)
    unknown = sorted(set(cfg) - {"graph", "k"})
    if unknown or "graph" not in cfg or "k" not in cfg:
        raise ConfigError(f"densest_k: needs 'graph' and 'k' (unknown fields: {unknown})")
    path = Path(base_dir) / cfg["graph"]
    rows = np.loadtxt(path, ndmin=2)
    n = int(rows[:, :2].max()) + 1
    return DensestKProblem(n, [(int(i), int(j), w) for i, j, w in rows], int(cfg["k"]), name=path.stem)


def random_graph(n: int, p: float, seed: int):
    rng = np.random.default_rng(seed)
    return [(i, j, float(rng.integers(1, 4))) for i, j in itertools.combinations(range(n), 2) if rng.random() < p]


def check(problem: DensestKProblem, library_name: str = "complete"):
    """Verify the move library and the compiled machine on every feasible state."""
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights
    from nimbus.transitions.verify import verify_library

    states = problem.enumerate_feasible()
    library = problem.transition_library(library_name)
    report = verify_library(library, states, is_feasible=problem.is_feasible, require=True)
    print(f"library {library_name}: {report.n_groups} moves on {report.n_states} feasible states, "
          f"feasibility kept: {report.feasibility_preserved}, involutions: {report.involution}, "
          f"connected: {report.connected}")
    model = compile_model(problem.structure(), library, identities=problem.identities())
    binding = bind_weights(model, problem.term_weights(0), problem.const(0))
    T = model.clause_values(states)
    v = model.inputs(T, binding)                        # v_r for every state and move
    E = np.array([problem.energy(s) for s in states])
    assert np.allclose(model.energy(states, binding), E)
    for r in range(model.n_groups):                     # 2 v_r = E(mu_r(s)) - E(s) on every feasible state
        moved = np.array([model.apply_action(r, s) for s in states])
        dE = np.array([problem.energy(s) for s in moved]) - E
        assert np.allclose(2 * v[:, r], dE), f"move {r}: 2 v != dE"
    assert all(abs(problem.native_objective(s) + e) < 1e-9 for s, e in zip(states, E))
    print(f"compiled: {model.n_clauses} stored clauses, {model.n_routes} connections; 2 v_r = dE on all states")


if __name__ == "__main__":
    import tempfile

    from nimbus import api
    from nimbus.problems.factory import register_problem_kind

    edges = random_graph(10, 0.5, seed=0)
    check(DensestKProblem(10, edges, 4), "complete")
    check(DensestKProblem(10, edges, 4), "path")

    register_problem_kind(PROBLEM_KIND, from_config, replace=True)
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "graph10.txt"
        np.savetxt(graph, np.array(edges), fmt="%d %d %g")
        config = {
            "schema": "nimbus.run/1", "name": "densest_k_demo",
            "problem": {"kind": PROBLEM_KIND, "graph": "graph10.txt", "k": 4},
            "transitions": {"library": "complete"},
            "annealing": {"schedule": "fn_endpoints", "params": "auto", "attempts": 2000,
                          "calibration": {"rule": "acceptance", "p_start": 0.8, "p_end": 0.01, "eta": 0.1,
                                          "min_ratio": 2, "min_ratio_mode": "lower_end"}},
            "run": {"seed": 1, "replicas": 8},
            "backend": {"name": "cpu"},
        }
        out = api.solve(api.load_config(config, base_dir=tmp), output_dir=str(Path(tmp) / "run"))
        best = max(r.best_feasible_native for r in out.rows)
        exact = max(DensestKProblem(10, edges, 4).native_objective(s)
                    for s in DensestKProblem(10, edges, 4).enumerate_feasible())
        print(f"status {out.status}: best edge weight found {best:g} (exact optimum {exact:g})")
