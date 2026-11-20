# A problem adapter


Write an adapter when the problem needs its own input format, its own native objective (for example a score in its
own units), or moves that are not built from the families of [a problem file](problem_file.md). The steps below build the complete adapter of
[`examples/new_problem.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/new_problem.py): **densest k-subgraph**, selecting exactly k of n vertices
to maximize the total weight of the edges with both ends selected.

## 1. The objective as a spin polynomial

Subclass `nimbus.problems.base.Problem` and return the energy as a `SpinPolynomial`:
E(s) = const + sum_t c_t prod_{i in S_t} s_i, lower is better. For an edge (i, j) of weight w, the reward
-w x_i x_j with x = (1 + s)/2 expands to -(w/4)(1 + s_i + s_j + s_i s_j). Derive the polynomial independently of
the compiler, so that the checks in step 5 are meaningful.

```python
import itertools

import numpy as np

from nimbus.core.errors import ConfigError
from nimbus.core.rng import INIT, stream_integers
from nimbus.problems.base import Problem, SpinPolynomial
from nimbus.transitions import builders as B


class DensestKProblem(Problem):
    family = "exact_cardinality"

    def __init__(self, n, edges, k, name="densest_k"):
        self.n_spins, self.k, self.name = int(n), int(k), name
        self.edges = [(int(i), int(j), float(w)) for i, j, w in edges]

    def polynomial(self, instance=0):
        terms, const = {}, 0.0
        for i, j, w in self.edges:            # -w x_i x_j = -(w / 4) (1 + s_i + s_j + s_i s_j)
            const -= w / 4
            for sup in ((min(i, j),), (max(i, j),), (min(i, j), max(i, j))):
                terms[sup] = terms.get(sup, 0.0) - w / 4
        supports = tuple(sorted(terms))
        return SpinPolynomial(self.n_spins, supports, np.array([terms[s] for s in supports]), const)
```

Supports are sorted tuples of distinct spin indices; equal supports must be merged. A problem with several weight
sets on the same supports (several instances of one size) sets `n_instances` and uses the `instance` argument.

## 2. Constraints, native objective and decoding

`is_feasible` checks the original constraints directly. `native_objective` evaluates the problem's own score
without the polynomial, and `energy_to_native` maps an energy to it (results report both).

```python
    def is_feasible(self, spins):
        return int(np.count_nonzero(np.asarray(spins) == 1)) == self.k

    def native_objective(self, spins, instance=0):
        sel = np.asarray(spins) == 1
        return float(sum(w for i, j, w in self.edges if sel[i] and sel[j]))

    def energy_to_native(self, energy, instance=0):
        return -float(energy)

    def decode(self, spins):
        return {"selected": [int(i) for i in np.flatnonzero(np.asarray(spins) == 1)]}
```

## 3. Feasible starting states

Every replica starts from a feasible state. Draw them from the `INIT` random stream so that they depend only on the
seed, the instance and the sample index. For small problems, `enumerate_feasible` lists the whole feasible set (used
by the checks in step 5).

```python
    def random_feasible(self, seed, n, instance=0):
        out = -np.ones((int(n), self.n_spins), dtype=np.int8)
        for m in range(int(n)):               # Fisher-Yates on INIT draws
            perm = list(range(self.n_spins))
            for t in range(self.n_spins - 1):
                j = t + int(stream_integers(seed, INIT, 1, self.n_spins - t, c1=t, c2=instance, c3=m)[0])
                perm[t], perm[j] = perm[j], perm[t]
            out[m, perm[: self.k]] = 1
        return out

    def enumerate_feasible(self):
        combos = list(itertools.combinations(range(self.n_spins), self.k))
        out = -np.ones((len(combos), self.n_spins), dtype=np.int8)
        for r, c in enumerate(combos):
            out[r, list(c)] = 1
        return out
```

## 4. The move library

A move is a grouped transition: a map of the spins that sends every feasible state to a feasible state, usually its
own inverse ([Your own moves](own_moves.md) covers moves that are not).
The compiler accepts moves whose effect on every product of spins is either fixed or set by one stored product of
spins (its switch). Build libraries from `nimbus.transitions.builders`:

| Builder | Moves | Keeps |
|---|---|---|
| `single_flips(n, spins)` | flip one spin | nothing (unconstrained spins) |
| `fixed_masks(masks, n)`, `nullspace_basis(H)` | flip a fixed set of spins | parity checks H s = syndrome |
| `pair_exchanges(positions, pattern)`, `simultaneous_exchanges(pairs)` | exchange two spins (or several disjoint pairs at once) | exact cardinality |
| `row_exchanges(P, pattern)`, `block_exchanges(blocks, pattern)`, `column_permutation`, `segment_reversals` | exchange rows of a one-hot matrix, blocks, columns, reverse segments | permutations |
| `row_cycles(P, cycles)` | cyclic exchanges of three rows of a one-hot matrix, each with its reverse | permutations |
| `chain_precedence`, `or_block`, `amo_block`, `contiguity_block` | the block moves of [a problem file](problem_file.md) | those families |
| `random_addition(candidates, count, seed)` | random extra moves from a candidate library | |

`pattern` is `complete` (every pair), `path` (consecutive pairs only: a backbone), `star`, `random_tree` or explicit
`edges`. Libraries combine with `TransitionLibrary.concat`.

```python
    def library_names(self):
        return ["complete", "path"]

    def transition_library(self, name="complete", **options):
        if name not in self.library_names() or options:
            raise ValueError(f"unknown library {name!r} or options {sorted(options)}")
        return B.pair_exchanges(range(self.n_spins), pattern=name, name=f"densest_k_{name}")
```

A library connects the feasible set when every feasible state can reach every other; the backbone (`path`) is the
smallest connected library, and shortcuts (added pairs) shorten the paths between states.

## 5. Verify the library and the compiled machine

On a small instance, check that every move keeps feasibility, is its own inverse and that the library connects the
feasible set (`verify_library`, exhaustive with the enumerated states), and that the compiled machine's input of every
move is half the energy change of that move, 2 v_r = E(mu_r(s)) - E(s), on every feasible state:

```python
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.transitions.verify import verify_library

rng = np.random.default_rng(0)
edges = [(i, j, float(rng.integers(1, 4))) for i, j in itertools.combinations(range(10), 2) if rng.random() < 0.5]
problem = DensestKProblem(10, edges, 4)
states = problem.enumerate_feasible()

library = problem.transition_library("path")
report = verify_library(library, states, is_feasible=problem.is_feasible, require=True)
print(report.summary()["ok"], report.connected)

model = compile_model(problem.structure(), library, identities=problem.identities())
binding = bind_weights(model, problem.term_weights(0), problem.const(0))
v = model.inputs(model.clause_values(states), binding)          # [states, moves]
E = np.array([problem.energy(s) for s in states])
for r in range(model.n_groups):
    dE = np.array([problem.energy(model.apply_action(r, s)) for s in states]) - E
    assert np.allclose(2 * v[:, r], dE)
print(f"{model.n_groups} moves, {model.n_clauses} stored clauses, {model.n_routes} connections: 2 v = dE")
```

For larger instances, `verify_library(library, sample_states=problem.random_feasible(...), is_feasible=...)` checks
sampled states (not a proof). Run a few replicas on the `oracle` backend (an independent recomputation) and on the
`emulation` or `cpu` backend and compare their results. If the compiler rejects a move, its effect on some product of
spins depends on the state in a way no single stored product can express; split the move or choose another
([Your own moves](own_moves.md) states the rule).

If the feasible set satisfies parity identities prod_{i in S} s_i = +1 or -1 (for example the product of all spins
under exact cardinality), return them from `identities()`: the compiler then reuses stored clauses as switches.

## 6. Register the kind and run it

`from_config(cfg, base_dir)` builds the problem from the `problem` section of a configuration (validate every field;
resolve relative paths against `base_dir`). Register it under a kind name, at run time or permanently in
`nimbus.problems.factory.BUILTIN_KINDS`:

```python
import tempfile
from pathlib import Path

from nimbus import api
from nimbus.problems.factory import register_problem_kind


def from_config(cfg, base_dir):
    cfg = dict(cfg)
    cfg.pop("kind", None)
    if set(cfg) != {"graph", "k"}:
        raise ConfigError("densest_k: needs exactly the fields 'graph' and 'k'")
    rows = np.loadtxt(Path(base_dir) / cfg["graph"], ndmin=2)
    n = int(rows[:, :2].max()) + 1
    return DensestKProblem(n, [(int(i), int(j), w) for i, j, w in rows], int(cfg["k"]))


register_problem_kind("densest_k", from_config, replace=True)

tmp = Path(tempfile.mkdtemp())
np.savetxt(tmp / "graph10.txt", np.array(edges), fmt="%d %d %g")
config = {
    "schema": "nimbus.run/1", "name": "densest_k_demo",
    "problem": {"kind": "densest_k", "graph": "graph10.txt", "k": 4},
    "transitions": {"library": "complete"},
    "annealing": {"schedule": "fn_endpoints", "params": "auto", "attempts": 2000,
                  "calibration": {"rule": "acceptance", "p_start": 0.8, "p_end": 0.01, "eta": 0.1,
                                  "min_ratio": 2, "min_ratio_mode": "lower_end"}},
    "run": {"seed": 1, "replicas": 8},
    "backend": {"name": "cpu"},
}
out = api.solve(api.load_config(config, base_dir=tmp), output_dir=str(tmp / "run"))
exact = max(problem.native_objective(s) for s in states)
print(out.status, max(r.best_feasible_native for r in out.rows), "exact optimum", exact)
```

A module that defines `PROBLEM_KIND` and `from_config` can also be registered by name,
`register_problem_kind("densest_k", "mypackage.densest_k")`, which imports it on first use.

## 7. Optional parts

* `penalty_formulation(instance, **strengths)`: the objective plus penalty terms, for the penalty baselines
  (`baseline.kind: penalty`); see `ToyCardinalityProblem` and `MatchingProblem`.
* `readout_default()`, `metadata()`: extra information for the manifest.
* A problem with several weight sets (`n_instances > 1`) runs all of them in one configuration (`run.instances`).

The built-in adapters are complete references: `nimbus.problems.toy_cardinality` (smallest), `matching`
(permutations, a cubic objective, shortcut libraries) and `surface_code` (parity checks, fixed-mask moves).
