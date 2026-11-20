# Quickstart

This page assumes NIMBUS is installed ([Installation](installation.md)). Prefer a notebook? The same steps run in
[First solve](../tutorials/01_first_solve.ipynb), [Your own problem](../tutorials/02_your_own_problem.ipynb) writes and
solves a new problem, and [Finding moves](../tutorials/03_finding_moves.ipynb) finds the moves of a constraint the package
does not know; all open in Google Colab.

## First solve

A run is described by one YAML configuration (schema `nimbus.run/1`). The six-spin toy of the paper selects 3 of 6
spins with two triangle rewards; its optimum has energy -7:

```bash
nimbus solve examples/toy_pair_exchanges.yaml --output local_runs/toy
```

The command compiles the problem and its move library into the machine, calibrates the start and end temperatures
from pilot proposals, anneals two replicas for 400 iterations each and writes the run folder. Without `--output`
the folder is `<runs root>/<date>_<name>_<hash>`, where the runs root is `$NIMBUS_RUNS_ROOT` or `local_runs/` in the
repository. A non-empty output folder is not overwritten unless `--set output.overwrite=true` is given.

Change any field from the command line with `--set <section.field>=<value>` (the value is parsed as YAML):

```bash
nimbus solve examples/toy_pair_exchanges.yaml --output local_runs/toy16 \
    --set run.replicas=16 --set run.targets=-7 --set run.stop_on_target=true
```

Other commands:

```bash
nimbus validate examples/matching_small.yaml        # check a configuration without running it
nimbus inspect examples/matching_small.yaml         # compile only: spins N, stored clauses M, moves R
nimbus analyze local_runs/toy local_runs/toy16      # summary of finished runs
```

`--json` (before the command, e.g. `nimbus --json analyze ...`) prints one JSON document instead of text, and
`--quiet` suppresses the progress lines.

## Reading the results

A run folder holds:

| File | Content |
|---|---|
| `results.parquet` | one row per replica (columns below) |
| `manifest.json` | status, configuration hashes, problem metadata, compiled-model size and resources, move library, per-instance calibration and resolved schedule, backend description, timings, host and git commit |
| `config.resolved.yaml` | the configuration as given, after overrides, and normalized with every default filled in |
| `checkpoint.npz` | only while a run is in progress or was interrupted (see [settings](../guides/settings.md#checkpoint-and-resume)) |

Columns of `results.parquet`:

| Column | Meaning |
|---|---|
| `instance_id`, `replica_id`, `seed` | which instance and replica |
| `best_feasible_objective` | lowest energy of a feasible state visited by the replica (empty if none) |
| `best_feasible_native` | the same state's objective in the problem's own units (for example the matching score F, the edge weight) |
| `best_feasible_attempt` | iteration at which that state was first reached (0 = the start state) |
| `best_energy`, `best_attempt` | lowest energy of the run slot and when it was reached |
| `final_energy`, `final_feasible` | the last state |
| `attempts`, `accepted`, `accepted_changed`, `self_loops`, `rejected`, `uphill_accepted` | counters: proposals, accepted proposals, accepted proposals that changed the state, accepted moves that change nothing at this state, rejections, accepted energy increases |
| `steps`, `candidate_evaluations`, `process` | proposal process and its clock (see [settings](../guides/settings.md#proposal-process)) |
| `termination` | `budget` (ran the whole budget), `target` (reached `run.targets` with `run.stop_on_target`), `frozen` (`nfold` process only: no allowed move left) |
| `target_hit`, `target_hit_attempt` | whether and when the target was reached; recorded whenever `run.targets` is set, with or without `run.stop_on_target` (empty without a target) |
| `timings_json`, `error` | per-replica notes |

Energies follow the convention E(s) = E_const - sum_k J_k T_k(s) (see [internals](../concepts/internals.md)); lower is better.

In Python:

```python
import pandas as pd

rows = pd.read_parquet("local_runs/toy16/results.parquet")
p = rows["target_hit"].mean()                          # fraction of replicas that reached energy -7
print(f"success fraction {p:.2f}, median hitting iteration {rows['target_hit_attempt'].median():.0f}")
```

`nimbus analyze` reports, per run, the number of replicas that ended with a feasible state, the best feasible energy
and the iterations to solution at 99%. When the run set `run.targets`, the success fraction is the fraction of replicas
that hit the target (`success_criterion: target`, whether or not the run stopped on it); without a target it is the
fraction of replicas that ended with a feasible state (`success_criterion: feasible`).

**Iterations to solution.** With a per-replica success fraction p at a budget of B iterations,
ITS(B) = B ln(0.01) / ln(1 - p): the expected number of iterations to reach the target with probability 0.99 by
independent repetitions. The paper reports the minimum over a grid of budgets of the median over instances.
[`examples/paper_protocol.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/paper_protocol.py) computes ITS per budget for one instance.

## The Python API

The CLI is a thin wrapper around `nimbus.api`:

```python
from nimbus import api

cfg = api.load_config("examples/toy_pair_exchanges.yaml", {"run.replicas": 4, "output.overwrite": True})
outcome = api.solve(cfg, output_dir="local_runs/toy_api")
print(outcome.status, outcome.run_dir)
print([row.best_feasible_objective for row in outcome.rows])

problem, model, timings = api.compile_problem("examples/matching_small.yaml")
print(model.n_spins, model.n_clauses, model.n_groups)    # N spins, stored clauses, moves (transition neurons)
```

A configuration can also be a Python dictionary (`api.load_config(mapping, base_dir=...)`); relative paths then
resolve against `base_dir`. For finer control (custom schedules, many budgets on one compiled model, your own
statistics), use the compiler and the backends directly; [`examples/paper_protocol.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/paper_protocol.py)
shows the complete sequence.

## Next

* [Settings](../guides/settings.md): every configuration field, with its default.
* [Solving a new problem](../guides/new_problem/index.md): your own objective, constraints and moves.
* [Surface-code decoding](../problems/surface_code.md): the surface-code decoders of the paper.
* [Internals](../concepts/internals.md): energy convention, compiled model, transaction semantics, random streams.
