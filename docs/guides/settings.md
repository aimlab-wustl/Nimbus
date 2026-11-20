# Settings

Every field of a run configuration (schema `nimbus.run/1`), with its meaning and default, plus the Python-level
settings used by the paper experiments that the configuration does not cover. The configuration is validated
strictly: an unknown field, a wrong type or an inconsistent combination stops the run with an error naming the
field. `nimbus validate <file>` checks a file without running it, and `config.resolved.yaml` in a run folder shows
every default that was applied.

```yaml
schema: nimbus.run/1          # required
name: my_run                  # optional label (default: the file name)
problem: {kind: ..., ...}     # required: what to solve
transitions: {library: ...}   # required: the move library
annealing: {...}              # required: schedule, temperatures, budget, proposal process
run: {...}                    # seed, replicas, instances, targets, checkpoints
backend: {...}                # which implementation runs the machine
compile: {...}                # compiler options and limits
recording: summary            # optional observations
output: {...}                 # run folder
baseline: {kind: none}        # penalty baselines
```

Contents: [problem](#problem) · [move libraries](#move-libraries) · [annealing schedule](#annealing-schedule) ·
[temperature calibration](#temperature-calibration) · [budget](#budget-and-iterations-to-solution) ·
[replicas and instances](#replicas-instances-and-targets) · [proposal process](#proposal-process) ·
[capacity batching](#capacity-batching) · [backends](#backends) · [event counters](#event-counters) ·
[penalty baselines](#penalty-baselines) · [compile](#compile) · [recording](#recording) ·
[output files](#output-files) · [checkpoint and resume](#checkpoint-and-resume) · [paths](#paths)

## Problem

`problem.kind` selects the adapter; the other fields belong to the adapter. Relative paths resolve against the
configuration file; a path `data:<relative>` resolves under the data root.

| Kind | Fields (default) | Constraint family |
|---|---|---|
| `toy_cardinality` | none | exact cardinality (the six-spin toy of the paper) |
| `matching` | `tensor`: path to a matching tensor `.npz` (required) | permutation: one-hot rows and columns |
| `surface_code` | `d` (required), `p` (required), `noise`: `independent` (X errors, one check type) or `depolarizing` (`independent`), `sector`: `X` or `Z` (`X`), `syndrome`: 0/1 list (all zero), `library` (`stabilizers_logical`) | parity checks (an affine coset of the code) |
| `spinpoly` | `path`: a `nimbus.spinpoly/1` file (required), `name` | any objective with disjoint blocks of exact cardinality, precedence, covering, packing and contiguity; free spins |
| `opb` | `path`: an OPB file (required), `name`, `contiguity`: list of `{spins: [1-based variable numbers], cyclic: bool}` | the same blocks, written in the OPB pseudo-Boolean format (contiguity only through the `contiguity` field) |

Problems you add yourself are registered with `nimbus.problems.factory.register_problem_kind`
([A problem adapter](new_problem/adapter.md)).

## Move libraries

`transitions.library` names the move library (the grouped transitions, one transition neuron each);
`transitions.options` (default `{}`) passes keyword options to the problem's `transition_library`.

| Kind | Libraries | Options |
|---|---|---|
| `toy_cardinality` | `pair_exchanges` (all 15 exchanges), `augmented` (plus the simultaneous double exchange (1,4)+(2,5)) | none |
| `matching` | `complete` (all L(L-1)/2 row exchanges: the backbone with all shortcuts), `path` (the backbone: the L - 1 exchanges of consecutive rows) | `double_exchanges: {count, seed}`: add composite moves that exchange two disjoint row pairs at once |
| `surface_code` | `stabilizers` (cannot change the logical class), `stabilizers_logical` (plus one logical operator per type: the backbone), `stabilizers_logical_all` (plus all d straight-line logical operators per type: the backbone with shortcuts) | none |
| `spinpoly`, `opb` | `default`: per block, the moves of its family (exact cardinality: pair exchanges; precedence: grow or shrink the selected prefix; covering: flip a literal while a neighboring literal is satisfied; packing: flip a spin while the rest of the block is unselected; contiguity: grow or shrink the interval at an end); single flips for free spins | `cardinality_pattern`: `complete` or `path` (`complete`), `or_tree`: `path`, `star` or `random_tree` (`path`) |

**Shortcuts and backbone share (graph matching).** The libraries of Fig. 4 (the path backbone plus K informed or
random shortcuts, optionally with a fixed backbone proposal share) are built with
`nimbus.problems.matching.shortcut_library(problem, spec, tensor)`:

| `spec` | Library |
|---|---|
| `{"base": "complete"}` | complete library |
| `{"base": "path"}` | path library (backbone) |
| `{"base": "path", "extra": "informed", "k": K}` | backbone plus the K row pairs whose candidate-target distributions overlap most, computed from the tensor alone (`informed_pairs`) |
| `{"base": "path", "extra": "random", "k": K, "seed": s}` | backbone plus K uniformly random row pairs (`random_pairs`) |
| add `"share": f` | proposals stay uniform over the move list, and the backbone is repeated c = max(1, round(f / (1 - f) K / (L - 1))) times so that it is proposed with probability close to f |

[`examples/paper_protocol.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/paper_protocol.py) runs any of them (`--library informed --k 4
--share 0.5`). Libraries of your own are built from `nimbus.transitions.builders` (single flips, fixed masks and
null-space bases, pair, row and block exchanges, simultaneous exchanges, segment reversals, column permutations,
`random_addition`) and combined with `TransitionLibrary.concat`; check them with
`nimbus.transitions.verify.verify_library` ([A problem adapter](new_problem/adapter.md),
[Your own moves](new_problem/own_moves.md)). `nimbus.transitions.grouping.group_transitions` finds moves from the
enumerated feasible states of a small problem.

## Annealing schedule

```yaml
annealing:
  schedule: fn_endpoints          # required
  params: {tau_start: 2.0, tau_end: 0.05, eta: 0.1}   # explicit temperatures, or: params: auto
  attempts: 100000                # budget B per replica
```

The temperature tau_n of iteration n = 0 .. B - 1 is evaluated in closed form, with frac = n / (B - 1):

| `schedule` | `params` (explicit) | tau_n |
|---|---|---|
| `fn_endpoints` | `tau_start`, `tau_end`, `eta` | Fowler-Nordheim (inverse-log) schedule through both endpoints: tau_n = tau0 / ln(1 + q_n), tau0 = tau_end ln(1 + eta), q_n = q_start + (eta - q_start) frac, q_start = exp(tau0 / tau_start) - 1. `eta` is the dimensionless shape parameter (0.1 in the paper); a smaller eta stays hot longer |
| `fn_legacy` | `tau0`, `c_fn`, `t_start`, `dt` | tau_n = tau0 / ln(1 + (t_start + n dt) / c_fn), the parameterization of earlier neuromorphic annealers |
| `geometric` | `tau_start`, `tau_end` | exponential interpolation between the endpoints |
| `constant` | `tau` | fixed temperature |
| `zero` | none | tau = 0: greedy descent (accept only moves that do not raise the energy) |

`fn_endpoints` is the schedule of every result in the paper. Acceptance is the Metropolis rule: a move with energy
change dE = 2 v_r is accepted iff v_r <= -(tau_n / 2) ln u_n with u_n uniform in (0, 1), that is with probability
min(1, exp(-dE / tau_n)); at tau = 0 iff v_r <= 0.

## Temperature calibration

With `params: auto`, the start and end temperatures are calibrated per instance from pilot proposals of the
production move library, drawn from feasible random states on their own random stream:

```yaml
annealing:
  schedule: fn_endpoints
  params: auto
  calibration: {rule: acceptance, p_start: 0.8, p_end: 0.01, eta: 0.1, min_ratio: 2, min_ratio_mode: lower_end}
  attempts: 100000
```

This is the automatically calibrated schedule of the paper (the pilot is part of the run and its iterations are
reported in the manifest). The fields of `annealing.calibration` (only allowed with `params: auto`):

| Field | Default | Meaning |
|---|---|---|
| `rule` | `acceptance` | `acceptance`: tau is the temperature at which the mean acceptance probability (1/L) sum_j exp(-dE_j / tau) of the L uphill pilot proposals equals the target. `quantile`: tau = Q_q(dE+) / ln(1/p) (the q-quantile uphill change is accepted with probability p). `legacy_max`: start amplitude slightly above the largest observed change (`legacy_factor` times the maximum) |
| `p_start` | 0.8 | target acceptance at the start |
| `p_end` | 0.01 | target acceptance at the end |
| `eta` | required for `fn_endpoints` and `fn_legacy` | FN shape parameter (`t_start`, default 1, also for `fn_legacy`; `constant_at: start` or `end` for an automatic `constant` schedule) |
| `min_ratio` | none | kappa >= 1: the end temperature is at least a factor kappa below the start temperature |
| `min_ratio_mode` | `raise_start` | `lower_end`: keep tau_start and set tau_end = min(tau_end, tau_start / kappa) (the paper's rule). `raise_start`: tau_start = max(tau_start, kappa tau_end) |
| `n_pilot` | 2048 | pilot proposals in total |
| `late_fraction` | 0.5 | share of the pilot used for the end temperature |
| `late_advance` | `descent` | `descent`: the end-phase pilot states are first advanced by a short FN descent from tau_start to 0.02 tau_start, so the end temperature is measured where the anneal ends. `none`: end phase at the random start states (on graph matching this gives far too cold end temperatures) |
| `late_advance_attempts` | `auto` (8192) | length of that descent |
| `budget_cap_fraction` | 0.05 | the pilot is capped at this fraction of the run budget (all replicas) |
| `min_pilot`, `min_positive`, `max_rounds` | 64, 8, 2 | smallest pilot worth running; uphill changes needed per phase; sampling rounds per phase |
| `quantile`, `legacy_factor` | 0.5, 1.1 | parameters of the `quantile` and `legacy_max` rules |
| `fallback` | `coefficients` | if a pilot finds no uphill change: use temperatures from the coefficient scale (with a warning), or `error` |
| `coefficient_scale`, `class_weights` | none | explicit fallback scale (number or `[start, end]`); production proposal probability per move class |

The calibration result (temperatures, pilot statistics, fallbacks) is stored per instance in `manifest.json`.

**Start factor m and minimum ratio kappa.** The paper's schedules use tau_start = m tau_start_cal and
tau_end = min(tau_end_cal, tau_start / kappa), from one raw calibration. The automatically calibrated schedule is
m = 1, kappa = 2 (the configuration above). For other values of m, and to reuse one calibration for a grid of
settings, use `nimbus.annealing.protocol`:

```python
from nimbus import api
from nimbus.annealing.protocol import calibrate_temperatures, scaled_temperatures
from nimbus.core.model import bind_weights

problem, model, _ = api.compile_problem("examples/matching_small.yaml")   # compiled machine, no weights yet
binding = bind_weights(model, problem.term_weights(0), problem.const(0))  # weights of instance 0

ts_cal, te_cal = calibrate_temperatures(model, binding, problem, seed=11, raw=True)
tau_start, tau_end = scaled_temperatures(ts_cal, te_cal, m=2.0, kappa=4.0)
print(f"tau_start = {tau_start:.4g}, tau_end = {tau_end:.4g}")
```

`calibrate_temperatures` uses the paper's constants (`P_START = 0.8`, `P_END = 0.01`, `ETA = 0.1`, 32 pilot states,
late descent of 8192 iterations to 0.02 tau_start); `schedule_temperatures(model, binding, problem, seed, m=1.0,
kappa=2.0)` does both steps. The configuration runner draws its pilot states with its own seeds, so its temperatures
are statistically equivalent to, not bit-identical with, these functions.

## Budget and iterations to solution

| Field | Default | Meaning |
|---|---|---|
| `annealing.attempts` | required | budget B: iterations (proposed moves) per replica; the schedule runs from tau_start to tau_end over exactly B iterations |

A replica succeeds at budget B if it reaches the target within its B iterations. With success fraction p over the
replicas, the iterations to solution are ITS(B) = B ln(0.01) / ln(1 - p). Because the schedule depends on B, each
budget is a separate run; the paper reports the minimum over a grid of budgets (powers of two) of the median ITS over
instances. Penalty machines count one single-spin update test as one iteration.

## Replicas, instances and targets

| Field | Default | Meaning |
|---|---|---|
| `run.seed` | 0 | root seed (0 .. 2^32 - 1) of all random streams |
| `run.replicas` | 1 | independent replicas per instance, each from its own random feasible state (independent anneals; not replica exchange) |
| `run.replica_offset` | 0 | first replica id: runs with offsets 0, 64, 128, ... continue one ensemble across jobs |
| `run.instances` | `all` | for problems with several weight sets: `all`, a count `n` (the first n), `{start, stop}` or a list of ids |
| `run.targets` | none | target energy: a number, a list (one per instance) or `{energy: value}`; a replica reaches it when it is feasible with energy <= target + 1e-9 scale |
| `run.target_tol` | 0 | extra tolerance on the target |
| `run.stop_on_target` | false | stop a replica when it reaches the target; its `termination` is `target` and `target_hit_attempt` the iteration |
| `run.chunk_attempts` | `auto` (min(B, 4096)) | iterations between checkpoints |

Every replica uses its own counter-based random stream (seed, replica id, instance id), so a replica's trajectory does
not depend on how many other replicas run, on the thread count or on where the run was cut into chunks.

**Replica exchange (parallel tempering).** Replicas at a ladder of temperatures that exchange states are used by the
surface-code decoders (`nimbus.decoding`, [surface_code.md](../problems/surface_code.md)), not by the configuration runner.

## Proposal process

`annealing.process` (default `select`) chooses how a replica picks the one move it commits per step. The compiled
machine is the same for all of them.

| Process | Rule | Clock of the budget | Options (`annealing.process_options`) |
|---|---|---|---|
| `select` | draw one move uniformly, Metropolis test; the reference process of the paper | iterations | none |
| `shared` | all replicas of a cohort use the same move sequence, each with its own acceptance uniform; per-replica law identical to `select`, replicas correlated | iterations | `cohort_size` (default: all replicas of an instance) |
| `sweep` | the moves in a random order per sweep, each tested once | iterations | `cohort_size` (default: one order per replica) |
| `colored` | moves colored so that no two moves of a color share a stored clause or switch; the moves of one color are tested together and the accepted ones commit together; colors in a fixed cycle | iterations (tested moves) | none |
| `parallel_trial` | every state-changing move tested with its own uniform; one of the accepted moves commits, chosen uniformly | steps | `pt_exclude_self_loops` (true) |
| `nfold` | rejection-free: choose a move with probability proportional to its acceptance probability and advance the clock by the number of `select` iterations it replaces | iterations (`nfold_clock: attempts`) or steps | `nfold_clock` (`attempts`) |

`parallel_trial`, `nfold` and `colored` need every move loaded at every step, so they cannot be combined with capacity
batching. Results record `steps` (parallel rounds) and `candidate_evaluations` (acceptance tests) next to `attempts`.

## Capacity batching

`annealing.batching: {capacity: C}` models a machine with C transition neurons when the library has R > C moves:
the move list is cut into contiguous batches of at most C moves, and the batch holding the proposed move is loaded
when it is needed. The proposal law (each move with probability 1/R) and the random draws are those of the fully
loaded machine, so the trajectory is unchanged; only the number of reloads differs. The emulation also accepts
`backend.options: {capacity: C}` or an explicit `batches` list.

## Backends

| `backend.name` | What it is | Use it for |
|---|---|---|
| `cpu` | Numba implementation of the machine, parallel over replicas | production runs; every result of the paper |
| `emulation` | event-driven reference implementation in Python: persistent clause signs, switches and maintained inputs, updated by events | checking the architecture semantics, the `events` and `trace` recording profiles, capacity batching; small problems |
| `oracle` | independent recomputation from a private spin array and the move descriptors | validating a new problem or library (`select` process only, no batching) |
| `gpu` | CUDA implementation (optional `gpu` extra, NVIDIA GPU); trajectories identical to `cpu` | large replica counts on a GPU |
| `auto` (default) | `cpu` if available, otherwise `emulation` | |

All backends execute the same transaction with the same random numbers, so `emulation`, `cpu` and `gpu` give the
same states and counters for the same seed (the oracle matches within floating-point rounding of the energies).
An explicit request for an unavailable backend is an error, never a silent substitution.

Further `backend` fields:

| Field | Default | Meaning |
|---|---|---|
| `strategy` | `auto` | `cpu`: how the input v_r of the proposed move is obtained. `recompute`: from its connections at every iteration (cost: one read per weighted connection of the proposed move). `incremental`: every input maintained, updated after each accepted move (event-driven updating). `lazy`: inputs touched by an accepted move are marked and recomputed when next read. `adaptive`: switches between `recompute` and `incremental` from the measured acceptance. `auto` chooses from the compiled structure. The emulation is always event-driven, the oracle always direct |
| `layout` | `auto` | `cpu`: `flat` or `gate_blocked` (connections grouped by their controlling switch, so a closed switch skips its whole block); `auto` uses `gate_blocked` when moves average more than 8 weighted connections |
| `threads` | all cores | CPU threads; results do not depend on it |
| `precision` | `float64` | every backend computes in float64; `float32` is rejected with an error |
| `device` | none | GPU device index |
| `options` | `{}` | backend-specific options, for example `scheduling` (`static`, `queue`, `auto`) and `event_counters` on `cpu`; `capacity`, `batches` and `validate` on the emulation |

## Event counters

The operation counts of Fig. 6 (event-driven updating versus recomputation) come from exact counters of the
architecture events, available on the CPU backend (`select` process) through the Python API:

```python
import numpy as np
from nimbus.annealing.schedule import fn_endpoints
from nimbus.backends import get_backend
from nimbus.backends._numba_kernels import log_windows

B, replicas = 2**14, 8
be = get_backend("cpu")
plan = be.prepare(model, strategy="recompute", event_counters=True, event_windows=log_windows(B, 8, 256))
state = be.init_state(plan, binding, problem.random_feasible(11, replicas, 0), replica_id=np.arange(replicas))
res = be.run(plan, state, fn_endpoints(tau_start, tau_end, 0.1, B), seed=11)
totals = res.state.counters                 # run totals per replica, including ev_<name>
windows = be.window_counters(state)         # per-window counts: [replicas, windows] arrays, plus "edges"
updates = totals["ev_input_corrections"].sum()           # event-driven updating
reads = windows["fanin_weighted"].sum()                   # recomputation of the proposed move's input
print(f"operations, event-driven / recomputation: {updates / reads:.3f}")
```

Totals (`ev_` prefix in `counters`): `clause_toggles` (stored clauses toggled by accepted moves),
`toggles_objective`, `toggles_gate`, `clause_route_events`, `clause_route_corrections`, `switch_reevaluations`,
`switch_changes`, `switch_route_events` and `input_corrections` (maintained-input updates: the cost of event-driven
updating). Per window (`event_windows`: strictly increasing iteration edges; `log_windows(B, per_doubling,
min_width)` or `linear_windows(B, k)`): `attempts`, `accepted`, `accepted_changed`, `neutral_accepted`,
`uphill_accepted`, `fanin_weighted` and `fanin_routes` (weighted and all connections of the proposed move: the cost
of recomputation), the event counts above, `feasible_attempts` and `accepted_changed_feasible`, plus the best energy
after each window. The counters do not depend on the strategy, so one run gives the cost of both updating schemes.
The `events` recording profile gives the same totals on the emulation. See `experiments/event_driven/` for the
complete study.

## Penalty baselines

Constraint-free machines that add a penalty term to the objective and use single spin flips.

**In a configuration** (`baseline.kind`, default `none`):

| Kind | What runs | `baseline.options` |
|---|---|---|
| `penalty` | objective plus penalty terms on the same compiled-machine infrastructure, single spin flips; needs explicit `annealing.params` | the problem's penalty strengths: `lam` (toy, spinpoly/OPB; default 6), `lambda_r`, `lambda_c` (matching; default 3 sum_k abs(J_k)) |
| `quadratic` | the same, after an exact reduction to second order with auxiliary spins (Rosenberg) | the same, plus `quad_strength` (reduction penalty M) |

Results report the best objective over feasible states visited (`best_feasible_objective`); see
`examples/toy_penalty.yaml`.

**The penalty machines of the graph-matching comparison** (N-HOIM style and NeuroSA style) are in
`nimbus.baselines.penalty_machines`:

```python
from nimbus.baselines.penalty_machines import PenaltyMachine

machine = PenaltyMachine(problem, kind="nhoim", lam=6.0)          # or kind="neurosa", mult=1.0
out = machine.run(budget=2**16, T_start=3.0, T_end=0.3, replicas=16, seed=11)
print(out["bestF"])          # best score F in a feasible state per replica (NaN if never feasible)
```

`nhoim`: the third-order terms are kept; every tick tests all spins against fresh noise thresholds and flips one of
those that pass, chosen uniformly. `neurosa`: the penalized polynomial is first reduced to second order, then one
spin per iteration is tested. One iteration is one single-spin update test; the FN-shaped schedule runs from
`T_start` to `T_end`. Penalty strengths and temperatures are in the units of the objective.

## Compile

| Field | Default | Meaning |
|---|---|---|
| `compile.limits` | none | `max_clauses`, `max_routes`, `max_memory_bytes`, `max_closure_rounds`, `time_limit_s`: stop with a resource report instead of compiling a machine that is too large |
| `compile.options` | `{}` | `use_identities` (use the parity identities that hold on the feasible set, supplied by the problem, to reuse stored clauses as switches), `gate_representative` (`best` or `requested`), `allow_non_involutive` (compile a move whose reverse is not in the library; by default such a move is refused, because its transitions could not be proposed in both directions), `table_max_states`, `run_preflight` |
| `compile.readout` | `default` | every spin is always stored as a singleton clause (its readout); `default` and `all` are equivalent |

`nimbus inspect <config>` compiles without running and prints the machine size.

## Recording

`recording` is a profile name or a mapping `{profile: ..., ...}`:

| Profile | Adds |
|---|---|
| `none` | nothing persisted beyond the result rows |
| `summary` (default) | manifest and one row per replica |
| `anytime` | best and current energy at log-spaced iterations, and the best-energy improvements |
| `events` | architecture event counters (emulation) and per-transition histograms |
| `trace` | per-iteration records of selected replicas, with a size cap |

Profiles beyond `summary` cost time; with `summary` the CPU backend runs its fast path.

## Output files

| Field | Default | Meaning |
|---|---|---|
| `output.path` | `<runs root>/<date>_<name>_<hash8>` | run folder (relative: under the runs root; the CLI `--output` is relative to the working directory) |
| `output.overwrite` | false | allow writing into a non-empty folder |
| `output.persist` | true | write the run folder |
| `output.keep_checkpoint` | false | keep `checkpoint.npz` after the run completes |

Files: `results.parquet`, `manifest.json`, `config.resolved.yaml` ([getting_started.md](../getting-started/quickstart.md#reading-the-results)).

## Checkpoint and resume

During a run, `checkpoint.npz` in the run folder is overwritten after every chunk (`run.chunk_attempts` iterations)
with the full replica state of the instance in progress, the finished instances and the hashes of the configuration,
compiled model and weights. An interrupted run continues with

```bash
nimbus resume local_runs/<run folder>
```

and produces exactly the trajectories of an uninterrupted run. A checkpoint made for a different configuration,
topology or weights is refused. The checkpoint is removed when the run completes (unless `output.keep_checkpoint`).

## Paths

| Environment variable | Default | Meaning |
|---|---|---|
| `NIMBUS_RUNS_ROOT` | `local_runs/` in the repository (or the working directory outside a checkout) | run outputs |
| `NIMBUS_DATA_ROOT` | `data/` in the repository | downloaded benchmark instances (`nimbus benchmarks fetch`) and `data:` paths |
| `NIMBUS_REPO_ROOT` | found from the package location | repository root |

Command-line overrides: `--set path.to.field=value` for any field; the short forms `seed`, `attempts`,
`replicas`, `threads`, `backend`, `record`, `output`, `device`, `strategy`, `instances` and
`process` are also accepted as `--set seed=3`.
