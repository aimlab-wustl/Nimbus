# Surface-code decoding

Decoding is a constrained problem: the error bits must reproduce the measured syndrome (parity-check constraints),
and the objective is the negative log-probability of the error. The moves are stabilizer operators (they keep the
syndrome) and logical operators (they change the logical class), so every state the machine visits is a valid
correction. Try it in the browser in the [live demo](../demos/surface_code.md). `nimbus` offers two ways to run it:

* **The generic machine** (`problem.kind: surface_code`): one syndrome, the ordinary configuration runner, any
  backend. See `examples/surface_code_d5.yaml` and [settings.md](../guides/settings.md#problem). Use this to study the
  formulation or to compare with other problems.
* **The decoders of the paper** (`nimbus.decoding`): bit-exact software models of the two FPGA decoders, which run
  replica exchange (parallel tempering) over many shots at once. They make exactly the decisions of the hardware on
  the same shots, so the logical error rates they give are those of the board.

| Module | Content |
|---|---|
| `nimbus.decoding.code_capacity` | shots of code-capacity noise, logical classes of a correction |
| `nimbus.decoding.one_shot` | one syndrome per shot: replica exchange, maximum-likelihood class from the occupancy of the beta = 1 replica |
| `nimbus.decoding.phenomenological` | the memory experiment with measurement errors and its shots |
| `nimbus.decoding.real_time` | real-time decoding of a syndrome stream in sliding windows |
| `nimbus.decoding.windows` | the sliding-window commit and syndrome forwarding, for any window decoder |
| `nimbus.decoding.baselines` | CPU reference decoders: PyMatching, correlated PyMatching, BP+matching, BP-OSD, union-find, tensor-network ML (`pip install -e ".[refs]"`) |

## Noise models

* **Code capacity** (one syndrome per shot): depolarizing noise, X, Y and Z each with probability p/3 on every data
  qubit of the rotated d x d code, one syndrome measured without error. Error bits are (x | z), 2 d^2 long. The
  generic machine also has `noise: independent` (X errors with probability p, one check type).
* **Phenomenological** (real time): the memory experiment of N = 4d noisy rounds (data errors as above before every
  round, measurement flips with probability p) followed by one round without error. The decoder sees the detectors
  (changes of the syndrome between rounds) and must predict the logical observables.

Shots are drawn from seeded generators (`sample_shots(problem, shots, seed)`, `memory_shots(d, p, shots, chunk,
seed)`), so every decoder can be run on the same shots by regenerating them.

## One syndrome per shot

`build_spec` compiles the move library (stabilizers and all d straight-line logical operators per type) and the
hardware tables for a code distance, an error rate and a machine configuration:

| `PTConfig` field | Default | Meaning |
|---|---|---|
| `NS` | 16 | replicas, on a geometric ladder of inverse temperatures from 1 to `bmin` |
| `bmin` | 0.1 | smallest inverse temperature |
| `K` | 1 | an exchange round after every K sweeps (the paper's designs use K = 2; a K above the sweep count disables replica exchange) |
| `B` | 16 | random bits per acceptance test |
| `USW`, `LF`, `TF` | 12, 16, 12 | random bits per exchange decision; fractional bits of the exchange tables |

A sweep proposes every move once: the moves are colored so that no two moves of a color share a qubit, and each color
is tested and committed in parallel. The paper's designs use `NS` = 16, 16, 64, 56, 40, 32, 24 replicas at d = 3, 5,
..., 15 (`one_shot.FPGA_REPLICAS`) and the sweep budgets of `one_shot.FPGA_SWEEPS`.

```python
import numpy as np
from nimbus.decoding.code_capacity import class_failures, sample_shots
from nimbus.decoding.one_shot import (PTConfig, build_spec, c0_from_syndrome, min_energy_class_by_budget,
                                      ml_class_by_budget, run_fixed_budget, run_pt)

d, p, shots = 5, 0.12, 1000
spec = build_spec(d, p, PTConfig(NS=16, bmin=0.1, K=2, B=16))
prob = spec.prob                                   # the SurfaceCodeProblem (depolarizing noise)
e, syn = sample_shots(prob, shots, seed=7)         # error bits and syndromes
c0 = c0_from_syndrome(spec, syn)                   # start state of every shot: a fixed chain per fired check
res = run_pt(spec, c0, 7, seed=1)                  # 2^7 sweeps per shot, all smaller budgets recorded
fail = class_failures(prob, c0, e)                 # [shots, 4]: would logical class k be a failure?

budgets = [16, 32, 64, 128]
rows = np.arange(shots)
for g, ml, me in zip(budgets, ml_class_by_budget(res["cs"], budgets), min_energy_class_by_budget(res["bs"], budgets)):
    print(f"{g:4d} sweeps: LER {fail[rows, ml].mean():.4f} (most visited class), {fail[rows, me].mean():.4f} (lowest energy)")
```

The decision at a budget of g sweeps is the logical class most visited by the beta = 1 replica over sweeps
[g/2, g) (maximum-likelihood decoding by sampling); the lowest-energy class is the minimum-weight decision. One run of
2^k sweeps gives every budget up to 2^k. `run_fixed_budget` runs any sweep count as the board does, optionally with
the board's early stop (at every power-of-two sweep count g >= `gmin`, stop when the leading class holds at least
`th32`/32 of the counts of the current block):

```python
ml, stops = run_fixed_budget(spec, c0, 128, seed=1, gmin=16, th32=24)
print(f"early stop: LER {fail[rows, ml].mean():.4f}, mean sweeps per shot {stops.mean():.1f}")
```

`run_shot` decodes one shot with a record of every move cycle (the reference trace for the hardware simulation).

## Real time

Rounds arrive one at a time. Windows of 2d rounds are decoded; a non-final window commits its first d rounds and
passes the corrected syndrome to the next window, and the final window commits all remaining rounds.
`fpga_config(d, R, bmin, K)` gives the configuration of the FPGA design for a distance (its layout of the moves in
hardware slabs, generators carried across the windows of a shot); `PTWConfig` holds the same fields as `PTConfig`
plus the fixed-point widths and the layout.

```python
from nimbus.decoding.phenomenological import memory_shots
from nimbus.decoding.real_time import decode_stream, failures, fpga_config

d, p = 5, 0.03
N = 4 * d                                               # noisy rounds
Dx, Dz, ox, oz = memory_shots(d, p, 100, chunk=0)       # detectors and true observables of 100 shots
res, info = decode_stream(d, p, Dx, Dz, N, d, d, 2**8, fpga_config(d, R=16), seed=1)   # C = B = d, 2^8 sweeps
f = failures(res, ox, oz)
print(f"real time: LER {f['ml'].mean():.3f} per shot of {N} rounds ({len(info['sched'])} windows)")
```

The arguments after the detectors are the number of rounds N, the committed rounds per window C, the buffer rounds
B (window = C + B rounds) and the sweeps per window (the paper: 2^12, with 16 replicas at d = 5, 7 and 24 at
d = 9, 11). Non-final windows commit their lowest-energy state; the final window gives both the most visited class
(`"ml"`) and the lowest-energy state (`"me"`). `decode_batch` reproduces the board's single generator stream over a
batch of shots.

## CPU baselines

One syndrome per shot (corrections are error bits of the problem):

```python
from nimbus.decoding import baselines as BL

for name, corr in (("PyMatching", BL.matching_correction(prob, syn)), ("BP-OSD", BL.bposd_correction(prob, syn)),
                   ("union-find", BL.union_find_correction(prob, syn))):
    print(f"{name}: LER {prob.logical_failure(corr, e).mean():.4f}")
```

`tensor_network_failures(prob, e, chi=16)` gives approximate maximum-likelihood decisions (qecsim matrix product
state decoder). In real time the baselines run in the same sliding windows as the hardware decoder:

```python
from nimbus.decoding.baselines import WINDOW_DECODERS
from nimbus.decoding.windows import correction_failures, decode_windows, schedule

corr = decode_windows(d, N, schedule(N, d, d), Dx, Dz, WINDOW_DECODERS["pmc"](p))
print({k: float(v.mean()) for k, v in correction_failures(d, N, corr, ox, oz).items()})
```

Window decoders: `"pm"` (PyMatching per sector), `"pmc"` (correlated PyMatching on the detector error model),
`"bpm"` (BP-weighted matching), `"bp"` and `"bp625"` (BP-OSD with min-sum scaling 1.0 and 0.625). `memory_baselines`
runs the same decoders on the whole memory at once, and `memory_union_find_failures` gives union-find. Any callable
`dec(window, Dx_window, Dz_window, final, k, t0) -> {variant: (ex, ez)}` can be used as a window decoder.

## Reproducing the paper

`experiments/surface_code/README.md` lists the task tables, seeds and analysis scripts of Fig. 5 and Fig. S7-S8, and
`hardware/README.md` the FPGA designs, their build and the board tools.
