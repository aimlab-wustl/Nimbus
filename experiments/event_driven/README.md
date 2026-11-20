# Event-driven updating and machine size: experiments, source data and panels

This folder holds the code for the operation counts of event-driven updating versus recomputation (Fig. 6, Fig. S9)
and for the size of the compiled machine (Fig. S3):

1. **Run** (`run_event.py`, `program_size.py`) writes raw outputs under a folder of your choice, called `RUNS` below.
2. **Analyze** (`analysis/*.py`) reads the raw outputs and writes one CSV per panel into `source_data/`.
   The raw outputs of the paper are `event_driven.zip` of the Zenodo record [10.5281/zenodo.24182700](https://doi.org/10.5281/zenodo.24182700); unpack it and use
   the folder as `RUNS`.
3. **Plot** (`figures/fig6*.py`, `figS9*.py`, `figS3*.py`, or `python figures/make_all.py`) reads only `source_data/`
   and writes `figures/out/<panel>.pdf` and `.png`.

| Paper item | Content | Run | Analysis -> source data | Panel script |
|---|---|---|---|---|
| Fig. 6a | cumulative operations, event-driven / recomputation, four workloads, break-even points | `run_event.py` (`m30c`, `m50p`, `s11`, `pair2d`) | `analysis/event_windows.py` -> `fig6/fig6a.csv`, `fig6/fig6_summary.csv` | `fig6a.py` |
| Fig. 6b | acceptance a against sharing ratio rho per window, colored by anneal progress | same | `analysis/event_windows.py` -> `fig6/fig6b.csv` | `fig6b.py` |
| Fig. S9a | operations per iteration, event-driven / recomputation, four workloads, 95% interval | same | `analysis/event_windows.py` -> `figS9/figS9a.csv` | `figS9a.py` |
| Fig. S9b | the same, synthetic family | `run_event.py` (`syn_*`) | `analysis/event_windows.py` -> `figS9/figS9b.csv` | `figS9b.py` |
| Fig. S9c | cumulative operations, synthetic family | same | `analysis/event_windows.py` -> `figS9/figS9c.csv` | `figS9c.py` |
| Fig. S9d | a against rho, synthetic family | same | `analysis/event_windows.py` -> `figS9/figS9d.csv`, `figS9/figS9_summary.csv` | `figS9d.py` |
| Fig. S3a, S3b, S3c | connections (surface code, matching) and clause counts (matching) | `program_size.py` | `analysis/figS3.py` -> `figS3/figS3a.csv`, `figS3b.csv`, `figS3c.csv` | `figS3a.py`, `figS3b.py`, `figS3c.py` |
| Fig. S3d, S3e | shares of the information needed to program and run the machine | `program_size.py` | `analysis/figS3.py` -> `figS3/figS3d.csv`, `figS3e.csv` | `figS3d.py`, `figS3e.py` |

Paths in the commands below are relative to this folder; `source_data/` and `figures/` are at the repository root.

## Terms

* **Recomputation**: at every iteration the input of the proposed move neuron is recomputed from its connections
  (one operation per weighted connection read). **Event-driven updating**: the inputs of the move neurons are
  maintained; an accepted spin transition updates the inputs it affects (one operation per input update).
* **Acceptance a**: accepted state-changing moves per iteration. **Sharing ratio rho**: input updates per accepted
  state change divided by connection reads per iteration. Per window, the ratio of operations event-driven /
  recomputation equals a * rho exactly (checked in `fig6_summary.csv`, column `identity_max_rel_dev`).
* **Cumulative ratio**: input updates summed up to the end of a window divided by connection reads summed up to the
  same point. **Break-even point**: the iteration after which the pooled cumulative ratio stays below 1 (0 if it is
  below 1 in every window). **Acceptance floor**: accepted moves with zero energy change per iteration, pooled over the
  last octave (last eight windows). **Recomputing every input**: the inputs of all moves recomputed at every iteration
  (`ratio_vs_recompute_all_max` in the summary).
* **Workloads**: `m30c` graph matching, complete library, L = 30 (five instances); `m50p` matching, path library,
  L = 50 (three instances); `s11` surface code, d = 11, depolarizing p_err = 0.12, all d logical operators per type
  (four shot sets); `pair2d` 2D +-J spin glass, 16 x 16 torus, single flips (four coupling sets). Synthetic family
  (`synthetic.SETTINGS`, four generator seeds each): single flips with terms of order 2, 3 and 4, one-hot exchanges
  with order-2 terms, and 9 x 9 and 16 x 16 permutations with adjacent row exchanges.

## Run

```bash
python run_event.py list
python run_event.py run --all --out RUNS/event [--only m30c s11 ...] [--threads N]
python run_event.py run --task m30c__o0_21 --out RUNS/event --log2 24 --replicas 128
```

Every task is one Fowler-Nordheim schedule (eta = 0.1) from the start temperature to 1/1,000 of it over 2^24 (matching,
complete) or 2^26 iterations per replica, never stopped at the optimum. The start temperature is that of the
automatically calibrated schedule of the matching runs (`configs/matching_temperatures.json`) or calibrated with the
same acceptance rule (the surface code, the synthetic problems; the spin glass uses the synthetic calibration).
Replicas: 128 (matching, surface code), 64 (others). The CPU backend counts per window, with eight windows per
doubling of the iteration count and at least 256 iterations per window, the proposed moves, accepted moves, accepted
state changes, accepted moves without energy change, connection reads and input updates. The counters do not depend
on the update strategy, so one trajectory gives the operations of both strategies, counted exactly on the same
run. Each task writes one `.npz`; a task whose file exists is skipped, so an interrupted run resumes by running the same
command again. The matching tasks need the staged instances (`../graph_matching/stage_instances.py`, `--inst` or
`NIMBUS_MATCHING_INST`); the surface-code task needs PyMatching.

Cost: about 415 core-hours for all 40 tasks (the largest: complete library L = 30, 21 min on 32 threads for 128
replicas at 2^24).

```bash
python analysis/event_windows.py --runs RUNS/event --out ../../source_data
```

The 95% intervals of the per-window ratio (Fig. S9a, S9b) resample the replicas with replacement within each instance
(instances fixed), 2,000 resamples, one random stream `default_rng(11)` used for the workloads in the order of the
sorted file names. The interquartile ranges (Fig. 6a, Fig. S9c) are over replicas of the per-replica cumulative ratio.
The source data hold the values plotted in the paper.

## Machine size

```bash
python program_size.py --out RUNS/size/sizes.jsonl [--cases surface:11:stabilizers matching:30:complete ...]
python analysis/figS3.py --sizes configs/program_sizes.jsonl --out ../../source_data
```

All counts are deterministic and obtained by compiling the machine (no annealing). `configs/program_sizes.jsonl` holds the
counts of the paper: 23 cases written by `program_size.py`, and the complete library at L = 50, which needs more
memory than a 16 GB laptop has (282 x 10^6 connections; the line was written by the same counting code on a larger
node). The other 23 cases take about 30 to 40 minutes on a laptop; the complete library at L = 40 needs about 5 GB. The
counting rule is in the docstring of `program_size.py`: connections and switch details are counted at their information
minimum, so the totals are a lower bound on the information stored, not a circuit or area estimate. Weights and
maintained inputs are counted at 16 bits.

## Checks against the paper text

`fig6/fig6_summary.csv` and `figS9/figS9_summary.csv`: break-even within 1-3% of the run for the complete matching
library, the surface code and the spin glass, and from the first window for the path library; cumulative ratio at the
end 0.2% (path library) to 18% (spin glass); at most 2.8% of the operations of recomputing every input; rho between 2
and 5 (path library 0.14); acceptance floor at most 0.08 (0.082 for the spin glass); the floor is below 1/rho for all
workloads.

## Raw outputs and equivalence

The 40 raw `.npz` files behind the source data (14 MB in total) are not part of the repository; `analysis/event_windows.py`
reproduces every CSV from them. For small cases (spin glass, surface code, one-hot and permutation problems, equal
seeds) the window counters written by `run_event.py` are identical to those of the production runs that produced the
paper's raw files. The bootstrap intervals recomputed here agree with the plotted values to 5 x 10^-6 (relative); the
source data keep the plotted values' resampling stream (`default_rng(11)`).
