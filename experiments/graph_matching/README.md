# Graph matching: experiments, source data and panels

This folder holds everything needed to rerun the graph-matching experiments of the paper, to recompute the plotted
numbers from the raw run outputs, and to draw each panel:

1. **Run** (`stage_instances.py`, `worker.py` with the task tables in `configs/`, and the study scripts below) writes
   raw outputs (one compressed shard per task) under a folder of your choice, called `RUNS` below.
2. **Analyze** (`analysis/*.py`) reads the raw outputs and writes one CSV per panel or table into `source_data/`.
   The raw outputs of the paper are `graph_matching.zip` and `instances.zip` of the Zenodo record
   [10.5281/zenodo.24182700](https://doi.org/10.5281/zenodo.24182700); unpack them (the README of the record maps each folder to its
   analysis command).
3. **Plot** (`figures/*.py`, or `python figures/make_all.py`) reads only `source_data/` and writes
   `figures/out/<panel>.pdf` and `.png`.

The CSVs in `source_data/` hold the plotted values; rerunning the analysis reproduces the means exactly and the
confidence and bootstrap intervals to within bootstrap resampling noise.

| Paper item | Content | Run | Analysis -> source data | Panel script |
|---|---|---|---|---|
| Fig. 3c | P(optimum) vs iterations, WILLOW, 4 methods + domain solvers | L = 10 method comparison | `analysis/fig3c.py` -> `fig3/fig3c.csv`, `fig3c_domain_solvers.csv` | `fig3c.py` |
| Fig. 3d | ITS vs L, complete library, per-size selected schedule | ITS per size | `analysis/fig3d.py` -> `fig3/fig3d.csv` | `fig3d.py` |
| Fig. 3e | Psi/Psi_ref vs iterations, L = 10-30, complete and path libraries | automatically calibrated schedule | `analysis/fig3e_figS1.py` -> `fig3/fig3e.csv` | `fig3e.py` |
| Fig. 3f | ITS at L = 10, synthetic and WILLOW, 4 methods | L = 10 method comparison | `analysis/fig3f.py` -> `fig3/fig3f.csv` | `fig3f.py` |
| Fig. 4a | P_s at 2^20 vs library size, L = 20 | libraries at fixed budget | `analysis/fig4a_figS4a.py` -> `fig4/fig4a.csv` | `fig4a.py` |
| Fig. 4b | relaxation time at infinite temperature, L = 20, 30 | landscapes | `analysis/fig4bc.py` -> `fig4/fig4b.csv` | `fig4b.py` |
| Fig. 4c | critical depth and Catoni difficulty, L = 10 | landscapes | `analysis/fig4bc.py` -> `fig4/fig4c.csv` | `fig4c.py` |
| Fig. 4d | ITS at L = 10 vs library | ITS of libraries at L = 10 | `analysis/fig4de_tableS4.py` -> `fig4/fig4d.csv` | `fig4d.py` |
| Fig. 4e | Spearman rank correlation at fixed K | landscapes + ITS at L = 10 | `analysis/fig4de_tableS4.py` -> `fig4/fig4e.csv` | `fig4e.py` |
| Fig. 4g | batched vs static libraries, heat map | batching | `analysis/fig4g_tableS3.py` -> `fig4/fig4g.csv` | `fig4g.py` |
| Fig. S1a | keypoint accuracy on WILLOW | L = 10 method comparison | `analysis/figS1a.py` -> `figS1/figS1a.csv` | `figS1a.py` |
| Fig. S1b, S1c | sigma = 0.10: P(0.98 Psi_ref) and Psi/Psi_ref | automatically calibrated schedule | `analysis/fig3e_figS1.py` -> `figS1/figS1b.csv`, `figS1c.csv` | `figS1b.py`, `figS1c.py` |
| Fig. S2a, S2b | anneal, no cooling, random walk | control | `analysis/figS2.py` -> `figS2/*.csv` | `figS2a.py`, `figS2b.py` |
| Fig. S4a | P_s at 2^22 vs library size, L = 30 | libraries at fixed budget | `analysis/fig4a_figS4a.py` -> `figS4/figS4a.csv` | `figS4a.py` |
| Fig. S4b | ITS at L = 20, informed vs random, selected schedules | ITS of libraries at L = 20 | `analysis/figS4b.py` -> `figS4/figS4b.csv`, `figS4b_ratio.csv` | `figS4b.py` |
| Table S2 | selected schedule settings | setting selection (all studies) | `analysis/tableS2.py` -> `tables/tableS2.csv` | `tableS2.py` |
| Table S3 | batched vs static | batching | `analysis/fig4g_tableS3.py` -> `tables/tableS3.csv` | `tableS3.py` |
| Table S4 | ITS at L = 10, automatic and optimized schedules | ITS of libraries at L = 10 | `analysis/fig4de_tableS4.py` -> `tables/tableS4.csv` | `tableS4.py` |

Paths in the commands below are relative to this folder; `source_data/` and `figures/` are at the repository root.

## Terms

* **Move library**: the row exchanges a machine can propose. **Path library**: the backbone of the L - 1 exchanges of
  consecutive rows. **Complete library**: the backbone with all shortcuts (all L(L-1)/2 exchanges). **Informed /
  random shortcuts**: K added exchanges chosen from the tensor alone (`nimbus.problems.matching.informed_pairs`) or
  uniformly at random (`random_pairs`). **Backbone share**: fixed probability of proposing a backbone exchange
  (`shortcut_library(..., {"share": 0.5})`).
* **Replicas**: independent anneals of one instance from different random starting assignments.
* **Iterations**: proposed spin transitions per replica (this work) or single-spin update tests (penalty machines).
* **Iterations to solution (ITS)**: with per-replica success fraction p at budget t, ITS(t) = t ln(0.01) / ln(1 - p);
  the reported ITS is the minimum over the budget grid of the median over instances (one budget per method and problem
  set), with a percentile bootstrap over instances (`common.optimal_median_its`).
* **Psi_ref**: the reference objective value: the certified optimum at L = 10 (enumeration of all L! assignments) and
  the best value found by three domain solvers at L > 10 (`configs/psi_ref.csv`; training instances:
  `configs/psi_ref_training.csv`; see `domain_solvers/README.md`).
* **Automatically calibrated schedule**: Fowler-Nordheim schedule (eta = 0.1) with temperatures from an
  acceptance-rule pilot calibration (p_start = 0.8, p_end = 0.01, end phase after a descent of 8192 attempts to
  0.02 tau_start), start factor m = 1 and minimum ratio kappa = 2 (`nimbus.annealing.protocol`). **Selected settings**:
  m and kappa chosen on separate training instances (synthetic indices 0-5, the 20 WILLOW training pairs) by the
  minimum median ITS, and then applied to the test instances (synthetic indices 6-25, the 40 WILLOW test pairs).

## Setup

```bash
pip install -e ".[dev]"                             # from the repository root
python -m nimbus.experiments.fetch --set matching   # WILLOW-ObjectClass keypoints -> data/willow (not redistributed)
python stage_instances.py --out INST                 # all instances used in the paper (about 425 MB, a few minutes)
export NIMBUS_MATCHING_INST=INST                     # or pass --inst INST to every script
```

`stage_instances.py` is deterministic: synthetic instances from seeds, WILLOW pairs from the keypoint files. The
staged tensors are identical to the ones used for the paper results.

All runs use the CPU backend (Numba), seed 11, and write one shard per task (`shard_<task_id>.npz`: JSON records plus
per-replica arrays such as the best objective, the iteration at which the target was first reached and the best
assignment). A finished shard is never recomputed, so an interrupted job is resumed by running the same command again.
`worker.py --worker j --workers K` runs every K-th task, for job arrays. Some tasks of the automatically calibrated
schedule and of the batching study were run on an NVIDIA H100 with a GPU implementation whose trajectories are
identical to the CPU backend (same states, counters and temperatures; best objectives equal to within 1e-12, from the
floating-point summation order); the release runs everything on the CPU.

Compute used for the paper results, in total: about 4,200 core-hours on 64-core CPU cluster nodes (mostly 16 threads
per task), about 180 H100-hours for the GPU shards, and about 12 hours of single-threaded MATLAB for the domain
solvers. Details per study below.

## L = 10 method comparison (Fig. 3c, 3f; Fig. S1a; Table S2, rows at L = 10)

**Setting selection on training instances.** 6 synthetic L = 10 instances (`syn_n10_s0.05_o0_00`-`05`) and the 20
WILLOW training pairs; 32 replicas, seed 11.

* This work: factors m and kappa applied to one raw calibration per instance and library
  (task kind `matching_sched`).

  | Problem set, library | m | kappa | Budgets (log2 iterations) |
  |---|---|---|---|
  | Synthetic, complete | 1/32-8 | 1-32 | 6-20 |
  | Synthetic, path | 0.5-32 | 1-32 | 8-24 |
  | WILLOW, complete | 0.5-8 | 1-128 | 6-20 |
  | WILLOW, path | 1/8-8 | 1-32 | 8-24 |

* Penalty machines (`nimbus.baselines.penalty_machines`), budgets 2^14-2^24: penalty strength lambda-bar 0.5-32 and
  end temperature 0.1-3 (both in units of Psi*/L), ratio of start to end temperature 10, 30, 100; the NeuroSA-style
  machine also uses the quadratization multiplier 1, 2, 4 and the ratios 300 and 1000 (synthetic) and 3 and 5 (WILLOW).

**Test runs** at the selected settings, seed 11: the 20 synthetic test instances (`o0_06`-`o0_25`) and the 40 WILLOW
test pairs. This work: 256 replicas, budgets 2^6-2^24 (complete library) and 2^8-2^24 (path library). Penalty
machines: 256 replicas (64 for the NeuroSA-style machine on WILLOW), budgets 2^14-2^24.

**Domain solvers** (`domain_solvers/README.md`): 500 restarts per pair on the 40 WILLOW test pairs.

```bash
python worker.py --tasks configs/willow_sym/tasks_select_round1.csv --out RUNS/willow_l10/select_round1 --worker J --workers K
# the same for tasks_select_round2 / _ext1 / _ext2 / _ext3 (one output folder each)
python penalty_baselines.py tune --kind nhoim --names syn_n10_s0.05_o0_00 --replicas 32 --seed 11 --out RUNS/willow_l10/btune/tune_nhoim_synthetic_0.jsonl
# ... one call per training instance (syn_n10_s0.05_o0_00..05, @willow:tuning:0..19) and per kind (nhoim, neurosa);
# NeuroSA style also with --grid configs/willow_sym/penalty_grid_ratio_high.json (synthetic) or
# penalty_grid_ratio_low.json (WILLOW), into separate output files
python select_settings.py --shards "RUNS/willow_l10/select_*" --penalty "RUNS/willow_l10/btune/*.jsonl" \
    --out configs/willow_sym/choice.json --baseline-out configs/willow_sym
python worker.py --tasks configs/willow_sym/tasks_test.csv --out RUNS/willow_l10/test --worker J --workers K
python penalty_baselines.py test --kind nhoim --choice configs/willow_sym/baseline_choice_synthetic.json --stage test \
    --names syn_n10_s0.05_o0_06 ... syn_n10_s0.05_o0_25 --replicas 256 --out RUNS/willow_l10/btest/test_nhoim_synthetic.jsonl
python penalty_baselines.py test --kind neurosa --choice configs/willow_sym/baseline_choice_synthetic.json --stage test \
    --names (same) --replicas 256 --out RUNS/willow_l10/btest/test_neurosa_synthetic.jsonl
python penalty_baselines.py test --kind nhoim --choice configs/willow_sym/baseline_choice_willow.json --stage willow_test \
    --names @willow:test:0 ... @willow:test:39 --replicas 256 --out RUNS/willow_l10/btest/test_nhoim_willow.jsonl
python penalty_baselines.py test --kind neurosa --choice configs/willow_sym/baseline_choice_willow.json --stage willow_test \
    --names (same) --replicas 64 --out RUNS/willow_l10/test_neurosa_willow.jsonl
# domain solvers (MATLAB)
python domain_solvers/export_mat.py INST RUNS/willow_domain/mat --family willow --role test
matlab -batch "run_refs('RUNS/willow_domain/mat','RUNS/willow_domain/refs_willow.mat',{'adapt_bcagm3','bcagm3','rrwhm'},500,'willow_*.mat')"
python domain_solvers/willow_domain_summary.py INST RUNS/willow_domain/refs_willow.mat RUNS/willow_domain
```

Analysis:

```bash
python analysis/fig3c.py --shards RUNS/willow_l10/test --nhoim "RUNS/willow_l10/btest/test_nhoim_willow*.jsonl" \
    --neurosa RUNS/willow_l10/test_neurosa_willow.jsonl --domain RUNS/willow_domain/willow_domain_solvers.csv --out ../../source_data/fig3
python analysis/fig3f.py --shards RUNS/willow_l10/test --penalty "RUNS/willow_l10/btest/*.jsonl" RUNS/willow_l10/test_neurosa_willow.jsonl \
    --out ../../source_data/fig3
python analysis/figS1a.py --shards RUNS/willow_l10/test --inst INST --out ../../source_data/figS1
```

Compute used (CPU cluster nodes, 64 cores each): selection of this work's settings about 525 core-hours (16-thread
tasks); test runs of this work about 650 core-hours; penalty-machine selection and 256-replica tests about 300
core-hours (4-thread jobs); domain solvers on WILLOW 60,000 restarts in about 6 minutes (MATLAB R2024a, one thread).

## Reference values Psi_ref

Domain solvers Adapt-BCAGM3, BCAGM3 and RRWHM on the 20 test instances of every (sigma, L) with L > 10 (sigma 0.05 at
L = 15-50, sigma 0.10 at L = 15-30; 500 restarts, 200 at L = 40 and 50) and on the 6 training instances per L
(sigma 0.05):

```bash
python domain_solvers/export_mat.py INST RUNS/refs/mat_test --family synthetic --role test
python domain_solvers/export_mat.py INST RUNS/refs/mat_tuning --family synthetic --role tuning
# one MATLAB call per (sigma, L), selected with the file pattern, for example:
matlab -batch "run_refs('RUNS/refs/mat_test','RUNS/refs/refs_test_s0.05_n15.mat',{'adapt_bcagm3','bcagm3','rrwhm'},500,'syn_n15_s0.05_*.mat')"
matlab -batch "run_refs('RUNS/refs/mat_tuning','RUNS/refs/refs_tuning_n15.mat',{'adapt_bcagm3','bcagm3','rrwhm'},500,'syn_n15_s0.05_*.mat')"
python domain_solvers/psi_ref_table.py --inst INST --mat RUNS/refs/refs_test_*.mat --out configs/psi_ref.csv
python domain_solvers/psi_ref_table.py --inst INST --mat RUNS/refs/refs_tuning_*.mat --training --out configs/psi_ref_training.csv
```

About 11.5 hours of single-threaded MATLAB (R2024a) on a standard workstation for all synthetic sets. `configs/psi_ref.csv` stores
6 significant digits (the values the test analyses use); `configs/psi_ref_training.csv` stores full precision (the
values the setting selection uses).

## Automatically calibrated schedule on synthetic instances (Fig. 3e, Fig. S1b, S1c, Fig. 4a, 4g, Fig. S4a, Table S3)

All runs use the 20 synthetic test instances per size (indices 6-25), 256 replicas, seed 11, and the automatically
calibrated schedule (m = 1, kappa = 2). Every budget is an independent anneal.

| Run | Task table / script | Content | Compute used |
|---|---|---|---|
| sigma005 | `configs/auto_schedule/tasks_sigma005.csv` | sigma = 0.05; complete library at L = 10-30, path library at L = 10-50; budgets 2^10-2^24 | about 100 H100-hours (a few tasks on 16 CPU cores) |
| sigma010 | `configs/auto_schedule/tasks_sigma010.csv` | sigma = 0.10, complete library, L = 10-30 | about 18 H100-hours |
| libraries | `configs/auto_schedule/tasks_libraries.csv` | backbone + K informed or random shortcuts, backbone share 0.5; L = 20: K = 10-160, budgets 2^16-2^20; L = 30: K = 15-240, budgets 2^18-2^22 | about 55 H100-hours |
| batching | `configs/batching/tasks_batching.csv` | L = 20; K = 20, 40, 80; batch replaced every D = 2^12, 2^14, 2^16 iterations; budgets 2^18, 2^20 | about 8.5 H100-hours |
| library control | `library_control.py --n 20` / `--n 30` | complete library with backbone share 0.5; L = 20: four calibrations, budgets 2^18, 2^20; L = 30: budgets 2^20, 2^22 | 16 CPU cores; L = 20 about 50 core-hours; L = 30 about 250 core-hours (compiling the L = 30 complete library needs 300-450 GB of RAM) |

```bash
python worker.py --tasks configs/auto_schedule/tasks_sigma005.csv --out RUNS/auto_schedule/sigma005 [--worker j --workers K]
python worker.py --tasks configs/auto_schedule/tasks_sigma010.csv --out RUNS/auto_schedule/sigma010
python worker.py --tasks configs/auto_schedule/tasks_libraries.csv --out RUNS/auto_schedule/libraries
python worker.py --tasks configs/batching/tasks_batching.csv --out RUNS/batching
python library_control.py --out RUNS/library_control --n 20
python library_control.py --out RUNS/library_control --n 30
python extract_shards.py --run RUNS/auto_schedule --subs sigma005 sigma010 libraries --dest RUNS/bundles
```

Analysis:

```bash
python analysis/fig3e_figS1.py --sigma005 RUNS/bundles/sigma005.npz --sigma010 RUNS/bundles/sigma010.npz --out ../../source_data
python analysis/fig4a_figS4a.py --sigma005 RUNS/bundles/sigma005.npz --libraries RUNS/bundles/libraries.npz \
    --control RUNS/library_control --out ../../source_data
python analysis/fig4g_tableS3.py --batching RUNS/batching --sigma005 RUNS/bundles/sigma005.npz \
    --libraries RUNS/bundles/libraries.npz --control RUNS/library_control --out ../../source_data
```

## ITS with schedules selected per size (Fig. 3d) and per library at L = 20 (Fig. S4b)

Both studies use the task kind `matching_sched`: one raw calibration per instance and library, and the variants set
tau_start = m tau_start_cal and tau_end = min(tau_end_cal, tau_start / kappa). Seed 11.

* **Training (selection of m, kappa):** 6 training instances per L (sigma 0.05, `syn_n{L}_s0.05_o0_00..05`), 32
  replicas. Grid kappa {2, 4, 8, 16} x m {0.5, 1, 2}, widened until the selection lies inside it (m down to 1/32 at
  L = 10, 0.125 and 0.25 at L = 30, and m = 0.25, 4, 8 at L = 20 for the libraries). Budgets 2^6 to 2^24 depending on L.
  Selected setting: the smallest training ITS (ties: lower mean log2 ITS, then the setting closest to kappa = 2,
  m = 1). For Fig. 3d the best kappa = 2 setting is kept when its training ITS lies inside the 95% bootstrap interval
  of the best setting. Targets on the training instances: `configs/psi_ref_training.csv`.
* **Test:** 20 test instances (`o0_06..25`), 256 replicas, target Psi >= 0.98 Psi_ref (the certified optimum at
  L = 10). ITS with a 95% percentile bootstrap over instances (5,000 resamples).
* **Libraries at L = 20:** backbone plus K = 40 or 80 informed or random shortcuts, backbone share 0.5 (random draw
  seed = instance index), and the complete library with backbone share 0.5. The complete library with uniform
  proposals reuses the L = 20 runs of the per-size study.

```bash
python worker.py --tasks configs/tts_per_size/tasks_training.csv --out RUNS/tts_per_size/training --worker J --workers K
python worker.py --tasks configs/tts_per_size/tasks_test.csv --out RUNS/tts_per_size/test --worker J --workers K
python worker.py --tasks configs/library_its_l20/tasks_training.csv --out RUNS/library_its_l20/training --worker J --workers K
python worker.py --tasks configs/library_its_l20/tasks_test.csv --out RUNS/library_its_l20/test --worker J --workers K
python analysis/fig3d.py --test RUNS/tts_per_size/test --train RUNS/tts_per_size/training \
    --selection-out RUNS/tts_per_size/selection.csv --out ../../source_data/fig3/fig3d.csv
python analysis/figS4b.py --test RUNS/library_its_l20/test RUNS/tts_per_size/test \
    --train RUNS/library_its_l20/training RUNS/tts_per_size/training --selection-out RUNS/library_its_l20/selection.csv \
    --out ../../source_data/figS4/figS4b.csv --ratio-out ../../source_data/figS4/figS4b_ratio.csv
```

With `--train`, the scripts recompute the selection from the training runs and compare it with `configs/*/choice.json`.

Compute used (CPU cluster nodes, 16 threads per task): per-size study at L = 15-30 about 1,300 core-hours in total
(training about 380 core-hours; a test task at L = 30 needs about 36 GB of RAM); L = 10 about 4 core-hours on a standard workstation;
libraries at L = 20 about 1,000 core-hours.

## Control: anneal, no cooling, random walk (Fig. S2)

Six test instances (L = 20 at 2^18 iterations, L = 30 at 2^20), complete library, 64 replicas, seed 11, with the same
starting assignments and random numbers in every variant: anneal (automatically calibrated schedule, m = 1,
kappa = 2), no cooling (constant temperature tau_start), random walk (every proposal accepted). One CPU node with 16
threads, about 15 minutes.

```bash
python control_random_walk.py --out RUNS/control/summary.jsonl --per-replica RUNS/control/per_replica
python analysis/figS2.py --per-replica RUNS/control/per_replica --summary RUNS/control/summary.jsonl \
    --out-a ../../source_data/figS2/figS2a.csv --out-b ../../source_data/figS2/figS2b.csv \
    --out-medians ../../source_data/figS2/figS2a_medians.csv
```

## Move libraries at L = 10 and relaxation time (Fig. 4b-e, Table S4)

**Relaxation time at infinite temperature (Fig. 4b).** Exact linear algebra on the 20 test instances at L = 20 and
L = 30. Libraries: the path backbone; the backbone plus K informed or random shortcuts at backbone share 0.5
(K = 10-160 at L = 20, 15-240 at L = 30; random seed = instance index); the complete library at backbone share 0.5.
Seconds on a standard workstation.

```bash
python landscape.py gap --out RUNS/landscape
```

**Exact energy landscapes at L = 10 (Fig. 4c, 4e).** All 10! assignments of each of the 20 test instances
(sigma = 0.05), for 46 libraries per instance: the path library; the complete library (backbone share 0.5); the backbone
plus K = 4, 9, 18, 27 informed shortcuts; the backbone plus K random shortcuts, ten draws per K (seeds = instance index
and 100-108). One task per instance, about 1 GB of RAM; about 2 core-hours in total.

```bash
python landscape.py depth --out RUNS/landscape/depth --task i     # i = 0..19
```

**Iterations to solution at L = 10 (Fig. 4d, 4e, Table S4).** Target: the certified optimum. Seed 11. Budgets 2^6 to
2^16 iterations per replica in factor-2 steps (2^6 to 2^20 for the path library).

```bash
# 1. automatically calibrated schedule: 20 test instances x 46 libraries, 256 replicas
python library_its.py run --role test --settings auto --task i --out RUNS/library_its/auto              # i = 0..19
# 2. setting selection: 6 training instances x 10 library classes, 32 replicas, m = 1/16..16 x kappa = 1..32;
#    random classes use 3 draws per instance (seeds i, 200 + i, 300 + i)
python library_its.py run --role tuning --task j --out RUNS/library_its/tune                            # j = 0..59
# 3. wider grid for random shortcuts at K = 9 (kappa 64 and 128 at every m)
python library_its.py run --role tuning --task j --variants 64,0.0625 64,0.125 ... 128,16 --out RUNS/library_its/tune_ext   # j = 6, 16, ..., 56
# 4. choose the settings -> configs/library_its_l10/choice.json
python library_its.py choose --tune RUNS/library_its/tune RUNS/library_its/tune_ext --out configs/library_its_l10/choice.json
# 5. test runs at the selected settings
python library_its.py run --role test --settings configs/library_its_l10/choice.json --task i --out RUNS/library_its/tuned
```

Steps 1-5: about 70 core-hours on 64-core CPU nodes (6 threads per task).

Analysis:

```bash
python analysis/fig4bc.py --gap RUNS/landscape/gap.json --depth RUNS/landscape/depth --out ../../source_data/fig4
python analysis/fig4de_tableS4.py --auto RUNS/library_its/auto --tuned RUNS/library_its/tuned \
    --gap RUNS/landscape/gap.json --depth RUNS/landscape/depth --out-fig ../../source_data/fig4 --out-table ../../source_data/tables
```

Statistics: ITS as above; for random shortcuts the median over the draws of each instance is taken first. ITS 95% CI:
percentile bootstrap over instances (5,000 resamples). Fig. 4b, 4c and 4e: bootstrap of the median (2,000 resamples),
all three from one random stream (`default_rng(0)`) in that order. Ratios: paired bootstrap (5,000 resamples).

## Table S2 (selected settings)

```bash
python analysis/tableS2.py --out ../../source_data/tables/tableS2.csv
```

reads the four selection outputs `configs/willow_sym/choice.json`, `configs/tts_per_size/choice.json`,
`configs/library_its_l20/choice.json` and `configs/library_its_l10/choice.json`.

## Bootstrap seeds

The confidence intervals use fixed seeds so that the source data are reproducible exactly; the seeds are arbitrary
integers (for example 20261015 and 20261023 in the ITS intervals and ratios).
