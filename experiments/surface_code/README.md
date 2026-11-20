# Surface code: experiments, source data and panels

This folder holds everything needed to rerun the surface-code decoding experiments of the paper, to recompute the
plotted numbers from the raw run outputs and the board result files, and to draw each panel:

1. **Run** (`one_shot.py`, `real_time.py`, `cpu_timing.py`, with the task tables in `configs/`) writes raw outputs (one
   `.npz` of per-shot logical failures and one `.json` summary per task) under a folder of your choice, called `RUNS`
   below. The board measurements come from the FPGA designs and host tools in `hardware/`; their result files are
   called `BOARD` below.
2. **Analyze** (`analysis/*.py`) reads the raw outputs, the board result files, the measured design data in `board/`
   and the published decoder times in `published/`, and writes one CSV per panel or table into `source_data/`.
   The raw outputs and board result files of the paper are `surface_code.zip` of the Zenodo record
   [10.5281/zenodo.24182700](https://doi.org/10.5281/zenodo.24182700); unpack it and use `surface_code/` as `RUNS` and `surface_code/board/` as `BOARD`.
3. **Plot** (`figures/fig5*.py`, `figS7*.py`, `figS8*.py`, `tableS5.py`, or `python figures/make_all.py`) reads only
   `source_data/` and writes `figures/out/<panel>.pdf` and `.png`.

| Paper item | Content | Run | Analysis -> source data | Panel script |
|---|---|---|---|---|
| Fig. 5c | LER vs sweeps per shot, d = 15, p_err = 0.12: backbone, shortcuts, replica exchange | one syndrome per shot: library | `analysis/one_shot.py` -> `fig5/fig5c.csv` | `fig5c.py` |
| Fig. 5d | LER vs d, p_err = 0.12, all decoders on the same shots | one syndrome per shot: accuracy, union-find | `analysis/one_shot.py` -> `fig5/fig5d.csv` | `fig5d.py` |
| Fig. 5e | LER vs decode time per shot, d = 3-15 | accuracy, CPU timing, board | `analysis/one_shot.py` -> `fig5/fig5e.csv` | `fig5e.py` |
| Fig. 5f | decoders of one syndrome per shot at d = 9, with published times | accuracy, CPU timing, board, `published/one_shot_times.csv` | `analysis/one_shot.py` -> `fig5/fig5f.csv` | `fig5f.py` |
| Fig. 5g | real time, LER vs d at p_err = 0.03, board points | real time: decoder, baselines; board | `analysis/real_time.py` -> `fig5/fig5g.csv` | `fig5g.py` |
| Fig. 5h | real time, LER vs decode time per round | decoder, baselines, CPU timing, board | `analysis/real_time.py` -> `fig5/fig5h.csv`, `fig5hi_rounds.csv` | `fig5h.py` |
| Fig. 5i | real-time decoders at d = 9, with published times | decoder, baselines, union-find, board, `published/real_time_times.csv` | `analysis/real_time.py` -> `fig5/fig5i.csv`, `fig5hi_rounds.csv` | `fig5i.py` |
| Fig. S7a | one syndrome per shot, LER vs p_err, this work and tensor-network ML | accuracy (incl. 2^12 sweeps near the threshold) | `analysis/one_shot.py` -> `figS7/figS7a.csv` | `figS7a.py` |
| Fig. S7b | real time, LER vs p_err, this work and best CPU baseline | decoder, baselines | `analysis/real_time.py` -> `figS7/figS7b.csv` | `figS7b.py` |
| Fig. S7c | real time, best CPU baseline / this work | decoder, baselines | `analysis/real_time.py` -> `figS7/figS7c.csv` | `figS7c.py` |
| Fig. S7d | real time, LER vs sweeps per window, d = 9, p_err = 0.03 | decoder (sweep and replica variants) | `analysis/real_time.py` -> `figS7/figS7d.csv` | `figS7d.py` |
| Fig. S7e | decode time per shot vs d, measured and published | CPU timing, board, `published/one_shot_times.csv` | `analysis/one_shot.py` -> `figS7/figS7e.csv` | `figS7e.py` |
| Fig. S8a | energy per decode on the board | board, `board/designs.csv` | `analysis/board.py` -> `figS8/figS8a.csv` | `figS8a.py` |
| Fig. S8b | post-selection on the board's confidence, p_err = 0.12 | board | `analysis/one_shot.py` -> `figS8/figS8b.csv` | `figS8b.py` |
| Fig. S8c | timeline of one real-time shot at d = 9, 1 ms rounds | board | `analysis/board.py` -> `figS8/figS8c.csv` | `figS8c.py` |
| Table S5 | board designs: replicas, layout, LUTs, clock, time, power | board, `board/designs.csv` | `analysis/board.py` -> `tables/tableS5.csv` | `tableS5.py` |

Paths in the commands below are relative to this folder; `source_data/` and `figures/` are at the repository root.

## Terms

* **One syndrome per shot**: code-capacity depolarizing noise (X, Y, Z each with probability p_err/3 on every data
  qubit), one syndrome measured without error per shot; the decoder outputs a logical class.
* **Real time**: phenomenological depolarizing noise, a memory experiment of 4d noisy rounds (data errors as above and
  measurement flips with probability p_err) and one round without error. **Windows** of 2d rounds commit their first d
  rounds and pass the corrected syndrome to the next window; the final window commits all remaining rounds.
* **This work**: the bit-exact software model of the FPGA design for each d (`nimbus.decoding.one_shot`,
  `nimbus.decoding.real_time`), which makes the same decisions as the board on the same shots.
* **Backbone**: stabilizer moves and one logical operator per error type. **Shortcuts**: the other d - 1 straight-line
  logical operators per error type. **Replica exchange**: R replicas on a geometric ladder of inverse temperatures from
  1 to 0.1, with an exchange round every second sweep (one syndrome per shot) or every sweep (real time).
* **Sweeps**: proposals of every move once. One syndrome per shot: the decision at a budget of g sweeps is the most
  visited class of the beta = 1 replica over sweeps [g/2, g). Real time: 2^12 sweeps per window; non-final windows
  commit the lowest-energy state, the final window the most visited class.
* **LER**: logical error rate, the fraction of shots whose correction differs from the error by a logical operator.
  Error bars: 95% Wilson score intervals (z = 1.96).
* **CPU baselines**: PyMatching (X and Z decoded separately), correlated PyMatching, BP+matching, BP-OSD (OSD-CS order
  10, min-sum scaling 0.625), union-find (peeling) and tensor-network ML decoding (qecsim matrix product state, bond
  dimension 16), all from `nimbus.decoding.baselines`. In real time the baselines run in the same sliding windows as
  this work. **Best CPU baseline**: the lowest number of logical errors among correlated PyMatching, BP+matching and
  BP-OSD at each (d, p_err).

## Setup

```bash
pip install -e ".[dev,refs]"     # from the repository root; refs: pymatching, ldpc, stim, qecsim
```

Shots are regenerated from seeds, so every decoder sees the same shots:

* one syndrome per shot: `sample_shots(problem, shots, 7 + 1000 * chunk)`, chunks of 1,000 shots (chunks 40-47:
  8,000 shots per point); decoder seed `1 + 7919 * chunk`;
* real time: `memory_shots(d, p, 250, chunk)` (seed 11), chunks of 250 shots; decoder seed
  `(11 << 40) ^ (chunk << 24) ^ (d << 16) ^ int(p * 1e5)`.

A task whose `.npz` exists is skipped, so an interrupted job is resumed by running the same command again;
`--worker j --workers K` runs every K-th line of a task table, for job arrays.

## One syndrome per shot (Fig. 5c-f, Fig. S7a, S7e, S8b)

Task table `configs/tasks_one_shot.txt` (888 lines):

* `accuracy d p chunk 1000`: the board configuration (replicas 16, 16, 64, 56, 40, 32, 24 at d = 3-15; 2^k sweeps with
  2^k the next power of two above the board budget of 46, 128, 216, 304, 430, 512, 1722 sweeps) and tensor-network ML,
  PyMatching and BP-OSD on the same shots; d = 3-15, p_err = 0.08-0.19, 8 chunks per point.
* `accuracy d p chunk 1000 12`: 2^12 sweeps per shot at p_err >= 0.14 for d >= 5 (the budget is a runtime register on
  the board); the reference decoders are taken from the run above.
* `library d 0.12 chunk 1000 variant 12`: d = 7, 11, 15 (the paper shows d = 15), variants `all` (backbone and shortcuts, the FPGA library), `one`
  (backbone only), `noswap` (no replica exchange: the beta = 1 replica alone), `stab` (stabilizer moves only).

```bash
python one_shot.py tasks configs/tasks_one_shot.txt --out RUNS/one_shot --worker J --workers K
# the library lines may go to their own folder: grep ^library configs/tasks_one_shot.txt > lib.txt
python one_shot.py tasks lib.txt --out RUNS/library
python one_shot.py union-find --dir RUNS/one_shot --p 0.12
```

Board results (written by the host tools in `hardware/`, p_err = 0.12, 20,000 shots per d): `res_d<d>_p0.12_20k.npz`
(fixed budget; d = 5 at 128 sweeps: `res_d5_p0.12_20k_n128.npz`; per shot: decision, class counts, on-chip cycles) and
`es/es_d<d>.jsonl` (the grid of early-stopping settings: first checkpoint, threshold, mean time, LER). The board time
per shot at the fixed budget is the on-chip time from syndrome to decision, (start + run + 2) cycles / clock. Early
stopping: the fastest setting of the grid whose LER differs from the fixed budget by at most 0.001.

## Real time (Fig. 5g-i, Fig. S7b-d)

Task table `configs/tasks_real_time.txt`:

* `decoder d p chunk R 12` (540 lines with the next item): the board designs, R = 16 at d = 5, 7 and R = 24 at d = 9, 11,
  2^12 sweeps per window. d = 5-11, p_err = 0.025-0.05; 12 chunks (3,000 shots) per point, 36 chunks (9,000 shots) at
  p_err = 0.0425-0.0475 for d = 7, 9 (d = 11: p_err = 0.025-0.04).
* `decoder 9 0.03 chunk R K`: sweeps per window 2^9-2^13 with 24 replicas and 2^12 with 16 replicas (Fig. S7d).
* `baselines d p chunk pm,pmc,bpm,bp` and `baselines d p chunk bp625` (960 lines): the windowed CPU baselines on the
  same chunks. The paper's BP-OSD in real time is `bp625` (min-sum scaling 0.625, as in one syndrome per shot); the
  analysis uses it in place of `bp` (library default scaling 1.0).

```bash
python real_time.py tasks configs/tasks_real_time.txt --out RUNS/real_time --worker J --workers K
python real_time.py union-find --d 9 --p 0.03 --chunks 0-11 --out RUNS/real_time_union_find
```

Board results (host tools in `hardware/`, p_err = 0.03, 1,000 other shots per d): `d<d>/check_thr.json` (rounds
released as fast as possible: LER, window decode time and time per round = window time / d) and `d<d>/d<d>_rt.npz`
(rounds released by the on-chip timer every 1 ms, at d = 11 every 4.45 ms: clock and, per window, start and commit
cycles).

## CPU decode times (Fig. 5e, 5f, 5h, Fig. S7e)

```bash
python cpu_timing.py all --out RUNS/cpu_timing
```

runs, on one core and one after another, the per-shot times of PyMatching, union-find, BP-OSD (200 shots) and
tensor-network ML (100 shots) at p_err = 0.12, d = 3-15 (`decoder_timing_one_shot.jsonl`), and the per-round times of
the window baselines on 60 shots at p_err = 0.03, d = 5-11 (`wbase_time_d<d>_p0.03.json`; time per round = mean time
of a non-final window / d). Decoder objects are built once and warmed up; only the decode calls are timed. The paper's
times were measured on one core of one CPU cluster node with an Intel Xeon Gold 6548Y+, as one job.

## Analysis

```bash
python analysis/one_shot.py --runs RUNS/one_shot --library RUNS/library \
    --timing RUNS/cpu_timing/decoder_timing_one_shot.jsonl --board BOARD/one_shot --out ../../source_data
python analysis/real_time.py --decoder RUNS/real_time --baselines RUNS/real_time --union-find RUNS/real_time_union_find \
    --timing RUNS/cpu_timing --board BOARD/real_time --out ../../source_data
python analysis/board.py --one-shot BOARD/one_shot --real-time BOARD/real_time --out ../../source_data
```

The scripts print the numbers quoted in the text (for example the ratios of logical errors at d = 15, the excess of
this work over ML decoding over all points of Fig. S7a, the ratio of the best CPU baseline to this work per d, the
smallest real-time slack on the board).

**Measured and published values typed from their sources.** `board/designs.csv` holds the parameters of each FPGA
design, the LUT counts from the Vivado implementation utilization reports and the core-rail power (decoding / idle)
from the board power logs, with the source in each row. `published/*.csv` hold the published decoder times and
platform round times, with the reference key of the paper in the `source` column; Fig. 5f and 5i place each published
decoder at the LER of its kind of algorithm on our shots (`algorithm_on_our_shots`: lowest-energy correction, matching,
union-find or BP-OSD).

**Board values in source data.** Board times, clocks and LERs are recomputed from the board result files at full
precision; the paper text and Table S5 round them (for example 1.05 µs, 646.3 µs per round).

## Compute used

One core per task on 64-core CPU cluster nodes, from the run times recorded in the outputs: one syndrome per shot
about 220 core-hours (accuracy; tensor-network ML is most of it) and 35 core-hours (library); real time about 710
core-hours (software model of the designs) and 16 core-hours (windowed baselines). CPU decode times: one exclusive node
(Intel Xeon Gold 6548Y+), one core, about 1 hour. Board runs: minutes per design on the RFSoC 4x2 board.

## Reproducibility notes

* The software model is bit-exact with the board and deterministic for the given seeds, so the LERs of this work
  are reproduced exactly on any platform.
* BP-OSD (ldpc 2.4.1, min-sum) gave slightly different decisions on Linux and on Windows for the same input: on 1,000
  test shots at d = 3, p_err = 0.12, 10 decisions differ between the cluster run (Linux) and the same code on Windows
  (window BP-OSD at d = 5, p_err = 0.025, 250 shots: 3 decisions).
  All paper results were computed on Linux. The other baselines (PyMatching, correlated PyMatching, BP+matching,
  union-find, tensor network) agreed exactly in the checks.
