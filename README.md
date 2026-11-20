# NIMBUS

**N**euromorphic **I**sing **M**achine with **B**lock **U**pdates of **S**pins

[![arXiv](https://img.shields.io/badge/arXiv-2611.25352-b31b1b.svg)](https://arxiv.org/abs/2611.25352)

This repository contains the software, experiments and hardware designs of the paper

> **Compiling Constraints into Neuromorphic Ising Machines**  
> Faiek Ahsan and Shantanu Chakrabartty, arXiv:2611.25352 (2026), [https://arxiv.org/abs/2611.25352](https://arxiv.org/abs/2611.25352)

NIMBUS is the software package of the paper: a higher-order neuromorphic Ising machine that enforces constraints
through programmable transitions instead of energy penalties. Each inner-layer neuron represents a move, a family of multi-spin transitions that maps
every feasible state to another feasible state, and receives the energy change of its move, for interactions of any
order, through connections that the outer-layer clause neurons switch on or off. A compiler turns the constraints and
the move library into this wiring, adding auxiliary clauses where a switch needs them, so the search stays on
feasible states and constraint handling becomes move design.

Besides the package (compiler, event-driven emulation, CPU and GPU backends, surface-code decoders), the repository
holds every experiment of the paper (graph matching, surface-code decoding on an FPGA, event-driven updating), the
source data and plotting script of every figure panel, and the FPGA designs.

Documentation: https://aimlab-wustl.github.io/Nimbus

## Repository layout

| Folder | Content |
|---|---|
| `src/nimbus/` | the Python package: problems, move libraries, compiler, backends, annealing and calibration, configuration runner, surface-code decoders, penalty baselines |
| `examples/` | small configurations that run in seconds, one per constraint family, and API scripts ([examples/README.md](examples/README.md)) |
| `docs/` | user guide (below) |
| `experiments/graph_matching/` | graph-matching experiments: Fig. 3, 4, S1, S2, S4, Tables S2-S4 ([README](experiments/graph_matching/README.md)) |
| `experiments/surface_code/` | surface-code experiments: Fig. 5, S7, S8, Table S5 ([README](experiments/surface_code/README.md)) |
| `experiments/event_driven/` | event-driven updating and machine size: Fig. 6, S3, S9 ([README](experiments/event_driven/README.md)) |
| `source_data/` | the plotted values of every panel and table, one CSV each |
| `figures/` | one script per panel; `make_all.py` draws all of them from `source_data/` |
| `hardware/` | RTL, generators, simulation testbenches, Vivado builds and board host code of the FPGA decoders ([README](hardware/README.md)) |
| `tests/` | test suite |
| `benchmarks/` | manifest of the public benchmark data that `nimbus benchmarks fetch` downloads |

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/aimlab-wustl/Nimbus.git
cd nimbus
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras: `refs` (PyMatching, ldpc, stim, qecsim: the CPU reference decoders of the surface-code
experiments), `plot` (matplotlib, pandas), `gpu` (CuPy for CUDA 12, for the optional GPU backend). For example
`pip install -e ".[dev,refs]"`.

## Quickstart

```bash
nimbus doctor                                            # available backends, data and run folders
nimbus solve examples/toy_pair_exchanges.yaml            # the six-spin toy of the paper
nimbus solve examples/matching_small.yaml --output local_runs/matching
nimbus analyze local_runs/matching                       # best score and iterations to solution
```

Each run writes `results.parquet` (one row per replica), `manifest.json` and `config.resolved.yaml`; see
[docs/getting-started/quickstart.md](docs/getting-started/quickstart.md).

## Documentation

| Guide | Content |
|---|---|
| [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) | install, first solve, reading the results, the Python API |
| [docs/guides/new_problem/index.md](docs/guides/new_problem/index.md) | solving your own problem: spinpoly or OPB input without code, a problem adapter (worked example), or your own moves, written from the problem's structure or found by grouping its transitions |
| [docs/guides/settings.md](docs/guides/settings.md) | every setting: move libraries, schedules, temperature calibration, replicas, budgets, proposal processes, backends, event counters, penalty baselines, checkpoints, output files |
| [docs/problems/surface_code.md](docs/problems/surface_code.md) | the surface-code decoders: one syndrome per shot, real time, noise models, CPU baselines |
| [docs/concepts/internals.md](docs/concepts/internals.md) | energy convention, compiled model, transaction semantics, random streams |

## Supported constraints

| Constraint family | Example application | Moves | Example |
|---|---|---|---|
| permutation (one-hot rows and columns) | graph matching | row exchanges: path backbone, complete, informed or random shortcuts | [`examples/matching_small.yaml`](examples/matching_small.yaml), [`examples/paper_protocol.py`](examples/paper_protocol.py) |
| parity checks | surface-code decoding | stabilizers and logical operators | [`examples/surface_code_d5.yaml`](examples/surface_code_d5.yaml); decoders: [docs/problems/surface_code.md](docs/problems/surface_code.md) |
| exact cardinality (k of n) | subset selection | pair exchanges | [`examples/toy_pair_exchanges.yaml`](examples/toy_pair_exchanges.yaml), [`examples/new_problem.py`](examples/new_problem.py) |
| precedence (chains of implications) | scheduling with prerequisites | grow or shrink the selected prefix | [`examples/precedence_small.yaml`](examples/precedence_small.yaml) |
| covering (at least one) | set cover, clauses | flip a literal while another is satisfied | [`examples/covering_small.yaml`](examples/covering_small.yaml) |
| packing (at most one) | independent choices | flip a spin while the rest of the block is unselected | [`examples/packing_small.yaml`](examples/packing_small.yaml) |
| contiguity (one interval or arc) | consecutive assignments | grow or shrink the interval at an end | [`examples/blocks_small.yaml`](examples/blocks_small.yaml) |
| unconstrained (free spins) | spin glass | single spin flips | [`examples/spin_glass.yaml`](examples/spin_glass.yaml) |

Blocks of different families can be combined in one problem as long as they do not share spins; the objective is a
polynomial of any order. Any move in which every new spin is a signed product of old spins can be used
(`AffineMove`), including cyclic exchanges together with their reverses (`row_cycles`, `with_reverses`). Other
constraint families need a move library of their own, written from the structure of
the problem or, for small problems, found by grouping its transitions ([docs/guides/new_problem/index.md](docs/guides/new_problem/index.md),
[`examples/find_moves.py`](examples/find_moves.py)).

## Paper figures and tables

Every panel is drawn by one script in `figures/` from one or more CSVs in `source_data/`. The experiment folder holds
the code that produced the raw outputs and the analysis that writes the CSVs.

| Paper item | Experiment folder | Panel scripts (`figures/`) | Source data (`source_data/`) |
|---|---|---|---|
| Fig. 3c-f | `experiments/graph_matching/` | `fig3c.py`, `fig3d.py`, `fig3e.py`, `fig3f.py` | `fig3/` |
| Fig. 4a-e | `experiments/graph_matching/` | `fig4a.py` - `fig4e.py` | `fig4/` |
| Fig. 4g | `experiments/graph_matching/` | `fig4g.py` | `fig4/fig4g.csv` |
| Fig. 5c-i | `experiments/surface_code/` | `fig5c.py` - `fig5i.py` | `fig5/` |
| Fig. 6a-b | `experiments/event_driven/` | `fig6a.py`, `fig6b.py` | `fig6/` |
| Fig. S1a-c | `experiments/graph_matching/` | `figS1a.py`, `figS1b.py`, `figS1c.py` | `figS1/` |
| Fig. S2a-b | `experiments/graph_matching/` | `figS2a.py`, `figS2b.py` | `figS2/` |
| Fig. S3a-e | `experiments/event_driven/` | `figS3a.py` - `figS3e.py` | `figS3/` |
| Fig. S4a-b | `experiments/graph_matching/` | `figS4a.py`, `figS4b.py` | `figS4/` |
| Fig. S7a-e | `experiments/surface_code/` | `figS7a.py` - `figS7e.py` | `figS7/` |
| Fig. S8a-c | `experiments/surface_code/` | `figS8a.py`, `figS8b.py`, `figS8c.py` | `figS8/` |
| Fig. S9a-d | `experiments/event_driven/` | `figS9a.py` - `figS9d.py` | `figS9/` |
| Table S2 | `experiments/graph_matching/` | `tableS2.py` | `tables/tableS2.csv` |
| Table S3 | `experiments/graph_matching/` | `tableS3.py` | `tables/tableS3.csv` |
| Table S4 | `experiments/graph_matching/` | `tableS4.py` | `tables/tableS4.csv` |
| Table S5 | `experiments/surface_code/` | `tableS5.py` | `tables/tableS5.csv` |

The experiment READMEs give, per panel, the run, the analysis script and the CSV it writes.

## Reproducing the figures

```bash
pip install -e ".[plot]"
python figures/make_all.py
```

draws every panel and table from `source_data/` into `figures/out/` (PDF and PNG); `python figures/fig5d.py` draws
one. Rerunning the analysis that writes `source_data/` needs the raw run outputs and board result files, which are
archived at Zenodo, DOI [10.5281/zenodo.24182700](https://doi.org/10.5281/zenodo.24182700) (one zip per experiment folder, the matching instances, the FPGA
bitstreams and a snapshot of this code; see the README of the record). Rerunning the experiments themselves is
described in each experiment README, with the task tables, seeds and the compute used.

## Hardware

The two FPGA surface-code decoders (one syndrome per shot, and real time in sliding windows) are in `hardware/`:
RTL, the generators of the size-dependent cores, the 12 pre-generated cores, xsim testbenches, Vivado build scripts
and the PYNQ host code for the RFSoC 4x2 board. Their bit-exact software models are `nimbus.decoding.one_shot` and
`nimbus.decoding.real_time`. The 12 bitstreams of the paper are a separate download, `nimbus_bitstreams.zip`, on the
release page https://github.com/aimlab-wustl/Nimbus/releases/tag/v1.0. See [hardware/README.md](hardware/README.md).

## Tests

```bash
pytest -q                 # the default suite (CPU only)
pytest -q -m acceptance   # bounded end-to-end pipelines
```

## Citation

This repository contains the code and data for the paper:

**Compiling Constraints into Neuromorphic Ising Machines**  
Faiek Ahsan and Shantanu Chakrabartty, arXiv:2611.25352 (2026), [https://arxiv.org/abs/2611.25352](https://arxiv.org/abs/2611.25352).

If you use this repository, please cite:

```bibtex
@misc{Ahsan2026Compiling,
  author        = {Ahsan, Faiek and Chakrabartty, Shantanu},
  title         = {Compiling Constraints into Neuromorphic Ising Machines},
  year          = {2026},
  eprint        = {2611.25352},
  archivePrefix = {arXiv},
  primaryClass  = {cs.ET},
  doi           = {10.48550/arXiv.2611.25352},
  url           = {https://arxiv.org/abs/2611.25352}
}
```

## License

Except where otherwise noted, original material in this repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for the full license text.

You are free to share and adapt the licensed material for non-commercial purposes, provided that appropriate credit is given and any changes are indicated. Third-party materials remain subject to their respective licenses.

## Acknowledgements

This project is supported by the National Science Foundation grants ECCS-2332166 and FET-2551604.
