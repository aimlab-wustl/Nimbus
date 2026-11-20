# Domain solvers (hypergraph matching) and the reference values Psi_ref

Three published hypergraph-matching solvers give the reference value Psi_ref at L > 10 and the one-restart success
probabilities on WILLOW-ObjectClass (Fig. 3c). Their code is not part of this repository (no license grants
redistribution); download it and build it as below.

| Solver | Source | Version |
|---|---|---|
| BCAGM3, Adapt-BCAGM3 (Nguyen, Tudisco, Gautier, Hein, T-PAMI 39(6), 2017; Nguyen, Gautier, Hein, CVPR 2015) | public mirror of the authors' 2017 code release: https://github.com/smallsmartboy/An-Efficient-Multilinear-Optimization-Framework-for-Hypergraph-Matching | commit `dd746b166dfc9dd652d7882743943eb3fbaf497d`, folders `methods/` and `utils/` |
| RRWHM (Lee, Cho, Lee, CVPR 2011) | the same mirror, `methods/RRWHM`: release 1.1 of the authors' code (https://cv.snu.ac.kr/research/~RRWHM/, `RRWHM_release_v1.1.zip`), changed only to accept a starting assignment (needed for random restarts) | as above |

Setup (MATLAB with a C++ compiler configured by `mex -setup C++`; the results in the paper used MATLAB R2024a):

```bash
cd experiments/graph_matching/domain_solvers
mkdir -p third_party && cd third_party
git clone --filter=blob:none --sparse \
  https://github.com/smallsmartboy/An-Efficient-Multilinear-Optimization-Framework-for-Hypergraph-Matching.git hypergraph_mirror
cd hypergraph_mirror && git sparse-checkout set methods utils && git checkout dd746b166dfc9dd652d7882743943eb3fbaf497d && cd ..
matlab -batch "cd('hypergraph_mirror/methods/BCAGM/cpp'); mex -largeArrayDims mexBCAGM3Matching.cpp; cd('../../RRWHM'); mex mexRRWHM.cpp"
```

Protocol: every solver is run with random restarts: restart 1 from the all-ones starting matrix of the solver code,
restarts 2, 3, ... from `0.5 + rand` (MATLAB `rng(12345)`). BCAGM3 and Adapt-BCAGM3 use at most 50 iterations per
restart. The continuous output is discretized by a Hungarian assignment (`matchpairs`). Every returned assignment is
evaluated again with the package's objective (`read_refs.py`), which also checks the solver's own score (6 Psi).

Files:

* `export_mat.py`: tensors of the staged instances -> solver input files (`indH3`, `valH3`, `nP`).
* `run_refs.m`: runs the solvers with restarts; one result file per call.
* `read_refs.py`: reads a result file and evaluates the assignments.
* `psi_ref_table.py`: builds `configs/psi_ref.csv` (test instances: certified optima at L = 10, best domain-solver value at L > 10; 6 significant digits) and, with `--training`, `configs/psi_ref_training.csv` (training instances used in the setting selection; best domain-solver value, full precision).
* `willow_domain_summary.py`: per-solver summary on the 40 WILLOW test pairs (Fig. 3c reference lines).
