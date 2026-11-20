# Changelog

## v1.0 (2026-11-19)

First release, with the paper *Compiling Constraints into Neuromorphic Ising Machines*.

- Compiler from a problem (objective polynomial of any order, constraints, move library) to the machine
- Event-driven emulation with CPU and optional GPU backends
- Annealing schedules, automatic temperature calibration, replicas and parallel tempering
- Constraint families: permutation, parity, exact cardinality, precedence, covering, packing, contiguity, free spins;
  spinpoly and OPB input
- Moves found automatically by grouping the transitions between the feasible states of a small problem
  (`group_transitions`)
- Moves in which every spin becomes a signed product of spins (`AffineMove`), and moves used together with their
  reverse, such as cyclic exchanges of three rows (`row_cycles`, `with_reverses`)
- Surface-code decoders (one syndrome per shot, real time) with CPU baselines
- FPGA designs of both decoders for the RFSoC 4x2 board; bitstreams on the
  [release page](https://github.com/aimlab-wustl/Nimbus/releases/tag/v1.0)
- Experiments, source data and panel scripts of every result figure and table of the paper
