# Small examples

Every configuration below runs in seconds on a laptop:

```bash
nimbus solve examples/<file>.yaml
```

Relative paths in a configuration resolve against the configuration file, so the commands work from any folder. The
files are in [`examples/`](https://github.com/aimlab-wustl/Nimbus/tree/main/examples).

| File | Problem | Constraint family | Move library |
|---|---|---|---|
| `toy_pair_exchanges.yaml` | six-spin toy of the paper (select 3 of 6, two triangle rewards) | exact cardinality | complete pair exchanges |
| `toy_augmented.yaml` | the same toy | exact cardinality | pair exchanges plus one simultaneous double exchange |
| `toy_penalty.yaml` | the same toy as a penalty baseline | none (penalty term) | single spin flips |
| `matching_small.yaml` | graph matching, 8 points per set | permutation (one-hot rows and columns) | complete row exchanges |
| `surface_code_d5.yaml` | decoding one syndrome of the d = 5 rotated surface code | parity checks | stabilizers plus one logical operator |
| `spin_glass.yaml` | 2D ±J spin glass, 8 x 8 torus | unconstrained (free spins) | single spin flips |
| `covering_small.yaml` | covering instance in OPB format | covering (at least one) | flip one literal while a neighboring literal is satisfied |
| `packing_small.yaml` | packing instance in OPB format | packing (at most one) | flip one spin while all others of the block are unselected |
| `precedence_small.yaml` | precedence chain in OPB format | precedence (chain) | grow or shrink the selected prefix of the chain by one |
| `blocks_small.yaml` | one block of every family plus a free spin | all of the above, including contiguity | composed per block |

## Scripts

| File | What it does |
|---|---|
| `paper_protocol.py` | the graph-matching protocol through the Python API: compile, calibrate, Fowler-Nordheim schedule, replicas, success fraction and iterations to solution per budget; path, informed and random shortcut libraries |
| `new_problem.py` | a complete new problem adapter (densest k-subgraph), its verification and its registration as a problem kind; the worked example of [A problem adapter](../guides/new_problem/adapter.md) |
| `find_moves.py` | moves for a constraint without a move builder (Latin squares of order 3), found by grouping the transitions between its feasible states, then compiled and checked on every feasible state; and a constraint whose transitions form no move ([Your own moves](../guides/new_problem/own_moves.md)) |
| `make_matching_instance.py` | writes a synthetic graph-matching tensor (`--n`, `--sigma`, `--seed`, `--out`) |
| `make_spin_glass.py` | writes a 2D ±J spin glass in the `nimbus.spinpoly/1` format (`--L`, `--seed`) |
