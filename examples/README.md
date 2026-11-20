# Examples

Every configuration below runs in seconds on a laptop (`nimbus solve examples/<file>.yaml`). Relative paths in a
configuration resolve against the configuration file, so the commands work from any folder.

| File | Problem | Constraint family | Move library |
|---|---|---|---|
| `toy_pair_exchanges.yaml` | six-spin toy of the paper (select 3 of 6, two triangle rewards) | exact cardinality | complete pair exchanges |
| `toy_augmented.yaml` | the same toy | exact cardinality | pair exchanges plus one simultaneous double exchange |
| `toy_penalty.yaml` | the same toy as a penalty baseline | none (penalty term) | single spin flips |
| `matching_small.yaml` | graph matching, 8 points per set (`problems/matching_n8.npz`) | permutation (one-hot rows and columns) | complete row exchanges (`--set transitions.library=path` for the path library) |
| `surface_code_d5.yaml` | decoding one syndrome of the d = 5 rotated surface code | parity checks | stabilizers plus one logical operator |
| `spin_glass.yaml` | 2D +-J spin glass, 8 x 8 torus (`problems/spin_glass_8x8.spinpoly.yaml`) | unconstrained (free spins) | single spin flips |
| `covering_small.yaml` | `problems/covering_small.opb` | covering (at least one) | flip one literal while a neighboring literal (path order) is satisfied |
| `packing_small.yaml` | `problems/packing_small.opb` | packing (at most one) | flip one spin while all others of the block are unselected |
| `precedence_small.yaml` | `problems/precedence_small.opb` | precedence (chain) | grow or shrink the selected prefix of the chain by one |
| `blocks_small.yaml` | `problems/blocks_small.spinpoly.yaml` | one block of every family, including contiguity, plus a free spin | composed per block |

Scripts:

| File | What it does |
|---|---|
| `paper_protocol.py` | the graph-matching protocol through the Python API: compile, calibrate (start factor m, minimum ratio kappa), FN schedule, replicas, success fraction and iterations to solution per budget; also path, informed and random shortcut libraries |
| `new_problem.py` | a complete new problem adapter (densest k-subgraph), its verification and its registration as a problem kind; the worked example of [docs/guides/new_problem/adapter.md](../docs/guides/new_problem/adapter.md) |
| `find_moves.py` | moves for a constraint without a move builder (Latin squares of order 3), found by grouping the transitions between its feasible states; compiled and checked on every feasible state; and a constraint whose transitions form no move |
| `make_matching_instance.py` | writes a synthetic graph-matching tensor (`--n`, `--sigma`, `--seed`) |
| `make_spin_glass.py` | writes a 2D +-J spin glass in the `nimbus.spinpoly/1` format (`--L`, `--seed`) |

A different syndrome for `surface_code_d5.yaml`: the syndrome of an error `e` (a 0/1 vector over the 25 qubits) is
`SurfaceCodeProblem(5, 0.1).syndrome_of(e[None])[0]`.

See [docs/getting-started/quickstart.md](../docs/getting-started/quickstart.md) for reading the results and [docs/guides/settings.md](../docs/guides/settings.md)
for every configuration field.
