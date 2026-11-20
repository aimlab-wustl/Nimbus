# Supported constraints

A constraint family is supported when the package has a move library for it: a set of multi-spin transitions that maps
every state satisfying the constraint to another state satisfying it. The compiler wires each move into the machine, so
every state the machine visits is feasible.

| Constraint family | Example application | Moves | Example | Status |
|---|---|---|---|---|
| permutation (one-hot rows and columns) | graph matching | row exchanges: path backbone, complete, informed or random shortcuts | [`matching_small.yaml`][matching], [`paper_protocol.py`][protocol] | paper experiments |
| parity checks | surface-code decoding | stabilizers and logical operators | [`surface_code_d5.yaml`][surface]; [Surface-code decoding](../problems/surface_code.md) | paper experiments |
| exact cardinality (k of n) | subset selection | pair exchanges | [`toy_pair_exchanges.yaml`][toy], [`new_problem.py`][newproblem] | paper worked example |
| precedence (chains of implications) | scheduling with prerequisites | grow or shrink the selected prefix | [`precedence_small.yaml`][precedence] | tested |
| covering (at least one) | set cover, clauses | flip a literal while another is satisfied | [`covering_small.yaml`][covering] | tested |
| packing (at most one) | independent choices | flip a spin while the rest of the block is unselected | [`packing_small.yaml`][packing] | tested |
| contiguity (one interval or arc) | consecutive assignments | grow or shrink the interval at an end | [`blocks_small.yaml`][blocks] | tested |
| unconstrained (free spins) | spin glass | single spin flips | [`spin_glass.yaml`][spinglass] | tested |

**Status.** *Paper experiments*: used for the results of the paper. *Paper worked example*: the example that
introduces the machine in the paper. *Tested*: covered by the test suite and a runnable example.

## Combining constraints

Blocks of different families can be combined in one problem as long as they do not share spins. The objective is a
polynomial of any order over all spins. [`blocks_small.yaml`][blocks] combines one block of every family with a free
spin.

## Input formats

Problems built from exact cardinality, precedence, covering, packing, contiguity and free spins need no code. They
are given as a spin polynomial file (`nimbus.spinpoly/1`) or in the pseudo-Boolean OPB format, with contiguity given
in the configuration for OPB. See [Solving a new problem](../guides/new_problem/problem_file.md).
Permutation and parity problems have their own problem kinds, `matching` and `surface_code`
([Settings](../guides/settings.md#problem)).

## Adding a constraint family

Other constraint families need a move library of their own. If you know which operations keep the constraint, write
them as moves directly. For a problem small enough to list its feasible states, `group_transitions` finds the moves
from the feasible states alone, by grouping the transitions between them. [Your own moves](../guides/new_problem/own_moves.md)
describes both, the four conditions every move must meet, and how to make a new family available without code.
[Which moves the machine can use](moves.md) explains which moves qualify.

[matching]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/matching_small.yaml
[protocol]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/paper_protocol.py
[surface]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/surface_code_d5.yaml
[toy]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/toy_pair_exchanges.yaml
[newproblem]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/new_problem.py
[precedence]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/precedence_small.yaml
[covering]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/covering_small.yaml
[packing]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/packing_small.yaml
[blocks]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/blocks_small.yaml
[spinglass]: https://github.com/aimlab-wustl/Nimbus/blob/main/examples/spin_glass.yaml
