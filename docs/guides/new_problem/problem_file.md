# A problem file (no code)


If the objective is a polynomial of any order and the constraints are disjoint blocks of the supported families,
write the problem as a file and point a configuration at it.

| Family | Meaning | spinpoly field | OPB form | Moves built for it |
|---|---|---|---|---|
| exact cardinality | exactly k of the block selected | `exact_cardinality: [{spins: [...], k: K}]` | `+1 x1 +1 x2 ... = K` | pair exchanges |
| precedence | x_b selected requires x_a selected, along a chain | `chains: [[a, b, c, ...]]` | `+1 xa -1 xb >= 0` (chained) | grow or shrink the selected prefix |
| covering | at least one literal true | `or_blocks: [{spins: [...], signs: [+1/-1, ...]}]` | `+1 l1 +1 l2 ... >= 1` (`l` = `x` or `~x`) | flip a literal while a neighboring literal is satisfied |
| packing | at most one of the block selected | `amo_blocks: [[...]]` | `-1 x1 -1 x2 ... >= -1` | flip a spin while the rest of the block is unselected |
| contiguity | the selected spins form one interval (or one arc of a cycle) | `contiguity: [{spins: [...], cyclic: false}]` | not expressible; use the configuration field `problem.contiguity` | grow or shrink the interval at an end |
| unconstrained | free spins | `free: [...]` (default: every spin in no block) | variables in no constraint | single spin flips |

Blocks must not share spins. The spin convention is s = +1 for a selected (true) variable, x = (1 + s)/2.

**spinpoly** (`nimbus.spinpoly/1`, YAML or JSON): the objective is a list of products of spins with weights. With
`energy_convention: minus_J` the energy is E(s) = const - sum_k J_k prod_{i in S_k} s_i (with `plus_c`:
E = const + sum_k c_k prod s). Spin indices are 0-based unless `index_base: 1`. A small example that selects 2 of
4 spins and rewards the pairs (0, 1) and (2, 3) differently, with one free spin:

```yaml
schema: nimbus.spinpoly/1
name: pick_two
n_spins: 5
spin_convention: plus_one_selected
energy_convention: minus_J
const: 0
terms:
  - {support: [0, 1], weight: 1.0}
  - {support: [2, 3], weight: 2.0}
  - {support: [0, 2, 4], weight: 0.5}
constraints:
  exact_cardinality: [{spins: [0, 1, 2, 3], k: 2}]
  free: [4]
```

Save it as `pick_two.yaml` next to a run configuration with

```yaml
problem: {kind: spinpoly, path: pick_two.yaml}
transitions: {library: default}
```

and the rest as in `examples/spin_glass.yaml`. Every family in one file: `examples/problems/blocks_small.spinpoly.yaml`
(run it with `nimbus solve examples/blocks_small.yaml`); generated input: `examples/make_spin_glass.py`.

**OPB** (the pseudo-Boolean competition format, PB24 subset): `min:` objective (products of literals allowed), then
constraints with `>=` or `=` only (write `sum x <= K` as `-1 x1 -1 x2 ... >= -K`). Variable `x<k>` is spin k - 1.
Examples: `examples/problems/{covering,packing,precedence}_small.opb`. A constraint that is not one of the forms
above is reported with its line, never dropped.
