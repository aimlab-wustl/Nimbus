# Your own moves


When no builder keeps your constraint, the [problem adapter](adapter.md) returns moves that you provide. Any move is accepted
that satisfies four conditions:

1. **Feasible.** It maps every feasible state to a feasible state, or leaves it unchanged (a self-loop).
2. **Reversible.** Applied twice, it returns to the start state, so one move neuron serves both directions. A move
   that is not, such as a cyclic exchange of three rows, is used together with its reverse as a separate move with
   the same proposal weight; `library.with_reverses()` adds the reverses.
3. **Connected.** Together, the moves of the library lead from every feasible state to every other (the backbone).
4. **One clause per switch.** For every spin, whether the move flips it is given, on the feasible states, by one
   product of spins with a fixed sign. The compiler then sets every switch from one stored clause, and adds that
   product as an auxiliary clause when no clause stores it.

`verify_library` checks conditions 1 to 3 and the compiler checks condition 4. There are two ways to obtain the
moves.

## From the structure of the problem

If you know which operations keep the constraint, write them directly. The built-in problems do this: graph
matching uses exchanges of two rows of the assignment matrix, and surface-code decoding uses the stabilizers and
logical operators as fixed flips. Four move types describe such operations at any problem size:

| Move type | Operation | Example |
|---|---|---|
| `FixedMask(spins)` | flip a fixed set of spins | a stabilizer of a code |
| `ConditionalMask(spins, cond, pol)` | flip a fixed set of spins when the product of the spins `cond`, times `pol`, is +1 | grow a chain by one: flip spin `i` when its neighbors differ |
| `SignedPermutation(mapping)` | move every spin to another position, optionally reversing its sign | exchange two rows of a one-hot matrix |
| `AffineMove(mapping)` | set every spin to a product of spins, with a sign | reverse `s0` where `s2 = -1` and `s1` where `s2 = +1` |

```python
from nimbus.transitions.descriptors import AffineMove, ConditionalMask, SignedPermutation
from nimbus.transitions.library import TransitionLibrary

swap_rows = SignedPermutation([(0, 3, 1), (3, 0, 1), (1, 4, 1), (4, 1, 1), (2, 5, 1), (5, 2, 1)])
grow = ConditionalMask([2], cond=[1, 3], pol=-1)          # flip s2 when s1 s3 = -1
two = AffineMove([(0, [0, 2], 1), (1, [1, 2], -1)])       # s0 <- s0 s2, s1 <- -s1 s2
library = TransitionLibrary(9, [swap_rows, grow, two], ["rows 0-1", "grow at 2", "two flips"])
```

These moves meet condition 4 by construction. An operation that has none of these forms in general may still have
one on the feasible states, because of relations that hold there (identities, such as one +1 per row). Write it in
the form it takes on the feasible states and return the identities from `identities()`, so that the compiler reduces
every switch with them. Check conditions 1 to 3 on a small instance as in [step 5 of the adapter](adapter.md#5-verify-the-library-and-the-compiled-machine).
[Which moves the machine can use](../../concepts/moves.md) explains the four types,
moves that are not their own inverse and the moves that are not supported.

## Found by grouping the transitions

Grouping is needed only when the structure is not known. The moves can then be found from the feasible states
alone, for problems small enough to list them. `group_transitions` carries out the derivation of [Derive it step by step](../../concepts/derivation.md)
automatically:

1. The **spin transitions** are the pairs of feasible states that differ in the fewest spins (or in at most
   `max_distance` spins).
2. Transitions that one spin transformation reproduces are collected into a **move**. A transformation is accepted
   only if it meets conditions 1, 2 and 4 and produces no other transitions.
3. Moves whose transitions connect all feasible states form the **backbone**; the others are shortcut candidates.

```python
from nimbus.transitions.grouping import group_transitions

grouping = group_transitions(problem.enumerate_feasible())      # mode="all" lists every move
print(grouping.summary())
library = grouping.library("backbone")                         # or "all", or a list of move indices
model = compile_model(problem.structure(), library, identities=grouping.identities)
```

Each move is returned as one of the four move types [above](#from-the-structure-of-the-problem) when it has that form on the feasible states, so it
compiles for any number of states, and as an explicit table otherwise (`TableTransition`, at most 65,536 feasible
states). The moves found are their own inverse; add cyclic moves and their reverses by hand. Return the library from the
adapter's `transition_library`; nothing else in the [adapter](adapter.md) changes.

[`examples/find_moves.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/find_moves.py) applies it to
Latin squares of order 3, a constraint for which the package has no builder: every row and every column holds each
of three symbols once. Grouping the 54 transitions between the 12 squares gives nine moves, each a permutation of the
spins such as the exchange of two symbols, and two of them form a backbone that connects all squares. On the bead
example of the concepts pages, `mode="all"` finds the three exchanges of two colours and the three exchanges of two
positions.

**Discover on a small instance, then write the move for any size.** A move found as a signed permutation often has
an obvious meaning, such as exchanging two symbols. Write that operation for the size you need, as [above](#from-the-structure-of-the-problem).

**When no move is found.** Some constraints have transitions that no move of this form contains. For example, when
k vertices of a cycle are selected and no two may be adjacent, moving a selected vertex one step is allowed only
when the vertex it moves next to is also unselected, and no single product of spins expresses that condition. `group_transitions`
reports such transitions as ungrouped instead of forcing them into moves: such a move would need a switch set by
more than one clause output ([Which moves the machine can use](../../concepts/moves.md)).

**How it works.** The transitions are found by hashing: states that differ in at most d spins agree exactly on one of
d + 1 blocks of spins, so only states sharing a block are compared. Fixed and conditional flips then follow from one
system of linear equations per flip pattern; permutations from a constraint search in which every feasible state keeps
a short list of possible destinations; and moves that combine two of these from their composition, since such moves
are closed under composition. A bounded search over single states covers what remains. On a laptop, grouping the
411,840 transitions between the 12,870 selections of 8 out of 16 spins, or the 52,920 transitions between the 5,040
permutations of seven items, takes about a second.

## Optional: a new family without code

To let others use a new constraint family from a spinpoly or OPB file, as in [a problem file](problem_file.md), add it to
`nimbus.problems.constrained_blocks.ConstrainedBlocksProblem`: a constraint field, its feasibility check, its random
feasible start, its move builder and its penalty form (for the baselines). Then extend the spinpoly and OPB readers
in `nimbus.problems.formats`. The existing families follow exactly this pattern.
