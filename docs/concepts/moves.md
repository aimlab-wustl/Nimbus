# Which moves the machine can use

Every switch of the machine is set by one stored clause output. This holds for every move in which **each spin after
the move is a product of spins before the move, up to sign**: whether a clause changes sign is then also one product
of spins, which the compiler stores as an auxiliary clause if no clause stores it yet. Four move types describe such
moves at any problem size.

## Four move types

| Move type | Each spin after the move | Example |
|---|---|---|
| `FixedMask` | the spin itself, reversed for a fixed set of spins | a stabilizer of a code |
| `SignedPermutation` | another spin, possibly reversed: \(s_i \to \pm s_{g(i)}\) | exchanging two beads; exchanging two rows of a permutation matrix |
| `ConditionalMask` | the spin, reversed for a fixed set of spins when a product of other spins has a given sign | growing a chain by one |
| `AffineMove` | any product of spins, with a sign: \(s_i \to \pm \prod_{j \in A_i} s_j\) | reversing \(s_1\) where \(s_3 = -1\) and \(s_2\) where \(s_3 = +1\) |

`AffineMove` contains the other three; `group_transitions` returns it for a move found by grouping that has none of
the first three forms.

## Moves that are not their own inverse

Rotating the colours of the [bead example](derivation.md), so that red takes the positions of
green, green those of blue and blue those of red, is a valid move, but applying it twice does not return to the start.
Its reverse, the rotation the other way, is a separate move neuron. Both neurons are connected to the same clauses,
through different switches: the connection from (red in \(p_1\)) is switched by (blue in \(p_1\)) for one rotation
and by (green in \(p_1\)) for the other. Their inputs and destinations differ too. Include both directions with the
same proposal weight, so that every transition is proposed as often as its reverse: `library.with_reverses()` adds
missing reverses, `row_cycles` builds the cyclic exchanges of three rows of a permutation matrix in both directions,
and the compiler refuses a move whose reverse is missing. Rotations keep the parity of the arrangement and reach only
half of the states, so they serve as shortcuts on top of exchanges. For all cyclic exchanges of three rows of an
\(L \times L\) permutation matrix under a quadratic objective, the auxiliary clauses grow as \(L^5/2\).

## Moves reduced by identities

For a particular problem, a move outside this class still
works when relations that hold on every feasible state (identities, such as one \(+1\) per row or column of a
permutation matrix) reduce each switch to one stored product. The same relations already save auxiliary clauses for
the moves of this class: in the bead example, exchanging red and green changes (red in \(p_1\)) when
\(s_1 s_4 = -1\), and because \(s_1 s_4 s_7 = +1\) in every feasible state, the switch reads the stored clause
\(s_7\). Write such a move in the form it takes on the feasible states and return the identities from the problem's
`identities()`; the compiler then reduces every switch with them. `group_transitions` tests moves on the feasible
states and derives the identities itself, so it finds such moves too.

## Moves that are not supported

A move whose switch needs more than one clause output is not supported. For example, when
\(k\) vertices of a cycle are selected and no two may be adjacent, moving a
selected vertex one step is allowed only when the vertex it moves next to is also unselected. The move depends on two
spins at once, through an AND, which no single product of spins expresses, and `group_transitions` reports such
transitions as ungrouped.

## Known structure first

When the structure of a problem shows which operations keep its constraints, as for the
row exchanges of graph matching and the stabilizers of the surface code, write those moves directly with the types
above, at any problem size. Grouping the transitions is needed only when no such structure is known
([Your own moves](../guides/new_problem/own_moves.md)).
