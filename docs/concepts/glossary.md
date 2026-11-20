# Glossary

Hover over an underlined term anywhere in the documentation to see its definition. The terms of the architecture are introduced one by one in [Derive it step by step](derivation.md).

Spin
:   A variable \(s_i \in \{-1, +1\}\). In constraint problems +1 means selected (or true).

Clause
:   A product of spins, \(T_k(\mathbf{s}) = \prod_{i \in S_k} s_i\). The energy is a weighted sum of clauses, of any order.

Clause output
:   The sign, +1 or −1, of a stored clause.

Clause neuron
:   Outer-layer neuron that stores the sign of one clause. The spins themselves are stored as one-spin clauses.

Spin transition
:   A change from one feasible state to another. A clause changes sign exactly when an odd number of its spins change.

Move
:   A family of multi-spin transitions that maps every feasible state to another feasible state, for example the
    exchange of two rows of a permutation matrix.

Reversible move
:   A move that returns to the start state when applied twice, so one move neuron serves both directions. A move that
    is not, such as a cyclic exchange of three rows, is used together with its reverse as a separate move neuron.

Affine move
:   A move in which every spin after the move is a product of spins before it, up to sign. Every switch of such a
    move is set by one clause output.

Move neuron
:   Inner-layer neuron of one move, also called a transition neuron. Its input is the energy change of its move; when
    it fires, every clause on its active connections toggles.

Move library
:   The set of moves the machine can propose for a problem.

Backbone
:   A set of moves that connects all feasible states, so that every feasible state can be reached.

Shortcut
:   A move added to a backbone. It joins in one transition two states that the backbone connects only through several,
    so the search can bypass energy barriers, at the cost of more connections and auxiliary clauses.

Switch
:   The on/off state of one connection between a clause neuron and a move neuron. A switch is fixed or set by the
    sign of one stored clause, so each connection needs one switch controlled by one clause neuron.

Gate clause
:   A stored clause whose sign sets switches, also called a switch-control clause or auxiliary clause. The compiler
    adds one when a switch needs a product of spins that is not yet stored; its energy weight is zero.

Encoder
:   The switched, weighted connections from the clause outputs to the move neurons. They give each move neuron half
    the energy change of its move.

Decoder (of the architecture)
:   The switched, unweighted connections from a firing move neuron back to the clause outputs it toggles. (The
    surface-code pages also use "decoder" for the decoding application.)

FN annealer
:   The source of the stochastic firing threshold; its temperature follows the Fowler-Nordheim schedule.

Self-loop
:   An accepted move that changes nothing in the current state, because all its switches are off.

Feasible state
:   A state that satisfies every constraint. Since every move maps feasible states to feasible states, the machine
    visits only feasible states.

Compiler
:   Turns the objective, the constraints and the move library into the wiring of the machine: the stored clauses,
    the connections and the switch of every connection.

Replica
:   One independent run of the machine on the same problem, with its own random stream.

Parallel tempering
:   Replicas at different temperatures that exchange their states, so that hot replicas help cold ones escape local
    minima.

Fowler-Nordheim schedule
:   The inverse-logarithmic annealing schedule \(\tau_n = \tau_0 / \ln(1 + q_n)\), named after Fowler-Nordheim
    tunnelling; the schedule of the experiments in this repository ([Settings](../guides/settings.md#annealing-schedule)).

Iterations to solution (ITS)
:   With a per-replica success fraction p at a budget of B iterations, \(\mathrm{ITS}(B) = B \ln(0.01) / \ln(1 - p)\): the expected
    number of iterations to reach the target with probability 0.99 by independent repetitions.
