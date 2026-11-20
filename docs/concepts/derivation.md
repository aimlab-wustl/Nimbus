---
hide:
  - toc
extra_css:
  - assets/demo/derive.css
extra_js:
  - assets/demo/derive.js
---

# Derive it step by step

Eight short steps take the bead example from a problem statement to the running machine. Each step introduces one
idea, gives you something to try, and ends with the key idea to remember. The formal derivation is folded into
**The math** boxes for readers who want it.

<nav class="nd-stepper" id="nd-stepper" data-s1="Problem" data-s2="Transitions" data-s3="Connect" data-s4="Moves"
data-s5="Switches" data-s6="Auxiliary" data-s7="Encoder" data-s8="Machine" aria-label="Steps"></nav>

## 1. A problem with a constraint

Three beads, red, green and blue, sit in a row of three positions \(p_1, p_2, p_3\), one bead per position. Each
arrangement has an energy, a sum of statements that are \(+1\) when true and \(-1\) when false:

\[
E = -(\text{red in } p_1) - 2\,(\text{blue in } p_3) - (\text{green in } p_1)(\text{blue in } p_2) - 2\,(\text{red in } p_3).
\]

The machine works with **spins**, one per colour and position (\(+1\) if that colour is in that position). Each
bracket, or product of brackets, is a **clause**; its value, \(+1\) or \(-1\), is the **clause output**. Swap beads,
or flip a single spin, and watch the energy and the constraint.

<div class="nd-widget" id="nd-problem" markdown></div>

!!! idea "Key idea"
    A swap of two beads keeps the arrangement valid; a single spin flip never does. A machine that flips one spin at
    a time needs penalty terms; NIMBUS changes several spins at once and stays on the valid arrangements, the
    **feasible states**.

??? math "The math"
    With \(s_{3(x-1)+a} = +1\) if colour \(x\) is in position \(p_a\) (red, green, blue \(= 1, 2, 3\)),

    \[
    E(\mathbf{s}) = -s_1 - 2 s_9 - s_4 s_8 - 2 s_3 = -\sum_{k=1}^{4} J_k T_k ,\qquad
    T_1 = s_1,\ T_2 = s_9,\ T_3 = s_4 s_8,\ T_4 = s_3,\ J = (1, 2, 1, 2),
    \]

    subject to \(s_1 + s_2 + s_3 = s_4 + s_5 + s_6 = s_7 + s_8 + s_9 = -1\) and
    \(s_1 + s_4 + s_7 = s_2 + s_5 + s_8 = s_3 + s_6 + s_9 = -1\). A clause of \(\ell\) spins is an interaction of
    order \(\ell\), so the same form covers interactions of any order.

## 2. Spin transitions

There are six arrangements, numbered 1 to 6. Going from one to another is a **spin transition**. A clause
changes sign exactly when an odd number of its spins change, so the energy change of a transition can be computed
from the clause outputs alone. Pick two states.

<div class="nd-widget" id="nd-pairs" markdown></div>

!!! idea "Key idea"
    The energy change of any transition is a weighted sum of the clause outputs it toggles. A neuron can compute it
    from clause outputs, without seeing the spins.

??? math "The math"
    The **switch variable** of clause \(k\) for the transition \(\boldsymbol\mu \to \boldsymbol\nu\) is

    \[
    \alpha_{k,\boldsymbol\mu\leftrightarrow\boldsymbol\nu} = \frac{1 - T_k(\boldsymbol\nu)\,T_k(\boldsymbol\mu)}{2} \in \{0, 1\},
    \qquad
    \Delta E_{\boldsymbol\mu \to \boldsymbol\nu} = 2 \sum_k J_k\, \alpha_{k,\boldsymbol\mu\leftrightarrow\boldsymbol\nu}\, T_k(\boldsymbol\mu) .
    \]

    \(\alpha\) is the same in both directions; \(\Delta E\) changes sign.

## 3. Connecting the states

The machine must be able to reach every feasible state, but it does not need all 15 transitions between the six
states: five that form a tree suffice. Compare two trees.

<div class="nd-widget" id="nd-trees" markdown></div>

!!! idea "Key idea"
    Connect the feasible states with the smallest changes first. Tree 2 uses only four-spin transitions, and every
    longer transition is a sequence of them.

## 4. Moves

One neuron per transition would not scale: the number of states grows exponentially with the problem. But
transitions repeat the same operation. \(1 \leftrightarrow 2\), \(3 \leftrightarrow 4\) and \(5 \leftrightarrow 6\)
all swap green and blue. Such an operation is a **move**, and each move becomes one **move neuron**. The two swaps of
tree 2, red ↔ green (\(G_1\)) and green ↔ blue (\(G_2\)), reach every state: together they form the **backbone**.
A further move, red ↔ blue (\(G_3\)), is a **shortcut**.

Start at state 6 and apply moves until you reach the optimum, state 1. Then add the shortcut and try again.

<div class="nd-widget" id="nd-moves" markdown></div>

!!! idea "Key idea"
    A move is one neuron for a whole family of transitions. The backbone guarantees that every state can be reached;
    shortcuts jump over energy barriers, here from \(E = 0\) straight to the optimum instead of through \(E = 4\).

??? math "The math"
    On the spins, the two backbone moves are the permutations

    \[
    \begin{aligned}
    G_1 &: (s_1, \ldots, s_9) \to (s_4, s_5, s_6, s_1, s_2, s_3, s_7, s_8, s_9), \\
    G_2 &: (s_1, \ldots, s_9) \to (s_1, s_2, s_3, s_7, s_8, s_9, s_4, s_5, s_6),
    \end{aligned}
    \]

    which give the transitions \(G_1\): \(1 \leftrightarrow 3,\ 2 \leftrightarrow 5,\ 4 \leftrightarrow 6\) and
    \(G_2\): \(1 \leftrightarrow 2,\ 3 \leftrightarrow 4,\ 5 \leftrightarrow 6\). The shortcut \(G_3\) gives
    \(1 \leftrightarrow 6,\ 2 \leftrightarrow 4,\ 3 \leftrightarrow 5\).

## 5. Switches

A move neuron must receive the energy change of its move from the current state, but which clauses a move toggles
depends on the state. Swapping red and green toggles (red in \(p_1\)) only if blue is not in \(p_1\): when blue is
there, neither red nor green enters or leaves \(p_1\). Look at the transitions of each move.

<div class="nd-widget" id="nd-switches" markdown></div>

!!! idea "Key idea"
    Each connection from a clause to a move neuron needs a **switch** that is on exactly when the move toggles that
    clause from the current state. Here every switch is set by one spin, the position of the bead the move leaves
    alone.

??? math "The math"
    \[
    \begin{array}{c|cc}
    k & \alpha_{k,G_1} & \alpha_{k,G_2} \\ \hline
    1 & (1 - s_7)/2 & 0 \\
    2 & 0 & (1 - s_3)/2 \\
    3 & (1 - s_7)/2 & (1 - s_3)/2 \\
    4 & (1 - s_9)/2 & 0
    \end{array}
    \]

## 6. Auxiliary clauses

The machine stores clause outputs, not spins, so every switch must be read from a stored clause output. Two of the
spins the switches need are already clause outputs. One is not.

<div class="nd-widget" id="nd-aux" markdown></div>

!!! idea "Key idea"
    When a switch needs a product of spins that no clause stores, the compiler appends it as an **auxiliary clause**
    with weight 0. Every switch is then set by one stored clause output: one switch per connection, one clause per
    switch.

??? math "The math"
    For moves that map each spin to a product of spins up to sign, every state-dependent switch has the form
    \(\alpha_{k,G_r} = (1 \pm T_j)/2\) for one clause \(T_j\). If \(T_j\) is not an objective clause, it is appended
    with \(J_j = 0\), and the compiler repeats this until the clause outputs supply every switch. The shortcut
    \(G_3\) needs five more auxiliary clauses in this example: the three positions of green, blue in \(p_2\) and red
    in \(p_2\).
    [Which moves the machine can use](moves.md) lists the moves of this kind,
    including moves that are not their own inverse.

## 7. Encoder and decoder

Put the switches into the energy change and each move neuron receives a weighted sum of clause outputs: the
**encoder**. The neuron fires when this input is below a random threshold from the **FN annealer**, and the firing
neuron toggles the clause outputs through the same switches, without the weights: the **decoder**. The toggled
outputs are the stored outputs of the next update, as in an autoencoder whose output becomes its next input; in
the machine below, encoder and decoder share the same connections, so both are drawn as one set of lines.

!!! idea "Key idea"
    The encoder computes every move's energy change in parallel from the stored clause outputs; the decoder applies
    the accepted move by toggling clause outputs. The spins are never read or written directly.

??? math "The math"
    \[
    v_{G_r} = \sum_k J_k\, \alpha_{k,G_r}\, T_k = \tfrac{1}{2}\,\Delta E_{G_r}, \qquad
    q_r = \begin{cases} 1 & v_{G_r} \le \xi_r \\ 0 & \text{otherwise} \end{cases}, \qquad
    \chi_k = \sum_r \alpha_{k,G_r}\, q_r, \qquad T_k \leftarrow (1 - 2\chi_k)\, T_k .
    \]

    For the backbone,
    \(v_{G_1} = \tfrac{1 - T_5}{2}(T_1 + T_3) + 2\,\tfrac{1 - T_2}{2}\, T_4\) and
    \(v_{G_2} = \tfrac{1 - T_4}{2}(2 T_2 + T_3)\). In state 4 these are \(-2\) and \(-1\); if \(G_2\) fires, the
    decoder toggles \(T_2\), \(T_3\) and \(T_5\), and the machine is in state 3. At most one move neuron fires at a
    time, and \(\xi\) makes the acceptance probability \(\min(1, e^{-\Delta E/\tau})\) at temperature \(\tau\).

## 8. The machine

Everything together: the derived network on the bead example. Every connection is drawn with its switch, and bright
connections are on in the current state. The encoder reads only the weighted connections; the zero-weight connections
of the auxiliary clauses, marked with one-way arrows, are used only by the decoder, and the thin red lines from each clause neuron to the
switches it sets appear when the switches are updated. Step through one update at a time, raise or lower the temperature, and add the shortcut to see the switches and
clauses it costs. The button at the top right opens it full screen.

<div class="nd-widget" id="nd-machine" markdown></div>

!!! idea "Key idea"
    Only the firing neuron and the clauses it toggles produce events (activity sparsity), and the connections that
    are on change with every accepted move (connectivity sparsity).

## What the compiler does for you

For a real problem the compiler carries out steps 4 to 7 automatically: from the objective, the constraints and a
move library, it derives every switch, appends the auxiliary clauses and loads the encoder and the decoder. When the
feasible states can be listed, `group_transitions` also carries out steps 2 to 4: it finds the transitions, groups
them into moves and selects a backbone ([Your own moves](../guides/new_problem/own_moves.md)). For
parity, cardinality, one-hot, partition and permutation constraints with objectives of bounded order, the number of
auxiliary clauses grows only polynomially with the problem size. Next: [Supported constraints](constraints.md) and
[Solving a new problem](../guides/new_problem/index.md).
