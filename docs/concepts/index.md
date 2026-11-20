---
hide:
  - toc
extra_css:
  - assets/demo/derive.css
extra_js:
  - assets/demo/derive.js
---

# How NIMBUS works

NIMBUS solves constrained problems with a network of neurons that only ever visits valid solutions. Below it runs on
a small example: three beads, red, green and blue, to be arranged in a row so that the energy is lowest.

<div class="nd-widget" id="nd-machine" data-autoplay="true" markdown></div>

<p class="nd-legend">Left: the <b>clause neurons</b>, each storing the sign of a product of spins such as (red in p₁).
Right: one <b>move neuron</b> per move, here “swap red and green” and “swap green and blue”. Every connection carries
a <b>switch</b> (a small lever, closed when on); bright connections are on in the current state. In the switch stage,
thin red lines show which clause neuron sets each switch. The connection of the zero-weight auxiliary clause, marked with a
one-way arrow, feeds nothing to the encoder and is used only by the decoder. Tick the shortcut to add a third move.</p>

## The idea in three parts

<div class="grid cards" markdown>

-   :material-swap-horizontal-bold:{ .lg .middle } __Moves, not penalties__

    ---

    A move changes several spins at once, such as swapping two beads, and always lands on another valid state. The
    energy needs no penalty terms, and the machine never crosses invalid states.

    [:octicons-arrow-right-24: Steps 1–4](derivation.md#1-a-problem-with-a-constraint)

-   :material-electric-switch:{ .lg .middle } __Switched connections__

    ---

    One neuron serves a whole move. Its connections carry switches that the stored clause outputs turn on or off, so
    the neuron always receives the energy change of the move from the current state.

    [:octicons-arrow-right-24: Steps 5–6](derivation.md#5-switches)

-   :material-cog-transfer-outline:{ .lg .middle } __A compiler builds the wiring__

    ---

    From the objective, the constraints and a set of moves, the compiler derives every switch and adds the auxiliary
    clauses the switches need. This works for interactions of any order.

    [:octicons-arrow-right-24: Steps 6–7](derivation.md#6-auxiliary-clauses)

</div>

## One update, in four events

1. **Encoder.** A move neuron sums the weights of the clause outputs whose switch is on. The sum is half the energy
   change its move would make.
2. **Threshold.** The neuron fires if this input is below a random threshold set by the annealer, which accepts the
   move with the Metropolis probability at the current temperature.
3. **Decoder.** The firing neuron toggles the clause outputs on its active connections, through the same switches.
4. **Feedback.** The new clause outputs set new switches, and the next move neuron is proposed.

Only the firing neuron and the clauses it toggles produce events, and the active connections change with every move.

## Backbone and shortcuts

A set of moves that can reach every valid state is a **backbone**. Extra moves, **shortcuts**, join distant states in
one transition and let the search bypass energy barriers, at the cost of more connections and clauses. In the
example, the backbone needs three transitions, over an energy barrier, to go from state 6 to the optimum; the shortcut
needs one.

## Learn more

<div class="grid cards" markdown>

-   :material-stairs:{ .lg .middle } __Derive it step by step__

    ---

    Eight short steps on the bead example, each with something to try: from the problem to the running machine.

    [:octicons-arrow-right-24: Start the tutorial](derivation.md)

-   :material-format-list-checks:{ .lg .middle } __Supported constraints__

    ---

    The constraint families that have move libraries, and how to combine them.

    [:octicons-arrow-right-24: Supported constraints](constraints.md)

-   :material-book-open-variant:{ .lg .middle } __Glossary__

    ---

    Every term in one place; hover over an underlined term anywhere on the site for its definition.

    [:octicons-arrow-right-24: Glossary](glossary.md)

</div>
