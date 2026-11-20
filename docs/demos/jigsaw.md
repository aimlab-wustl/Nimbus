---
hide:
  - toc
extra_css:
  - assets/demo/jigsaw.css
extra_js:
  - assets/demo/jigsaw.js
---

# Jigsaw puzzle

<div id="nimbus-jigsaw-demo"></div>

## At a glance

<div class="nb-glance" markdown>
<div markdown>:material-image-outline: Given</div>
<div markdown>A picture, generated in your browser, cut into \(n \times n\) square tiles and shuffled.</div>
<div markdown>:material-target: Find</div>
<div markdown>The arrangement of the tiles that restores the picture.</div>
<div markdown>:material-dots-grid: Spins</div>
<div markdown>One per tile and slot: \(x_{t,p} = 1\) when tile \(t\) is in slot \(p\); \(n^4\) spins.</div>
<div markdown>:material-lock-outline: Constraint</div>
<div markdown>Every tile in exactly one slot, every slot holding one tile: a permutation. Kept by the moves.</div>
<div markdown>:material-chart-bell-curve: Energy</div>
<div markdown>The mismatch of the pixels along the edges of neighbouring tiles: second-order clauses.</div>
<div markdown>:material-swap-horizontal: Moves</div>
<div markdown>Backbone: exchanges of two tiles. Shortcuts: exchanges of two blocks of tiles and shifts of a block.</div>
<div markdown>:material-play-outline: Start</div>
<div markdown>A random arrangement.</div>
<div markdown>:material-thermometer: Run</div>
<div markdown>Metropolis acceptance; Fowler-Nordheim annealing between temperatures calibrated on the puzzle.</div>
<div markdown>:material-flag-checkered: Output</div>
<div markdown>The final arrangement, and how many tiles are in place.</div>
</div>

## From start to end

<ol class="nb-timeline" markdown>
<li markdown>
**The puzzle.** A landscape is painted from a random seed and cut into tiles. The machine never sees the picture as
a whole: for every ordered pair of tiles it receives only how badly their edges fit, side by side and one above the
other.
</li>
<li markdown>
**Spins and constraint.** The arrangement is an \(N \times N\) matrix of spins, \(N = n^2\), with exactly one 1 in
every row (tile) and every column (slot). The panel on the right of the demo shows this matrix; since tiles are
numbered by their position in the picture, the solution is the diagonal.
</li>
<li markdown>
**Energy.** For every pair of neighbouring slots \(p, q\) and tiles \(a, b\), the energy adds the edge mismatch
\(D(a, b)\) when tile \(a\) is in \(p\) and tile \(b\) in \(q\). The mismatch compares each edge line of pixels with
the line predicted from the two lines before it, on both sides. A correct arrangement has the lowest energy.
</li>
<li markdown>
**Moves.** Every move is a fixed permutation of the slots: exchange two tiles, exchange two blocks of up to
\(n/2 \times n/2\) tiles, or shift a block along its rows or columns, wrapping around. Each maps a complete arrangement
to another, so every state the machine visits uses each tile exactly once. Half of the proposals are exchanges of two
tiles, the backbone; the other half are block moves, the shortcuts that move whole solved regions at once.
</li>
<li markdown>
**Calibration.** The start temperature accepts 80% of uphill moves at random arrangements; the end temperature accepts
1% after a short descent.
</li>
<li markdown>
**Annealing.** A move is proposed at random and accepted with the Metropolis rule while the temperature falls, for
\(5000 N\) iterations (\(20000 N\) for \(8 \times 8\)). The counters show the iteration, the temperature, the energy
and the tiles in place.
</li>
<li markdown>
**Result.** At the end of the schedule the picture is restored, or nearly so; **Run again** reshuffles and solves
again.
</li>
</ol>

## In the architecture

!!! idea "Key idea"
    The constraint never enters the energy. Every move permutes the slots, so in terms of the spins it is a fixed
    permutation that maps each feasible state to another feasible state, and the energy holds only the edge mismatch.

??? math "The math"
    With \(x_{t,p} \in \{0, 1\}\),

    \[
    E = \sum_{(p, q)\ \text{neighbours}}\ \sum_{a \ne b} D_{pq}(a, b)\; x_{a,p}\, x_{b,q} ,
    \]

    where \(D_{pq}\) is the horizontal or the vertical mismatch. With spins \(s = 2x - 1\) each product
    \(x_{a,p} x_{b,q}\) becomes a sum of clauses of order at most 2, so the move neurons receive the energy change
    through the switched connections of these clauses. A slot permutation maps every spin to another spin, so each
    switch is set by one product of spins, and the compiler adds the products that are not yet stored as auxiliary
    clauses, as for the row exchanges of graph matching.

## Try this

- Choose **8 × 8** tiles: 64 tiles, 4,096 spins, and a longer schedule.
- Press **New picture** for another landscape, or **Shuffle** to restart from a new arrangement.
- Watch the spin matrix: as the tiles find their places, the selected spins gather on the diagonal.
