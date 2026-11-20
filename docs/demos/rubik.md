---
hide:
  - toc
extra_css:
  - assets/demo/rubik.css
extra_js:
  - assets/demo/rubik.js
---

# Rubik's cube

<div id="nimbus-rubik-demo"></div>

## At a glance

<div class="nb-glance" markdown>
<div markdown>:material-cube-outline: Given</div>
<div markdown>A cube scrambled by 40 random face turns.</div>
<div markdown>:material-target: Find</div>
<div markdown>A solved cube: every face a single colour.</div>
<div markdown>:material-dots-grid: Spins</div>
<div markdown>One per sticker position and colour: \(x_{j,c} = 1\) when position \(j\) shows colour \(c\); \(54 \times 6\) spins.</div>
<div markdown>:material-lock-outline: Constraint</div>
<div markdown>The stickers form a cube that can be reached by turning. Kept by the moves, never written into the energy.</div>
<div markdown>:material-chart-bell-curve: Energy</div>
<div markdown>The number of stickers whose colour differs from the centre of their face: first-order clauses.</div>
<div markdown>:material-swap-horizontal: Moves</div>
<div markdown>Backbone: the 18 face turns. Shortcuts: about 3,000 short sequences that change only two to four pieces.</div>
<div markdown>:material-play-outline: Start</div>
<div markdown>The scrambled cube.</div>
<div markdown>:material-thermometer: Run</div>
<div markdown>Metropolis acceptance; Fowler-Nordheim schedule from temperature 3 to 0.25 over 300,000 iterations, restarted if needed.</div>
<div markdown>:material-flag-checkered: Output</div>
<div markdown>The solved cube, and how many moves of each kind were accepted.</div>
</div>

## From start to end

<ol class="nb-timeline" markdown>
<li markdown>
**The scramble.** Forty random face turns mix the cube. Every state the cube can reach this way is a valid cube; most
colourings of 54 stickers are not.
</li>
<li markdown>
**Spins and constraint.** Each of the 54 sticker positions shows one of six colours. Which colourings are valid is a
complicated condition, the reachable states of the cube, and NIMBUS never writes it down: it starts from a valid cube
and only makes moves the cube itself can make.
</li>
<li markdown>
**Energy.** The energy counts the stickers whose colour differs from the centre of their face. It is 0 for the solved
cube and only the solved cube.
</li>
<li markdown>
**Backbone.** The 18 face turns (U, U′, U2, …) reach every state of the cube, but each turn moves 20 stickers at once.
From a scrambled cube, runs of 3 million iterations with face turns alone came no closer than 13 wrong stickers.
</li>
<li markdown>
**Shortcuts.** Commutators, short sequences that change only a few pieces and restore everything else:

- a 3-cycle of three corners or three edges,
- a pair of twisted corners or a pair of flipped edges,
- a swap of two corners and two edges, which changes the permutation parity.

Each is conjugated by up to four setup turns, so that it acts on any group of pieces. Proposals pick a kind first (face
turns 30%, 3-cycles 20% each, twists, flips and swaps 10% each), then a move of that kind.
</li>
<li markdown>
**Annealing.** Uphill moves are accepted often at first, so the search can cross rough ground, and rarely at the end.
A run that ends unsolved restarts the schedule from where it stopped.
</li>
<li markdown>
**Result.** A 40-move scramble is solved in about 100,000 iterations. The counters under the cube show how the work
splits between face turns and the shortcuts.
</li>
</ol>

## In the architecture

!!! idea "Key idea"
    The cube is the purest case of "constraints as moves": the constraint is too complicated to write as an energy,
    but every legal move keeps it automatically. Shortcuts turn a hopeless landscape into a solvable one.

??? math "The math"
    With \(x_{j,c} \in \{0, 1\}\) and \(h(j)\) the colour of the centre of the face of position \(j\),

    \[
    E = \sum_{j} \bigl(1 - x_{j,h(j)}\bigr) ,
    \]

    a sum of first-order clauses. A move is a permutation \(\pi\) of the 54 positions and maps \(x_{j,c}\) to
    \(x_{\pi(j),c}\). Clause \(x_{j,c}\) changes exactly when the colour at \(j\) and the colour that arrives there
    differ, so each switch is set by the product of two spins, which the compiler stores as an auxiliary clause.

## Try this

- Press **Scramble** and watch a new cube being solved.
- Move the **speed** slider to the right to solve in about a second, or to the left to follow single moves.
- Compare the counters: most accepted moves near the end are 3-cycles, twists and flips, not face turns.
