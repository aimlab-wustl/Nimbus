---
hide:
  - toc
extra_css:
  - assets/demo/matching.css
extra_js:
  - assets/demo/matching.js
---

# Graph matching

<div id="nimbus-matching-demo"></div>

## At a glance

<div class="nb-glance" markdown>
<div markdown>:material-image-outline: Given</div>
<div markdown>Two sets of \(n\) points: image 2 is image 1 rotated, scaled, perturbed by noise and relabelled.</div>
<div markdown>:material-target: Find</div>
<div markdown>The one-to-one matching of the points that maximizes the similarity of matched triangles.</div>
<div markdown>:material-dots-grid: Spins</div>
<div markdown>One per pair of points: \(x_{i,a} = 1\) when point \(i\) of image 1 is matched to point \(a\) of image 2; \(n^2\) spins.</div>
<div markdown>:material-lock-outline: Constraint</div>
<div markdown>Every point matched exactly once: a permutation. Kept by the moves.</div>
<div markdown>:material-chart-bell-curve: Energy</div>
<div markdown>Minus the sum of triangle similarities: third-order clauses.</div>
<div markdown>:material-swap-horizontal: Moves</div>
<div markdown>Exchange the targets of two points: all \(n(n-1)/2\) pairs, the complete library.</div>
<div markdown>:material-play-outline: Start</div>
<div markdown>A random matching.</div>
<div markdown>:material-thermometer: Run</div>
<div markdown>Metropolis acceptance; Fowler-Nordheim annealing between calibrated temperatures, \(1000\, n^2\) iterations.</div>
<div markdown>:material-flag-checkered: Output</div>
<div markdown>The matching, and how many matches are correct.</div>
</div>

## From start to end

<ol class="nb-timeline" markdown>
<li markdown>
**The instance.** Image 1 is a random set of points. Image 2 is the same set rotated, scaled, moved by a little noise and
listed in a random order, so the true matching is known but hidden from the machine.
</li>
<li markdown>
**Triangle similarity.** Every triangle of three points has a shape, described by the sines of its three angles, which
does not change under rotation or scaling. For each triangle of image 1 (up to \(n^2\) of them, chosen at random), the
instance keeps the \(3n\) triangles of
image 2 with the most similar shapes and gives each pair a similarity \(H = e^{-d^2/\gamma}\), with \(d\) the
distance between the shapes.
</li>
<li markdown>
**Spins, constraint and energy.** The matching is an \(n \times n\) matrix of spins with one 1 per row and per column.
The energy is minus the total similarity of the matched triangles: a sum of products of three spins, one per
corner of the triangle.
</li>
<li markdown>
**Moves.** A move exchanges the targets of two points, which keeps every point matched exactly once. The demo uses
all pairs: the backbone of exchanges of consecutive points together with every shortcut.
</li>
<li markdown>
**Calibration and annealing.** The start temperature accepts 80% of uphill moves at random matchings, the end
temperature 1% after a descent. A move is proposed at random and accepted with the Metropolis rule while the
temperature falls.
</li>
<li markdown>
**Result.** Solid lines are correct matches. With low noise the most similar matching is the true one; with more noise
it can differ from it, so a few lines may stay dashed even at the lowest energy.
</li>
</ol>

## In the architecture

!!! idea "Key idea"
    Each energy term links three matches, an interaction of order 3. NIMBUS stores it as one clause and never reduces
    it to pairwise couplings; the exchanges keep the permutation, so no penalty is needed.

??? math "The math"
    \[
    E = -\sum_{(i,j,k)} \sum_{(a,b,c)} H_{ijk,abc}\; x_{i,a}\, x_{j,b}\, x_{k,c} .
    \]

    With spins \(s = 2x - 1\), each product becomes a sum of clauses of order at most 3. An exchange of the targets of
    rows \(i\) and \(j\) maps every spin to another spin, so each switch is set by one product of spins; the compiler
    adds the products that are not yet stored as auxiliary clauses. The full problem, with real images and the
    benchmarks, is on the [Graph matching](../problems/graph_matching.md) page.

## Try this

- **Drag any point** of either image: the problem changes under the machine, and it continues from its current matching.
- Raise the **noise** and watch where the most similar matching stops being the true one.
- Increase the number of **points** to 15: 225 spins and 105 moves.
