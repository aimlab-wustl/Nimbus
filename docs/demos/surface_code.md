---
hide:
  - toc
extra_css:
  - assets/demo/surface.css
extra_js:
  - assets/demo/surface.js
---

# Surface-code decoding

<div id="nimbus-surface-demo"></div>

## At a glance

<div class="nb-glance" markdown>
<div markdown>:material-grid: Given</div>
<div markdown>The syndrome of a rotated surface code of distance \(d\): which checks are violated by unknown errors.</div>
<div markdown>:material-target: Find</div>
<div markdown>The logical class of the correction that most likely undoes the errors.</div>
<div markdown>:material-dots-grid: Spins</div>
<div markdown>Two per data qubit: the \(X\) part \(x_i\) and the \(Z\) part \(z_i\) of the correction; \(2d^2\) spins.</div>
<div markdown>:material-lock-outline: Constraint</div>
<div markdown>The correction reproduces the syndrome: one parity check per stabilizer. Kept by the moves.</div>
<div markdown>:material-chart-bell-curve: Energy</div>
<div markdown>\(\omega\) times the number of qubits the correction touches: first- and second-order clauses.</div>
<div markdown>:material-swap-horizontal: Moves</div>
<div markdown>Every stabilizer (keeps the logical class) and the two logical operators (change it).</div>
<div markdown>:material-play-outline: Start</div>
<div markdown>Any correction with the right syndrome, found by Gaussian elimination.</div>
<div markdown>:material-thermometer: Run</div>
<div markdown>Sampling at temperature 1 with parallel tempering: 12 replicas from \(\tau = 1\) to 10, 4,000 sweeps.</div>
<div markdown>:material-flag-checkered: Output</div>
<div markdown>The most visited logical class, its share of the visits as the confidence, and whether it corrects the error.</div>
</div>

## From start to end

<ol class="nb-timeline" markdown>
<li markdown>
**Errors and syndrome.** Each data qubit suffers an \(X\), \(Y\) or \(Z\) error with probability \(p\) (red letters),
or you place errors by clicking. The decoder does not see them, only the checks they violate (amber): \(Z\) checks
detect \(X\) errors and \(X\) checks detect \(Z\) errors.
</li>
<li markdown>
**What to find.** Many corrections reproduce the syndrome. Two corrections that differ by a stabilizer act the same on
the encoded qubit; corrections that differ by a logical operator do not. So the decoder must pick one of four logical
classes, I, X, Z or Y, and the best choice is the class with the largest total probability: maximum-likelihood
decoding.
</li>
<li markdown>
**Constraint and moves.** A stabilizer flips the correction on its qubits without changing the syndrome, and so do
the two logical operators, an \(X\) string down a column and a \(Z\) string along a row. Starting from one valid
correction, these moves visit every correction with the same syndrome, and no other.
</li>
<li markdown>
**Energy.** For depolarizing noise, a correction that touches \(w\) qubits has probability proportional to
\((p/3)^w (1-p)^{d^2 - w}\). The energy \(\omega w\), with \(\omega = \ln(3(1-p)/p)\), makes sampling at temperature
1 visit every correction with exactly this probability.
</li>
<li markdown>
**Parallel tempering.** Changing the class costs a whole logical string, a large energy barrier. Twelve replicas run
at temperatures from 1 to 10 and regularly exchange their states, so the replica at temperature 1 keeps receiving
corrections from other classes.
</li>
<li markdown>
**Decision.** After 200 sweeps of burn-in, the demo counts the class of the replica at temperature 1 after every
sweep. The bars show the counts; the decision is the most visited class and the confidence is its share. Red marks the
class of the actual error: the decoder succeeds when the two agree.
</li>
</ol>

## In the architecture

!!! idea "Key idea"
    Here the machine is a sampler, not an optimizer: it stays at temperature 1 and reads the answer from how often it
    visits each class. Every move is a fixed set of spin flips, so every switch is fixed and no auxiliary clause is
    needed.

??? math "The math"
    With bits \(x_i, z_i \in \{0, 1\}\) and \(H_X, H_Z\) the check matrices, the constraint is
    \(H_Z x = \sigma_Z,\ H_X z = \sigma_X\) and the energy is

    \[
    E = \omega \sum_i (x_i \lor z_i) = \frac{\omega}{4} \sum_i \bigl(3 - s^x_i - s^z_i - s^x_i s^z_i\bigr),
    \qquad s = 1 - 2b ,
    \]

    first-order clauses and one second-order clause per qubit, which captures that a \(Y\) error is a single error.
    A stabilizer or logical move flips a fixed set of spins, so it toggles a clause exactly when the clause contains
    an odd number of those spins, in every state. The decision \(\arg\max_\gamma \hat P(\gamma)\) estimates the class
    with the lowest free energy, \(F_\gamma = -\ln \sum_{c \in \gamma} e^{-E(c)}\).

## Try this

- Raise the **error rate** to 0.15 and press **New errors** a few times: the confidence drops and some decisions fail.
- **Click qubits** to place a string of errors across the lattice: a long enough chain makes the decoder choose the
  wrong class.
- Switch to **distance 7**: at the same error rate, fewer decisions fail than at distance 3.
- Untick **show the errors** to see only what the decoder sees.

The [FPGA decoders](../hardware/index.md) run this scheme in hardware, and the
[Surface-code decoding](../problems/surface_code.md) page describes the decoders of the package.
