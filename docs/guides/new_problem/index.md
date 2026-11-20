# Solving a new problem

The compiler, the backends, the annealing and the runner do not depend on the problem. A problem enters through
three things: a spin polynomial (the objective), a feasibility test (the constraints) and a move library (grouped
transitions that keep every feasible state feasible). Which route you take depends on where the moves come from.

| Your constraints | Route | What you write |
|---|---|---|
| Disjoint blocks of the supported families: exact cardinality, precedence, covering, packing, contiguity, free spins | [A problem file](problem_file.md) | a spinpoly or OPB file, no Python |
| Kept by moves the package already builds: pair, row or block exchanges, fixed flips, the block moves of a problem file | [A problem adapter](adapter.md) | the objective, the feasibility test and the starting states, in Python; the moves come from a builder |
| Kept by no existing builder | [Your own moves](own_moves.md) | an adapter, plus the moves: from the structure of the problem, or found automatically by grouping its transitions |

All three routes end in the same place: a compiled machine that runs on every backend, through the same
configuration files and the same Python API as the built-in problems.

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } __A problem file__

    ---

    Constraints from the supported families, written as a spinpoly or OPB file. No Python.

    [:octicons-arrow-right-24: A problem file](problem_file.md)

-   :material-language-python:{ .lg .middle } __A problem adapter__

    ---

    Your own objective, feasibility test and input format in Python, with moves from the package's builders.

    [:octicons-arrow-right-24: A problem adapter](adapter.md)

-   :material-swap-horizontal-bold:{ .lg .middle } __Your own moves__

    ---

    Moves written from the structure of the problem, or found automatically by grouping its transitions.

    [:octicons-arrow-right-24: Your own moves](own_moves.md)

</div>
