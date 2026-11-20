# API reference

The reference is generated from the docstrings of the package. Most work needs only the functions of `nimbus.api`;
the other modules give finer control over each stage of a run.

## Most used

| Function | What it does |
|---|---|
| [`nimbus.api.load_config`](running.md#nimbus.api.load_config) | read a run configuration (YAML file or dictionary), with overrides |
| [`nimbus.api.compile_problem`](running.md#nimbus.api.compile_problem) | build the problem and compile it with its move library |
| [`nimbus.api.solve`](running.md#nimbus.api.solve) | compile, calibrate, anneal every replica and write the run folder |
| [`nimbus.api.analyze`](running.md#nimbus.api.analyze) | summarize finished runs: best energies and iterations to solution |
| [`nimbus.compiler.compile.compile_model`](compiler.md#nimbus.compiler.compile.compile_model) | compile an objective structure and a move library into a machine |
| [`nimbus.transitions.verify.verify_library`](moves.md#nimbus.transitions.verify.verify_library) | check that a move library keeps every feasible state feasible |

## By topic

<div class="grid cards" markdown>

-   :material-play-circle-outline:{ .lg .middle } __Running problems__

    ---

    Configurations, the run loop, result files.

    [:octicons-arrow-right-24: `nimbus.api`, `nimbus.experiments`](running.md)

-   :material-cog-transfer-outline:{ .lg .middle } __Compiler and model__

    ---

    From objective and moves to clauses, connections and switches.

    [:octicons-arrow-right-24: `nimbus.compiler`, `nimbus.core.model`](compiler.md)

-   :material-swap-horizontal-bold:{ .lg .middle } __Moves__

    ---

    Move-library builders, move descriptors, verification and grouping transitions into moves.

    [:octicons-arrow-right-24: `nimbus.transitions`](moves.md)

-   :material-puzzle-outline:{ .lg .middle } __Problems__

    ---

    The problem interface, built-in kinds and input formats.

    [:octicons-arrow-right-24: `nimbus.problems`](problems.md)

-   :material-thermometer:{ .lg .middle } __Annealing__

    ---

    Schedules, calibration, acceptance and the annealing protocol.

    [:octicons-arrow-right-24: `nimbus.annealing`](annealing.md)

-   :material-lightning-bolt-outline:{ .lg .middle } __Backends__

    ---

    CPU, GPU, event-driven emulation and the reference backend.

    [:octicons-arrow-right-24: `nimbus.backends`](backends.md)

-   :material-atom:{ .lg .middle } __Surface-code decoding__

    ---

    Software models of the FPGA decoders, noise models, CPU baselines.

    [:octicons-arrow-right-24: `nimbus.decoding`](decoding.md)

-   :material-chart-line:{ .lg .middle } __Analysis__

    ---

    Iterations to solution, intervals and reports.

    [:octicons-arrow-right-24: `nimbus.analysis`](analysis.md)

</div>
