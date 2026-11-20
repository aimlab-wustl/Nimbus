"""Python API mirroring the CLI: ``load_config``, ``compile_problem``, ``bind``,
``solve``, ``resume``, ``analyze``. The CLI (:mod:`nimbus.cli`) is a thin wrapper around these.
"""

from __future__ import annotations

from pathlib import Path

from nimbus.core.model import bind_weights
from nimbus.experiments.config import load_config
from nimbus.experiments.runner import SolveOutcome
from nimbus.experiments.runner import compile_problem as _compile_problem
from nimbus.experiments.runner import resume as _resume
from nimbus.experiments.runner import solve as _solve

__all__ = ["load_config", "compile_problem", "bind", "solve", "resume", "analyze"]


def compile_problem(source, overrides=None, *, base_dir=None):
    """``(problem, model, timings)`` for a configuration (no weights bound yet)."""
    cfg = load_config(source, overrides, base_dir=base_dir)
    return _compile_problem(cfg)


def bind(source, overrides=None, instance: int = 0, *, base_dir=None):
    """``WeightBinding`` for one instance of a configuration's problem/model."""
    problem, model, _ = compile_problem(source, overrides, base_dir=base_dir)
    return bind_weights(model, problem.term_weights(instance), problem.const(instance))


def solve(source, overrides=None, *, output_dir=None, progress=None) -> SolveOutcome:
    """Run a configuration end to end: compile, calibrate, anneal every replica and write the run folder.

    ``source`` is a configuration file, a mapping or a ``RunConfig``; ``overrides`` maps dotted field names to
    values. Returns a ``SolveOutcome`` with the status, the run folder, the manifest and one result row per replica.
    ``progress(event, info)`` is called at chunk and instance boundaries.
    """
    return _solve(source, overrides, output_dir=output_dir, progress=progress)


def resume(run_dir, *, progress=None) -> SolveOutcome:
    """Resume an interrupted run from its folder (which must contain ``checkpoint.npz``)."""
    return _resume(run_dir, progress=progress)


def analyze(run_dirs, *, kind: str = "summary"):
    """Analysis of one or more completed run directories; see :mod:`nimbus.analysis`."""
    from nimbus.analysis import report

    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    return report.summarize_runs(run_dirs, kind=kind)
