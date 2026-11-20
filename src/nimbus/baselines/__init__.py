"""Baselines: the generic penalty and quadratic (quadratized penalty) baselines, built on the
shared architecture infrastructure, and the N-HOIM-style and NeuroSA-style penalty machines
(:mod:`nimbus.baselines.penalty_machines`). They never supply an acceptance delta to an architecture backend.

This module is intentionally light: submodules are imported lazily on first attribute access so
importing ``nimbus.baselines`` never pulls in optional dependencies (e.g. scipy/numba-heavy
solver code) or does any work.
"""

from __future__ import annotations

import importlib

__all__ = ["penalty", "penalty_machines", "quadratize"]

_SUBMODULES = frozenset(__all__)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
