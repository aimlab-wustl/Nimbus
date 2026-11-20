"""Figures from stored results only. ``matplotlib`` is an optional dependency,
imported lazily; every function raises a clear ``ImportError``-derived message if it is missing
rather than silently doing nothing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["anytime_curve", "success_vs_budget"]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("figures need the optional 'plot' extra: pip install -e .[plot]") from exc
    return plt


def anytime_curve(checkpoints, best, *, ax=None, label=None):
    """Best energy vs attempts (one or more replicas' ``anytime.best`` rows)."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots()
    best = np.atleast_2d(np.asarray(best))
    for row in best:
        ax.plot(checkpoints, row, label=label, drawstyle="steps-post")
    ax.set_xscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("best energy")
    return ax


def success_vs_budget(budgets, success_rate, *, ax=None, label=None):
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(budgets, success_rate, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("attempt budget")
    ax.set_ylabel("success probability")
    return ax
