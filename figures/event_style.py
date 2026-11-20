"""Shared by the event-driven panels (Fig. 6, Fig. S9): workload styles and the four panel types."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

import _style as S

STYLE = {  # workload: (label, color, line style, marker)
    "m30c": ("matching, complete library, $L$ = 30", "#0072B2", "-", "o"),
    "m50p": ("matching, path library, $L$ = 50", "#56B4E9", (0, (5, 1.5)), "s"),
    "s11": ("surface code, $d$ = 11, $p_{err}$ = 0.12", "#CC79A7", (0, (5, 1.5, 1, 1.5)), "^"),
    "pair2d": (r"2D $\pm J$ spin glass, single flips", "#555555", (0, (1, 1.2)), "D"),
    "syn_k2_rowswap_path_q16": ("16 x 16 permutation, path row exchanges", "#56B4E9", "-", "s"),
    "syn_k2_rowswap_path_q9": ("9 x 9 permutation, path row exchanges", "#0072B2", "-", "o"),
    "syn_k2_flip": ("single flips, order-2 terms", "#E69F00", "-", "^"),
    "syn_k3_flip": ("single flips, order-3 terms", "#D55E00", "-", "v"),
    "syn_k4_flip": ("single flips, order-4 terms", "#CC79A7", "-", "D"),
    "syn_k2_onehot": ("one-hot exchanges, order-2 terms", "#009E73", "-", "P"),
}


def _new(cbar):
    """Figure with room above the axes for the key and, if ``cbar``, right of them for the color bar."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.subplots_adjust(left=0.16, right=0.84 if cbar else 0.96, top=0.80, bottom=0.14)
    return fig, ax


def _key(ax, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=S.FS - 2, columnspacing=1.0, **kw)


def _iter_axis(ax, lo=2 ** 7, hi=2 ** 26.3):
    S.pow2_axis(ax, range(8, 27, 4))
    ax.set_xlim(lo, hi)
    ax.set_xlabel("iterations")


def cumulative(name, csv, workloads, break_even=None):
    """Cumulative operations of event-driven updating / recomputation, with the interquartile range over replicas."""
    d = S.read(*csv)
    fig, ax = _new(cbar=False)
    for k in workloads:
        lab, col, ls, mk = STYLE[k]
        w = d[d.workload == k].sort_values("win_hi")
        ok = (w.cum_ratio_q25 > 0) & np.isfinite(w.cum_ratio_q75)
        ax.fill_between(w.win_hi[ok], w.cum_ratio_q25[ok], w.cum_ratio_q75[ok], color=col, alpha=0.15, lw=0)
        ax.plot(w.win_hi, w.cum_ratio, color=col, ls=ls, label=lab)
        if break_even is not None and break_even.get(k, 0) > 0:
            ax.plot([break_even[k]], [1.0], "o", ms=5, mfc="white", mec=col, mew=1.2, zorder=5)
    ax.axhline(1.0, color="0.2", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 12)
    _iter_axis(ax)
    ax.set_ylabel("cumulative operations,\nevent-driven / recomputation")
    ax.text(2 ** 7.4, 8.0, "recomputation cheaper", va="center", color="0.35")
    ax.text(2 ** 7.4, 0.75, "event-driven cheaper", va="top", color="0.35")
    _key(ax)
    S.save(fig, name)


def plane(name, csv, workloads):
    """Acceptance a against sharing ratio rho, one dot per window, colored by anneal progress."""
    d = S.read(*csv)
    fig, ax = _new(cbar=True)
    aa = np.logspace(-6, 0.3, 100)
    ax.fill_between(aa, 1e-3, np.minimum(1 / aa, 1e3), color="#eef4fa", lw=0, zorder=0)
    ax.plot(aa, 1 / aa, color="0.2", lw=0.9, zorder=2)
    sc = None
    for k in workloads:
        lab, col, ls, mk = STYLE[k]
        w = d[d.workload == k]
        sc = ax.scatter(w.a, w.rho, c=w.progress, cmap="viridis", vmin=0, vmax=1, marker=mk, s=9, linewidths=0.25,
                        edgecolors="0.15", zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6 if workloads[0].startswith("syn") else 1e-4, 1.6)
    ax.set_ylim(0.08, 14)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
    ax.text(0.3, 0.3, r"$a\rho$ = 1", color="0.2")
    ax.set_xlabel("acceptance $a$")
    ax.set_ylabel(r"sharing ratio $\rho$")
    h = [Line2D([], [], color="0.15", marker=STYLE[k][3], ls="", ms=4, mfc="0.7", label=STYLE[k][0]) for k in workloads]
    _key(ax, handles=h)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1], pad=0.02, fraction=0.06)
    cb.ax.set_yticklabels(["start", "end"])
    cb.set_label("anneal progress")
    S.save(fig, name)


def per_window(name, csv, workloads):
    """Per-window ratio of input updates to connection reads, with the 95% bootstrap interval."""
    d = S.read(*csv)
    fig, ax = _new(cbar=False)
    for k in workloads:
        lab, col, ls, mk = STYLE[k]
        w = d[d.workload == k].sort_values("mid")
        ax.fill_between(w.mid, w.ratio_ci_low, w.ratio_ci_high, color=col, alpha=0.25, lw=0)
        ax.plot(w.mid, w.ratio, color=col, ls=ls, label=lab)
    ax.axhline(1.0, color="0.2", lw=0.9)
    ax.set_yscale("log")
    _iter_axis(ax)
    ax.set_ylabel("operations per iteration,\nevent-driven / recomputation")
    _key(ax)
    S.save(fig, name)
