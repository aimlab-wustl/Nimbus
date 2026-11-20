"""Shared style of the surface-code panels (Fig. 5, Fig. S7, Fig. S8): decoder colors and markers, code-distance colors,
time axes. Imported by the panel scripts; draws nothing itself."""
from __future__ import annotations

import matplotlib
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as S

DS = [3, 5, 7, 9, 11, 13, 15]
#: decoders: (label, color, line style, marker, filled)
DEC = {
    "hw_ml": ("This work (FPGA)", S.C["ours"], "-", "o", True),
    "hw": ("This work (FPGA)", S.C["ours"], "-", "o", True),
    "tn": ("Tensor-network ML", "#111111", (0, (4, 1.5)), "s", False),
    "bposd": ("BP-OSD", "#7B3294", (0, (1, 1.2)), "D", False),
    "bp": ("BP-OSD", "#7B3294", (0, (3, 1.5)), "D", False),
    "pm": ("PyMatching", "#C0392B", (0, (5, 1.5, 1, 1.5)), "^", False),
    "pmc": ("correlated PyMatching", "#D55E00", (0, (3, 1.5)), "v", False),
    "bpm": ("BP+matching", "#111111", (0, (3, 1.5)), "s", False),
    "uf": ("Union-find", "#009E73", (0, (2, 1.2)), "v", False),
}
LOWEST_ENERGY = "#CC79A7"
TIME_LAB = {1e-2: "10 ns", 1e-1: "100 ns", 1: "1 µs", 10: "10 µs", 100: "100 µs", 1e3: "1 ms", 1e4: "10 ms",
            1e5: "100 ms", 1e6: "1 s", 1e7: "10 s"}


def dcol(d, ds=DS):
    """Code distance -> color (light blue at d = 3 to dark blue at d = 15)."""
    return matplotlib.colors.to_hex(matplotlib.colormaps["Blues"](0.35 + 0.6 * DS.index(d) / (len(DS) - 1)))


def style(k, **kw):
    lab, c, ls, mk, fill = DEC[k]
    s = dict(color=c, ls=ls, lw=1.4 if k in ("hw_ml", "hw") else 1.1, marker=mk, ms=3.5, mfc=c if fill else "white",
             mec=c, mew=1.0)
    s.update(kw)
    return s


def time_axis(ax, lo, hi, axis="x", label=None):
    """Log time axis in microseconds with unit labels."""
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    a = ax.xaxis if axis == "x" else ax.yaxis
    maj = [v for v in TIME_LAB if lo <= v <= hi]
    a.set_major_locator(FixedLocator(maj))
    a.set_major_formatter(FixedFormatter([TIME_LAB[v] for v in maj]))
    a.set_minor_formatter(NullFormatter())
    if label:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)(label)


def log_y(ax, lo, hi, ticks):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def err(df):
    return S.yerr(df, "ler", "ci_lo", "ci_hi")
