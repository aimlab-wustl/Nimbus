"""Shared style of the panel scripts: colors and symbols per method, font sizes, input and output folders."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUT = Path(__file__).resolve().parent / "out"

FS = 8.0  # labels, ticks and legends (pt)
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS - 1, "axes.titlesize": FS, "mathtext.default": "regular",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "errorbar.capsize": 1.5, "legend.frameon": False, "legend.handlelength": 1.8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
})

#: colors of the paper figures
C = {
    "ours": "#0072B2",           # this work, complete library / informed shortcuts
    "ours_path": "#56B4E9",      # this work, path library
    "nhoim": "#009E73",          # N-HOIM with penalty
    "neurosa": "#D55E00",        # NeuroSA with quadratized penalty
    "random": "#8C8C8C",         # random shortcuts
    "complete_ref": "#111111",   # complete library, uniform proposals (library figures)
    "share": "#2CA02C",          # complete library, backbone share 0.5
    "hot": "#E69F00",            # no cooling (control)
    "shortcut": "#E69F00",       # shortcut exchanges
}

#: methods of the matching comparison: (label, color, marker, line style)
METHOD = {
    "complete": ("This work, complete library", C["ours"], "o", "-"),
    "path": ("This work, path library", C["ours_path"], "^", "--"),
    "nhoim": ("N-HOIM with penalty", C["nhoim"], "s", "-"),
    "neurosa": ("NeuroSA with quadratized penalty", C["neurosa"], "D", "-"),
}

#: domain solvers: (label, dash pattern)
DOMAIN = {"adapt_bcagm3": ("Adapt-BCAGM3", (0, (5, 1.5))), "bcagm3": ("BCAGM3", (0, (2, 1.2))),
          "rrwhm": ("RRWHM", (0, (0.8, 1.2)))}

#: number of keypoints L -> color
SIZE = {10: "#440154", 15: "#3b528b", 20: "#21908d", 25: "#5dc863", 30: "#b5a300"}

#: move libraries of the library figures: (label, color, marker, filled)
LIBRARY = {
    "informed": ("backbone + $K$ informed shortcuts", C["ours"], "o", True),
    "random": ("backbone + $K$ random shortcuts", C["random"], "s", False),
    "path": ("path library (backbone alone)", C["ours_path"], "^", True),
    "complete": ("complete library, uniform proposals", C["complete_ref"], "D", True),
    "share": ("complete library, backbone share 0.5", C["share"], "*", True),
}

PANEL = (3.3, 2.6)   # default panel size (inches)


def read(*parts) -> pd.DataFrame:
    """Read ``source_data/<parts...>``."""
    return pd.read_csv(SOURCE.joinpath(*parts))


def new(size=PANEL):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    return fig, ax


def pow2_axis(ax, ks, axis="x"):
    """Ticks at ``2**k`` for ``k`` in ``ks``, labelled ``2^k`` (log2 scale)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    a.set_major_locator(FixedLocator([2.0 ** k for k in ks]))
    a.set_major_formatter(FixedFormatter([f"$2^{{{k}}}$" for k in ks]))
    a.set_minor_locator(FixedLocator([]))


def yerr(df, y, lo, hi):
    """Asymmetric error bars from columns ``y``, ``lo``, ``hi``."""
    return np.vstack([np.clip(df[y] - df[lo], 0, None), np.clip(df[hi] - df[y], 0, None)])


def save(fig, name: str):
    """Save ``figures/out/<name>.pdf`` and ``.png``."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def write_table(name: str, header, rows, note: str = ""):
    """Print a markdown table and save it as ``figures/out/<name>.md``."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = "\n".join(lines) + ("\n\n" + note if note else "") + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)
    return text
