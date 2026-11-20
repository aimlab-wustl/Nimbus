"""Shared by the machine-size panels (Fig. S3): colors and the stacked share bars."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch

import _style as S

SEG = [("bits_pi_obj", "connections to objective clauses", "#0072B2", None),
       ("bits_pi_gate", "connections to auxiliary clauses", "#56B4E9", None),
       ("bits_switch", "switch details", "#E69F00", None),
       ("bits_weights", "weights (16 bit)", "#009E73", None),
       ("bits_signs_obj", "objective-clause signs (1 bit), run time", "#F0E442", "////"),
       ("bits_signs_gate", "auxiliary-clause signs (1 bit), run time", "#D55E00", "////"),
       ("bits_inputs", "move-neuron inputs (16 bit), run time", "#CC79A7", "////")]
LIB = {"complete": ("complete library", S.C["ours"], "o"), "path": ("path library", S.C["ours_path"], "^"),
       "stabilizers": ("stabilizer moves only", "#B03A2E", "s"),
       "stabilizers_logical_all": ("+ $d$ logical operators per type", S.C["ours"], "o")}


def total_label(bits):
    kb = bits / 8e3
    return f"{kb / 1e3:.0f} MB" if kb >= 1e4 else (f"{kb / 1e3:.2g} MB" if kb >= 100 else f"{kb:.2g} kB")


def shares(name, csv, libs, xlabel, size):
    d = S.read(*csv)
    sizes = sorted(d["size"].unique())
    fig, ax = S.new(size)
    ticks, tl, x0 = [], [], 0.0
    for lib in libs:
        s = d[d.lib == lib].set_index("size").reindex(sizes)
        x = x0 + np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for key, _, col, hat in SEG:
            v = (s[key] / s.bits_total).to_numpy()
            ax.bar(x, v, 0.78, bottom=bottom, color=col, hatch=hat, edgecolor="white" if hat is None else "0.25", lw=0.3)
            bottom += v
        for xi, t in zip(x, s.bits_total):
            ax.text(xi, 1.02, total_label(t), ha="center", va="bottom", rotation=90, fontsize=S.FS - 2)
        ax.text(x0 + (len(sizes) - 1) / 2, -0.17, LIB[lib][0], ha="center", va="top", transform=ax.get_xaxis_transform())
        ticks += list(x)
        tl += [str(v) for v in sizes]
        x0 += len(sizes) + 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl)
    ax.set_xlabel(xlabel, labelpad=30)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of total size")
    ax.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="0.25" if h else "white", label=lab) for _, lab, c, h in SEG],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=S.FS - 2)
    S.save(fig, name)
