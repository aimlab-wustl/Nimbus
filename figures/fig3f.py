"""Fig. 3f: iterations to solution at L = 10 (target: the certified optimum), synthetic instances and WILLOW pairs."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as S

KEY = {"this work, complete library": "complete", "this work, path library": "path", "N-HOIM with penalty": "nhoim",
       "NeuroSA with quadratized penalty": "neurosa"}
OFFSET = {"complete": -0.21, "path": -0.07, "nhoim": 0.07, "neurosa": 0.21}


def main():
    d = S.read("fig3", "fig3f.csv")
    fig, ax = S.new((3.0, 2.7))
    for g, set_ in enumerate(("synthetic", "willow")):
        for _, r in d[d.problem_set == set_].iterrows():
            k = KEY[r.method]
            lab, col, mk, _ = S.METHOD[k]
            hi = r.its_ci_high if np.isfinite(r.its_ci_high) else 3e10
            ax.errorbar([g + OFFSET[k]], [r.its_median_iterations],
                        yerr=[[r.its_median_iterations - r.its_ci_low], [max(hi - r.its_median_iterations, 0)]],
                        fmt=mk, color=col, ms=3.5 if mk != "D" else 3.0, elinewidth=0.9, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1e2, 1e4, 1e6, 1e8, 1e10]))
    ax.yaxis.set_major_formatter(FixedFormatter([f"$10^{{{k}}}$" for k in (2, 4, 6, 8, 10)]))
    ax.yaxis.set_minor_locator(FixedLocator([10.0 ** k for k in (3, 5, 7, 9)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(1e2, 3e10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["synthetic\n(20 instances)", "WILLOW\n(40 pairs)"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("iterations to solution")
    hs = [Line2D([], [], color=S.METHOD[k][1], marker=S.METHOD[k][2], ls="none", label=S.METHOD[k][0]) for k in OFFSET]
    fig.legend(handles=hs, loc="outside upper center", ncol=2, fontsize=6)
    S.save(fig, "fig3f")


if __name__ == "__main__":
    main()
