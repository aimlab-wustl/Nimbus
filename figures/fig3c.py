"""Fig. 3c: probability that a replica reaches the certified optimum versus iterations per replica (40 WILLOW pairs)."""
from matplotlib.lines import Line2D

import _style as S

KEY = {"this work, complete library": "complete", "this work, path library": "path", "N-HOIM with penalty": "nhoim",
       "NeuroSA with quadratized penalty": "neurosa"}
DOMKEY = {"Adapt-BCAGM3": "adapt_bcagm3", "BCAGM3": "bcagm3", "RRWHM": "rrwhm"}


def main():
    d = S.read("fig3", "fig3c.csv")
    dom = S.read("fig3", "fig3c_domain_solvers.csv")
    fig, ax = S.new((3.6, 2.9))
    handles = []
    for meth in ("path", "complete", "nhoim", "neurosa"):
        lab, col, mk, ls = S.METHOD[meth]
        s = d[d.method.map(KEY) == meth].sort_values("budget_log2")
        ax.errorbar(2.0 ** s.budget_log2, s.p_optimum, yerr=S.yerr(s, "p_optimum", "p_optimum_ci_low", "p_optimum_ci_high"),
                    color=col, marker=mk, ls=ls, ms=3.5 if mk != "D" else 3.0, elinewidth=0.9, zorder=3)
        handles.append(Line2D([], [], color=col, marker=mk, ls=ls, label=lab))
    for _, r in dom.iterrows():
        lab, dash = S.DOMAIN[DOMKEY[r.solver]]
        ax.axhline(r.p_restart_optimum, color="0.15", ls=dash, lw=0.9, zorder=1)
        handles.append(Line2D([], [], color="0.15", ls=dash, lw=0.9, label=f"{lab} (one restart)"))
    S.pow2_axis(ax, [6, 12, 18, 24])
    ax.set_xlim(2 ** 5.4, 2 ** 24.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("iterations per replica")
    ax.set_ylabel("probability of reaching\nthe optimum")
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=6)
    S.save(fig, "fig3c")


if __name__ == "__main__":
    main()
