"""Supplementary Fig. S1b: probability that a replica reaches 0.98 Psi_ref versus iterations per replica, deformation
sigma = 0.10, complete library, L = 10-30."""
from __future__ import annotations

from matplotlib.lines import Line2D

import _style as st


def main():
    df = st.read("figS1", "figS1b.csv")
    fig, ax = st.new()
    for n in sorted(df.L.unique()):
        s = df[df.L == n].sort_values("iterations")
        ax.errorbar(s.iterations, s.p_reach_098_mean, yerr=st.yerr(s, "p_reach_098_mean", "ci95_lo", "ci95_hi"),
                    color=st.SIZE[n], marker="o", ms=3.0, lw=1.1, elinewidth=0.9)
    st.pow2_axis(ax, [10, 14, 18, 22])
    ax.set_xlim(2 ** 9.5, 2 ** 22.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("iterations per replica")
    ax.set_ylabel("probability of reaching\n0.98 $\\Psi_{\\rm ref}$")
    ax.text(0.97, 0.05, "$\\sigma$ = 0.10", transform=ax.transAxes, ha="right", va="bottom")
    hs = [Line2D([], [], color=st.SIZE[n], marker="o", ms=3.0, lw=1.1, label=f"$L$ = {n}") for n in sorted(df.L.unique())]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, fontsize=6.5, handlelength=1.2,
              columnspacing=0.6)
    st.save(fig, "figS1b")


if __name__ == "__main__":
    main()
