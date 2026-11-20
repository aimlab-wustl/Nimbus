"""Supplementary Fig. S1c: Psi / Psi_ref (mean over replicas, then over instances) versus iterations per replica,
deformation sigma = 0.10, complete library, L = 10-30."""
from __future__ import annotations

from matplotlib.lines import Line2D

import _style as st


def main():
    df = st.read("figS1", "figS1c.csv")
    fig, ax = st.new()
    for n in sorted(df.L.unique()):
        s = df[df.L == n].sort_values("iterations")
        ax.fill_between(s.iterations, s.ci95_lo, s.ci95_hi, color=st.SIZE[n], alpha=0.15, lw=0)
        ax.plot(s.iterations, s.psi_over_psi_ref_mean, color=st.SIZE[n], marker="o", ms=3.0, lw=1.1)
    st.pow2_axis(ax, [10, 14, 18, 22])
    ax.set_xlim(2 ** 9.5, 2 ** 22.5)
    ax.axhline(1.0, color="0.25", lw=0.8, zorder=0)
    ax.set_ylim(0.4, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("iterations per replica")
    ax.set_ylabel("$\\Psi$ / $\\Psi_{\\rm ref}$")
    ax.text(0.97, 0.05, "$\\sigma$ = 0.10", transform=ax.transAxes, ha="right", va="bottom")
    hs = [Line2D([], [], color=st.SIZE[n], lw=1.6, label=f"$L$ = {n}") for n in sorted(df.L.unique())]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, fontsize=6.5, handlelength=1.2,
              columnspacing=0.6)
    st.save(fig, "figS1c")


if __name__ == "__main__":
    main()
