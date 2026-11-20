"""Fig. 3e: Psi / Psi_ref (mean over replicas and instances) versus iterations per replica, complete (solid) and path
(dashed) libraries, L = 10, 15, 20, 30, automatically calibrated schedule."""
from __future__ import annotations

from matplotlib.lines import Line2D

import _style as st


def main():
    df = st.read("fig3", "fig3e.csv")
    fig, ax = st.new()
    for n in (10, 15, 20, 30):
        for lib, ls, mk in (("complete", "-", "o"), ("path", "--", "^")):
            s = df[(df.L == n) & (df.library == lib)].sort_values("iterations")
            ax.fill_between(s.iterations, s.ci95_lo, s.ci95_hi, color=st.SIZE[n], alpha=0.15, lw=0)
            ax.plot(s.iterations, s.psi_over_psi_ref_mean, color=st.SIZE[n], ls=ls, marker=mk, ms=3.0, lw=1.1)
    st.pow2_axis(ax, [10, 14, 18, 22])
    ax.set_xlim(2 ** 9.5, 2 ** 22.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("iterations per replica")
    ax.set_ylabel("$\\Psi$ / $\\Psi_{\\rm ref}$")
    hs = [Line2D([], [], color=st.SIZE[n], lw=1.6, label=f"$L$ = {n}") for n in (10, 15, 20, 30)]
    hs += [Line2D([], [], color="0.2", ls="-", marker="o", ms=3.0, lw=1.1, label="complete library"),
           Line2D([], [], color="0.2", ls="--", marker="^", ms=3.0, lw=1.1, label="path library")]
    ax.legend(handles=hs, loc="lower right", ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=0.8)
    st.save(fig, "fig3e")


if __name__ == "__main__":
    main()
