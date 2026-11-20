"""Fig. 4a: probability of reaching 0.98 Psi_ref within 2^20 iterations versus library size (L = 20).
Supplementary Fig. S4a uses the same drawing (figS4a.py)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as st


def draw(df, name, xticks, xlim, size=(3.6, 2.7)):
    L = int(df.L.iloc[0])
    complete = L * (L - 1) // 2
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.17, 0.17, 0.66, 0.62])
    axr = fig.add_axes([0.86, 0.17, 0.10, 0.62], sharey=ax)
    for lib, f in (("informed", 0.97), ("random", 1.03)):
        lab, col, mk, filled = st.LIBRARY[lib]
        t = df[df.library == lib].sort_values("exchanges")
        ax.errorbar(t.exchanges * f, t.p_reach_098_mean, yerr=st.yerr(t, "p_reach_098_mean", "ci95_lo", "ci95_hi"),
                    color=col, marker=mk, mfc=col if filled else "white", lw=1.2, elinewidth=1.0, label=lab, zorder=3)
    t = df[df.library == "path"]
    lab, col, mk, _ = st.LIBRARY["path"]
    ax.errorbar(t.exchanges, t.p_reach_098_mean, yerr=st.yerr(t, "p_reach_098_mean", "ci95_lo", "ci95_hi"), color=col,
                marker=mk, ls="", label=lab, zorder=4)
    for lib, x in (("complete", -0.25), ("share", 0.25)):
        t = df[df.library == lib]
        lab, col, mk, _ = st.LIBRARY[lib]
        axr.errorbar([x], t.p_reach_098_mean, yerr=st.yerr(t, "p_reach_098_mean", "ci95_lo", "ci95_hi"), color=col,
                     marker=mk, ms=6.5 if mk == "*" else 3.4, mec="black" if mk == "*" else col, ls="", zorder=4)
        ax.plot([], [], color=col, marker=mk, ls="", ms=6.5 if mk == "*" else 3.4, mec="black" if mk == "*" else col,
                label=lab)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    axr.set_xlim(-0.75, 0.75)
    axr.set_xticks([0])
    axr.set_xticklabels([str(complete)])
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", left=False, labelleft=False)
    for a_, xs, w in ((ax, 1.0, 0.02), (axr, 0.0, 0.12)):     # axis-break marks
        a_.plot([xs - w, xs + w], [-0.03, 0.03], transform=a_.transAxes, color="black", lw=0.8, clip_on=False)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(f"exchanges in the move library ($L$ = {L})")
    ax.set_ylabel("probability of reaching\n0.98 $\\Psi_{\\rm ref}$")
    hs, labs = ax.get_legend_handles_labels()
    order = [st.LIBRARY[k][0] for k in ("informed", "random", "path", "complete", "share")]
    pick = [labs.index(o) for o in order]
    ax.legend([hs[i] for i in pick], [labs[i] for i in pick], loc="lower left", bbox_to_anchor=(-0.2, 1.02), ncol=2,
              fontsize=6, handlelength=1.6, columnspacing=0.8)
    st.save(fig, name)


def main():
    draw(st.read("fig4", "fig4a.csv"), "fig4a", [20, 50, 100], (16, 205))


if __name__ == "__main__":
    main()
