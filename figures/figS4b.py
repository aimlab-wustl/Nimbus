"""Fig. S4b: iterations to solution at L = 20 for informed and random shortcuts and the two complete references, each with
its selected schedule; brackets: ratio random / informed with its 95% paired bootstrap interval."""
import _style as S


def main():
    t = S.read("figS4", "figS4b.csv")
    pr = S.read("figS4", "figS4b_ratio.csv").set_index("shortcuts_K")
    key = {("informed", 40): "informed", ("random", 40): "random", ("informed", 80): "informed", ("random", 80): "random",
           ("complete, uniform proposals", 171): "complete", ("complete, backbone share 0.5", 171): "share"}
    row = {(r.library, r.shortcuts_K): r for r in t.itertuples()}
    dx = 0.2
    groups = [("$K$ = 40", [("informed", 40), ("random", 40)], 40),
              ("$K$ = 80", [("informed", 80), ("random", 80)], 80),
              ("complete", [("complete, uniform proposals", 171), ("complete, backbone share 0.5", 171)], None)]
    fig, ax = S.new((3.3, 2.8))
    for g, (_, items, k) in enumerate(groups):
        for item, off in zip(items, (-dx, dx)):
            r = row[item]
            lab, col, mk, filled = S.LIBRARY[key[item]]
            star = mk == "*"
            ax.errorbar([g + off], [r.its_iterations], yerr=[[r.its_iterations - r.its_ci_lo], [r.its_ci_hi - r.its_iterations]],
                        color=col, marker=mk, ms=6.5 if star else 3.6, mfc=col if filled else "white",
                        mec="black" if star else col, mew=0.6 if star else 1.0, elinewidth=1.0, ls="", zorder=3,
                        label=lab if g != 1 else None)
        if k is not None:
            q = pr.loc[k]
            ytop = max(row[items[0]].its_ci_hi, row[items[1]].its_ci_hi) * 1.35
            ax.plot([g - dx, g - dx, g + dx, g + dx], [ytop / 1.15, ytop, ytop, ytop / 1.15], color="0.25", lw=0.75)
            ax.text(g, ytop * 1.12, f"{q.ratio_random_over_informed:.1f}$\\times$ ({q.ratio_ci_lo:.1f}–{q.ratio_ci_hi:.1f})",
                    ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(7e3, 6e8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g for g, _, _ in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("iterations to solution ($L$ = 20)")
    ax.legend(loc="upper right", ncol=2, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    S.save(fig, "figS4b")


if __name__ == "__main__":
    main()
