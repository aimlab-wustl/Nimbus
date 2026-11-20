"""Fig. 4e: Spearman rank correlation between each landscape property and the iterations to solution among the 11
move libraries with K shortcuts of one instance (median over 80 cells of instance and K, 95% bootstrap CI)."""
import _style as S

LABEL = {"catoni_difficulty": "difficulty", "critical_depth": "critical depth", "local_minima": "local minima",
         "relaxation_time_infinite_temperature": "mixing, $\\tau\\to\\infty$"}


def main():
    d = S.read("fig4", "fig4e.csv")
    fig, ax = S.new((3.0, 2.4))
    n = len(d)
    for i, r in enumerate(d.itertuples()):
        y = n - 1 - i
        col = S.C["ours"] if r.property in ("catoni_difficulty", "critical_depth") else "0.6"
        ax.barh(y, r.spearman_rho_median, height=0.6, color=col, zorder=2)
        ax.errorbar([r.spearman_rho_median], [y], xerr=[[r.spearman_rho_median - r.ci95_lo],
                                                       [r.ci95_hi - r.spearman_rho_median]],
                    color="black", ls="", zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL[p] for p in d.property][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.12, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("rank correlation with\niterations to solution")
    S.save(fig, "fig4e")


if __name__ == "__main__":
    main()
