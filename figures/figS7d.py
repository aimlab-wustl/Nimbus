"""Fig. S7d: real time, d = 9, p_err = 0.03: LER versus sweeps per window for 24 replicas (the board design) and 16
replicas. Dashed line: best CPU baseline. Dotted line: the board's budget (2^12 sweeps)."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("figS7", "figS7d.csv")
    fig, ax = S.new()
    a = t[t.replicas == 24].sort_values("sweeps")
    ax.errorbar(a.sweeps, a.ler, yerr=Q.err(a), color=S.C["ours"], lw=1.3, marker="o", ms=3.2, elinewidth=0.8,
                label="24 replicas")
    b = t[t.replicas == 16]
    ax.errorbar(b.sweeps, b.ler, yerr=Q.err(b), ls="none", marker="s", ms=3.0, mfc="white", mec="0.45", ecolor="0.45",
                elinewidth=0.8, label="16 replicas")
    best = t[t.series == "best_cpu"].ler.iloc[0]
    ax.axhline(best, color="#111111", ls=(0, (4, 1.5)), lw=1.0, label="best CPU baseline")
    ax.axvline(4096, color="0.6", lw=0.8, ls=":", label="board budget")
    S.pow2_axis(ax, [9, 10, 11, 12, 13])
    ax.set_xlim(380, 10000)
    Q.log_y(ax, 0.04, 0.14, [0.05, 0.1])
    ax.set_xlabel("sweeps per window")
    ax.set_ylabel("logical error rate")
    ax.legend(loc="upper right")
    S.save(fig, "figS7d")


if __name__ == "__main__":
    main()
