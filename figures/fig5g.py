"""Fig. 5g: real time, LER per shot versus d at p_err = 0.03: this work (software model of the board designs), best CPU
baseline (same shots, same windows) and the board on 1,000 other shots (squares)."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("fig5", "fig5g.csv")
    fig, ax = S.new()
    for k, lab, col, ls, mfc in (("this_work_model", "This work (bit-exact model)", S.C["ours"], "-", S.C["ours"]),
                                 ("best_cpu", "best CPU baseline", "#111111", (0, (3, 1.5)), "white")):
        a = t[t.series == k].sort_values("d")
        ax.errorbar(a.d, a.ler, yerr=Q.err(a), color=col, ls=ls, lw=1.1, marker="o", ms=3, mfc=mfc, mec=col,
                    elinewidth=0.6, label=lab)
    b = t[t.series == "this_work_board"].sort_values("d")
    b = b[b.d >= 5]
    ax.errorbar(b.d, b.ler, yerr=Q.err(b), ls="none", marker="s", ms=4.5, mfc="none", mec=S.C["ours"], mew=0.9,
                ecolor=S.C["ours"], elinewidth=0.6, label="This work, on the board")
    Q.log_y(ax, 0.02, 0.62, [0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticks([5, 7, 9, 11])
    ax.set_xlim(4, 12)
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("logical error rate ($p_{err}$ = 0.03)")
    ax.legend(loc="upper right")
    S.save(fig, "fig5g")


if __name__ == "__main__":
    main()
