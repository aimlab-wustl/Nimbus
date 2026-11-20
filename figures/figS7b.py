"""Fig. S7b: real time, LER versus p_err, d = 5-11: this work (solid) and the best CPU baseline (dashed), same shots and
windows."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("figS7", "figS7b.csv")
    fig, ax = S.new()
    for d in (5, 7, 9, 11):
        for k, ls, mfc in (("hw", "-", None), ("best", (0, (3, 1.5)), "white")):
            a = t[(t.decoder == k) & (t.d == d)].sort_values("p")
            ax.errorbar(a.p, a.ler, yerr=Q.err(a), color=Q.dcol(d), ls=ls, lw=1.1, marker="o", ms=2.4, mfc=mfc or Q.dcol(d),
                        mec=Q.dcol(d), elinewidth=0.6, label=f"$d$ = {d}" if k == "hw" else None)
    ax.plot([], [], color="0.3", marker="o", ms=2.4, label="This work")
    ax.plot([], [], color="0.3", ls=(0, (3, 1.5)), marker="o", ms=2.4, mfc="white", label="best CPU baseline")
    Q.log_y(ax, 0.006, 0.8, [0.01, 0.1])
    ax.set_xlim(0.022, 0.052)
    ax.set_xlabel("physical error rate $p_{err}$")
    ax.set_ylabel("logical error rate ($4d$ rounds)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    S.save(fig, "figS7b")


if __name__ == "__main__":
    main()
