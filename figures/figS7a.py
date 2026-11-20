"""Fig. S7a: one syndrome per shot, LER versus p_err, d = 3-15, 8,000 shots per point. Filled: this work (fixed budget
of each design; 2^12 sweeps at p_err >= 0.14 for d >= 5). Open: tensor-network ML, bond dimension 16."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("figS7", "figS7a.csv")
    fig, ax = S.new()
    for d in Q.DS:
        for k, ls, mk, mfc in (("hw_ml", "-", "o", None), ("tn", (0, (3, 1.5)), "s", "white")):
            a = t[(t.decoder == k) & (t.d == d)].sort_values("p")
            ax.errorbar(a.p, a.ler, yerr=Q.err(a), color=Q.dcol(d), ls=ls, lw=1.1, marker=mk, ms=2.6, mfc=mfc or Q.dcol(d),
                        mec=Q.dcol(d), elinewidth=0.6, label=f"$d$ = {d}" if k == "hw_ml" else None)
    ax.plot([], [], color="0.3", marker="o", ms=2.6, label="This work")
    ax.plot([], [], color="0.3", ls=(0, (3, 1.5)), marker="s", ms=2.6, mfc="white", label="Tensor-network ML")
    Q.log_y(ax, 3e-4, 0.7, [0.001, 0.01, 0.1])
    ax.set_xlim(0.07, 0.195)
    ax.set_xlabel("physical error rate $p_{err}$")
    ax.set_ylabel("logical error rate")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    S.save(fig, "figS7a")


if __name__ == "__main__":
    main()
