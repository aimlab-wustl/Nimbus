"""Fig. S8b: one syndrome per shot, p_err = 0.12, board: LER of the shots kept when the least confident are discarded
(confidence: largest class count / total). 20,000 board shots per d."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("figS8", "figS8b.csv")
    fig, ax = S.new()
    for d in Q.DS:
        q = t[(t.d == d) & (t.fails > 0)]
        ax.plot(100 * q.discarded, q.ler, color=Q.dcol(d), lw=1.2, label=f"$d$ = {d}")
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_ylim(1e-4, 0.3)
    ax.set_xlabel("least confident shots discarded (%)")
    ax.set_ylabel("logical error rate of kept shots")
    ax.legend(loc="lower left", ncol=2, fontsize=6)
    S.save(fig, "figS8b")


if __name__ == "__main__":
    main()
