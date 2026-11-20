"""Fig. 5i: real-time decoders at d = 9, p_err = 0.03: published times per round (pentagons), at the LER of their
algorithm on our shots (green: union-find, red: matching, purple: BP-OSD), and this work on the board (star)."""
import _style as S
import fig5_style as Q
from fig5h import rounds

COL = {"uf": Q.DEC["uf"][1], "pm": Q.DEC["pm"][1], "bp": Q.DEC["bp"][1]}


def main():
    t = S.read("fig5", "fig5i.csv")
    fig, ax = S.new((4.6, 2.8))
    rounds(ax, ("trapped ions", "neutral atoms", "superconducting"))
    for i, r in t.iterrows():
        if r.kind == "board":
            ax.plot(r.time_per_round_us, r.ler, marker="*", ms=9, color=S.C["ours"], ls="none")
        else:
            ax.plot(r.time_per_round_us, r.ler, marker="p", ms=5, color=COL[r.algorithm], ls="none")
        ax.annotate(r["name"].replace(" (", "\n("), (r.time_per_round_us, r.ler), xytext=(0, -7 - 14 * (i % 2)), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color="0.25")
    Q.time_axis(ax, 0.006, 2e4, label="decode time per round")
    Q.log_y(ax, 0.04, 1.4, [0.05, 0.1, 0.2, 0.5])
    ax.set_ylabel("logical error rate ($d$ = 9, $p_{err}$ = 0.03)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    S.save(fig, "fig5i")


if __name__ == "__main__":
    main()
