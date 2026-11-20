"""Fig. 5f: decoders of one syndrome per shot at d = 9, p_err = 0.12: LER on our shots versus decode time per shot.
Pentagons: published times, at the LER of their kind of algorithm on our shots (pink: lowest-energy correction, red:
matching)."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("fig5", "fig5f.csv")
    fig, ax = S.new()
    ml = t[t.algorithm == "tn"].ler.iloc[0]
    ax.axhline(ml, color="0.6", lw=0.7, ls="--")
    for _, r in t.iterrows():
        if r.kind == "board":
            ax.plot([r.time_early_stop_us, r.time_us], [r.ler, r.ler], color=S.C["ours"], lw=1.4)
            ax.plot(r.time_early_stop_us, r.ler, marker="*", ms=9, color=S.C["ours"], ls="none")
            ax.plot([r.time_us] * 2, [r.ler / 1.03, r.ler * 1.03], color=S.C["ours"], lw=1.0)
            x = r.time_early_stop_us
        elif r.kind == "cpu":
            ax.plot(r.time_us, r.ler, ls="none", **{k: v for k, v in Q.style(r.algorithm).items() if k != "ls"})
            x = r.time_us
        else:
            col = Q.LOWEST_ENERGY if r.algorithm == "lowest_energy" else Q.DEC["pm"][1]
            ax.plot(r.time_us, r.ler, marker="p", ms=5, color=col, ls="none")
            x = r.time_us
        ax.annotate(r["name"].split(",")[0].replace(" [", "\n["), (x, r.ler), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color="0.25")
    Q.time_axis(ax, 0.3, 3e6, label="decode time per shot")
    Q.log_y(ax, 0.046, 0.2, [0.05, 0.1, 0.2])
    ax.set_ylabel("logical error rate ($d$ = 9, $p_{err}$ = 0.12)")
    S.save(fig, "fig5f")


if __name__ == "__main__":
    main()
