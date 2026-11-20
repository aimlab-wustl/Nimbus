"""Fig. 5e: LER versus decode time per shot, d = 3-15, p_err = 0.12. This work: board, mean with early stopping (dot)
to the fixed budget (bar end); CPU decoders: one core."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("fig5", "fig5e.csv")
    fig, ax = S.new((4.4, 2.6))
    for k in ("uf", "pm", "bposd", "tn", "hw_ml"):
        a = t[t.decoder == k].sort_values("d")
        ax.plot(a.time_us, a.ler, label=Q.DEC[k][0], **Q.style(k))
        if k == "hw_ml":
            for _, r in a.iterrows():
                ax.plot([r.time_us, r.time_max_us], [r.ler, r.ler], color=S.C["ours"], lw=1.0)
                ax.plot([r.time_max_us] * 2, [r.ler / 1.04, r.ler * 1.04], color=S.C["ours"], lw=1.0)
            for _, r in a[a.d.isin([3, 15])].iterrows():
                ax.text(r.time_us * 0.7, r.ler, f"$d$ = {r.d}", ha="right", va="center", color=S.C["ours"])
    Q.time_axis(ax, 0.1, 3e6, label="decode time per shot")
    Q.log_y(ax, 0.015, 0.2, [0.02, 0.05, 0.1, 0.2])
    ax.set_ylabel("logical error rate ($p_{err}$ = 0.12)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    S.save(fig, "fig5e")


if __name__ == "__main__":
    main()
