"""Fig. 5d: LER versus code distance d, p_err = 0.12, one syndrome per shot, all decoders on the same shots."""
import _style as S
import fig5_style as Q


def main():
    t = S.read("fig5", "fig5d.csv")
    fig, ax = S.new()
    for k in ("uf", "pm", "bposd", "tn", "hw_ml"):
        a = t[t.decoder == k].sort_values("d")
        ax.errorbar(a.d, a.ler, yerr=Q.err(a), elinewidth=0.8, label=Q.DEC[k][0], **Q.style(k))
    ax.set_xticks(Q.DS)
    ax.set_xlim(2, 16)
    Q.log_y(ax, 0.015, 0.2, [0.02, 0.05, 0.1, 0.2])
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("logical error rate ($p_{err}$ = 0.12)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="lower left")
    S.save(fig, "fig5d")


if __name__ == "__main__":
    main()
