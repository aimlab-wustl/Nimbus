"""Fig. S7e: decode time per shot versus d for decoders of one syndrome per shot (p_err = 0.12 for the decoders run by
us; published times at their own noise models and error rates)."""
import _style as S
import fig5_style as Q

SERIES = {
    "this_work_fixed_budget": ("This work, board, fixed budget", S.C["ours"], "-", "o", True),
    "this_work_early_stop_mean": ("This work, mean with early stopping", S.C["ours"], (0, (2, 1.2)), "o", False),
    "tn": ("tensor-network ML, CPU", Q.DEC["tn"][1], "-", "s", False),
    "bposd": ("BP-OSD, CPU", Q.DEC["bposd"][1], "-", "D", False),
    "pm": ("PyMatching, CPU", Q.DEC["pm"][1], "-", "^", False),
    "uf": ("union-find, CPU", Q.DEC["uf"][1], "-", "v", False),
    "neural_software": ("neural decoder, software [a]", "#882255", (0, (3, 1.5)), "p", True),
    "neural_gpu": ("neural decoder, GPU [b]", "#882255", (0, (3, 1.5)), "h", True),
    "simulated_annealing": ("simulated annealing, CPU, XZZX [c]", "#CC79A7", (0, (3, 1.5)), "p", True),
    "dwave": ("D-Wave annealer [d]", "#8C6D31", (0, (3, 1.5)), "p", True),
}


def main():
    t = S.read("figS7", "figS7e.csv")
    fig, ax = S.new((4.6, 2.8))
    for k, (lab, col, ls, mk, fill) in SERIES.items():
        a = t[t.series == k].sort_values("d")
        ax.plot(a.d, a.time_us, color=col, ls=ls, lw=1.1, marker=mk, ms=3.2, mfc=col if fill else "white", mec=col,
                label=lab)
    Q.time_axis(ax, 0.1, 2e7, axis="y", label="decode time per shot")
    ax.set_xticks(Q.DS)
    ax.set_xlim(2.5, 15.5)
    ax.set_xlabel("code distance $d$")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6)
    S.save(fig, "figS7e")


if __name__ == "__main__":
    main()
