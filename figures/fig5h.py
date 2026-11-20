"""Fig. 5h: real time, LER versus decode time per round (window time / d), p_err = 0.03. This work: board; CPU
baselines: one core. Shaded: trapped-ion rounds of 1-5 ms; dotted line: the 4.45 ms neutral-atom round."""
import _style as S
import fig5_style as Q


def rounds(ax, which=("trapped ions", "neutral atoms")):
    r = S.read("fig5", "fig5hi_rounds.csv").set_index("platform")
    if "trapped ions" in which:
        ax.axvspan(r.loc["trapped ions", "round_lo_us"], r.loc["trapped ions", "round_hi_us"], color="0.9", lw=0, zorder=0,
                   label="trapped-ion rounds")
    if "neutral atoms" in which:
        ax.axvline(r.loc["neutral atoms", "round_lo_us"], color="0.4", lw=0.8, ls=":", label="neutral-atom round")
    if "superconducting" in which:
        ax.axvline(r.loc["superconducting", "round_lo_us"], color="0.4", lw=0.8, ls="--", label="superconducting round")


def main():
    t = S.read("fig5", "fig5h.csv")
    fig, ax = S.new((4.6, 2.6))
    rounds(ax)
    for k in ("bpm", "pmc", "bp", "pm"):
        a = t[t.decoder == k].sort_values("d")
        ax.plot(a.time_per_round_us, a.ler, label=Q.DEC[k][0] + " (CPU)", **Q.style(k, ls=(0, (3, 1.5)), lw=1.0))
    a = t[t.decoder == "hw"].sort_values("d")
    ax.plot(a.time_per_round_us, a.ler, label="This work (FPGA)", **Q.style("hw"))
    for _, r in a.iterrows():
        ax.text(r.time_per_round_us / 1.3, r.ler, f"{r.d}", ha="right", va="center", color=S.C["ours"])
    Q.time_axis(ax, 1.0, 3e4, label="decode time per round")
    Q.log_y(ax, 0.006, 0.7, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_ylabel("logical error rate ($p_{err}$ = 0.03)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    S.save(fig, "fig5h")


if __name__ == "__main__":
    main()
