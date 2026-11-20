"""Fig. 5c: LER versus sweeps per shot, d = 15, p_err = 0.12: move library and replica exchange."""
import _style as S
import fig5_style as Q

SERIES = {"backbone_and_shortcuts": ("backbone + 14 shortcuts", S.C["ours"], "-"),
          "backbone": ("backbone only", S.C["ours_path"], "-"),
          "no_replica_exchange": ("no replica exchange", S.C["ours"], (0, (2, 1.2)))}


def main():
    t = S.read("fig5", "fig5c.csv")
    fig, ax = S.new()
    for k, (lab, col, ls) in SERIES.items():
        s = t[t.series == k].sort_values("sweeps")
        ax.plot(s.sweeps, s.ler, color=col, ls=ls, lw=1.4, label=lab)
    st = t[t.series == "stabilizer_moves_only"].ler.iloc[0]
    ax.axhline(st, color="0.5", lw=1.1, label="stabilizer moves only")
    ml = t[t.series == "tensor_network_ml"].ler.iloc[0]
    ax.axhline(ml, color="#111111", ls=(0, (4, 1.5)), lw=1.0, label="ML (tensor network)")
    ax.axvline(t.fpga_sweeps.iloc[0], color="0.6", lw=0.8, ls=":", label="board budget")
    S.pow2_axis(ax, [2, 4, 6, 8, 10, 12])
    ax.set_xlim(3.5, 4800)
    Q.log_y(ax, 1.5e-4, 1.4, [0.001, 0.01, 0.1, 0.5])
    ax.set_xlabel("sweeps per shot")
    ax.set_ylabel("logical error rate ($d$ = 15, $p_{err}$ = 0.12)")
    ax.legend(loc="lower left", fontsize=6.5, ncol=2)
    S.save(fig, "fig5c")


if __name__ == "__main__":
    main()
