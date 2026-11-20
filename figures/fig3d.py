"""Fig. 3d: iterations to solution of the complete library versus the number of keypoints L (per-size selected schedule)."""
import _style as S


def main():
    d = S.read("fig3", "fig3d.csv").sort_values("L")
    fig, ax = S.new()
    col = S.C["ours"]
    ax.plot(d.L, d.its_iterations, color=col, lw=1.0, zorder=2)
    ax.errorbar(d.L, d.its_iterations, yerr=S.yerr(d, "its_iterations", "its_ci_lo", "its_ci_hi"), fmt="o", color=col,
                elinewidth=1.0, zorder=3, label=S.METHOD["complete"][0])
    ax.set_yscale("log")
    ax.set_ylim(6e2, 1.2e6)
    ax.set_xlim(7.5, 32.5)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_xlabel("number of keypoints $L$")
    ax.set_ylabel("iterations to solution")
    ax.legend(loc="upper left")
    S.save(fig, "fig3d")


if __name__ == "__main__":
    main()
