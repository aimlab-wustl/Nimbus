"""Supplementary Fig. S1a: keypoint accuracy on the 40 WILLOW test pairs versus iterations per replica."""
import numpy as np
from matplotlib.lines import Line2D

import _style as S


def main():
    d = S.read("figS1", "figS1a.csv")
    fig, ax = S.new((3.4, 2.7))
    opt = float(d.optimum_accuracy.iloc[0])
    ax.axhspan(float(d.optimum_accuracy_ci_low.iloc[0]), float(d.optimum_accuracy_ci_high.iloc[0]), color="0.88", lw=0, zorder=0)
    ax.axhline(opt, color="0.1", lw=1.0, ls=(0, (4, 1.5, 1, 1.5)), zorder=5)
    for lib in ("complete", "path"):
        lab, col, mk, _ = S.METHOD[lib]
        s = d[d.library == lib].sort_values("budget_log2")
        x = 2.0 ** s.budget_log2
        ax.fill_between(x, s.accuracy_mean_ci_low, s.accuracy_mean_ci_high, color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, s.accuracy_mean_over_replicas, color=col, marker=mk, ls=(0, (1.2, 1.2)), lw=1.1, ms=3.3, mfc="white",
                mew=0.9, zorder=3)
        if lib == "path":
            ax.errorbar(x * 2 ** 0.15, s.accuracy_best_replica,
                        yerr=S.yerr(s, "accuracy_best_replica", "accuracy_best_ci_low", "accuracy_best_ci_high"),
                        color=col, marker=mk, ls="-", lw=1.1, ms=3.3, elinewidth=0.8, zorder=4)
        else:
            ax.plot(x, s.accuracy_best_replica, color=col, marker=mk, ls="-", lw=1.1, ms=3.3, zorder=4)
    S.pow2_axis(ax, [6, 12, 18, 24])
    ax.set_xlim(2 ** 5.3, 2 ** 24.7)
    ax.set_ylim(0.0, 0.88)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_xlabel("iterations per replica")
    ax.set_ylabel("keypoint accuracy (WILLOW)")
    hs = [Line2D([], [], color=S.METHOD[k][1], marker=S.METHOD[k][2], label=S.METHOD[k][0]) for k in ("complete", "path")]
    hs += [Line2D([], [], color="0.3", ls="-", label="best replica"),
           Line2D([], [], color="0.3", ls=(0, (1.2, 1.2)), label="mean over replicas"),
           Line2D([], [], color="0.1", ls=(0, (4, 1.5, 1, 1.5)), label=f"certified optimum ({opt:.3f})")]
    ax.legend(handles=hs, loc="lower right", fontsize=6)
    S.save(fig, "figS1a")


if __name__ == "__main__":
    main()
