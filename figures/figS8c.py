"""Fig. S8c: real time, timeline of one shot at d = 9 with rounds every 1 ms (board). Ticks: arriving rounds. Bars: each
window from its start to its commit. Orange: the start of the next window (the deadline)."""
import numpy as np
from matplotlib.patches import Rectangle

import _style as S


def main():
    t = S.read("figS8", "figS8c.csv").sort_values("window")
    fig, ax = S.new()
    tr = t.round_period_ms.iloc[0]
    end = t.commit_ms.max()
    for k in range(int(np.ceil(t.start_ms.max() / tr)) + 1):
        ax.plot([k * tr] * 2, [3.15, 3.45], color="0.35", lw=0.8)
    ax.text(0.0, 3.6, f"rounds arrive (every {tr:.0f} ms)", ha="left", va="bottom")
    for i, r in enumerate(t.itertuples()):
        y = 2.4 - 0.8 * i
        ax.add_patch(Rectangle((r.start_ms, y - 0.22), r.commit_ms - r.start_ms, 0.44, fc=S.C["ours"], ec=S.C["ours"]))
        ax.text(r.start_ms - 0.4, y, "final window" if r.final == "yes" else f"window {r.window}", ha="right", va="center")
        if r.final != "yes":
            ax.plot([r.deadline_ms] * 2, [y - 0.35, y + 0.35], color="#D55E00", lw=1.2)
            ax.annotate("", xy=(r.deadline_ms, y), xytext=(r.commit_ms, y), arrowprops=dict(arrowstyle="<->", lw=0.7))
            ax.text(r.deadline_ms + 0.6, y, f"{r.slack_ms:.1f} ms\nslack", ha="left", va="center")
    ax.plot([], [], color="#D55E00", lw=1.2, label="deadline (start of the next window)")
    ax.set_xlim(-12, end + 6)
    ax.set_ylim(0.2, 4.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (ms)")
    ax.legend(loc="lower right", fontsize=6)
    S.save(fig, "figS8c")


if __name__ == "__main__":
    main()
