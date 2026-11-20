"""Fig. S7c: real time, LER of the best CPU baseline divided by that of this work, versus p_err, d = 5-11."""
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as S
import fig5_style as Q


def main():
    t = S.read("figS7", "figS7c.csv")
    fig, ax = S.new()
    for d in (5, 7, 9, 11):
        a = t[t.d == d].sort_values("p")
        ax.plot(a.p, a.ratio, color=Q.dcol(d), lw=1.1, marker="o", ms=2.4, label=f"$d$ = {d}")
    ax.axhline(1.0, color="0.6", lw=0.6, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 2.6)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 2.5]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "1.5", "2", "2.5"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.022, 0.052)
    ax.set_xlabel("physical error rate $p_{err}$")
    ax.set_ylabel("best CPU baseline / This work")
    ax.legend(loc="upper right")
    S.save(fig, "figS7c")


if __name__ == "__main__":
    main()
