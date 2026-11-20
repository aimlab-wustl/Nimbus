"""Fig. 4b: relaxation time at infinite temperature versus exchanges in the library, L = 20 (solid) and L = 30 (dashed)."""
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as S


def draw(ax, df, y, lo, hi, L, ls="-"):
    """Path, informed, random and complete (backbone share 0.5) for one L; x = exchanges in the library."""
    for lib in ("informed", "random"):
        t = df[df.library == lib].sort_values("K")
        lab, col, mk, filled = S.LIBRARY[lib]
        x = t.exchanges * (0.97 if lib == "informed" else 1.03)
        ax.errorbar(x, t[y], yerr=S.yerr(t, y, lo, hi), color=col, marker=mk, ls=ls, mfc=col if filled else "white",
                    zorder=3)
    for lib, key, ms in (("path", "path", 4), ("complete", "share", 7)):
        t = df[df.library == lib]
        lab, col, mk, filled = S.LIBRARY[key]
        ax.errorbar(t.exchanges, t[y], yerr=S.yerr(t, y, lo, hi), color=col, marker=mk, ms=ms, ls="",
                    mec="black" if key == "share" else col, mew=0.5, zorder=4)


def handles():
    hs = []
    for key in ("path", "informed", "random", "share"):
        lab, col, mk, filled = S.LIBRARY[key]
        hs.append(Line2D([], [], color=col, marker=mk, ls="-" if key in ("informed", "random") else "",
                         mfc=col if filled else "white", mec="black" if key == "share" else col,
                         ms=7 if key == "share" else 3.5, mew=0.5, label=lab))
    return hs


def legend(ax, loc="upper right"):
    ax.legend(handles=handles(), loc=loc, fontsize=6.5)


def main():
    d = S.read("fig4", "fig4b.csv")
    y, lo, hi = ("relaxation_time_median_iterations", "relaxation_time_ci95_lo_iterations",
                 "relaxation_time_ci95_hi_iterations")
    fig, ax = S.new((4.0, 3.2))
    for L, ls in ((20, "-"), (30, "--")):
        draw(ax, d[d.L == L], y, lo, hi, L, ls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(15, 520)
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(FixedFormatter(["20", "50", "100", "200", "500"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(8, 5000)
    ax.text(21.5, 772, "$L$ = 20 (solid)", va="center", ha="left")
    ax.text(32.0, 2647, "$L$ = 30 (dashed)", va="center", ha="left")
    ax.set_xlabel("exchanges in the library")
    ax.set_ylabel("relaxation time at infinite\ntemperature (iterations)")
    fig.legend(handles=handles(), loc="outside upper center", ncol=2, fontsize=6.5)
    S.save(fig, "fig4b")


if __name__ == "__main__":
    main()
