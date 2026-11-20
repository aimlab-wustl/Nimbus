"""Fig. S8a: energy per decode on the board: core-rail power while decoding times the decode time, per shot at the fixed
budget (one syndrome per shot) and per round (real time). Triangles: lower bounds."""
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import _style as S


def main():
    t = S.read("figS8", "figS8a.csv")
    fig, ax = S.new()
    for mode, ls, mfc, lab in (("one_shot", "-", S.C["ours"], "one syndrome per shot, per shot"),
                               ("real_time", (0, (3, 1.5)), "white", "real time, per round")):
        s = t[t["mode"] == mode].sort_values("d")
        ax.plot(s.d, s.energy_j, color=S.C["ours"], ls=ls, lw=1.2, label=lab)
        for lb, mk in (("no", "o" if mode == "one_shot" else "s"), ("yes", "^")):
            q = s[s.lower_bound == lb]
            ax.plot(q.d, q.energy_j, ls="none", marker=mk, ms=3.6, mfc=mfc, mec=S.C["ours"], mew=0.8)
    ax.plot([], [], ls="none", marker="^", ms=3.6, color=S.C["ours"], label="lower bound")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3.0)
    lab = {1e-6: "1 µJ", 1e-5: "10 µJ", 1e-4: "100 µJ", 1e-3: "1 mJ", 1e-2: "10 mJ", 1e-1: "100 mJ"}
    ax.yaxis.set_major_locator(FixedLocator(list(lab)))
    ax.yaxis.set_major_formatter(FixedFormatter(list(lab.values())))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([3, 5, 7, 9, 11, 13, 15])
    ax.set_xlim(2.5, 15.5)
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("energy per decode")
    ax.legend(loc="upper left")
    S.save(fig, "figS8a")


if __name__ == "__main__":
    main()
