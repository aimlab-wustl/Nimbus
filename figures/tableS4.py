"""Supplementary Table S4: iterations to solution at L = 10 with the automatically calibrated and optimized schedules,
and ratios of the iterations to solution of two libraries (95% bootstrap CI)."""
import math

import _style as S

ORDER = ["path"] + [f"{t}, K={k}" for k in (4, 9, 18, 27) for t in ("informed", "random")] + ["complete"]
NAME = {"path": "Path", "complete": "Complete, backbone share 0.5"}


def fmt(x):
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}e{e}" if e >= 4 else f"{x:,.0f}"


def rfmt(x, big):
    return f"{x:.0f}" if big else f"{x:.2f}"


def main():
    t = S.read("tables", "tableS4.csv")
    lib = t[t.section == "library"].set_index(["item", "schedule"])
    rows = []
    for it in ORDER:
        a, o = lib.loc[(it, "automatic")], lib.loc[(it, "optimized")]
        name = NAME.get(it, it[0].upper() + it[1:])
        rows.append([name, f"{fmt(a.value)} [{fmt(a.ci95_lo)}, {fmt(a.ci95_hi)}]",
                     f"({float(o.m):g}, {float(o.kappa):g})", f"{fmt(o.value)} [{fmt(o.ci95_lo)}, {fmt(o.ci95_hi)}]"])
    rat = t[t.section == "ratio"]
    for it in rat.item.unique():
        g = rat[rat.item == it].set_index("schedule")
        big = g.value.max() >= 10
        cells = [f"{rfmt(g.loc[s].value, big)} [{rfmt(g.loc[s].ci95_lo, big)}, {rfmt(g.loc[s].ci95_hi, big)}]"
                 for s in ("automatic", "optimized")]
        rows.append([f"Ratio {it}", cells[0], "", cells[1]])
    S.write_table("tableS4", ["Library", "ITS automatic schedule [95% CI]", "(m, kappa) optimized",
                              "ITS optimized schedule [95% CI]"], rows,
                  note="ITS: iterations to solution (target: certified optimum), 20 synthetic test instances, "
                       "256 replicas per instance. Ratios: ITS of the first library over the second.")


if __name__ == "__main__":
    main()
