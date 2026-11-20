"""Supplementary Table S2: selected schedule settings (markdown)."""
from _style import read, write_table


def setting(r):
    if r.m == r.m and str(r.m) != "nan":
        return f"m = {r.m:g}, kappa = {r.kappa:g}"
    s = f"lambda-bar = {r.penalty_strength_per_psi_unit:g}, tau_end = {r.end_temperature_per_psi_unit:g}, " \
        f"ratio {r.start_to_end_ratio:g}"
    return s


def main():
    df = read("tables", "tableS2.csv")
    rows = [(r.problem_set, r.L, r.group, r.method, setting(r)) for r in df.itertuples()]
    write_table("tableS2", ["Problem set", "L", "Experiment", "Method", "Selected setting"], rows,
                note="This work: start factor m and minimum ratio kappa. Penalty machines: penalty strength lambda-bar "
                     "and end temperature tau_end in units of Psi*/L, and the ratio of start to end temperature.")


if __name__ == "__main__":
    main()
