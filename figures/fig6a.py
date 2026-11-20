"""Fig. 6a: cumulative operations of event-driven updating / recomputation along one anneal, four workloads."""
import _style as S
import event_style as E

WL = ["m30c", "m50p", "s11", "pair2d"]


def main():
    s = S.read("fig6", "fig6_summary.csv").set_index("workload")
    E.cumulative("fig6a", ("fig6", "fig6a.csv"), WL, break_even=s.break_even_iteration.to_dict())


if __name__ == "__main__":
    main()
