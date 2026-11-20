"""Fig. S9a: operations per iteration, event-driven / recomputation, four workloads (95% interval over replicas)."""
import event_style as E


def main():
    E.per_window("figS9a", ("figS9", "figS9a.csv"), ["m30c", "m50p", "s11", "pair2d"])


if __name__ == "__main__":
    main()
