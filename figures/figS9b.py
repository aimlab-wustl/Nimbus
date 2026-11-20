"""Fig. S9b: operations per iteration, event-driven / recomputation, synthetic family (95% interval over replicas)."""
import event_style as E

SYN = ["syn_k2_rowswap_path_q16", "syn_k2_rowswap_path_q9", "syn_k2_flip", "syn_k3_flip", "syn_k4_flip", "syn_k2_onehot"]


def main():
    E.per_window("figS9b", ("figS9", "figS9b.csv"), SYN)


if __name__ == "__main__":
    main()
