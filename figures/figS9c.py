"""Fig. S9c: cumulative operations of event-driven updating / recomputation, synthetic family."""
import event_style as E

SYN = ["syn_k2_rowswap_path_q16", "syn_k2_rowswap_path_q9", "syn_k2_flip", "syn_k3_flip", "syn_k4_flip", "syn_k2_onehot"]


def main():
    E.cumulative("figS9c", ("figS9", "figS9c.csv"), SYN)


if __name__ == "__main__":
    main()
