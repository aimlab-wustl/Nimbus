"""Fig. S9d: acceptance a against sharing ratio rho, synthetic family, colored by anneal progress."""
import event_style as E

SYN = ["syn_k2_rowswap_path_q16", "syn_k2_rowswap_path_q9", "syn_k2_flip", "syn_k3_flip", "syn_k4_flip", "syn_k2_onehot"]


def main():
    E.plane("figS9d", ("figS9", "figS9d.csv"), SYN)


if __name__ == "__main__":
    main()
