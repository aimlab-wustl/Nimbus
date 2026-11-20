"""Fig. 6b: acceptance a against sharing ratio rho, one dot per window, colored by anneal progress."""
import event_style as E


def main():
    E.plane("fig6b", ("fig6", "fig6b.csv"), ["m30c", "m50p", "s11", "pair2d"])


if __name__ == "__main__":
    main()
