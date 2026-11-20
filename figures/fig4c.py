"""Fig. 4c: critical depth h*/Psi* (top) and Catoni difficulty (bottom) of the exact landscapes at L = 10."""
import matplotlib.pyplot as plt

import _style as S
from fig4b import draw, handles


def main():
    d = S.read("fig4", "fig4c.csv")
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.0, 3.6), constrained_layout=True)
    for a, q in ((ax, "critical_depth_over_psi_opt"), (ax2, "catoni_difficulty")):
        draw(a, d[d.quantity == q], "median", "ci95_lo", "ci95_hi", 10)
    ax.set_ylim(0, 0.72)
    ax.set_yticks([0, 0.3, 0.6])
    ax.set_ylabel("$h^{*}/\\Psi^{*}$")
    ax.text(0.97, 0.95, "critical depth", transform=ax.transAxes, ha="right", va="top")
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 14)
    ax2.set_ylabel("difficulty")
    ax2.text(0.97, 0.95, "Catoni difficulty", transform=ax2.transAxes, ha="right", va="top")
    ax2.set_xlim(7.5, 48)
    ax2.set_xticks([9, 18, 27, 36, 45])
    ax2.set_xlabel("exchanges ($L$ = 10)")
    fig.legend(handles=handles(), loc="outside upper center", ncol=2, fontsize=6.5)
    S.save(fig, "fig4c")


if __name__ == "__main__":
    main()
