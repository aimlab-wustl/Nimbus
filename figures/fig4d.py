"""Fig. 4d: iterations to solution at L = 10 (target Psi*, automatically calibrated schedule) versus library size."""
import _style as S
from fig4b import draw, legend


def main():
    d = S.read("fig4", "fig4d.csv")
    fig, ax = S.new((3.3, 2.8))
    draw(ax, d, "its_iterations", "its_ci95_lo", "its_ci95_hi", 10)
    ax.set_yscale("log")
    ax.set_ylim(1e3, 3e7)
    ax.set_xlim(7.5, 48)
    ax.set_xticks([9, 18, 27, 36, 45])
    ax.set_xlabel("exchanges ($L$ = 10)")
    ax.set_ylabel("iterations to solution")
    legend(ax)
    S.save(fig, "fig4d")


if __name__ == "__main__":
    main()
