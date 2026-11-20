"""Fig. S2b: best value per replica, Psi/Psi_ref, versus the fraction of proposals the replica accepted over the run."""
import _style as S
from figS2a import ARMS


def main():
    d = S.read("figS2", "figS2b.csv")
    fig, ax = S.new((3.3, 2.6))
    for arm, (col, mk) in ARMS.items():
        s = d[d.variant == arm]
        ax.scatter(s.acceptance_fraction, s.psi_over_psi_ref, s=3.0, color=col, marker=mk, linewidths=0, alpha=0.6,
                   zorder=3 if arm == "anneal" else 2, label=arm)
    ax.axhline(0.98, color="0.3", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fraction of proposals accepted over the run")
    ax.set_ylabel("best value per replica, $\\Psi/\\Psi_{\\rm ref}$")
    ax.legend(loc="lower left", markerscale=2.5, handletextpad=0.2)
    S.save(fig, "figS2b")


if __name__ == "__main__":
    main()
