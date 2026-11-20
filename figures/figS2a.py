"""Fig. S2a: best value per replica, Psi/Psi_ref, of the anneal, no-cooling and random-walk runs (bars: medians)."""
import numpy as np

import _style as S

ARMS = {"anneal": (S.C["ours"], "o"), "no cooling": (S.C["hot"], "s"), "random walk": (S.C["random"], "^")}


def main():
    d = S.read("figS2", "figS2a.csv")
    fig, ax = S.new((3.6, 2.6))
    rng = np.random.default_rng(11)               # horizontal jitter of the dots only
    keys = sorted({(r.L, r.instance_index) for r in d.itertuples()})
    xs = []
    for i, (L, idx) in enumerate(keys):
        x0 = i + (1 if L == 30 else 0)            # gap between L = 20 and L = 30
        xs.append(x0)
        for j, (arm, (col, mk)) in enumerate(ARMS.items()):
            v = d[(d.L == L) & (d.instance_index == idx) & (d.variant == arm)].psi_over_psi_ref.values
            x = x0 + (j - 1) * 0.27
            ax.scatter(x + rng.uniform(-0.08, 0.08, v.size), v, s=2.5, color=col, marker=mk, linewidths=0, alpha=0.55,
                       zorder=2, label=arm if i == 0 else None)
            ax.plot([x - 0.11, x + 0.11], [np.median(v)] * 2, color="0.1", lw=1.2, zorder=3)
    ax.axhline(0.98, color="0.3", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(7.45, 0.955, "0.98 $\\Psi_{\\rm ref}$", ha="right", va="top", color="0.3")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(idx) for _, idx in keys])
    ax.tick_params(axis="x", length=0)
    for x, L in ((1, 20), (5, 30)):
        ax.text(x, -0.13, f"$L$ = {L}", ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("best value per replica, $\\Psi/\\Psi_{\\rm ref}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, markerscale=2.5, handletextpad=0.2, columnspacing=1.0)
    S.save(fig, "figS2a")


if __name__ == "__main__":
    main()
