"""Fig. 4g: probability P_s of reaching 0.98 Psi_ref (L = 20) for static and batched libraries and the complete library,
as a heat map: columns K at 2^18 (left) and 2^20 (right) iterations, rows the library condition."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import _style as st

ROWS = [("static", "static"), ("batched, D = 2^12", "$D$ = $2^{12}$"), ("batched, D = 2^14", "$D$ = $2^{14}$"),
        ("batched, D = 2^16", "$D$ = $2^{16}$"), ("complete, backbone share 0.5", "complete, share 0.5"),
        ("complete, uniform proposals", "complete, uniform")]
KS = [20, 40, 80]


def main():
    df = st.read("fig4", "fig4g.csv")
    cmap = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.5), constrained_layout=True, sharey=True)
    for ax, B in zip(axes, (18, 20)):
        M = np.full((len(ROWS), len(KS)), np.nan)
        for i, (cond, _) in enumerate(ROWS):
            for j, k in enumerate(KS):
                r = df[(df.iterations_log2 == B) & (df.K == k) & (df.condition == cond)]
                M[i, j] = r.p_reach_098_mean.iloc[0]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] < 0.55 else "black")
        ax.axhline(3.5, color="white", lw=2.0)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([str(k) for k in KS])
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(f"$K$ ($2^{{{B}}}$ iterations)")
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([lab for _, lab in ROWS])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[0, 0.5, 1])
    cb.set_label("$P_{\\rm s}$")
    st.save(fig, "fig4g")


if __name__ == "__main__":
    main()
