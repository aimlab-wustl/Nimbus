"""Fig. S3c: objective and auxiliary clauses of the matching machine versus L (auxiliary: R (L + L(L-1)/2) exactly)."""
import _style as S
import size_style as Z


def main():
    d = S.read("figS3", "figS3c.csv")
    fig, ax = S.new()
    o = d[d.lib == "path"].sort_values("size")
    ax.plot(o["size"], o.M_obj, color="0.1", marker="s", label="objective clauses")
    for lib in ("complete", "path"):
        lab, col, mk = Z.LIB[lib]
        s = d[d.lib == lib].sort_values("size")
        ax.plot(s["size"], s.M_gate, color=col, marker=mk, label=f"auxiliary clauses, {lab}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels(["10", "20", "30", "40", "50"])
    ax.minorticks_off()
    ax.set_xlabel("keypoints $L$")
    ax.set_ylabel("clauses")
    ax.legend(loc="upper left")
    S.save(fig, "figS3c")


if __name__ == "__main__":
    main()
