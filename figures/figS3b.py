"""Fig. S3b: connections of the matching machine versus the number of keypoints L, complete and path library."""
import _style as S
import size_style as Z


def main():
    d = S.read("figS3", "figS3b.csv")
    fig, ax = S.new()
    for lib in ("complete", "path"):
        lab, col, mk = Z.LIB[lib]
        s = d[d.lib == lib].sort_values("size")
        ax.plot(s["size"], s.routes, color=col, marker=mk, label=lab)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels(["10", "20", "30", "40", "50"])
    ax.minorticks_off()
    ax.set_xlabel("keypoints $L$")
    ax.set_ylabel("connections")
    ax.legend(loc="lower right")
    S.save(fig, "figS3b")


if __name__ == "__main__":
    main()
