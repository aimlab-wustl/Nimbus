"""Fig. S3a: connections of the surface-code machine (one syndrome per shot) versus code distance d."""
import _style as S
import size_style as Z


def main():
    d = S.read("figS3", "figS3a.csv")
    fig, ax = S.new()
    for lib in ("stabilizers", "stabilizers_logical_all"):
        lab, col, mk = Z.LIB[lib]
        s = d[d.lib == lib].sort_values("size")
        ax.plot(s["size"], s.routes, color=col, marker=mk, label=lab)
    ax.set_yscale("log")
    ax.set_xticks([3, 5, 7, 9, 11])
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("connections")
    ax.legend(loc="upper left")
    S.save(fig, "figS3a")


if __name__ == "__main__":
    main()
