"""Fig. S3e: shares of the information needed to program and run the surface-code machine (total above each bar)."""
import size_style as Z


def main():
    Z.shares("figS3e", ("figS3", "figS3e.csv"), ["stabilizers", "stabilizers_logical_all"], "code distance $d$", (5.0, 3.0))


if __name__ == "__main__":
    main()
