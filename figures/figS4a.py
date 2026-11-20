"""Supplementary Fig. S4a: probability of reaching 0.98 Psi_ref within 2^22 iterations versus library size (L = 30)."""
from __future__ import annotations

import _style as st
from fig4a import draw


def main():
    draw(st.read("figS4", "figS4a.csv"), "figS4a", [30, 50, 100, 200], (25, 300))


if __name__ == "__main__":
    main()
