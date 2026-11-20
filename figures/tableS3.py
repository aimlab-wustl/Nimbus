"""Supplementary Table S3: batched versus static libraries (L = 20, 20 test instances, 256 replicas per instance)."""
from __future__ import annotations

import _style as st


def sgn(x):
    return f"+{x:.2f}" if x >= 0 else f"{x:.2f}"


def main():
    df = st.read("tables", "tableS3.csv").sort_values(["K", "iterations_log2", "dwell_log2"])
    rows = [(r.K, f"2^{r.iterations_log2}", f"2^{r.dwell_log2}", f"{r.static_p_reach_098:.2f}",
             f"{r.batched_p_reach_098:.2f}", sgn(r.difference),
             f"[{sgn(r.difference_ci95_lo)}, {sgn(r.difference_ci95_hi)}]") for r in df.itertuples()]
    st.write_table("tableS3", ["K", "budget", "D", "static", "batched", "difference", "95% CI"], rows,
                   "Probability that a replica reaches 0.98 Psi_ref within the budget (mean over instances); difference = "
                   "batched minus static, paired over instances, with its 95% bootstrap confidence interval.")


if __name__ == "__main__":
    main()
