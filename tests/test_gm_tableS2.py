"""Supplementary Table S2 is generated from the selection outputs in experiments/graph_matching/configs and lists the
settings printed in the paper."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPER = {  # (problem set, L, method) -> (m, kappa) or (lambda-bar, tau_end, ratio)
    ("Synthetic", 10, "This work, complete"): (0.125, 2), ("Synthetic", 10, "This work, path"): (8, 2),
    ("Synthetic", 10, "N-HOIM, penalty"): (1, 0.3, 30), ("Synthetic", 10, "NeuroSA, quadratized penalty"): (2, 0.3, 100),
    ("WILLOW", 10, "This work, complete"): (2, 32), ("WILLOW", 10, "This work, path"): (0.5, 1.5),
    ("WILLOW", 10, "N-HOIM, penalty"): (1, 0.3, 30), ("WILLOW", 10, "NeuroSA, quadratized penalty"): (2, 0.3, 10),
    ("Synthetic", 15, "This work, complete"): (0.5, 2), ("Synthetic", 20, "This work, complete"): (0.5, 2),
    ("Synthetic", 25, "This work, complete"): (0.5, 2), ("Synthetic", 30, "This work, complete"): (0.25, 2),
    ("Synthetic", 20, "Informed, K=40"): (2, 2), ("Synthetic", 20, "Random, K=40"): (4, 2),
    ("Synthetic", 20, "Informed, K=80"): (1, 2), ("Synthetic", 20, "Random, K=80"): (1, 2),
    ("Synthetic", 20, "Complete, uniform proposals"): (0.5, 2), ("Synthetic", 20, "Complete, backbone share 0.5"): (0.5, 2),
    ("Synthetic", 10, "Path"): (4, 8), ("Synthetic", 10, "Informed, K=4"): (2, 8), ("Synthetic", 10, "Random, K=4"): (2, 8),
    ("Synthetic", 10, "Informed, K=9"): (2, 2), ("Synthetic", 10, "Random, K=9"): (8, 32),
    ("Synthetic", 10, "Informed, K=18"): (0.5, 4), ("Synthetic", 10, "Random, K=18"): (1, 4),
    ("Synthetic", 10, "Informed, K=27"): (0.25, 2), ("Synthetic", 10, "Random, K=27"): (0.5, 2),
    ("Synthetic", 10, "Complete, backbone share 0.5"): (0.5, 4),
}


def test_table_s2_matches_paper():
    spec = importlib.util.spec_from_file_location("tableS2", ROOT / "experiments/graph_matching/analysis/tableS2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    got = {}
    for r in mod.rows():
        key = (r["problem_set"], int(r["L"]), r["method"])
        if r["m"]:
            got[key] = (float(r["m"]), float(r["kappa"]))
        else:
            got[key] = (float(r["penalty_strength_per_psi_unit"]), float(r["end_temperature_per_psi_unit"]),
                        float(r["start_to_end_ratio"]))
    assert got == {k: tuple(float(x) for x in v) for k, v in PAPER.items()}
