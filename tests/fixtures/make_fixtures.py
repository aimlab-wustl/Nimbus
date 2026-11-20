"""Regenerate the original tiny format fixtures under tests/fixtures (run from the repo root).

    .venv/Scripts/python tests/fixtures/make_fixtures.py

All fixtures written here are original to this project. Files are written with LF line
endings; tests/fixtures/.gitattributes disables any end-of-line conversion so checksums are stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from nimbus.problems.formats.alist import ParityCheckMatrix, format_alist  # noqa: E402


def w(rel: str, text: str) -> Path:
    p = HERE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("ascii"))
    return p


# =============================================================================================
# alist: original 10-bit parity-check matrix, 3 independent checks + 1 redundant row (r4 = r1 xor r2)
# =============================================================================================
ROWS = [[0, 1, 2, 4], [2, 3, 5, 6], [4, 5, 7, 8, 9], [0, 1, 3, 4, 5, 6]]
H = ParityCheckMatrix.from_rows(10, ROWS)
padded = format_alist(H, zero_padding=True)
w("alist/toy_10_4.alist", padded)
w("alist/toy_10_4_aff3ct.alist", format_alist(H, zero_padding=False, comments=[
    "nimbus original fixture: N=10 variables, M=4 checks, rank 3 (row 4 = row 1 xor row 2), K=7",
    "AFF3CT-style layout: '#' comments, unpadded lists; same matrix as toy_10_4.alist",
]))
vn_only = "\n".join(padded.splitlines()[: 4 + 10]) + "\n"  # header + VN-to-CN block only
w("alist/toy_10_4_vn_only.alist", vn_only)

lines = padded.splitlines()
# line indices (0-based): 0 "10 4", 1 "3 6", 2 d_v, 3 d_c, 4..13 VN lists, 14..17 CN lists


def variant(rel: str, edits: dict, keep: int | None = None, extra: list[str] = ()) -> None:
    ls = list(lines)
    for i, txt in edits.items():
        ls[i] = txt
    if keep is not None:
        ls = ls[:keep]
    w(rel, "\n".join(ls + list(extra)) + "\n")


variant("malformed/alist_truncated.alist", {}, keep=9)
variant("malformed/alist_duplicate.alist", {4: "1 1 0", 2: "2 2 2 2 3 3 2 1 1 1"})
variant("malformed/alist_degree_mismatch.alist", {2: "2 2 2 2 3 3 2 2 0 1"})
variant("malformed/alist_degree_sum.alist", {3: "4 4 4 6"})
variant("malformed/alist_out_of_range.alist", {13: "5 0 0"})
variant("malformed/alist_disagree.alist", {16: "5 6 7 9 10 0"})
variant("malformed/alist_zero_inside.alist", {4: "1 0 4"})
variant("malformed/alist_dmax.alist", {1: "2 6"})
variant("malformed/alist_trailing.alist", {}, extra=["1 2 3"])
variant("malformed/alist_not_integer.alist", {5: "1 x 0"})

# =============================================================================================
# OPB (documented PB24 subset)
# =============================================================================================
w("opb/chain_precedence.opb", """* #variable= 6 #constraint= 3 #equal= 0 intsize= 3 #product= 2 sizeproduct= 5
* nimbus original fixture: two precedence chains x1 <- x2 <- x3 (x2 -> x1, x3 -> x2) and x4 <- x5; x6 free.
* Boolean-to-spin: x = (1+s)/2, x=1 <-> s=+1 (selected).
min: -3 x3 +2 x1 x5 -1 x2 x4 ~x6 +1 x6 ;
+1 x1 -1 x2 >= 0 ;
+1 x2 -1 x3 >= 0 ;
+1 x4 -1 x5 >= 0 ;
""")
w("opb/or_cover_signed.opb", """* #variable= 7 #constraint= 3 #equal= 0 intsize= 4
* nimbus original fixture: disjoint hard OR requirements with signed literals.
* (x1 or ~x2 or x3), written with a negated literal (general OPB);
* (x4 or ~x5 or ~x6), written in normalized linear form +1 x4 -1 x5 -1 x6 >= -1;
* (x7), a one-literal requirement that fixes x7 = 1.
min: +2 x1 +1 x2 -1 x3 +3 x4 x5 -2 x6 +1 x7 ;
+1 x1 +1 ~x2 +1 x3 >= 1 ;
+1 x4 -1 x5 -1 x6 >= -1 ;
+1 x7 >= 1 ;
""")
w("opb/amo_packing.opb", """* #variable= 6 #constraint= 2 #equal= 0 intsize= 3
* nimbus original fixture: unit-capacity packing blocks {x1,x2,x3} and {x4,x5} (at most one selected each); x6 free.
* sum x <= 1 is written -1 x1 -1 x2 ... >= -1 because PB24 OPB has no '<='.
min: -1 x1 -2 x2 -3 x3 +1 x1 x4 -2 x5 -1 x6 ;
-1 x1 -1 x2 -1 x3 >= -1 ;
-1 x4 -1 x5 >= -1 ;
""")
w("opb/cardinality.opb", """* #variable= 5 #constraint= 1 #equal= 1 intsize= 3
* nimbus original fixture: exact cardinality (choose exactly 2 of 5).
min: +1 x1 x2 -2 x3 +1 x4 x5 ;
+1 x1 +1 x2 +1 x3 +1 x4 +1 x5 = 2 ;
""")
w("opb/unsupported_overlap.opb", """* #variable= 3 #constraint= 2 #equal= 0
* nimbus original fixture: overlapping covers (x2 in both) are outside the supported disjoint-cover family.
min: +1 x1 +1 x2 +1 x3 ;
+1 x1 +1 x2 >= 1 ;
+1 x2 +1 x3 >= 1 ;
""")
w("opb/unsupported_forms.opb", """* #variable= 8 #constraint= 4 #equal= 0
* nimbus original fixture: constraint forms outside the supported subset.
+2 x1 +3 x2 >= 3 ;
+1 x3 +1 x4 +1 x5 >= 2 ;
+1 x6 x7 >= 1 ;
+1 x8 -1 x8 >= 0 ;
""")
w("opb/unsupported_branching.opb", """* #variable= 3 #constraint= 2 #equal= 0
* nimbus original fixture: x3 -> x1 and x3 -> x2 is a precedence DAG, not a chain.
+1 x1 -1 x3 >= 0 ;
+1 x2 -1 x3 >= 0 ;
""")
w("malformed/opb_missing_semicolon.opb", "* #variable= 2 #constraint= 1\nmin: +1 x1 ;\n+1 x1 +1 x2 >= 1\n")
w("malformed/opb_le_operator.opb", "* #variable= 2 #constraint= 1\n+1 x1 +1 x2 <= 1 ;\n")
w("malformed/opb_max_objective.opb", "* #variable= 2 #constraint= 1\nmax: +1 x1 ;\n+1 x1 +1 x2 >= 1 ;\n")
w("malformed/opb_bad_literal.opb", "* #variable= 2 #constraint= 1\n+1 x1 +1 y2 >= 1 ;\n")
w("malformed/opb_header_mismatch.opb", "* #variable= 2 #constraint= 3\n+1 x1 +1 x2 >= 1 ;\n-1 x1 -1 x2 >= -1 ;\n")
w("malformed/opb_objective_not_first.opb", "* #variable= 2 #constraint= 1\n+1 x1 +1 x2 >= 1 ;\nmin: +1 x1 ;\n")
w("malformed/opb_no_coefficient.opb", "* #variable= 2 #constraint= 1\nx1 + x2 >= 1 ;\n")
w("malformed/opb_undeclared_variable.opb", "* #variable= 2 #constraint= 1\n+1 x1 +1 x3 >= 1 ;\n")

# =============================================================================================
# spinpoly (project schema nimbus.spinpoly/1)
# =============================================================================================
TOY = """# Six-spin exact-cardinality toy (Results, Visualizing operating principle using a toy problem; Methods, An example toy problem).
# Exactly three of six vertices are selected; completing triangle {1,2,3} rewards 4 and triangle {3,4,5}
# rewards 7; vertex 6 completes neither. A selected vertex has s_i = +1:
#   E0(s) = -4 prod_{i in {1,2,3}} (1+s_i)/2 - 7 prod_{i in {3,4,5}} (1+s_i)/2,   sum_i (1+s_i)/2 = 3.
# Sign convention (Methods, Derivation of the transition-programmable architecture): E = E_const - sum_k J_k T_k with T_k = prod_{i in S_k} s_i,
# so each `weight` below is J_k. Expansion: 13 nonconstant clauses (5 unary, 6 pair, 2 cubic); the first
# triangle contributes J = 1/2 to each of its clauses and the second 7/8, so the shared unary clause s3 has
# J = 1/2 + 7/8 = 11/8, and E_const = -1/2 - 7/8 = -11/8. Spin indices are 1-based (index_base: 1).
# Start {1,2,3}: E = -4. Unique optimum {3,4,5}: E = -7. 20 feasible states.
schema: nimbus.spinpoly/1
name: six_spin_toy
description: Paper six-spin exact-cardinality toy (two rewarded triples, choose 3 of 6)
n_spins: 6
index_base: 1
spin_convention: plus_one_selected
energy_convention: minus_J
const: -11/8
terms:
  - {support: [1], weight: 1/2}
  - {support: [2], weight: 1/2}
  - {support: [3], weight: 11/8}
  - {support: [4], weight: 7/8}
  - {support: [5], weight: 7/8}
  - {support: [1, 2], weight: 1/2}
  - {support: [1, 3], weight: 1/2}
  - {support: [2, 3], weight: 1/2}
  - {support: [3, 4], weight: 7/8}
  - {support: [3, 5], weight: 7/8}
  - {support: [4, 5], weight: 7/8}
  - {support: [1, 2, 3], weight: 1/2}
  - {support: [3, 4, 5], weight: 7/8}
constraints:
  exact_cardinality:
    - {spins: [1, 2, 3, 4, 5, 6], k: 3}
labels: [v1, v2, v3, v4, v5, v6]
metadata:
  source: Results, Visualizing operating principle using a toy problem and Methods, An example toy problem
  start_selection: [1, 2, 3]
  optimal_selection: [3, 4, 5]
  start_energy: -4
  optimal_energy: -7
  penalty_lambda: 6
  barriers: {penalty_single_flip: 6, all_pair_exchanges: 4, with_double_exchange_shortcut: 0}
"""
w("spinpoly/six_spin_toy.yaml", TOY)
CARD = {
    "schema": "nimbus.spinpoly/1",
    "name": "exact_card_5_2",
    "description": "nimbus original fixture: choose exactly 2 of 5 spins; E = const + sum c T (plus_c convention), 0-based",
    "n_spins": 5,
    "index_base": 0,
    "spin_convention": "plus_one_selected",
    "energy_convention": "plus_c",
    "const": 0.25,
    "terms": [
        {"support": [0], "weight": -1.0},
        {"support": [1, 2], "weight": 0.5},
        {"support": [0, 3, 4], "weight": -2.0},
        {"support": [1, 2, 3, 4], "weight": "-3/4"},
    ],
    "constraints": {"exact_cardinality": [{"spins": [0, 1, 2, 3, 4], "k": 2}]},
}
w("spinpoly/exact_card_5_2.json", json.dumps(CARD, indent=1) + "\n")
MIXED = """# nimbus original fixture: one block of each scoped family plus a free spin (0-based).
schema: nimbus.spinpoly/1
name: mixed_blocks
n_spins: 9
spin_convention: plus_one_selected
energy_convention: minus_J
const: 0
terms:
  - {support: [0, 3], weight: 1.5}
  - {support: [2, 5, 8], weight: -0.5}
  - {support: [6], weight: 2}
constraints:
  chains: [[0, 1, 2]]
  or_blocks: [{spins: [3, 4], signs: [1, -1]}]
  amo_blocks: [[5, 6, 7]]
  free: [8]
"""
w("spinpoly/mixed_blocks.yaml", MIXED)


def toy_variant(rel: str, old: str, new: str) -> None:
    assert old in TOY, old
    w(rel, TOY.replace(old, new, 1))


toy_variant("malformed/spinpoly_unknown_field.yaml", "n_spins: 6\n", "n_spins: 6\nn_clauses: 13\n")
toy_variant("malformed/spinpoly_nan_weight.yaml", "{support: [4], weight: 7/8}", "{support: [4], weight: .nan}")
toy_variant("malformed/spinpoly_out_of_range.yaml", "{support: [3, 4, 5], weight: 7/8}", "{support: [3, 4, 7], weight: 7/8}")
toy_variant("malformed/spinpoly_bad_schema.yaml", "schema: nimbus.spinpoly/1", "schema: nimbus.spinpoly/2")
toy_variant("malformed/spinpoly_duplicate_support.yaml", "{support: [2, 3], weight: 1/2}", "{support: [3, 1], weight: 1/2}")
toy_variant("malformed/spinpoly_repeated_index.yaml", "{support: [2, 3], weight: 1/2}", "{support: [2, 2], weight: 1/2}")
toy_variant("malformed/spinpoly_bad_k.yaml", "k: 3}", "k: 7}")
toy_variant("malformed/spinpoly_unknown_term_key.yaml", "{support: [1], weight: 1/2}", "{support: [1], weight: 1/2, role: gate}")
toy_variant("malformed/spinpoly_bad_convention.yaml", "energy_convention: minus_J", "energy_convention: J_plus")
toy_variant("malformed/spinpoly_overlap.yaml", "    - {spins: [1, 2, 3, 4, 5, 6], k: 3}\n",
            "    - {spins: [1, 2, 3, 4, 5, 6], k: 3}\n  amo_blocks: [[5, 6]]\n")
bad = dict(CARD)
bad["terms"] = [{"support": [0], "weight": True}]
w("malformed/spinpoly_bool_weight.json", json.dumps(bad, indent=1) + "\n")
w("malformed/spinpoly_invalid.yaml", "schema: nimbus.spinpoly/1\nterms: [ {support: [1}\n")

# =============================================================================================
# matching: one small synthetic point-matching tensor (n = 5, one-hot 5 x 5 permutation)
# =============================================================================================
from nimbus.problems.formats.matching_tensor import (point_pair_tensor, save_tensor,  # noqa: E402
                                                      similarity_deformation_pair)

_P1, _P2, _gt, _valid = similarity_deformation_pair(5, 0.03, np.random.default_rng(3))
(HERE / "matching").mkdir(exist_ok=True)
save_tensor(HERE / "matching" / "five.npz", point_pair_tensor(_P1, _P2, seed=3), gt=_gt)
# n = 7 (penalty-machine, calibration and shortcut-library equivalence tests)
_P1, _P2, _gt, _valid = similarity_deformation_pair(7, 0.05, np.random.default_rng(7))
save_tensor(HERE / "matching" / "seven.npz", point_pair_tensor(_P1, _P2, seed=7), gt=_gt)

print("fixtures written to", HERE)
