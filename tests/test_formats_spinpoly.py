"""Project schema nimbus.spinpoly/1: the paper's six-spin toy, an independent instance, strictness."""

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from nimbus.core.errors import FormatError, UnsupportedProblem
from nimbus.problems.formats.spinpoly import parse_spinpoly, read_spinpoly, write_spinpoly

pytestmark = pytest.mark.quick

SP = Path(__file__).parent / "fixtures" / "spinpoly"
BAD = Path(__file__).parent / "fixtures" / "malformed"


def all_states(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.int8)


def toy_product_form(s):
    """E0 = -4 prod_{123} (1+s)/2 - 7 prod_{345} (1+s)/2 (Results, Visualizing operating principle using a toy problem), 1-based vertices."""
    x = (1 + s.astype(int)) // 2
    return -4 * x[:, 0] * x[:, 1] * x[:, 2] - 7 * x[:, 2] * x[:, 3] * x[:, 4]


def test_toy_expansion_matches_the_manuscript():
    doc = read_spinpoly(SP / "six_spin_toy.yaml")
    assert doc.n_spins == 6 and doc.n_terms == 13
    orders = sorted(len(s) for s in doc.supports)
    assert orders.count(1) == 5 and orders.count(2) == 6 and orders.count(3) == 2
    assert doc.exact["const"] == Fraction(-11, 8) and doc.const == -1.375
    j = dict(zip(doc.supports, doc.exact["weights"]))
    assert j[(2,)] == Fraction(11, 8)  # shared singleton s3 (0-based spin 2)
    assert all(j[s] == Fraction(1, 2) for s in [(0,), (1,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
    assert all(j[s] == Fraction(7, 8) for s in [(3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)])
    states = all_states(6)
    assert np.array_equal(doc.energy(states), toy_product_form(states))  # every one of the 64 states


def test_toy_feasible_set_and_targets():
    doc = read_spinpoly(SP / "six_spin_toy.yaml")
    feas = doc.enumerate_feasible()
    assert len(feas) == 20 and np.all((feas > 0).sum(axis=1) == 3)
    e = doc.energy(feas)
    start = np.array([1, 1, 1, -1, -1, -1])
    opt = np.array([-1, -1, 1, 1, 1, -1])
    assert doc.energy(start) == -4 and doc.energy(opt) == -7
    assert np.sum(e == -7) == 1 and sorted(set(e.tolist())) == [-7.0, -4.0, 0.0]
    assert doc.is_feasible(start) and not doc.is_feasible(np.ones(6, dtype=int))
    assert doc.metadata["barriers"] == {"penalty_single_flip": 6, "all_pair_exchanges": 4,
                                        "with_double_exchange_shortcut": 0}
    assert doc.labels == ("v1", "v2", "v3", "v4", "v5", "v6") and doc.constraint_family == "exact_cardinality"


def test_independent_exact_cardinality_instance_plus_c_convention():
    doc = read_spinpoly(SP / "exact_card_5_2.json")
    # file uses E = const + sum c T; internally J = -c
    assert doc.weights.tolist() == [1.0, -0.5, 2.0, 0.75]
    feas = doc.enumerate_feasible()
    assert len(feas) == 10
    for s in feas:
        s = s.astype(int)
        direct = 0.25 - 1.0 * s[0] + 0.5 * s[1] * s[2] - 2.0 * s[0] * s[3] * s[4] - 0.75 * s[1] * s[2] * s[3] * s[4]
        assert doc.energy(s) == pytest.approx(direct)


def test_mixed_blocks_feasibility_semantics():
    doc = read_spinpoly(SP / "mixed_blocks.yaml")
    assert doc.constraint_family == "mixed" and doc.free == (8,)
    states = all_states(9)
    sel = states > 0
    chain_ok = np.array([not any((not r[0] and r[1], not r[1] and r[2])) for r in sel[:, :3]])
    or_ok = (states[:, 3] == 1) | (states[:, 4] == -1)
    amo_ok = sel[:, 5:8].sum(axis=1) <= 1
    assert np.array_equal(doc.is_feasible(states), chain_ok & or_ok & amo_ok)
    assert len(doc.enumerate_feasible()) == 4 * 3 * 4 * 2


@pytest.mark.parametrize("index_base, suffix", [(0, ".yaml"), (1, ".json"), (1, ".yml")])
def test_write_read_roundtrip(tmp_path, index_base, suffix):
    doc = read_spinpoly(SP / "six_spin_toy.yaml")
    p = write_spinpoly(tmp_path / f"toy{suffix}", doc, index_base=index_base)
    back = read_spinpoly(p)
    assert back.supports == doc.supports and np.array_equal(back.weights, doc.weights)
    assert back.const == doc.const and back.exact_cardinality == doc.exact_cardinality
    assert back.exact["weights"] == doc.exact["weights"]
    text = p.read_text()
    assert "11/8" in text  # exact rationals are written as 'p/q'


def test_to_spin_polynomial_bridge():
    base = pytest.importorskip("nimbus.problems.base")
    if not hasattr(base, "SpinPolynomial"):
        pytest.skip("nimbus.problems.base.SpinPolynomial not available")
    doc = read_spinpoly(SP / "six_spin_toy.yaml")
    poly = doc.to_spin_polynomial()
    states = all_states(6)
    assert np.allclose(poly.evaluate(states), doc.energy(states))
    assert np.array_equal(poly.clause_weights(), doc.weights)


@pytest.mark.parametrize("name, exc, fragment", [
    ("spinpoly_unknown_field.yaml", FormatError, "unknown field(s) ['n_clauses']"),
    ("spinpoly_nan_weight.yaml", FormatError, "terms[3].weight: non-finite value nan"),
    ("spinpoly_out_of_range.yaml", FormatError, "spin index 7 out of range 1..6"),
    ("spinpoly_bad_schema.yaml", FormatError, "expected 'nimbus.spinpoly/1'"),
    ("spinpoly_duplicate_support.yaml", FormatError, "same support as terms[6]"),
    ("spinpoly_repeated_index.yaml", FormatError, "spin index 2 repeated"),
    ("spinpoly_bad_k.yaml", FormatError, "k=7 outside 0..6"),
    ("spinpoly_unknown_term_key.yaml", FormatError, "terms[0]: unknown field(s) ['role']"),
    ("spinpoly_bad_convention.yaml", FormatError, "energy_convention: must be one of"),
    ("spinpoly_bool_weight.json", FormatError, "expected a number, found boolean True"),
    ("spinpoly_invalid.yaml", FormatError, "invalid YAML"),
    ("spinpoly_overlap.yaml", UnsupportedProblem, "constraint blocks must be disjoint"),
])
def test_malformed_spinpoly(name, exc, fragment):
    path = BAD / name
    with pytest.raises(exc) as err:
        read_spinpoly(path)
    assert str(err.value).startswith(str(path)) and fragment in str(err.value)


def test_string_parsing_and_validation_messages():
    base = ("schema: nimbus.spinpoly/1\nn_spins: 2\nspin_convention: plus_one_selected\n"
            "energy_convention: minus_J\nconst: 0\nterms: [{support: [0, 1], weight: '3/4'}]\n")
    doc = parse_spinpoly(base, "inline")
    assert doc.weights.tolist() == [0.75] and doc.free == (0, 1)
    with pytest.raises(FormatError, match=r"^inline: terms\[0\]\.support: empty list"):
        parse_spinpoly(base.replace("[0, 1]", "[]"), "inline")
    with pytest.raises(FormatError, match="spin_convention"):
        parse_spinpoly(base.replace("plus_one_selected", "minus_one_selected"), "inline")
    with pytest.raises(FormatError, match="explicitly but spins"):
        parse_spinpoly(base + "constraints: {amo_blocks: [[0]], free: []}\n", "inline")
    with pytest.raises(FormatError, match="unknown extension"):
        read_spinpoly(Path(__file__).parent / "fixtures" / "opb" / "cardinality.opb")
