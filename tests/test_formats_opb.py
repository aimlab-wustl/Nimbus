"""OPB subset parser, constraint-family recognizer and Boolean-to-spin conversion."""

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from nimbus.core.errors import FormatError, UnsupportedProblem
from nimbus.problems.formats.opb import (
    format_opb,
    objective_spin_terms,
    parse_opb,
    read_opb,
    recognize_constraints,
    to_spinpoly,
)

pytestmark = pytest.mark.quick

OPB = Path(__file__).parent / "fixtures" / "opb"
BAD = Path(__file__).parent / "fixtures" / "malformed"
SUPPORTED = ["chain_precedence.opb", "or_cover_signed.opb", "amo_packing.opb", "cardinality.opb"]


def assignments(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)


def test_parse_chain_fixture():
    m = read_opb(OPB / "chain_precedence.opb")
    assert m.n_vars == 6 and len(m.constraints) == 3
    assert m.header == {"#variable": 6, "#constraint": 3, "#equal": 0, "intsize": 3, "#product": 2, "sizeproduct": 5}
    assert [t.literals for t in m.objective] == [(3,), (1, 5), (2, 4, -6), (6,)]
    assert [t.coeff for t in m.objective] == [-3, 2, -1, 1]
    c = m.constraints[0]
    assert (c.op, c.rhs, c.line, [(t.coeff, t.literals) for t in c.terms]) == (">=", 0, 5, [(1, (1,)), (-1, (2,))])


def test_recognized_blocks_of_each_fixture():
    r = recognize_constraints(read_opb(OPB / "chain_precedence.opb"))
    assert r.chains == ((0, 1, 2), (3, 4)) and r.free == (5,) and r.families == ("chain_precedence",)
    assert r.kinds == ("implication",) * 3
    r = recognize_constraints(read_opb(OPB / "or_cover_signed.opb"))
    assert r.or_blocks == (((0, 1, 2), (1, -1, 1)), ((3, 4, 5), (1, -1, -1)), ((6,), (1,)))
    assert r.free == () and r.families == ("disjoint_or_cover",)
    r = recognize_constraints(read_opb(OPB / "amo_packing.opb"))
    assert r.amo_blocks == ((0, 1, 2), (3, 4)) and r.free == (5,)
    r = recognize_constraints(read_opb(OPB / "cardinality.opb"))
    assert r.exact_cardinality == (((0, 1, 2, 3, 4), 2),) and r.kinds == ("cardinality",)


@pytest.mark.parametrize("name", SUPPORTED)
def test_spin_conversion_is_exact_and_feasibility_is_preserved(name):
    m = read_opb(OPB / name)
    doc = to_spinpoly(m)
    terms, const = objective_spin_terms(m)
    for x in assignments(m.n_vars):
        s = 2 * x - 1  # x = (1 + s)/2 : x = 1 <-> s = +1
        exact = const + sum(c * int(np.prod(s[list(sup)])) for sup, c in terms.items())
        assert exact == m.objective_value(x)
        assert doc.energy(s) == pytest.approx(float(m.objective_value(x)), abs=1e-12)
        assert doc.is_feasible(s) == m.is_feasible(x)


def test_negated_literal_expansion():
    m = parse_opb("min: +4 ~x1 x2 ;\n+1 x1 +1 x2 >= 1 ;\n")
    terms, const = objective_spin_terms(m)
    # 4 (1-s1)/2 (1+s2)/2 = 1 - s1 + s2 - s1 s2
    assert const == 1 and terms == {(0,): Fraction(-1), (1,): Fraction(1), (0, 1): Fraction(-1)}
    m = parse_opb("min: +3 x1 ~x1 +2 x2 x2 ;\n+1 x1 +1 x2 >= 1 ;\n")
    terms, const = objective_spin_terms(m)
    assert const == 1 and terms == {(1,): Fraction(1)}  # x1 ~x1 = 0, x2 x2 = x2


@pytest.mark.parametrize("text, kind", [
    ("+2 x1 +2 x2 >= 2 ;", "cover"),             # equal weights divided out
    ("-1 ~x1 -1 ~x2 >= -1 ;", "cover"),          # (1-x1)+(1-x2) <= 1  <=>  x1 + x2 >= 1
    ("+1 ~x1 +1 ~x2 >= 1 ;", "packing"),         # at most one selected, written as a clause
    ("-1 x1 -1 x2 -1 x3 = -1 ;", "cardinality"),
    ("+1 x2 +1 ~x1 >= 1 ;", "implication"),      # x1 -> x2
])
def test_equivalent_normal_forms(text, kind):
    m = parse_opb(text + "\n")
    r = recognize_constraints(m)
    assert r.kinds == (kind,)
    doc = to_spinpoly(m, r)
    for x in assignments(m.n_vars):
        assert doc.is_feasible(2 * x - 1) == m.is_feasible(x)


@pytest.mark.parametrize("name, n_listed, fragments", [
    ("unsupported_overlap.opb", 2, ["variable x2 appears in more than one constraint block"]),
    ("unsupported_forms.opb", 4, ["weighted inequality", "at-least-2-of-3", "products of literals",
                                  "no variables remain"]),
    ("unsupported_branching.opb", 2, ["precedence DAG"]),
])
def test_unsupported_constraints_are_listed(name, n_listed, fragments):
    m = read_opb(OPB / name)
    with pytest.raises(UnsupportedProblem) as exc:
        recognize_constraints(m)
    assert len(exc.value.details) == n_listed
    for f in fragments:
        assert f in str(exc.value)
    assert all(d.startswith("line ") for d in exc.value.details)


@pytest.mark.parametrize("text, fragment", [
    ("+1 x1 -1 x2 >= 0 ;\n+1 x2 -1 x1 >= 0 ;", "cyclic implications"),
    ("+1 x1 -1 x2 >= 0 ;\n+1 x1 -1 x2 >= 0 ;", "duplicate of the implication"),
    ("+1 x1 +1 x2 = 3 ;", "infeasible"),
    ("+1 x1 +1 x2 >= 0 ;", "tautology"),
    ("+1 x1 +1 x2 >= 2 ;", "forces every literal"),
    ("+1 x1 -1 x2 >= 0 ;\n+1 x2 +1 x3 >= 1 ;", "more than one constraint block"),
    ("+1 x1 -2 x2 = 0 ;", "not an unweighted cardinality"),
])
def test_unsupported_inline(text, fragment):
    with pytest.raises(UnsupportedProblem, match=fragment):
        recognize_constraints(parse_opb(text + "\n"))


def test_statement_layout_and_comments():
    m = parse_opb("* #variable= 3 #constraint= 2\nmin: +1 x1\n +2 x2 ;\n* a comment between statements\n"
                  "+1 x1 +1 x2 >= 1 ; +1 x3 >= 1 ;\n")
    assert len(m.objective) == 2 and len(m.constraints) == 2 and m.constraints[1].line == 5
    m = parse_opb("1 x1 +4 x2 -2 x5 >=2;\n")  # operator attached to the integer, as in the PB24 example
    assert m.constraints[0].rhs == 2 and m.n_vars == 5


def test_format_roundtrip():
    m = read_opb(OPB / "or_cover_signed.opb")
    back = parse_opb(format_opb(m))
    assert back.n_vars == m.n_vars
    assert [(c.op, c.rhs, [(t.coeff, t.literals) for t in c.terms]) for c in back.constraints] == \
           [(c.op, c.rhs, [(t.coeff, t.literals) for t in c.terms]) for c in m.constraints]
    assert [(t.coeff, t.literals) for t in back.objective] == [(t.coeff, t.literals) for t in m.objective]


@pytest.mark.parametrize("name, line, fragment", [
    ("opb_missing_semicolon.opb", 3, "statement not terminated by ';'"),
    ("opb_le_operator.opb", 2, "'<=' is not part of the PB24 OPB format"),
    ("opb_max_objective.opb", 2, "'max:' objectives are not part of the PB24 OPB format"),
    ("opb_bad_literal.opb", 2, "not followed by a literal x<k> or ~x<k> (found 'y2')"),
    ("opb_header_mismatch.opb", 1, "header '#constraint= 3' but the file has 2 constraints"),
    ("opb_objective_not_first.opb", 3, "must be the first non-comment statement"),
    ("opb_no_coefficient.opb", 2, "expected an integer coefficient, found 'x1'"),
    ("opb_undeclared_variable.opb", 1, "header '#variable= 2' but variable x3 is used"),
])
def test_malformed_opb(name, line, fragment):
    path = BAD / name
    with pytest.raises(FormatError) as exc:
        read_opb(path)
    assert str(exc.value).startswith(f"{path}:{line}: ") and fragment in str(exc.value)


def test_wbo_rejected():
    with pytest.raises(FormatError, match="WBO"):
        parse_opb("soft: 6 ;\n[2] +1 x1 >= 1 ;\n")


def test_to_spinpoly_metadata_and_weights():
    m = read_opb(OPB / "cardinality.opb")
    doc = to_spinpoly(m)
    assert doc.labels == ("x1", "x2", "x3", "x4", "x5") and doc.constraint_family == "exact_cardinality"
    # J = -c (manuscript convention); x1 x2 = (1 + s1 + s2 + s1 s2)/4 -> c(s1 s2) = 1/4 -> J = -1/4
    k = doc.supports.index((0, 1))
    assert doc.weights[k] == -0.25 and doc.exact["weights"][k] == Fraction(-1, 4)
