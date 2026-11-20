"""A documented subset of the PB24 OPB format and a recognizer for the scoped constraint families.

Format reference: O. Roussel, "Restricted OPB Format in Use in the PB Competitions" (PB24,
https://www.cril.univ-artois.fr/PB24/OPBcompetition.pdf, rev. 4524, 2024-03-11).

Accepted syntax (a subset of the general format; deviations are rejected with FormatError):

- ``*`` comment lines anywhere. The first line may carry the hints ``* #variable= N #constraint= M
  [#equal= E] [intsize= S] [#product= P sizeproduct= Q]``; ``#variable``, ``#constraint`` and
  ``#equal`` are verified when present, the others are recorded.
- Optional objective ``min: <sum> ;`` as the first non-comment statement (``max:`` is not part of
  PB24 and is rejected). The objective may contain products of literals (non-linear terms).
- Constraints ``<sum> >= <integer> ;`` and ``<sum> = <integer> ;``. ``<=`` is **not** part of the
  PB24 format (the grammar has only ``=`` and ``>=``) and is rejected: write ``sum x <= K`` as
  ``-1 x1 -1 x2 ... >= -K``.
- ``<sum>`` is a list of ``<integer> <literal>+``: every term starts with a signed integer coefficient
  (no space between sign and digits) followed by one literal (linear term) or several literals (a
  product). Literals are ``x<k>`` or ``~x<k>`` with 1 <= k < 2^32. Integers have arbitrary size.
- Statements end with ``;``; a statement may span lines (the general format allows long lines).
- WBO files (``soft:`` header, ``[cost]`` soft constraints) are not supported.

Boolean-to-spin convention (project convention): variable ``x<k>`` is spin
``k-1``, and ``x = (1 + s)/2``: ``x = 1`` (true, *selected*) is ``s = +1``. A negated literal is
``~x = (1 - s)/2``. :func:`objective_spin_terms` expands the objective exactly (rationals).

Recognized constraint forms (:func:`recognize_constraints`); everything else raises
:class:`~nimbus.core.errors.UnsupportedProblem` listing each offending constraint:

====================  ==========================================  =====================================
family                written form (any equivalent normalization)  meaning
====================  ==========================================  =====================================
exact cardinality      ``+1 x1 +1 x2 ... +1 xm = K``                exactly K of the block selected
implication / chain    ``+1 xa -1 xb >= 0``  (== ``xa + ~xb >= 1``)  xb -> xa; the implications must
                                                                    form vertex-disjoint simple chains
covering (OR)          ``+1 l1 +1 l2 ... >= 1`` with signed         at least one literal true; a 1-literal
                       literals, or its linear normalization        cover fixes that variable
                       ``sum_pos x - sum_neg x >= 1 - #neg``
unit packing (AMO)     ``-1 x1 -1 x2 ... -1 xm >= -1`` (m >= 2)     at most one of the block selected
====================  ==========================================  =====================================

Normalization: each ``>=`` constraint is rewritten over literals with positive coefficients
(``c x = c + |c| ~x`` for ``c < 0``); equal coefficients ``w`` are divided out
(``sum w l >= d  <=>  sum l >= ceil(d / w)``). A two-literal clause with one negated literal is an
implication; a clause whose literals are all negated with degree ``m - 1`` is packing (this
includes ``~x1 + ~x2 >= 1``); any other clause with degree 1 is a cover. The blocks (cardinality
blocks, covers, packing blocks and whole chains) must be variable-disjoint; variables in no block
are free. Weighted (knapsack-type) inequalities, at-least-k for 1 < k < m - 1, tautologies,
infeasible constraints, products inside constraints, overlapping blocks, branching or cyclic
implications are outside the supported subset.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

import numpy as np

from nimbus.core.errors import UnsupportedProblem
from nimbus.problems.formats import SourceText, format_error, read_source, source_from_text

__all__ = [
    "OPBTerm",
    "OPBConstraint",
    "OPBModel",
    "parse_opb",
    "read_opb",
    "format_opb",
    "RecognizedConstraints",
    "recognize_constraints",
    "objective_spin_terms",
    "to_spinpoly",
]

_MAX_VAR = 2 ** 32 - 1
_MAX_PRODUCT_EXPANSION = 20


@dataclass(frozen=True)
class OPBTerm:
    """``coeff * prod(literals)``; a literal is ``+k`` for ``x<k>`` and ``-k`` for ``~x<k>`` (1-based)."""

    coeff: int
    literals: tuple

    @property
    def is_linear(self) -> bool:
        return len(self.literals) == 1

    def value(self, x: np.ndarray) -> int:
        v = 1
        for lit in self.literals:
            xi = int(x[abs(lit) - 1])
            v *= xi if lit > 0 else 1 - xi
        return self.coeff * v

    def text(self) -> str:
        lits = " ".join(("~" if l < 0 else "") + f"x{abs(l)}" for l in self.literals)
        return f"{self.coeff:+d} {lits}"


@dataclass(frozen=True)
class OPBConstraint:
    terms: tuple
    op: str  # ">=" or "="
    rhs: int
    line: int
    text: str

    @property
    def is_linear(self) -> bool:
        return all(t.is_linear for t in self.terms)

    def lhs(self, x: np.ndarray) -> int:
        return sum(t.value(x) for t in self.terms)

    def satisfied(self, x: np.ndarray) -> bool:
        v = self.lhs(x)
        return v >= self.rhs if self.op == ">=" else v == self.rhs


@dataclass(frozen=True)
class OPBModel:
    n_vars: int
    objective: tuple | None  # tuple[OPBTerm] or None (satisfaction instance)
    constraints: tuple
    header: dict = field(default_factory=dict)
    objective_line: int | None = None
    source: str | None = None
    sha256: str | None = None

    def _x(self, x) -> np.ndarray:
        a = np.asarray(x)
        if a.shape != (self.n_vars,) or not np.isin(a, (0, 1)).all():
            raise ValueError(f"expected a 0/1 assignment of length {self.n_vars}")
        return a.astype(np.int64)

    def objective_value(self, x) -> int:
        """Objective of a 0/1 assignment (0 for satisfaction instances)."""
        a = self._x(x)
        return sum(t.value(a) for t in (self.objective or ()))

    def violated(self, x) -> list[int]:
        a = self._x(x)
        return [k for k, c in enumerate(self.constraints) if not c.satisfied(a)]

    def is_feasible(self, x) -> bool:
        return not self.violated(x)


# ---------------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"min:|max:|soft:|>=|<=|=|~?x\d+|[+-]?\d+|\S+")
_HINT = re.compile(r"(#variable|#constraint|#equal|intsize|#product|sizeproduct|#soft|mincost|maxcost|sumcost)=\s*(-?\d+)")


def _parse_literal(tok: str, name: str, line: int) -> int:
    neg = tok.startswith("~")
    k = int(tok[2:] if neg else tok[1:])
    if k < 1 or k > _MAX_VAR:
        raise format_error(name, line, f"variable index in {tok!r} outside 1..2^32-1")
    return -k if neg else k


def _parse_sum(toks: list[str], i: int, stop: set, name: str, line: int) -> tuple[list[OPBTerm], int]:
    terms = []
    while i < len(toks) and toks[i] not in stop:
        tok = toks[i]
        if not re.fullmatch(r"[+-]?\d+", tok):
            if tok in ("+", "-"):
                raise format_error(name, line, "a sign must be attached to its coefficient (e.g. '+1 x1', not '+ 1 x1')")
            raise format_error(name, line, f"expected an integer coefficient, found {tok!r}")
        coeff = int(tok)
        i += 1
        lits = []
        while i < len(toks) and re.fullmatch(r"~?x\d+", toks[i]):
            lits.append(_parse_literal(toks[i], name, line))
            i += 1
        if not lits:
            found = toks[i] if i < len(toks) else "end of statement"
            raise format_error(name, line, f"coefficient {coeff} is not followed by a literal x<k> or ~x<k> (found {found!r})")
        terms.append(OPBTerm(coeff=coeff, literals=tuple(lits)))
    return terms, i


def parse_opb(src: SourceText | str, name: str | None = None) -> OPBModel:
    """Parse OPB text (see the module docstring for the accepted subset)."""
    if isinstance(src, str):
        src = source_from_text(src, name or "<string>")
    fname = src.name
    header: dict = {}
    statements: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_line = None
    first_content_seen = False
    for lineno, raw in enumerate(src.text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("*"):
            if not first_content_seen and not header:
                for key, val in _HINT.findall(stripped):
                    header[key] = int(val)
            continue
        if not stripped:
            continue
        first_content_seen = True
        if stripped.startswith("[") or stripped.startswith("soft:"):
            raise format_error(fname, lineno, "WBO soft constraints / 'soft:' headers are not supported")
        parts = stripped.split(";")
        for k, part in enumerate(parts):
            if part.strip():
                if buf_line is None:
                    buf_line = lineno
                buf.append(part)
            if k < len(parts) - 1:  # a ';' terminated this part
                if not buf:
                    raise format_error(fname, lineno, "empty statement (stray ';')")
                statements.append((buf_line, " ".join(buf)))
                buf, buf_line = [], None
    if buf:
        raise format_error(fname, buf_line, "statement not terminated by ';' (truncated file?)")

    objective = None
    objective_line = None
    constraints: list[OPBConstraint] = []
    for idx, (line, text) in enumerate(statements):
        toks = _TOKEN.findall(text)
        if not toks:
            raise format_error(fname, line, "empty statement")
        if toks[0] in ("max:",):
            raise format_error(fname, line, "'max:' objectives are not part of the PB24 OPB format (use 'min:' with negated coefficients)")
        if toks[0] == "min:":
            if idx != 0:
                raise format_error(fname, line, "the objective 'min:' must be the first non-comment statement")
            terms, i = _parse_sum(toks, 1, set(), fname, line)
            if i != len(toks):
                raise format_error(fname, line, f"unexpected token {toks[i]!r} in the objective")
            objective = tuple(terms)
            objective_line = line
            continue
        terms, i = _parse_sum(toks, 0, {">=", "<=", "="}, fname, line)
        if i >= len(toks):
            raise format_error(fname, line, "constraint without a relational operator ('>=' or '=')")
        op = toks[i]
        if op == "<=":
            raise format_error(fname, line, "'<=' is not part of the PB24 OPB format; write 'sum <= K' as "
                                            "'-1 x1 -1 x2 ... >= -K'")
        if not terms:
            raise format_error(fname, line, "constraint with an empty left-hand side")
        rest = toks[i + 1:]
        if len(rest) != 1 or not re.fullmatch(r"[+-]?\d+", rest[0]):
            raise format_error(fname, line, f"expected one integer right-hand side after {op!r}, found {' '.join(rest) or 'nothing'!r}")
        constraints.append(OPBConstraint(terms=tuple(terms), op=op, rhs=int(rest[0]), line=line, text=text.strip() + " ;"))

    max_var = 0
    for t in (objective or ()):
        max_var = max(max_var, *(abs(l) for l in t.literals))
    for c in constraints:
        for t in c.terms:
            max_var = max(max_var, *(abs(l) for l in t.literals))
    n_vars = max_var
    if "#variable" in header:
        if header["#variable"] < max_var:
            raise format_error(fname, 1, f"header '#variable= {header['#variable']}' but variable x{max_var} is used")
        n_vars = header["#variable"]
    if "#constraint" in header and header["#constraint"] != len(constraints):
        raise format_error(fname, 1, f"header '#constraint= {header['#constraint']}' but the file has {len(constraints)} constraints")
    n_eq = sum(1 for c in constraints if c.op == "=")
    if "#equal" in header and header["#equal"] != n_eq:
        raise format_error(fname, 1, f"header '#equal= {header['#equal']}' but the file has {n_eq} equality constraints")
    if n_vars == 0:
        raise format_error(fname, None, "no variables")
    return OPBModel(n_vars=n_vars, objective=objective, constraints=tuple(constraints), header=header,
                    objective_line=objective_line, source=str(src.path) if src.path else src.name, sha256=src.sha256)


def read_opb(path: str | os.PathLike) -> OPBModel:
    return parse_opb(read_source(path))


def format_opb(model: OPBModel) -> str:
    """OPB text of a model (header hints, objective, constraints)."""
    n_eq = sum(1 for c in model.constraints if c.op == "=")
    lines = [f"* #variable= {model.n_vars} #constraint= {len(model.constraints)} #equal= {n_eq}"]
    if model.objective is not None:
        lines.append("min: " + " ".join(t.text() for t in model.objective) + " ;")
    for c in model.constraints:
        lines.append(" ".join(t.text() for t in c.terms) + f" {c.op} {c.rhs} ;")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RecognizedConstraints:
    """Constraint blocks in 0-based spin indices (spin i = variable x_{i+1})."""

    n_vars: int
    exact_cardinality: tuple  # ((spins...), k)
    chains: tuple  # (spins in prefix order: selecting a later position requires all earlier ones)
    or_blocks: tuple  # ((spins...), (signs...)); sign +1 = literal x (satisfied by s=+1), -1 = ~x
    amo_blocks: tuple  # (spins...) with at most one s=+1
    free: tuple
    kinds: tuple  # per OPB constraint: "cardinality" | "implication" | "cover" | "packing"

    @property
    def families(self) -> tuple:
        out = []
        if self.exact_cardinality:
            out.append("exact_cardinality")
        if self.chains:
            out.append("chain_precedence")
        if self.or_blocks:
            out.append("disjoint_or_cover")
        if self.amo_blocks:
            out.append("unit_packing")
        return tuple(out)


def _linear_form(c: OPBConstraint) -> tuple[dict, int]:
    """sum_v a_v x_v (op) d over variables (1-based), with ~x = 1 - x substituted."""
    a: dict[int, int] = {}
    d = c.rhs
    for t in c.terms:
        (lit,) = t.literals
        v = abs(lit)
        if lit > 0:
            a[v] = a.get(v, 0) + t.coeff
        else:
            a[v] = a.get(v, 0) - t.coeff
            d -= t.coeff
    return {v: w for v, w in a.items() if w != 0}, d


def _classify(c: OPBConstraint):
    """Return (kind, payload) or ("unsupported", reason)."""
    if not c.is_linear:
        return "unsupported", "products of literals inside a constraint are not supported"
    a, d = _linear_form(c)
    if not a:
        return "unsupported", "no variables remain after simplification"
    if c.op == "=":
        coeffs = set(a.values())
        if coeffs == {1}:
            k = d
        elif coeffs == {-1}:
            k = -d
        else:
            return "unsupported", "equality is not an unweighted cardinality constraint over positive literals"
        m = len(a)
        if k < 0 or k > m:
            return "unsupported", f"cardinality {k} is infeasible for a block of {m} variables"
        return "cardinality", (tuple(sorted(a)), k)
    # '>=' : literal form with positive weights
    lits = []
    deg = d
    for v, w in sorted(a.items()):
        if w > 0:
            lits.append((v, +1, w))
        else:
            lits.append((v, -1, -w))
            deg += -w
    weights = {w for _, _, w in lits}
    if len(weights) != 1:
        return "unsupported", "weighted inequality (unequal coefficients; knapsack-type constraints are not supported)"
    w = weights.pop()
    m = len(lits)
    if deg <= 0:
        return "unsupported", "always satisfied (tautology)"
    deg = -(-deg // w)  # ceil(deg / w)
    if deg > m:
        return "unsupported", "infeasible inequality"
    signs = [s for _, s, _ in lits]
    if m == 2 and deg == 1 and sorted(signs) == [-1, 1]:
        pos = next(v for v, s, _ in lits if s > 0)
        neg = next(v for v, s, _ in lits if s < 0)
        return "implication", (neg, pos)  # x_neg -> x_pos
    if m >= 2 and all(s < 0 for s in signs) and deg == m - 1:
        return "packing", tuple(v for v, _, _ in lits)
    if deg == 1:
        return "cover", (tuple(v for v, _, _ in lits), tuple(signs))
    if deg == m:
        return "unsupported", "forces every literal (write fixed variables as one-literal covers)"
    return "unsupported", f"at-least-{deg}-of-{m} constraint (only covers, implications and unit packing are supported)"


def _fmt(c: OPBConstraint, reason: str) -> str:
    return f"line {c.line}: {c.text}  -- {reason}"


def _raise(problems: list[str], src: str | None):
    msg = f"{src or 'OPB model'}: {len(problems)} constraint(s) outside the supported OPB subset:\n  " + "\n  ".join(problems)
    exc = UnsupportedProblem(msg)
    exc.details = list(problems)
    raise exc


def recognize_constraints(model: OPBModel) -> RecognizedConstraints:
    """Classify every constraint into the supported families and check block disjointness."""
    problems: list[str] = []
    kinds: list[str] = []
    cards, covers, packs, imps = [], [], [], []  # (constraint index, payload)
    for k, c in enumerate(model.constraints):
        kind, payload = _classify(c)
        kinds.append(kind)
        if kind == "unsupported":
            problems.append(_fmt(c, payload))
        elif kind == "cardinality":
            cards.append((k, payload))
        elif kind == "cover":
            covers.append((k, payload))
        elif kind == "packing":
            packs.append((k, payload))
        else:
            imps.append((k, payload))
    if problems:
        _raise(problems, model.source)

    # ---- implications -> vertex-disjoint simple chains ---------------------------------------
    requires: dict[int, list[int]] = {}  # b -> [a]: x_b -> x_a (a precedes b)
    required_by: dict[int, list[int]] = {}
    imp_of: dict[tuple, int] = {}
    for k, (b, a) in imps:
        if (b, a) in imp_of:
            problems.append(_fmt(model.constraints[k], f"duplicate of the implication on line {model.constraints[imp_of[(b, a)]].line}"))
            continue
        imp_of[(b, a)] = k
        requires.setdefault(b, []).append(a)
        required_by.setdefault(a, []).append(b)
    for b, lst in requires.items():
        if len(lst) > 1:
            for a in lst:
                problems.append(_fmt(model.constraints[imp_of[(b, a)]], f"x{b} requires several variables (a precedence DAG, not a chain)"))
    for a, lst in required_by.items():
        if len(lst) > 1:
            for b in lst:
                problems.append(_fmt(model.constraints[imp_of[(b, a)]], f"x{a} is required by several variables (a precedence tree, not a chain)"))
    if problems:
        _raise(sorted(set(problems)), model.source)
    chain_vars = set(requires) | set(required_by)
    chains = []
    chain_cons: list[list[int]] = []
    seen: set[int] = set()
    heads = sorted(v for v in chain_vars if v not in requires)
    for h in heads:
        chain = [h]
        cons = []
        seen.add(h)
        while chain[-1] in required_by:
            nxt = required_by[chain[-1]][0]
            cons.append(imp_of[(nxt, chain[-1])])
            chain.append(nxt)
            seen.add(nxt)
        chains.append(tuple(chain))
        chain_cons.append(cons)
    cyc = sorted(chain_vars - seen)
    if cyc:
        bad = sorted({imp_of[(b, a)] for (b, a) in imp_of if b in cyc or a in cyc})
        _raise([_fmt(model.constraints[k], "cyclic implications (equivalences are not a chain)") for k in bad], model.source)

    # ---- block disjointness ------------------------------------------------------------------
    owner: dict[int, list[tuple[str, list[int]]]] = {}
    blocks: list[tuple[str, tuple, list[int]]] = []
    for k, (vs, kk) in cards:
        blocks.append(("cardinality", vs, [k]))
    for k, (vs, _sg) in covers:
        blocks.append(("cover", vs, [k]))
    for k, vs in packs:
        blocks.append(("packing", vs, [k]))
    for ch, cons in zip(chains, chain_cons):
        blocks.append(("chain", ch, cons))
    for kind, vs, cons in blocks:
        for v in vs:
            owner.setdefault(v, []).append((kind, cons))
    overlap = []
    for v in sorted(owner):
        if len(owner[v]) > 1:
            for kind, cons in owner[v]:
                for k in cons:
                    overlap.append(_fmt(model.constraints[k], f"variable x{v} appears in more than one constraint block "
                                                              "(overlapping blocks are outside the supported families)"))
    if overlap:
        _raise(sorted(set(overlap), key=overlap.index), model.source)

    used = set(owner)
    free = tuple(v - 1 for v in range(1, model.n_vars + 1) if v not in used)
    return RecognizedConstraints(
        n_vars=model.n_vars,
        exact_cardinality=tuple((tuple(v - 1 for v in vs), k) for _, (vs, k) in cards),
        chains=tuple(tuple(v - 1 for v in ch) for ch in chains),
        or_blocks=tuple((tuple(v - 1 for v in vs), tuple(sg)) for _, (vs, sg) in covers),
        amo_blocks=tuple(tuple(v - 1 for v in vs) for _, vs in packs),
        free=free,
        kinds=tuple(kinds),
    )


# ---------------------------------------------------------------------------------------------
# Objective -> spin polynomial
# ---------------------------------------------------------------------------------------------
def objective_spin_terms(model: OPBModel) -> tuple[dict, Fraction]:
    """Exact expansion of the objective under x = (1+s)/2, ~x = (1-s)/2.

    Returns ``({support: c}, const)`` with ``E(s) = const + sum_S c_S prod_{i in S} s_i`` (0-based
    spins, rational coefficients, zero coefficients dropped) and E equal to the OPB objective.
    """
    acc: dict[tuple, Fraction] = {}
    const = Fraction(0)
    for t in (model.objective or ()):
        pol: dict[int, int] = {}
        zero = False
        for lit in t.literals:
            v, s = abs(lit) - 1, (1 if lit > 0 else -1)
            if v in pol and pol[v] != s:
                zero = True  # x * ~x = 0
                break
            pol[v] = s  # x * x = x
        if zero:
            continue
        vars_ = sorted(pol)
        m = len(vars_)
        if m > _MAX_PRODUCT_EXPANSION:
            raise UnsupportedProblem(f"objective product of {m} literals is too large to expand (limit {_MAX_PRODUCT_EXPANSION})")
        scale = Fraction(t.coeff, 2 ** m)
        for r in range(m + 1):
            for sub in combinations(vars_, r):
                sign = 1
                for v in sub:
                    sign *= pol[v]
                if r == 0:
                    const += scale
                else:
                    acc[sub] = acc.get(sub, Fraction(0)) + sign * scale
    return {k: v for k, v in sorted(acc.items(), key=lambda kv: (len(kv[0]), kv[0])) if v != 0}, const


def to_spinpoly(model: OPBModel, recognized: RecognizedConstraints | None = None, *, name: str | None = None):
    """Convert to the project schema (:class:`nimbus.problems.formats.spinpoly.SpinPolyDocument`).

    Weights use the manuscript convention ``E = E_const - sum_k J_k T_k`` (``J = -c``).
    """
    from nimbus.problems.formats.spinpoly import SpinPolyDocument

    rec = recognized if recognized is not None else recognize_constraints(model)
    terms, const = objective_spin_terms(model)
    supports = tuple(terms)
    exact = {"const": const, "weights": tuple(-c for c in terms.values())}
    return SpinPolyDocument(
        n_spins=model.n_vars,
        supports=supports,
        weights=np.array([float(-c) for c in terms.values()], dtype=np.float64),
        const=float(const),
        exact_cardinality=rec.exact_cardinality,
        chains=rec.chains,
        or_blocks=rec.or_blocks,
        amo_blocks=rec.amo_blocks,
        free=rec.free,
        labels=tuple(f"x{i + 1}" for i in range(model.n_vars)),
        name=name or (os.path.basename(model.source) if model.source else None),
        description="converted from OPB (x = (1+s)/2; x=1 <-> s=+1 selected)",
        metadata={"opb_source": model.source, "opb_sha256": model.sha256},
        exact=exact,
    )
