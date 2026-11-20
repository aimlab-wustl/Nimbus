"""Finding moves for a constraint that no move builder covers, by grouping spin transitions.

The constraint here is a Latin square of order 3: every cell holds one of three symbols, and every row and every
column holds each symbol once. Spin 9 r + 3 c + v is +1 when cell (r, c) holds symbol v. The script lists the 12
feasible squares, groups the transitions between them into moves, checks the compiled machine on every feasible
state and shows the case in which no move exists. See docs/guides/new_problem/own_moves.md.

    python examples/find_moves.py
"""
from __future__ import annotations

import itertools

import numpy as np

from nimbus.compiler import compile_model
from nimbus.core.model import ObjectiveStructure, bind_weights
from nimbus.transitions.grouping import group_transitions
from nimbus.transitions.verify import verify_library


def latin_squares(P: int = 3) -> np.ndarray:
    rows = list(itertools.permutations(range(P)))
    out = []
    for square in itertools.product(rows, repeat=P):
        if all(len({square[r][c] for r in range(P)}) == P for c in range(P)):
            s = -np.ones(P * P * P, dtype=np.int8)
            for r in range(P):
                for c in range(P):
                    s[(r * P + c) * P + square[r][c]] = 1
            out.append(s)
    return np.array(out)


def independent_sets_on_cycle(n: int, k: int) -> np.ndarray:
    """k selected vertices of a cycle of n vertices, no two adjacent."""
    out = []
    for sel in itertools.combinations(range(n), k):
        if all((b - a) % n not in (1, n - 1) for a, b in itertools.combinations(sel, 2)):
            s = -np.ones(n, dtype=np.int8)
            s[list(sel)] = 1
            out.append(s)
    return np.array(out)


def main() -> None:
    # 1. group the transitions of the Latin squares into moves
    states = latin_squares(3)
    grouping = group_transitions(states)
    print(grouping.summary())

    # 2. check the backbone library on every feasible state
    library = grouping.library("backbone")
    report = verify_library(library, states, require=True)
    print("backbone keeps feasibility and connects all squares:", report.ok)

    # 3. compile it with an objective and check 2 v = dE for every move and every feasible state
    rng = np.random.default_rng(0)
    terms = [tuple(sorted(rng.choice(27, size=int(rng.integers(1, 4)), replace=False).tolist())) for _ in range(12)]
    weights = rng.normal(size=len(terms))
    structure = ObjectiveStructure(n_spins=27, terms=terms)
    model = compile_model(structure, library, identities=grouping.identities)
    binding = bind_weights(model, weights, 0.0)

    def energy(S):
        return -np.array([sum(w * np.prod(s[list(t)]) for t, w in zip(terms, weights)) for s in S])

    v = model.inputs(model.clause_values(states), binding)
    E = energy(states)
    for r in range(model.n_groups):
        dE = energy(np.array([model.apply_action(r, s) for s in states])) - E
        assert np.allclose(2 * v[:, r], dE)
    print(f"{model.n_groups} moves, {model.n_clauses} stored clauses: 2 v = dE on all {len(states)} squares")

    # 4. a constraint whose transitions form no move
    none = group_transitions(independent_sets_on_cycle(8, 3))
    print(none.summary())


if __name__ == "__main__":
    main()
