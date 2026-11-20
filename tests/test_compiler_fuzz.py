"""Randomized libraries mixing every symbolic descriptor kind (fixed masks, conditional masks with
overlapping condition/mask, signed permutations with sign flips), compiled and checked
exhaustively against direct recomputation, on the full cube and on a parity-constrained subset
(with its identity used for gate reuse)."""

import numpy as np
import pytest

from helpers_core import all_states, check_compiled_exhaustive, direct_energy, random_polynomial
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.transitions.descriptors import ConditionalMask, FixedMask, SignedPermutation
from nimbus.transitions.library import TransitionLibrary
from nimbus.transitions.verify import verify_library


def _random_involutive_perm(rng, pool):
    pool = list(pool)
    rng.shuffle(pool)
    mp = {}
    n_pairs = int(rng.integers(1, max(2, len(pool) // 2 + 1)))
    i = 0
    for _ in range(n_pairs):
        if i + 1 >= len(pool):
            break
        a, b = pool[i], pool[i + 1]
        e = int(rng.choice([-1, 1]))
        mp[a] = (b, e)
        mp[b] = (a, e)  # eps_a eps_b = 1 keeps it an involution
        i += 2
    if i < len(pool) and rng.random() < 0.5:
        mp[pool[i]] = (pool[i], -1)  # sign flip
    return SignedPermutation(mp)


def _random_condmask(rng, n, allowed=None):
    while True:
        mask = sorted(rng.choice(n, size=int(rng.integers(1, 4)), replace=False).tolist())
        cond = sorted(rng.choice(n, size=int(rng.integers(0, 4)), replace=False).tolist())
        if len(set(mask) & set(cond)) % 2 == 0 and (allowed is None or allowed(mask)):
            return ConditionalMask(mask, cond, int(rng.choice([-1, 1])))


@pytest.mark.parametrize("seed", range(25))
def test_mixed_library_full_cube(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 8))
    groups = []
    for _ in range(int(rng.integers(3, 8))):
        u = rng.random()
        if u < 0.25:
            groups.append(FixedMask(sorted(rng.choice(n, size=int(rng.integers(1, 4)), replace=False).tolist())))
        elif u < 0.55:
            groups.append(_random_condmask(rng, n))
        else:
            groups.append(_random_involutive_perm(rng, range(n)))
    lib = TransitionLibrary(n, groups, name=f"fuzz{seed}")
    structure, w, c = random_polynomial(n, 12, 4, seed)
    model = compile_model(structure, lib, readout="all" if seed % 3 == 0 else None)
    S = all_states(n)
    check_compiled_exhaustive(model, lib, S, binding=bind_weights(model, w, c),
                              energy=lambda X: direct_energy(structure, w, c, X))


@pytest.mark.parametrize("seed", range(15))
def test_mixed_library_on_parity_subset(seed):
    """Omega = {s : prod_{i in Q} s_i = q}; groups preserve it; the identity enables gate reuse."""
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(5, 8))
    Q = sorted(rng.choice(n, size=int(rng.integers(2, n + 1)), replace=False).tolist())
    q = int(rng.choice([-1, 1]))
    S = all_states(n)
    Om = S[np.prod(S[:, Q].astype(int), axis=1) == q]
    inQ = set(Q)
    groups = []
    while len(groups) < 6:
        u = rng.random()
        if u < 0.3:
            m = sorted(rng.choice(n, size=int(rng.integers(1, 5)), replace=False).tolist())
            if len(set(m) & inQ) % 2 == 0:
                groups.append(FixedMask(m))
        elif u < 0.6:
            groups.append(_random_condmask(rng, n, allowed=lambda m: len(set(m) & inQ) % 2 == 0))
        else:
            # permutation inside Q and inside its complement (maps Q to itself); signs: even number of
            # -1 on Q so that prod_Q is preserved
            g = _random_involutive_perm(rng, Q) if rng.random() < 0.5 else None
            rest = [i for i in range(n) if i not in inQ]
            h = _random_involutive_perm(rng, rest) if len(rest) >= 2 else None
            mp = {}
            for part in (g, h):
                if part is not None:
                    mp.update(part.mapping)
            if not mp:
                continue
            negQ = sum(1 for d, (s_, e) in mp.items() if d in inQ and e < 0)
            if negQ % 2:
                continue
            groups.append(SignedPermutation(mp))
    lib = TransitionLibrary(n, groups, name=f"parity_fuzz{seed}")
    assert verify_library(lib, Om, check_connectivity=False).feasibility_preserved
    structure, w, c = random_polynomial(n, 12, 4, 50 + seed)
    for ids in ((), [(tuple(Q), q)]):
        model = compile_model(structure, lib, identities=ids)
        check_compiled_exhaustive(model, lib, Om, binding=bind_weights(model, w, c),
                                  energy=lambda X: direct_energy(structure, w, c, X))
