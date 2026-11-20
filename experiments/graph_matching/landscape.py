"""Properties of the move libraries that do not involve annealing: relaxation time and exact energy landscapes.

``gap``: relaxation time at infinite temperature of each library (Fig. 4b). With every proposal accepted, a library of
    row exchanges drives the interchange process on its exchange graph; by the Caputo-Liggett-Richthammer theorem its
    spectral gap equals the second-smallest eigenvalue lambda_2 of the weighted graph Laplacian, with edge weights equal
    to the proposal probabilities. Relaxation time t_rel = 1 / lambda_2 (iterations). Libraries: the path backbone,
    the backbone plus K informed or random shortcuts at backbone share 0.5 (random seed = instance index) and the
    complete library at backbone share 0.5, on the 20 test instances at L = 20 (K = 10-160) and L = 30 (K = 15-240).
    Exact linear algebra, seconds on a standard CPU.

``depth``: exact energy landscape of the L = 10 test instances (all 10! assignments; Fig. 4c, 4e). For each library:
    number of nonoptimal local minima, critical depth h* (the largest, over nonoptimal local minima, of the smallest
    energy rise needed to reach a strictly lower state) and Catoni difficulty D = max_x depth(x) / (E(x) - E_min), with
    E = -Psi. The barrier of one transition is the larger of its two endpoint energies (vertex-ordered union-find).
    Libraries: path, backbone + K informed shortcuts and backbone + K random shortcuts (K = 4, 9, 18, 27; random draws
    with seeds equal to the instance index and 100-108), complete (backbone share 0.5). One task per instance.

    python landscape.py gap   --inst INST --out RUNS/landscape
    python landscape.py depth --inst INST --out RUNS/landscape/depth --task 0      # tasks 0-19, one per test instance

Outputs: ``gap.json``; ``depth/depth_n10_task<ii>.json`` (records: instance, lib, k, seed, n_minima, d_star, difficulty,
t_rel, F_opt).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from nimbus.problems.formats.matching_tensor import load_tensor
from nimbus.problems.matching import backbone_edges, informed_pairs, random_pairs


def test_instances(inst: Path, n: int, sigma: str = "0.05"):
    man = json.loads((Path(inst) / "manifest.json").read_text())
    return [m for m in man if m["family"] == "synthetic" and m["n"] == n and f"_s{sigma}_" in m["name"]
            and m["role"] == "test"]


def library_edges(tensor, n: int, kind: str, k: int, seed: int):
    """``(edges, weights)``: path backbone (weight c, the number of copies that gives backbone share 0.5, as in
    ``shortcut_library``) plus ``k`` shortcuts of weight 1; ``complete_uniform``: all exchanges with weight 1."""
    bb = backbone_edges(n)
    if kind == "path":
        return bb, [1.0] * len(bb)
    excl = set(bb)
    rest = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in excl]
    if kind == "complete_uniform":
        e = bb + rest
        return e, [1.0] * len(e)
    if kind == "informed":
        add = informed_pairs(tensor, k, excl)
    elif kind == "random":
        add = random_pairs(n, k, excl, seed)
    elif kind == "complete":                 # complete library with backbone share 0.5
        add = rest
    else:
        raise ValueError(kind)
    c = max(1, int(round(len(add) / (n - 1))))
    return bb + add, [float(c)] * len(bb) + [1.0] * len(add)


def t_rel(n: int, edges, w) -> float:
    """Relaxation time ``1 / lambda_2`` of the interchange process with proposal probabilities ``w / sum(w)``."""
    w = np.asarray(w, float) / np.sum(w)
    L = np.zeros((n, n))
    for (i, j), x in zip(edges, w):
        L[i, i] += x
        L[j, j] += x
        L[i, j] -= x
        L[j, i] -= x
    return float(1.0 / np.sort(np.linalg.eigvalsh(L))[1])


def run_gap(args):
    rows = []
    for n, ks in ((20, (10, 20, 40, 80, 160)), (30, (15, 30, 60, 120, 240))):
        for m in test_instances(args.inst, n):
            t = load_tensor(Path(args.inst) / f"{m['name']}.npz")
            idx = int(m["name"].rsplit("_", 1)[-1])
            base = {"n": n, "instance": m["name"]}
            rows.append({**base, "lib": "path", "k": 0, "t_rel": t_rel(n, *library_edges(t, n, "path", 0, 0))})
            rows.append({**base, "lib": "complete", "k": n * (n - 1) // 2 - (n - 1),
                         "t_rel": t_rel(n, *library_edges(t, n, "complete", 0, 0))})
            for k in ks:
                for kind in ("informed", "random"):
                    rows.append({**base, "lib": kind, "k": k, "t_rel": t_rel(n, *library_edges(t, n, kind, k, idx))})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gap.json").write_text(json.dumps(rows))
    print(len(rows), "rows ->", args.out / "gap.json")


# ------------------------------------------------------------------------------------------- exact landscape
def all_perms(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n))), dtype=np.int8)   # lexicographic order


def perm_rank(P: np.ndarray) -> np.ndarray:
    """Lexicographic rank of each permutation (row of ``P``)."""
    n = P.shape[1]
    r = np.zeros(P.shape[0], dtype=np.int64)
    for i in range(n):
        c = (P[:, i + 1:] < P[:, i:i + 1]).sum(axis=1)
        r += c * math.factorial(n - 1 - i)
    return r


def energies(tensor, P: np.ndarray) -> np.ndarray:
    """``E = -Psi`` of every assignment (row of ``P``)."""
    import numba as nb

    n = tensor.n
    tr = tensor.triples.astype(np.int64)
    key = tr[:, 0, 0] * n + tr[:, 0, 1]
    order = np.argsort(key, kind="stable")
    tr, val = tr[order], tensor.values[order].astype(np.float64)
    ptr = np.searchsorted(key[order], np.arange(n * n + 1))

    @nb.njit(cache=False)
    def ev(P, tr, val, ptr, n):
        out = np.zeros(P.shape[0])
        for s in range(P.shape[0]):
            f = 0.0
            for i in range(n):
                kk = i * n + P[s, i]
                for t in range(ptr[kk], ptr[kk + 1]):
                    if P[s, tr[t, 1, 0]] == tr[t, 1, 1] and P[s, tr[t, 2, 0]] == tr[t, 2, 1]:
                        f += val[t]
            out[s] = f
        return out

    return -ev(P.astype(np.int64), tr, val, ptr, n)


def landscape(E: np.ndarray, nbr: np.ndarray):
    """Vertex-ordered union-find over the states sorted by energy; ``nbr[v, c]`` is the c-th neighbor of state v.
    Returns ``(number of nonoptimal local minima, critical depth h*, Catoni difficulty)``."""
    import numba as nb

    @nb.njit(cache=False)
    def go(E, nbr):
        V = E.shape[0]
        order = np.argsort(E)
        parent = np.full(V, -1, np.int64)        # -1: not yet added
        cmin = np.zeros(V)                        # component minimum energy (at root)
        emin = E[order[0]]
        dstar = 0.0
        diff = 0.0
        nmin = 0
        for v in order:
            parent[v] = v
            cmin[v] = E[v]
            lower = False
            for c in range(nbr.shape[1]):
                u = nbr[v, c]
                if parent[u] < 0:
                    continue
                lower = True
                ru = u
                while parent[ru] != ru:
                    parent[ru] = parent[parent[ru]]
                    ru = parent[ru]
                rv = v
                while parent[rv] != rv:
                    parent[rv] = parent[parent[rv]]
                    rv = parent[rv]
                if ru == rv:
                    continue
                hi, lo = (ru, rv) if cmin[ru] > cmin[rv] else (rv, ru)
                if cmin[hi] < E[v]:                # hi's minimum is a real local minimum (not v itself)
                    d = E[v] - cmin[hi]
                    if d > dstar:
                        dstar = d
                    g = cmin[hi] - emin
                    if g > 0 and d / g > diff:
                        diff = d / g
                parent[hi] = lo
            if not lower and E[v] > emin:
                nmin += 1
        return nmin, dstar, diff

    return go(E, nbr)


def neighbor_tables(n: int):
    """``(P, {(i, j): rank of P with positions i, j exchanged})`` over all ``n!`` assignments."""
    P = all_perms(n)
    col = {}
    for i in range(n):
        for j in range(i + 1, n):
            Q = P.copy()
            Q[:, [i, j]] = Q[:, [j, i]]
            col[(i, j)] = perm_rank(Q).astype(np.int32)
    return P, col


def run_depth(args):
    n = args.n
    t0 = time.time()
    P, col = neighbor_tables(n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"L={n}: {len(P)} states, neighbor tables {time.time() - t0:.0f} s", flush=True)
    rest_n = len(pairs) - (n - 1)
    ks = sorted({max(1, round(f * rest_n)) for f in (0.1, 0.25, 0.5, 0.75)})
    rows = []
    insts = test_instances(args.inst, n)
    if args.task >= 0:
        insts = insts[args.task:args.task + 1]
    for m in insts:
        t = load_tensor(Path(args.inst) / f"{m['name']}.npz")
        idx = int(m["name"].rsplit("_", 1)[-1])
        E = energies(t, P)
        specs = [("path", 0, 0), ("complete", 0, 0)]
        for k in ks:
            specs.append(("informed", k, 0))
            specs += [("random", k, s) for s in ([idx] + list(range(100, 100 + args.random_seeds)))]
        for kind, k, seed in specs:
            edges, w = library_edges(t, n, kind, k, seed)
            nbr = np.stack([col[e] for e in edges], axis=1)
            nmin, d, D = landscape(E, nbr)
            rows.append({"n": n, "instance": m["name"], "lib": kind, "k": k if kind != "complete" else rest_n,
                         "seed": seed, "n_minima": int(nmin), "d_star": float(d), "difficulty": float(D),
                         "t_rel": t_rel(n, edges, w), "F_opt": float(-E.min())})
        print(f"{m['name']}: done {time.time() - t0:.0f} s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"depth_n{n}.json" if args.task < 0 else f"depth_n{n}_task{args.task:02d}.json"
    (args.out / name).write_text(json.dumps(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Relaxation time and exact energy landscapes of the move libraries.")
    ap.add_argument("what", choices=["gap", "depth"])
    ap.add_argument("--inst", type=Path, default=None, help="staged instance directory (default: $NIMBUS_MATCHING_INST)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=int, default=-1, help="depth: one test instance (index 0-19); default all")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--random-seeds", type=int, default=9, help="random draws beyond the instance-index seed")
    a = ap.parse_args(argv)
    if a.inst is None:
        import os

        if not os.environ.get("NIMBUS_MATCHING_INST"):
            raise SystemExit("set --inst or NIMBUS_MATCHING_INST to the staged instance directory")
        a.inst = Path(os.environ["NIMBUS_MATCHING_INST"])
    run_gap(a) if a.what == "gap" else run_depth(a)


if __name__ == "__main__":
    main()
