"""Size of the compiled machine (Fig. S3): information needed to program and run it, counted at its minimum.

  python program_size.py --cases surface:3:stabilizers matching:30:complete ... [--inst INST] --out RUNS/size/sizes.jsonl

Cases: ``surface:<d>:<library>`` (rotated surface code, one syndrome per shot, depolarizing noise; library
``stabilizers`` or ``stabilizers_logical_all``) and ``matching:<L>:<library>`` (library ``complete`` or ``path``; the
synthetic instance ``syn_n<L>_s0.05_o0_06`` of the staged set). One JSON line per case is appended to ``--out``.

Counting rule. The run-time state (clause outputs, maintained inputs) is added in the analysis; here the program:
  1. connections to objective clauses      log2 C(M_obj * R, connections to objective clauses)
  2. connections to auxiliary clauses      log2 C(M_aux * R, ...) (+ the same for readout-only clauses, none here)
  3. switch details                        which connections are switched: log2 C(connections, switched);
                                           per move neuron: switched connections x log2(auxiliary clauses used by that
                                           neuron) (+ log2 C(G, clauses of the neuron) when an auxiliary clause
                                           controls switches of several neurons); which clause state opens each
                                           switch: log2 C(switched, opens on +1)
  4. weights                               counted in the analysis (16 bits per objective clause)
Binary structures are counted at their information minimum (a lower bound; the structure is treated as
unstructured). Free by convention: clause numbering in role blocks, auxiliary clauses private to one neuron numbered
in one block per neuron, weighted connections listed first per neuron. Every count is deterministic (compile only).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "graph_matching"))

from nimbus.compiler import compile_model  # noqa: E402
from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE, ROLE_READOUT  # noqa: E402

#: the cases of Fig. S3
CASES = ([f"surface:{d}:{lib}" for d in (3, 5, 7, 9, 11) for lib in ("stabilizers", "stabilizers_logical_all")]
         + [f"matching:{n}:{lib}" for n in (10, 15, 20, 25, 30) for lib in ("path", "complete")]
         + ["matching:40:path", "matching:50:path", "matching:40:complete", "matching:50:complete"])


def log2_choose(n, k):
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def program_bits(model) -> dict:
    role = model.clause_role.astype(np.int64)
    obj = (role & ROLE_OBJECTIVE) != 0
    gate = ((role & ROLE_GATE) != 0) & ~obj
    read = ((role & ROLE_READOUT) != 0) & ~obj & ~gate
    can_ctrl = (role & ROLE_GATE) != 0
    R = int(model.n_groups)
    rp = model.route_ptr.astype(np.int64)
    rc = model.route_clause
    ctrl = model.route_ctrl
    pol = model.route_pol
    nnz = int(rc.size)
    out = dict(R=R, M_obj=int(obj.sum()), M_gate=int(gate.sum()), M_read=int(read.sum()), routes=nnz,
               G=int(can_ctrl.sum()))
    for name, mask in (("obj", obj), ("gate", gate), ("read", read)):
        k = int(np.count_nonzero(mask[rc]))
        out[f"routes_{name}"] = k
        out[f"bits_pi_{name}"] = log2_choose(int(mask.sum()) * R, k)
    owner = np.full(int(model.n_clauses), -1, dtype=np.int64)
    shared = False
    nc_tot, npos, bits_ctrl, g_used, per_neuron = 0, 0, 0.0, [], []
    for r in range(R):                                   # per neuron: keeps the memory small for large models
        c = ctrl[rp[r]:rp[r + 1]]
        m = c >= 0
        n_r = int(np.count_nonzero(m))
        if not n_r:
            continue
        u = np.unique(c[m]).astype(np.int64)
        prev = owner[u]
        if np.any((prev >= 0) & (prev != r)):
            shared = True
        owner[u] = r
        per_neuron.append((n_r, int(u.size)))
        nc_tot += n_r
        npos += int(np.count_nonzero(pol[rp[r]:rp[r + 1]][m] == 1))
        g_used.append(int(u.size))
    for n_r, g_r in per_neuron:
        bits_ctrl += n_r * math.log2(max(g_r, 1)) + (log2_choose(out["G"], g_r) if shared else 0.0)
    out.update(controlled=nc_tot, gates_private=not shared,
               gates_per_neuron_mean=float(np.mean(g_used)) if g_used else 0.0,
               switch_open_plus_frac=(npos / nc_tot) if nc_tot else None)
    out["bits_switch"] = log2_choose(nnz, nc_tot) + bits_ctrl + log2_choose(nc_tot, npos)
    return out


def build(case, inst):
    prob, size, lib = case.split(":")
    size = int(size)
    if prob == "surface":
        from nimbus.problems.surface_code import SurfaceCodeProblem
        p = SurfaceCodeProblem(size, 0.10, noise="depolarizing", library=lib)
        return prob, size, lib, "depolarizing", compile_model(p.structure(), p.transition_library(), identities=())
    import common as GM
    name = f"syn_n{size}_s0.05_o0_06"
    _, t, p = GM.load_instance(GM.inst_dir(inst), name)
    model, _, _ = GM.compile_instance(p, t, {"base": lib})
    return prob, size, lib, name, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--inst", default=None, help="staged matching instances (default: $NIMBUS_MATCHING_INST)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for case in a.cases:
        prob, size, lib, inst, model = build(case, a.inst)
        rec = dict(problem=prob, size=size, lib=lib, instance=inst, **program_bits(model))
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(case, "R", rec["R"], "connections", rec["routes"], flush=True)
        del model


if __name__ == "__main__":
    main()
