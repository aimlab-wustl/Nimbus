"""The annealing protocol of the graph-matching results, step by step, through the Python API.

1. compile the problem and a move library (complete, path, or the path backbone plus K shortcuts);
2. calibrate the start and end temperatures once (acceptance rule) and apply the start factor m and the minimum
   ratio kappa;
3. run independent replicas on the Fowler-Nordheim (FN) schedule for several budgets B, each replica from its own
   random feasible assignment, with the certified optimum as target;
4. report the success fraction p and the iterations to solution ITS(B) = B ln(0.01) / ln(1 - p).

    python examples/paper_protocol.py                        # examples/problems/matching_n8.npz, complete library
    python examples/paper_protocol.py --library path --m 2 --kappa 4
    python examples/paper_protocol.py --library informed --k 4 --share 0.5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from nimbus.annealing import schedule as S
from nimbus.annealing.protocol import ETA, calibrate_temperatures, scaled_temperatures
from nimbus.backends import get_backend
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.problems.formats.matching_tensor import load_tensor
from nimbus.problems.matching import MatchingProblem, enumerate_optimum, shortcut_library

HERE = Path(__file__).resolve().parent


def its(budget: int, p: float) -> float:
    """Iterations to reach the target with probability 0.99 at budget ``budget`` and success fraction ``p``."""
    if p >= 1.0:
        return float(budget)
    return budget * math.log(0.01) / math.log(1.0 - p) if p > 0 else math.inf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tensor", default=str(HERE / "problems" / "matching_n8.npz"))
    ap.add_argument("--library", choices=("complete", "path", "informed", "random"), default="complete")
    ap.add_argument("--k", type=int, default=4, help="shortcuts added to the path backbone (informed / random)")
    ap.add_argument("--share", type=float, default=None, help="fixed backbone proposal share (informed / random)")
    ap.add_argument("--m", type=float, default=1.0, help="start factor m")
    ap.add_argument("--kappa", type=float, default=2.0, help="minimum ratio kappa of start to end temperature")
    ap.add_argument("--replicas", type=int, default=64)
    ap.add_argument("--budgets", default="8,10,12,14", help="log2 of the budgets")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)

    tensor = load_tensor(a.tensor)
    prob = MatchingProblem(tensor, name=Path(a.tensor).stem)
    spec = {"base": a.library} if a.library in ("complete", "path") else \
        {"base": "path", "extra": a.library, "k": a.k, "seed": a.seed, "share": a.share}
    library, info = shortcut_library(prob, spec, tensor)
    model = compile_model(prob.structure(), library, identities=tuple(prob.identities()))
    binding = bind_weights(model, prob.term_weights(0), prob.const(0))
    print(f"{prob.name}: n = {prob.P}, library {a.library}: R = {model.n_groups} moves, "
          f"{model.n_clauses} stored clauses, {model.n_routes} connections")

    # one raw calibration, then the start factor m and the minimum ratio kappa
    ts0, te0 = calibrate_temperatures(model, binding, prob, a.seed, raw=True)
    tau_start, tau_end = scaled_temperatures(ts0, te0, a.m, a.kappa)
    print(f"calibrated tau_start = {ts0:.4g}, tau_end = {te0:.4g}; with m = {a.m:g}, kappa = {a.kappa:g}: "
          f"tau_start = {tau_start:.4g}, tau_end = {tau_end:.4g}")

    opt = enumerate_optimum(tensor)                    # certified optimum (n <= 11)
    target = -opt["F_opt"]                             # energy E = -F
    be = get_backend("cpu")
    plan = be.prepare(model, threads=a.threads)
    spins0 = prob.random_feasible(a.seed, a.replicas, 0)
    print(f"optimum F* = {opt['F_opt']:.6g}")
    print("budget    p(optimum)   ITS")
    for lb in (int(x) for x in a.budgets.split(",")):
        B = 2 ** lb
        state = be.init_state(plan, binding, spins0, replica_id=np.arange(a.replicas), is_feasible=prob.is_feasible,
                              targets=np.full(a.replicas, target))
        res = be.run(plan, state, S.fn_endpoints(tau_start, tau_end, ETA, B), seed=a.seed)
        hit = np.asarray(res.state.counters["target_hit_attempt"]) >= 0
        p = float(hit.mean())
        print(f"2^{lb:<6d}  {p:10.3f}   {its(B, p):.4g}")


if __name__ == "__main__":
    main()
