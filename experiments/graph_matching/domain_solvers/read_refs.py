"""Read the results of run_refs.m and evaluate every returned assignment with the package's objective."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

from nimbus.problems.formats.matching_tensor import evaluate_F, load_tensor


def load_refs(mat_path, inst_dir):
    """One dict per (instance, solver): instance, solver, F (objective per restart), times (s), perms, meta (manifest).

    The solver's own score (6 Psi) is checked against the objective evaluated here.
    """
    inst_dir = Path(inst_dir)
    man = {m["name"]: m for m in json.loads((inst_dir / "manifest.json").read_text())}
    res = sio.loadmat(mat_path, simplify_cells=True)["res"]
    rows = []
    for key, solvers in res.items():
        m = man[key] if key in man else man[next(k for k in man if k.replace(".", "_") == key or
                                                  re.sub(r"\W", "_", k) == key)]
        t = load_tensor(inst_dir / m["tensor"])
        for s, d in solvers.items():
            perms = np.atleast_2d(d["perms"]).astype(int)
            F = evaluate_F(t, perms)
            assert np.allclose(6 * F, np.atleast_1d(d["score"]), rtol=1e-9), (key, s)
            rows.append({"instance": m["name"], "solver": s, "F": F, "times": np.atleast_1d(d["times"]),
                         "perms": perms, "meta": m})
    return rows
