"""Host-side tools for the one-syndrome-per-shot board runs (any computer with the nimbus package).

  make-shots: syndromes for the board, with the failure flag of each of the 4 classes (for the logical error rate)
      python hardware/host/pt_host.py make-shots --d 7 --p 0.15 --shots 1000 --seed 7 --out shots_d7_p0.15.npz
  check: the board's results against the golden model, bit for bit (same tables, seed and shots, shots back to back on one
      generator stream as on the board), plus the cycle model for every shot's cycle counts
      python hardware/host/pt_host.py check --res res.npz [--max 300]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nimbus.decoding.one_shot import PTConfig, build_spec, c0_from_syndrome, run_shot, seed_states, with_tables  # noqa: E402
import pt_board as D  # noqa: E402


def make_shots(d: int, p: float, shots: int, seed: int):
    from nimbus.decoding.code_capacity import class_correction, sample_shots
    spec = build_spec(d, p, PTConfig(NS=2))
    prob = spec.prob
    e, syn = sample_shots(prob, shots, seed)
    c0 = c0_from_syndrome(spec, syn)
    fail_cls = np.stack([prob.logical_failure(class_correction(prob, c0, np.full(shots, k)), e) for k in range(4)], 1)
    return dict(syn=syn.astype(np.uint8), fail_cls=fail_cls.astype(bool), d=d, p=p, seed=seed)


def golden_for(res: dict, meta: dict, syn: np.ndarray, d: int, p: float = 0.12):
    """Golden decisions for the board session described by meta (tables, seed, budget, early stop)."""
    NS, K, B = int(meta["NS"]), int(meta["K"]), int(meta["B"])
    spec = with_tables(build_spec(d, p, PTConfig(NS=NS, K=K, B=B)), meta["thr"], meta["theta"])
    ga, gb, sw = seed_states(spec, 1, int(meta["seed"]))
    ga, gb, sw = ga[0], gb[0], sw[0]
    heads = np.zeros((NS, 2), np.int64)
    c0 = c0_from_syndrome(spec, syn)
    nsw, gmin, th = int(meta["nsw"]), int(meta["gmin"]), int(meta["th32"])
    out = {k: [] for k in ("ml", "me", "sweeps", "emin", "counts")}
    for s in range(len(syn)):
        r = run_shot(spec, c0[s], nsw, ga, gb, heads, sw, gmin, th if gmin else 0)
        kb = int(r["stop"]).bit_length() - 1                     # best snapshot row (last power-of-two checkpoint)
        s1 = int(r["stop"]) - 1                                  # count block of the last sweep, as the RTL: sweeps >= nsw / 2
        cnt = r["counts"][max(1, s1 if s1 < (nsw >> 1) else nsw - 1).bit_length()]   # form the last block (also after an early stop)
        out["ml"].append(int(np.argmax(cnt))); out["me"].append(int(r["bestC"][np.argmin(r["bestE"])])); out["sweeps"].append(int(r["stop"]))
        out["emin"].append(int(r["bestE"].min())); out["counts"].append(cnt.copy())
    out = {k: np.array(v) for k, v in out.items()}
    out["stopped"] = out["sweeps"] < nsw
    out["cycles"] = D.cycle_model(out["sweeps"], out["stopped"], nsw, K, NS, gmin, spec.cls_mat.shape[0])
    return out


def compare(res: dict, gold: dict) -> dict:
    S = len(gold["ml"])
    bad = np.zeros(S, bool)
    per = {}
    for k in ("ml", "me", "sweeps", "emin", "stopped"):
        m = np.asarray(res[k][:S]).astype(np.int64) != gold[k].astype(np.int64)
        per[k] = int(m.sum()); bad |= m
    m = (np.asarray(res["counts"][:S]) != gold["counts"]).any(1)
    per["counts"] = int(m.sum()); bad |= m
    cyc = np.asarray(res["cyc_start"][:S]) + np.asarray(res["cyc_run"][:S]) + 2
    m = cyc != gold["cycles"]
    per["cycles"] = int(m.sum()); bad |= m
    return dict(shots=S, mismatched_shots=int(bad.sum()), per_field=per, first_bad=np.flatnonzero(bad)[:10].tolist())


def load_board_npz(path: str):
    z = np.load(path, allow_pickle=False)
    res = {k[2:]: z[k] for k in z.files if k.startswith("r_")}
    meta = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
    meta = {k: (v.item() if v.ndim == 0 else v) for k, v in meta.items()}
    return res, meta, z


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("make-shots")
    a1.add_argument("--d", type=int, required=True); a1.add_argument("--p", type=float, required=True)
    a1.add_argument("--shots", type=int, default=1000); a1.add_argument("--seed", type=int, default=7)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("check")
    a2.add_argument("--res", required=True); a2.add_argument("--max", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "make-shots":
        z = make_shots(a.d, a.p, a.shots, a.seed)
        np.savez_compressed(a.out, **z)
        print(f"{a.shots} shots d={a.d} p={a.p} -> {a.out} (M = {z['syn'].shape[1]})")
        return
    res, meta, z = load_board_npz(a.res)
    syn = z["syn"]
    S = a.max or len(syn)
    d = int(round(math.sqrt(int(meta["N"]))))
    gold = golden_for(res, meta, syn[:S], d)
    rep = compare(res, gold)
    print(json.dumps(rep))
    sys.exit(0 if rep["mismatched_shots"] == 0 else 1)


if __name__ == "__main__":
    main()
