"""Host-side tools for the real-time board runs (any computer with the nimbus package).

  make-shots: raw syndromes of N = 4 d noisy rounds + the perfect round per shot (nimbus.decoding.phenomenological, the
      sampler of the accuracy runs), with the true observables for the logical error rate
      python hardware/host/ptw_host.py make-shots --d 7 --p 0.04 --shots 300 --seed 7 --out shots_ptw_d7_p0.04.npz
  check: every window record of a board run against the golden model (nimbus.decoding.real_time.decode_batch: same tables,
      seeds and shots, the batch's shots on one generator stream as on the board), bit for bit, plus the cycle model of
      every window, the LER and the timing summary. Batches seeded separately are checked in parallel (--workers).
      python hardware/host/ptw_host.py check --res res.npz [--max 300] [--workers 12]
  same: two board runs of the same shots, seeds and tables (e.g. real-time and throughput) give identical decisions
      python hardware/host/ptw_host.py same --a rt.npz --b thr.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptw_board as D  # noqa: E402

DEC_FIELDS = ("shot", "win", "fin", "best_e", "best_c", "dec_ml", "frame", "fr_ml", "counts", "sigma")


def make_shots(d: int, p: float, shots: int, seed: int) -> dict:
    from nimbus.decoding import real_time as g
    from nimbus.decoding.phenomenological import memory_model
    N = 4 * d
    Dx, Dz, ox, oz = memory_model(d, N).sample(p, shots, np.random.default_rng([seed, d, 4242]))
    m = (d * d - 1) // 2
    sx, sz = g.syndromes(Dx, N, m), g.syndromes(Dz, N, m)
    syn = np.concatenate([sx, sz], 2).astype(np.uint8)
    return dict(syn=syn, obs=np.stack([ox, oz], 1).astype(np.uint8), d=d, p=p, seed=seed)


def load_board_npz(path: str):
    z = np.load(path, allow_pickle=False)
    res = {k[2:]: z[k] for k in z.files if k.startswith("r_")}
    meta = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
    meta = {k: (v.item() if v.ndim == 0 else v) for k, v in meta.items()}
    return res, meta, z


def golden_cfg(meta: dict):
    from nimbus.decoding import real_time as g
    info = json.loads(meta.get("info") or "{}")
    pi = info.get("ptw_info", {})
    KS, RS, LDC = int(meta["KS"]), int(meta["RS"]), int(meta["LDC"])
    lram = bool(pi["lram"]) if "lram" in pi else (KS > 1 and LDC == RS * KS)
    bmin = float(meta["bmin"]) if meta.get("bmin") not in (None, -1) else 0.1
    return g.PTWConfig(R=int(meta["R"]), K=int(meta["K"]), B=int(meta["B"]), bmin=bmin, k=KS, lram=lram, carry=True)


def _golden_batch(args):
    d, p, syn, nsw, cfg, seed, tables = args
    from nimbus.decoding import real_time as g
    m = syn.shape[2] // 2
    recs, _ = g.decode_batch(d, p, syn[..., :m], syn[..., m:], nsw, cfg, seed=seed, tables=tables)
    return recs


def golden_for(meta: dict, res: dict, syn: np.ndarray, S: int, workers: int = 1) -> list:
    """Golden window records of the first S shots of a board run (batch structure, seeds and tables from the result file)."""
    from nimbus.decoding import real_time as g
    cfg = golden_cfg(meta)
    d, p, nsw = int(meta["d"]), float(meta["p"]), int(meta["nsw"])
    tables = (np.asarray(meta["thr"]), np.asarray(meta["theta"]), int(meta["Wq"]), int(meta["wq"]), int(meta["mode"]))
    bsz = np.asarray(res["b_shots"]); seeds = np.asarray(res["b_seed"])
    reseed = int(meta.get("reseed", 1))
    jobs, i = [], 0
    for b, n in enumerate(bsz):
        if i >= S:
            break
        n = min(int(n), S - i)
        jobs.append((d, p, syn[i:i + n], nsw, cfg, int(seeds[b]), tables))
        i += n
    if not reseed:
        state, out = None, []
        for j in jobs:
            m = j[2].shape[2] // 2
            r, state = g.decode_batch(d, p, j[2][..., :m], j[2][..., m:], nsw, cfg, seed=j[5], state=state, tables=tables)
            out += r
        return out
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_golden_batch, jobs)
    else:
        parts = [_golden_batch(j) for j in jobs]
    out, off = [], 0
    for j, r in zip(jobs, parts):
        for e in r:
            e = dict(e); e["shot"] += off
            out.append(e)
        off += len(j[2])
    return out


def compare(res: dict, gold: list, meta: dict) -> dict:
    n = len(gold)
    bad = np.zeros(n, bool)
    per = {}
    exp = dict(shot=[e["shot"] for e in gold], win=[e["win"] for e in gold], fin=[e["fin"] for e in gold],
               best_e=[e["best_e"] for e in gold], best_c=[e["best_c"] for e in gold], dec_ml=[e["dec_ml"] for e in gold],
               frame=[e["frame"] for e in gold], fr_ml=[e["fr_ml"] for e in gold])
    for k, v in exp.items():
        m = np.asarray(res[k][:n]).astype(np.int64) != np.asarray(v, np.int64)
        per[k] = int(m.sum()); bad |= m
    m = (np.asarray(res["counts"][:n]) != np.array([e["counts"] for e in gold])).any(1)
    per["counts"] = int(m.sum()); bad |= m
    m = (np.asarray(res["sigma"][:n]) != np.array([D.sigma_words(e["sigma"]) for e in gold])).any(1)
    per["sigma"] = int(m.sum()); bad |= m
    cyc = int(meta["cyc_model"])
    m = np.asarray(res["cyc_win"][:n]) != cyc
    per["cyc_win"] = int(m.sum()); bad |= m
    m = np.asarray(res["d_done"][:n]) != cyc + 1
    per["d_done"] = int(m.sum()); bad |= m
    return dict(windows=n, shots=n // D.NWIN, mismatched_windows=int(bad.sum()), per_field=per,
                first_bad=np.flatnonzero(bad)[:10].tolist(), cyc_model=cyc)


def ler(res: dict, obs: np.ndarray) -> dict:
    fin = np.asarray(res["fin"]) == 1
    S = int(fin.sum())
    out = {}
    for k in ("fr_ml", "frame"):
        v = np.asarray(res[k])[fin]
        f = (np.stack([v & 1, v >> 1], 1) != np.asarray(obs)[:S]).any(1)
        out["ml" if k == "fr_ml" else "me"] = dict(shots=S, fails=int(f.sum()), ler=float(f.mean()) if S else float("nan"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("make-shots")
    a1.add_argument("--d", type=int, required=True); a1.add_argument("--p", type=float, required=True)
    a1.add_argument("--shots", type=int, default=300); a1.add_argument("--seed", type=int, default=7)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("check")
    a2.add_argument("--res", required=True); a2.add_argument("--max", type=int, default=0)
    a2.add_argument("--workers", type=int, default=1); a2.add_argument("--json", default="")
    a3 = sub.add_parser("same")
    a3.add_argument("--a", required=True); a3.add_argument("--b", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-shots":
        z = make_shots(a.d, a.p, a.shots, a.seed)
        np.savez_compressed(a.out, **z)
        print(f"{a.shots} shots d={a.d} p={a.p} -> {a.out} (syn {z['syn'].shape})")
        return
    if a.cmd == "same":
        ra, ma, _ = load_board_npz(a.a)
        rb, mb, _ = load_board_npz(a.b)
        n = min(len(ra["shot"]), len(rb["shot"]))
        diff = {k: int((np.asarray(ra[k][:n]) != np.asarray(rb[k][:n])).reshape(n, -1).any(1).sum()) for k in DEC_FIELDS}
        print(json.dumps(dict(windows=n, differing=diff)))
        sys.exit(0 if not any(diff.values()) else 1)
    res, meta, z = load_board_npz(a.res)
    syn = z["syn"]
    S = a.max or len(syn)
    gold = golden_for(meta, res, syn, S, a.workers)
    rep = compare(res, gold, meta)
    if "obs" in z.files:
        rep["ler_all"] = ler(res, z["obs"])
    mhz = float(z["clock_mhz"]) if "clock_mhz" in z.files else 0
    if mhz:
        rep["timing"] = D.timing_summary(res, meta, mhz)
    print(json.dumps(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if rep["mismatched_windows"] == 0 else 1)


if __name__ == "__main__":
    main()
