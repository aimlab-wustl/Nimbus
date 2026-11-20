"""Decode times of the CPU decoders on one core (decode call only; decoder objects built once and warmed up).

  python cpu_timing.py all --out RUNS/cpu_timing          # both parts, one core, one after another (as in the paper)
  python cpu_timing.py one-shot --d 3 5 7 9 11 13 15 --p 0.12 --shots 200 --tn-shots 100 --out RUNS/cpu_timing/one_shot.jsonl
  python cpu_timing.py real-time --d 5 7 9 11 --p 0.03 --shots 60 --out RUNS/cpu_timing

one-shot
    Per-shot time of PyMatching (one decode per check type), union-find (ldpc, peeling, one decode per check type),
    BP-OSD (joint X / Y / Z columns, OSD-CS order 10) and tensor-network ML (qecsim, bond dimension 16) on the shots of
    the accuracy runs (chunk 40). Each record holds (median, mean) seconds per shot; the paper uses the mean.
real-time
    Time per round of the window decoders (2d rounds, commit d) on the first shots of chunk 0: total time over all
    windows / (shots x N) ("per_round"), and the mean time of a non-final window / d ("per_commit_round", the time to
    commit d rounds, the quantity compared with the board's time per round). Output ``wbase_time_d<d>_p<p>.json``.

The paper's times come from one node of a CPU cluster (Intel Xeon Gold 6548Y+), one core (``all`` pins the process to
core 0 with taskset when available and sets every thread count to 1).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def time_calls(fn, items):
    ts = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.mean(ts))


def one_shot(ds, p, shots, tn_shots, out, chunk=40):
    import pymatching
    from ldpc.union_find_decoder import UnionFindDecoder
    from qecsim import paulitools as pt
    from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder

    from nimbus.decoding.baselines import bposd_decoder, to_qecsim
    from nimbus.decoding.code_capacity import sample_shots
    from nimbus.problems.surface_code import SurfaceCodeProblem
    for d in ds:
        prob = SurfaceCodeProblem(d, p, noise="depolarizing")
        e, syn = sample_shots(prob, shots, 7 + 1000 * chunk)
        rec = dict(d=d, p=p, shots=shots, cpu="one core, " + C.cpu_model())
        (Hz, *_), (Hx, *_) = prob._checks[0], prob._checks[1]
        sz, sx = prob._split_syndrome(syn)
        mz, mx = pymatching.Matching.from_check_matrix(Hz), pymatching.Matching.from_check_matrix(Hx)
        for s in sz[:20]:
            mz.decode(s)
        rec["pm_s"] = time_calls(lambda k: (mz.decode(sz[k]), mx.decode(sx[k])), range(shots))
        uz, ux = UnionFindDecoder(Hz, uf_method="peeling"), UnionFindDecoder(Hx, uf_method="peeling")
        for s in sz[:20]:
            uz.decode(s)
        rec["uf_s"] = time_calls(lambda k: (uz.decode(sz[k]), ux.decode(sx[k])), range(shots))
        bo = bposd_decoder(prob)
        for s in syn[:20]:
            bo.decode(s)
        rec["bposd_s"] = time_calls(lambda k: bo.decode(syn[k]), range(shots))
        code = RotatedPlanarCode(d, d)
        dec = RotatedPlanarMPSDecoder(chi=16)
        syns = [pt.bsp(b, code.stabilizers.T) for b in to_qecsim(prob, code, e[:tn_shots])]
        dec.decode(code, syns[0])
        rec["tn_s"] = time_calls(lambda s: dec.decode(code, s), syns)
        print(json.dumps(rec), flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def real_time(ds, p, shots, out):
    from nimbus.decoding.baselines import WINDOW_DECODERS
    from nimbus.decoding.phenomenological import memory_shots
    from nimbus.decoding.windows import correction_failures, decode_windows, schedule
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for d in ds:
        N = 4 * d
        sched = schedule(N, d, d)
        Dx, Dz, ox, oz = memory_shots(d, p, 250, chunk=0)
        Dx, Dz, ox, oz = Dx[:shots], Dz[:shots], ox[:shots], oz[:shots]
        rec = dict(d=d, p=p, N=N, tshots=shots, windows=len(sched), cpu="one core, " + C.cpu_model(), per_round={},
                   per_commit_round={}, ler={})
        for q, fac in WINDOW_DECODERS.items():
            dec = fac(p)
            decode_windows(d, N, sched, Dx[:1], Dz[:1], dec)            # build and cache the decoder objects
            tw = []

            def timed(w, Dxw, Dzw, final, k, t0, _dec=dec):
                s = time.perf_counter()
                r = _dec(w, Dxw, Dzw, final, k, t0)
                tw.append((final, time.perf_counter() - s))
                return r
            s = time.perf_counter()
            corr = decode_windows(d, N, sched, Dx, Dz, timed)
            tot = time.perf_counter() - s
            nf = [t for fin, t in tw if not fin]
            rec["per_round"][q] = tot / (shots * N)
            rec["per_commit_round"][q] = (np.mean(nf) / shots) / d if nf else None
            rec["ler"][q] = float(correction_failures(d, N, corr, ox, oz)[q].mean())
        (out / f"wbase_time_d{d}_p{p}.json").write_text(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


def run_all(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    pin = ["taskset", "-c", "0"] if Path("/usr/bin/taskset").exists() else []
    me = [sys.executable, str(Path(__file__).resolve())]
    subprocess.run(pin + me + ["one-shot", "--out", str(out / "decoder_timing_one_shot.jsonl")], check=True, env=env)
    subprocess.run(pin + me + ["real-time", "--out", str(out)], check=True, env=env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("one-shot")
    a1.add_argument("--d", type=int, nargs="+", default=C.DS_ONE_SHOT)
    a1.add_argument("--p", type=float, default=C.P_ONE_SHOT)
    a1.add_argument("--shots", type=int, default=200)
    a1.add_argument("--tn-shots", type=int, default=100)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("real-time")
    a2.add_argument("--d", type=int, nargs="+", default=C.DS_REAL_TIME)
    a2.add_argument("--p", type=float, default=C.P_REAL_TIME)
    a2.add_argument("--shots", type=int, default=60)
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("all")
    a3.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "one-shot":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        one_shot(a.d, a.p, a.shots, a.tn_shots, a.out)
    elif a.cmd == "real-time":
        real_time(a.d, a.p, a.shots, a.out)
    else:
        run_all(a.out)


if __name__ == "__main__":
    main()
