"""Real time (sliding windows, phenomenological depolarizing noise): runs of the paper.

  python real_time.py decoder   --d 9 --p 0.03 --chunk 0 --R 24 --K 12 --out RUNS/real_time
  python real_time.py baselines --d 9 --p 0.03 --chunk 0 --decoders pm pmc bpm bp --out RUNS/real_time_baselines
  python real_time.py union-find --d 9 --p 0.03 --chunks 0-11 --out RUNS/real_time_union_find
  python real_time.py tasks configs/tasks_real_time.txt --out RUNS/real_time [--worker j --workers K]

Memory experiment: N = 4d noisy rounds (X, Y, Z data errors each with probability p/3, measurement flips with
probability p) and one round without error. Shots: ``memory_shots(d, p, 250, chunk)`` (seed 11, 250 shots per chunk),
so chunk c of every script below is the same 250 shots.

decoder
    The software model of the FPGA design for d (bit-exact: hardware schedule of its slab layout, generators carried
    across the windows of a shot, R replicas, beta from 1 to 0.1), windows of 2d rounds that commit their first d,
    2^K sweeps per window. Non-final windows commit the lowest-energy state, the final window the most visited class.
    Decoder seed ``(11 << 40) ^ (chunk << 24) ^ (d << 16) ^ int(p * 1e5)``. Output
    ``ptw_d<d>_p<p>_c<chunk>_g_hwc_r<R>_k<K>.npz`` with the per-shot failures ``f_<var>:ml`` and ``f_<var>:me``.
baselines
    CPU decoders in the same sliding windows (2d rounds, commit d): PyMatching (pm), correlated PyMatching (pmc),
    BP+matching (bpm), BP-OSD with the library's min-sum scaling 1.0 (bp) and with scaling 0.625 (bp625, the setting of
    the one-shot BP-OSD; the paper's BP-OSD in real time). Output ``win_d<d>_p<p>_c<chunk>_wbase_<names>.npz``.
union-find
    Union-find (ldpc, peeling) on each sector of the whole memory, over the chunks given. Output
    ``b_uf_d<d>_p<p>.json`` (failures, LER, seconds per shot).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

from nimbus.decoding import real_time as RT  # noqa: E402
from nimbus.decoding.phenomenological import memory_model, memory_shots  # noqa: E402
from nimbus.decoding.windows import correction_failures, decode_windows, schedule  # noqa: E402

SEED, SHOTS = 11, 250


def decoder_seed(d, p, chunk):
    return (SEED << 40) ^ (chunk << 24) ^ (d << 16) ^ int(p * 1e5)


def decoder(d, p, chunk, R, K, out, shots=SHOTS, bmin=0.1, rounds_per_d=4):
    var = f"g_hwc_r{R}_k{K}" + (f"_n{rounds_per_d}" if rounds_per_d != 4 else "") + (f"_bm{bmin:g}" if bmin != 0.1 else "")
    tag = f"ptw_d{d}_p{p}_c{chunk}_{var}"
    if C.done(out, tag):
        return f"skip {tag}"
    N = rounds_per_d * d
    Dx, Dz, ox, oz = memory_shots(d, p, shots, chunk=chunk, seed=SEED, rounds=N)
    t0 = time.perf_counter()
    res, info = RT.decode_stream(d, p, Dx, Dz, N, d, d, 1 << K, RT.fpga_config(d, R, bmin), seed=decoder_seed(d, p, chunk),
                                 start=("chain",))
    f = {f"{var}:{k}": v for k, v in RT.failures(res, ox, oz).items()}
    rec = dict(d=d, p=p, chunk=chunk, var=var, shots=shots, N=N, K=K, R=R, bmin=bmin,
               sec=round(time.perf_counter() - t0, 1), ler={k: float(v.mean()) for k, v in f.items()}, Emax=info["Emax"])
    C.save_task(out, tag, {f"f_{k}": v for k, v in f.items()}, rec)
    return json.dumps(rec)


def baselines(d, p, chunk, names, out, shots=SHOTS):
    from nimbus.decoding.baselines import WINDOW_DECODERS
    tag = f"win_d{d}_p{p}_c{chunk}_wbase_" + "_".join(names)
    if C.done(out, tag):
        return f"skip {tag}"
    N = 4 * d
    Dx, Dz, ox, oz = memory_shots(d, p, shots, chunk=chunk, seed=SEED)
    f, sec = {}, {}
    t0 = time.perf_counter()
    for q in names:
        tq = time.perf_counter()
        f.update(correction_failures(d, N, decode_windows(d, N, schedule(N, d, d), Dx, Dz, WINDOW_DECODERS[q](p)), ox, oz))
        sec[q] = round(time.perf_counter() - tq, 3)
    rec = dict(d=d, p=p, chunk=chunk, dec="wbase_" + "_".join(names), shots=shots, N=N,
               sec=round(time.perf_counter() - t0, 1), ler={k: float(v.mean()) for k, v in f.items()}, sec_dec=sec)
    C.save_task(out, tag, {f"f_{k}": v for k, v in f.items()}, rec)
    return json.dumps(rec)


def union_find(d, p, chunks, out):
    from nimbus.decoding.baselines import memory_union_find_failures
    f = Path(out) / f"b_uf_d{d}_p{p}.json"
    if f.exists():
        return f"skip {f.name}"
    N = 4 * d
    G = memory_model(d, N)
    fails, n, t = 0, 0, 0.0
    for ch in chunks:
        Dx, Dz, ox, oz = memory_shots(d, p, SHOTS, chunk=ch, seed=SEED)
        t0 = time.perf_counter()
        fu = memory_union_find_failures(G, Dx, Dz, ox, oz)
        t += time.perf_counter() - t0
        fails += int(fu.sum())
        n += len(fu)
    rec = dict(d=d, p=p, N=N, shots=n, chunks=[min(chunks), max(chunks)], uf_fails=fails, ler_uf=fails / n, sec_uf=t / n)
    Path(out).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec) + "\n")
    return json.dumps(rec)


def chunk_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def run_line(line, out):
    """One line of a task table: ``decoder d p chunk R K`` or ``baselines d p chunk name1,name2,...``."""
    f = line.split()
    d, p, chunk = int(f[1]), float(f[2]), int(f[3])
    if f[0] == "decoder":
        return decoder(d, p, chunk, int(f[4]), int(f[5]), out)
    return baselines(d, p, chunk, f[4].split(","), out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("decoder")
    a2 = sub.add_parser("baselines")
    a3 = sub.add_parser("union-find")
    for q in (a1, a2, a3):
        q.add_argument("--d", type=int, required=True)
        q.add_argument("--p", type=float, required=True)
        q.add_argument("--out", required=True)
    for q in (a1, a2):
        q.add_argument("--chunk", type=int, required=True)
    a1.add_argument("--R", type=int, required=True, help="replicas")
    a1.add_argument("--K", type=int, default=C.K_REAL_TIME, help="2^K sweeps per window")
    a1.add_argument("--bmin", type=float, default=0.1)
    a1.add_argument("--rounds-per-d", type=int, default=4, help="noisy rounds per d (4: N = 4d)")
    a2.add_argument("--decoders", nargs="+", default=["pm", "pmc", "bpm", "bp"])
    a3.add_argument("--chunks", default="0-11")
    a4 = sub.add_parser("tasks")
    a4.add_argument("table")
    a4.add_argument("--out", required=True)
    a4.add_argument("--worker", type=int, default=0)
    a4.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "decoder":
        print(decoder(a.d, a.p, a.chunk, a.R, a.K, a.out, bmin=a.bmin, rounds_per_d=a.rounds_per_d))
    elif a.cmd == "baselines":
        print(baselines(a.d, a.p, a.chunk, a.decoders, a.out))
    elif a.cmd == "union-find":
        print(union_find(a.d, a.p, chunk_range(a.chunks), a.out))
    else:
        for line in C.select_tasks(C.task_lines(a.table), a.worker, a.workers):
            print(run_line(line, a.out), flush=True)


if __name__ == "__main__":
    main()
