"""One syndrome per shot: source data of Fig. 5c-f, Fig. S7a, S7e and S8b.

  python analysis/one_shot.py --runs RUNS/one_shot --library RUNS/library --timing RUNS/cpu_timing/decoder_timing_one_shot.jsonl \
      --board BOARD/one_shot --out ../../source_data

Inputs
  --runs     accuracy runs (``afin_*.npz``, the 2^12-sweep runs ``afin_*_l12.npz`` and the union-find files ``*_uf.npz``)
  --library  library and replica-exchange runs (``abl_*.npz``)
  --timing   CPU decode times per shot (``cpu_timing.py one-shot``)
  --board    board results at p = 0.12, 20,000 shots per d: ``res_d<d>_p0.12_20k.npz`` (fixed budget; d = 5:
             ``res_d5_p0.12_20k_n128.npz``, 128 sweeps) and ``es/es_d<d>.jsonl`` (early-stop settings grid)
  published times: ``published/one_shot_times.csv``

Outputs (``--out``): fig5/fig5c.csv, fig5d.csv, fig5e.csv, fig5f.csv, figS7/figS7a.csv, figS7e.csv, figS8/figS8b.csv.

This work = the software model of each FPGA design (bit-exact with the board) at the board budget (d = 5: 128 sweeps).
Board time per shot at the fixed budget: on-chip cycles from syndrome to decision (start + run + 2 cycles) / clock,
averaged over the 20,000 shots (the same for every shot). Early stopping: of the settings grid (first checkpoint gmin,
threshold th32), the fastest whose LER differs from the fixed budget by at most 0.001; its mean time per shot. Error bars:
95% Wilson intervals (z = 1.96).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

DEC = ("hw_ml", "tn", "bposd", "pm", "uf")


def ci(k, n):
    lo, hi = C.wilson(k, n)
    return dict(fails=int(k), shots=int(n), ler=k / n, ci_lo=float(lo), ci_hi=float(hi))


def load_runs(runs):
    """{(d, p): {"grid", "f_ml", "f_me", "f_pm", "f_bo", "f_tn", "f_uf", "l12_grid", "l12_ml"}} concatenated over chunks."""
    parts = defaultdict(list)
    for f in sorted(Path(runs).glob("afin_d*_p*_c*.npz")):
        tail = f.stem.split("_c")[-1]
        if not tail.isdigit():
            continue
        d = int(f.stem.split("_d")[1].split("_")[0])
        p = float(f.stem.split("_p")[1].split("_")[0])
        parts[(d, p)].append(f)
    out = {}
    for key, fs in parts.items():
        zs = [np.load(f) for f in fs]
        r = dict(grid=list(zs[0]["grid"]), f_ml=np.concatenate([z["f_ml"] for z in zs], 1),
                 f_me=np.concatenate([z["f_me"] for z in zs], 1))
        for k in ("f_pm", "f_bo", "f_tn"):
            r[k] = np.concatenate([z[k] for z in zs])
        uf = [f.with_name(f.stem + "_uf.npz") for f in fs]
        if all(u.exists() for u in uf):
            r["f_uf"] = np.concatenate([np.load(u)["f_uf"] for u in uf])
        l12 = [f.with_name(f.stem + "_l12.npz") for f in fs]
        if all(u.exists() for u in l12):
            zl = [np.load(u) for u in l12]
            g = list(zl[0]["grid"]).index(4096)
            r["l12_ml"] = np.concatenate([z["f_ml"][g] for z in zl])
        out[key] = r
    return out


def hw_board(r, d):
    return r["f_ml"][r["grid"].index(C.SWEEPS_BOARD[d])]


def board_times(board):
    """d -> (fixed-budget time per shot, early-stop mean, early-stop setting, clock MHz) from the board result files."""
    out = {}
    for d in C.DS_ONE_SHOT:
        f = Path(board) / (f"res_d{d}_p0.12_20k_n128.npz" if d == 5 else f"res_d{d}_p0.12_20k.npz")
        z = np.load(f)
        clk = float(z["clock_mhz"])
        fixed = float(((z["r_cyc_start"] + z["r_cyc_run"] + 2) / clk).mean())
        grid = [json.loads(s) for s in open(Path(board) / "es" / f"es_d{d}.jsonl")]
        ok = [g for g in grid if abs(g["d_ler"]) <= 0.001 + 1e-12]
        es = min(ok, key=lambda g: g["lat_mean"])
        out[d] = dict(fixed_us=fixed, es_us=es["lat_mean"], gmin=es["gmin"], th32=es["th32"], clock_mhz=clk,
                      es_ler_change=es["d_ler"])
    return out


def cpu_times(path):
    t = {}
    for line in open(path):
        j = json.loads(line)
        for k in ("pm", "uf", "bposd", "tn"):
            t[(j["d"], k)] = j[f"{k}_s"][1]                  # mean seconds per shot
        t[(j["d"], "cpu")] = j["cpu"]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--published", default=str(C.HERE / "published" / "one_shot_times.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    R = load_runs(a.runs)
    P = C.P_ONE_SHOT

    # ---- Fig. 5d: LER vs d at p = 0.12, all decoders
    acc = {}
    rows = []
    for d in C.DS_ONE_SHOT:
        r = R[(d, P)]
        f = dict(hw_ml=hw_board(r, d), tn=r["f_tn"], bposd=r["f_bo"], pm=r["f_pm"], uf=r["f_uf"])
        for k in DEC:
            acc[(d, k)] = ci(f[k].sum(), f[k].size)
            rows.append(dict(d=d, p=P, decoder=k, sweeps=C.SWEEPS_BOARD[d] if k == "hw_ml" else "", **acc[(d, k)]))
    C.write_csv(out / "fig5" / "fig5d.csv", rows)
    a15 = {k: acc[(15, k)]["fails"] for k in DEC}
    print("d = 15 logical errors relative to this work: " + ", ".join(f"{k} {a15[k] / a15['hw_ml']:.2f}"
                                                                     for k in ("bposd", "pm", "uf")))

    # ---- Fig. 5c: library and replica exchange, d = 15
    lib = defaultdict(lambda: [0, 0])
    grid_of = {}
    for f in sorted(Path(a.library).glob("abl_d15_p0.12_c*_*.npz")):
        var = f.stem.split("_")[-2]
        z = np.load(f)
        grid_of[var] = list(z["grid"])
        for g, fr in zip(z["grid"], z["f_ml"]):
            lib[(var, int(g))][0] += int(fr.sum())
            lib[(var, int(g))][1] += fr.size
    name = {"all": "backbone_and_shortcuts", "one": "backbone", "noswap": "no_replica_exchange", "stab": "stabilizer_moves_only"}
    rows = []
    for (var, g), (k, n) in sorted(lib.items(), key=lambda x: (list(name).index(x[0][0]), x[0][1])):
        rows.append(dict(series=name[var], d=15, p=P, sweeps=g if var != "stab" else "", **ci(k, n), fpga_sweeps=C.SWEEPS_BOARD[15]))
    rows.append(dict(series="tensor_network_ml", d=15, p=P, sweeps="", **acc[(15, "tn")], fpga_sweeps=C.SWEEPS_BOARD[15]))
    C.write_csv(out / "fig5" / "fig5c.csv", rows)
    nr = lib[("noswap", 4096)]
    print(f"no replica exchange at 2^12 sweeps: LER {nr[0] / nr[1]:.4f}")

    # ---- Fig. 5e: LER vs decode time per shot
    bt = board_times(a.board)
    ct = cpu_times(a.timing)
    rows = []
    for d in C.DS_ONE_SHOT:
        rows.append(dict(d=d, decoder="hw_ml", ler=acc[(d, "hw_ml")]["ler"], time_us=bt[d]["es_us"],
                         time_max_us=bt[d]["fixed_us"], timing="board: mean with early stopping (time_us), fixed budget (time_max_us)"))
        for k in ("tn", "bposd", "pm", "uf"):
            rows.append(dict(d=d, decoder=k, ler=acc[(d, k)]["ler"], time_us=ct[(d, k)] * 1e6, time_max_us="",
                             timing="mean per shot, " + ct[(d, "cpu")]))
    C.write_csv(out / "fig5" / "fig5e.csv", rows)
    print("board fixed / early-stop mean (us): " + ", ".join(f"d{d} {bt[d]['fixed_us']:.2f}/{bt[d]['es_us']:.2f}"
                                                              for d in C.DS_ONE_SHOT))
    print(f"d = 9: TN {ct[(9, 'tn')]:.3f} s per shot = {ct[(9, 'tn')] * 1e6 / bt[9]['fixed_us']:.0f} x the fixed budget, "
          f"{ct[(9, 'tn')] * 1e6 / bt[9]['es_us']:.0f} x the early-stop mean")

    # ---- Fig. 5f: d = 9, decoders of one syndrome per shot, measured and published
    d = 9
    r = R[(d, P)]
    me = r["f_me"][-1]                                   # lowest-energy class at the largest recorded budget
    me_ci = ci(me.sum(), me.size)
    print(f"d = 9: LER of the lowest-energy correction {me_ci['ler']:.4f}, most visited class {acc[(9, 'hw_ml')]['ler']:.4f}")
    rows = [dict(name="This work (FPGA)", algorithm="hw_ml", kind="board", time_us=bt[d]["fixed_us"],
                 time_early_stop_us=bt[d]["es_us"], ler=acc[(d, "hw_ml")]["ler"], source="board")]
    rows += [dict(name=C.BASELINE_NAMES[k] + " (CPU)", algorithm=k, kind="cpu", time_us=ct[(d, k)] * 1e6,
                  time_early_stop_us="", ler=acc[(d, k)]["ler"], source="this work, " + ct[(d, "cpu")])
             for k in ("tn", "bposd", "pm", "uf")]
    pub = list(csv.DictReader(open(a.published, encoding="utf-8")))
    for q in pub:
        if int(q["d"]) != d:
            continue
        cls = q["algorithm_on_our_shots"]
        ler = me_ci["ler"] if cls == "lowest_energy" else acc[(d, "pm")]["ler"]
        rows.append(dict(name=q["label"], algorithm=cls, kind="published", time_us=float(q["time_per_shot_us"]),
                         time_early_stop_us="", ler=ler, source=q["source"]))
    C.write_csv(out / "fig5" / "fig5f.csv", rows)

    # ---- Fig. S7a: LER vs p, this work and TN
    rows = []
    tot = [0, 0]
    for (d, p), r in sorted(R.items()):
        if p >= 0.135 and "l12_ml" in r:
            hw, nsw = r["l12_ml"], 4096
        else:
            hw, nsw = hw_board(r, d), C.SWEEPS_BOARD[d]
        rows.append(dict(d=d, p=p, decoder="hw_ml", sweeps=nsw, **ci(hw.sum(), hw.size)))
        rows.append(dict(d=d, p=p, decoder="tn", sweeps="", **ci(r["f_tn"].sum(), r["f_tn"].size)))
        tot[0] += int(hw.sum())
        tot[1] += int(r["f_tn"].sum())
    C.write_csv(out / "figS7" / "figS7a.csv", rows)
    print(f"Fig. S7a: {len(rows) // 2} points; this work makes {100 * (tot[0] / tot[1] - 1):.2f}% more logical errors than TN")

    # ---- Fig. S7e: decode time per shot vs d
    rows = []
    for d in C.DS_ONE_SHOT:
        rows.append(dict(series="this_work_fixed_budget", d=d, time_us=bt[d]["fixed_us"], kind="measured", source="board"))
        rows.append(dict(series="this_work_early_stop_mean", d=d, time_us=bt[d]["es_us"], kind="measured", source="board"))
        for k in ("tn", "bposd", "pm", "uf"):
            rows.append(dict(series=k, d=d, time_us=ct[(d, k)] * 1e6, kind="measured", source="this work, " + ct[(d, "cpu")]))
    for q in pub:
        rows.append(dict(series=q["series"], d=int(q["d"]), time_us=float(q["time_per_shot_us"]), kind="published",
                         source=q["source"]))
    C.write_csv(out / "figS7" / "figS7e.csv", rows)

    # ---- Fig. S8b: post-selection on the board's confidence
    rows = []
    for d in C.DS_ONE_SHOT:
        z = np.load(Path(a.board) / (f"res_d{d}_p0.12_20k_n128.npz" if d == 5 else f"res_d{d}_p0.12_20k.npz"))
        c = z["r_counts"].astype(float)
        conf = c.max(1) / c.sum(1)
        fail = z["fail_ml"].astype(bool)
        o = np.argsort(conf, kind="stable")
        n = len(fail)
        for fr in np.round(np.arange(0, 0.501, 0.025), 3):
            keep = o[int(round(fr * n)):]
            k = int(fail[keep].sum())
            lo, hi = C.wilson(k, keep.size)
            rows.append(dict(d=d, discarded=fr, fails=k, kept=int(keep.size), ler=k / keep.size, ci_lo=float(lo), ci_hi=float(hi)))
    C.write_csv(out / "figS8" / "figS8b.csv", rows)
    for d in (3, 15):
        q = {r["discarded"]: r["ler"] for r in rows if r["d"] == d}
        print(f"post-selection d = {d}: 10% -> {q[0.0] / q[0.1]:.1f}x, 25% -> {q[0.0] / q[0.25]:.1f}x")


if __name__ == "__main__":
    main()
