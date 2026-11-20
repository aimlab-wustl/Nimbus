"""Real time: source data of Fig. 5g-i and Fig. S7b-d.

  python analysis/real_time.py --decoder RUNS/real_time --baselines RUNS/real_time_baselines \
      --union-find RUNS/real_time_union_find --timing RUNS/cpu_timing --board BOARD/real_time --out ../../source_data

Inputs
  --decoder     software model runs (``ptw_*.json``; one or more folders)
  --baselines   window baselines (``win_*_wbase_*.json``; one or more folders). BP-OSD with min-sum scaling 0.625
                (``bp625``) replaces the library default as the paper's BP-OSD ("bp")
  --union-find  ``b_uf_d9_p0.03.json`` (union-find on the whole memory, 3,000 shots)
  --timing      ``wbase_time_d<d>_p0.03.json`` (``cpu_timing.py real-time``; one or more folders, the first has
                priority for a decoder timed in more than one)
  --board       board results per d: ``d<d>/check_thr.json`` (1,000 shots at p = 0.03, decided as fast as possible:
                LER and time per round)
  published times: ``published/real_time_times.csv``, ``published/round_times.csv``

Outputs (``--out``): fig5/fig5g.csv, fig5h.csv, fig5i.csv, fig5hi_rounds.csv, figS7/figS7b.csv, figS7c.csv, figS7d.csv.

This work = the software model of the FPGA design for d (replicas 16 at d = 5, 7 and 24 at d = 9, 11; 2^12 sweeps per
window), on the same shots as the baselines (chunks of 250 shots present for both). Best CPU baseline: the lowest number
of logical errors among correlated PyMatching, BP+matching and BP-OSD at each (d, p). Error bars: 95% Wilson intervals
(z = 1.96). CPU time per round: mean time of a non-final window / d (one core).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

BASE = ("pm", "pmc", "bpm", "bp")


def ci(k, n):
    lo, hi = C.wilson(k, n)
    return dict(fails=int(k), shots=int(n), ler=k / n, ci_lo=float(lo), ci_hi=float(hi))


def decoder_records(dirs):
    """[(d, p, chunk, R, K, fails, shots)] of the software-model runs (memory of 4d rounds, beta_min 0.1)."""
    out = {}
    for dp in dirs:
        for f in Path(dp).glob("ptw_*.json"):
            j = json.loads(f.read_text())
            m = re.fullmatch(r"g_hwc_r(\d+)(?:_k(\d+))?", j.get("var", ""))
            if not m or j["N"] != 4 * j["d"] or j.get("bmin", 0.1) != 0.1:
                continue
            key = (j["d"], round(j["p"], 4), j["chunk"], int(m.group(1)), j["K"])
            out[key] = (round(j["ler"][j["var"] + ":ml"] * j["shots"]), j["shots"])
    return out


def baseline_records(dirs):
    base = {}
    bp625 = {}
    for dp in dirs:
        for f in Path(dp).glob("win_*.json"):
            j = json.loads(f.read_text())
            key = (j["d"], round(j["p"], 4), j["chunk"])
            if "bp625" in j["ler"]:
                bp625[key] = round(j["ler"]["bp625"] * j["shots"])
            if all(q in j["ler"] for q in BASE):
                base[key] = ({q: round(j["ler"][q] * j["shots"]) for q in BASE}, j["shots"])
    for k, v in bp625.items():                      # the paper's BP-OSD: min-sum scaling 0.625
        if k in base:
            base[k][0]["bp"] = v
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decoder", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--union-find", required=True)
    ap.add_argument("--timing", nargs="+", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--published", default=str(C.HERE / "published"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    hw = decoder_records(a.decoder)
    base = baseline_records(a.baselines)

    # ---- LER per (d, p): this work, every baseline, best baseline (same chunks)
    g = defaultdict(list)
    for (d, p, ch, R, K) in hw:
        if R == C.R_REAL_TIME.get(d) and K == C.K_REAL_TIME and (d, p, ch) in base:
            g[(d, p)].append(((d, p, ch, R, K), (d, p, ch)))
    T = {}
    rows = []
    for (d, p), ks in sorted(g.items()):
        n = sum(hw[k][1] for k, _ in ks)
        fails = {"hw": sum(hw[k][0] for k, _ in ks)}
        for q in BASE:
            fails[q] = sum(base[b][0][q] for _, b in ks)
        best = min(C.BEST_OF, key=fails.get)
        fails["best"] = fails[best]
        for q, f in fails.items():
            T[(d, p, q)] = ci(f, n)
            rows.append(dict(d=d, p=p, decoder=q, **T[(d, p, q)], chunks=len(ks), best_cpu_decoder=best if q == "best" else ""))
    C.write_csv(out / "figS7" / "figS7b.csv", rows)
    rows = [dict(d=d, p=p, best_cpu_decoder=r["best_cpu_decoder"], ratio=r["ler"] / T[(d, p, "hw")]["ler"])
            for r in rows if r["decoder"] == "best" for d, p in [(r["d"], r["p"])]]
    C.write_csv(out / "figS7" / "figS7c.csv", rows)
    for d in C.DS_REAL_TIME:
        rr = [r["ratio"] for r in rows if r["d"] == d and 0.025 - 1e-9 <= r["p"] <= 0.04 + 1e-9]
        print(f"d = {d}: best CPU baseline / this work, geometric mean over p = 0.025-0.04 ({len(rr)} points): "
              f"{np.exp(np.mean(np.log(rr))):.3f}")

    # ---- board results
    board = {}
    for d in (3, 5, 7, 9, 11):
        f = Path(a.board) / f"d{d}" / "check_thr.json"
        if f.exists():
            j = json.loads(f.read_text())
            board[d] = dict(fails=j["ler_all"]["ml"]["fails"], shots=j["ler_all"]["ml"]["shots"],
                            us_per_round=j["timing"]["round_us"]["mean"])

    # ---- Fig. 5g: LER vs d at p = 0.03
    P = C.P_REAL_TIME
    rows = []
    for d in C.DS_REAL_TIME:
        for q in ("hw", "best"):
            r = T[(d, P, q)]
            rows.append(dict(series="this_work_model" if q == "hw" else "best_cpu", d=d, p=P, **r,
                             best_cpu_decoder=[x for x in C.BEST_OF if T[(d, P, x)]["fails"] == r["fails"]][0] if q == "best" else ""))
        if d in board:
            rows.append(dict(series="this_work_board", d=d, p=P, **ci(board[d]["fails"], board[d]["shots"]), best_cpu_decoder=""))
    C.write_csv(out / "fig5" / "fig5g.csv", rows)

    # ---- Fig. 5h: LER vs decode time per round
    ct = {}
    cpu = ""
    for dp in a.timing:
        for f in sorted(Path(dp).glob("wbase_time_d*_p0.03.json")):
            j = json.loads(f.read_text())
            c = ct.setdefault(j["d"], {})
            for q, v in j["per_commit_round"].items():     # the first folder given has priority
                c.setdefault(q, v)
            cpu = j["cpu"]
    rows = []
    for d in C.DS_REAL_TIME:
        rows.append(dict(d=d, decoder="hw", ler=T[(d, P, "hw")]["ler"], time_per_round_us=board[d]["us_per_round"],
                         timing="board, window decode time / d"))
        c = dict(ct[d])
        if "bp625" in c:
            c["bp"] = c["bp625"]
        for q in BASE:
            rows.append(dict(d=d, decoder=q, ler=T[(d, P, q)]["ler"], time_per_round_us=c[q] * 1e6,
                             timing="mean non-final window time / d, " + cpu))
    C.write_csv(out / "fig5" / "fig5h.csv", rows)
    print("d = 11 CPU time per round (us): " + ", ".join(f"{q} {r['time_per_round_us']:.1f}" for r in rows
                                                          if r["d"] == 11 for q in [r["decoder"]]))
    rd = list(csv.DictReader(open(Path(a.published) / "round_times.csv", encoding="utf-8")))
    C.write_csv(out / "fig5" / "fig5hi_rounds.csv", rd)

    # ---- Fig. 5i: real-time decoders at d = 9, p = 0.03
    uf = json.loads((Path(a.union_find) / "b_uf_d9_p0.03.json").read_text())
    ler = {"hw": T[(9, P, "hw")]["ler"], "uf": uf["uf_fails"] / uf["shots"], "pm": T[(9, P, "pm")]["ler"],
           "bp": T[(9, P, "bp")]["ler"]}
    rows = [dict(name="This work (FPGA)", algorithm="hw", kind="board", time_per_round_us=board[9]["us_per_round"],
                 ler=ler["hw"], source="board")]
    for q in csv.DictReader(open(Path(a.published) / "real_time_times.csv", encoding="utf-8")):
        rows.append(dict(name=q["label"], algorithm=q["algorithm_on_our_shots"], kind="published",
                         time_per_round_us=float(q["time_per_round_us"]), ler=ler[q["algorithm_on_our_shots"]],
                         source=q["source"]))
    C.write_csv(out / "fig5" / "fig5i.csv", rows)
    print(f"d = 9: union-find LER {ler['uf']:.3f} ({ler['uf'] / ler['hw']:.1f}x), PyMatching {ler['pm'] / ler['hw']:.1f}x, "
          f"BP-OSD {ler['bp'] / ler['hw']:.2f}x the logical errors of this work")

    # ---- Fig. S7d: sweeps per window, d = 9, p = 0.03
    sw = defaultdict(lambda: [0, 0])
    for (d, p, ch, R, K), (f, n) in hw.items():
        if d == 9 and abs(p - P) < 1e-9:
            sw[(R, K)][0] += f
            sw[(R, K)][1] += n
    rows = [dict(series=f"this_work_{R}_replicas", replicas=R, sweeps=1 << K, **ci(*sw[(R, K)])) for R, K in sorted(sw)]
    rows.append(dict(series="best_cpu", replicas="", sweeps="", **T[(9, P, "best")]))
    C.write_csv(out / "figS7" / "figS7d.csv", rows)


if __name__ == "__main__":
    main()
