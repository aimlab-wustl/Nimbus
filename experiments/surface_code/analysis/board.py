"""Board designs and measurements: source data of Table S5, Fig. S8a and Fig. S8c.

  python analysis/board.py --one-shot BOARD/one_shot --real-time BOARD/real_time --out ../../source_data

Inputs
  ``board/designs.csv``  design parameters, LUTs (Vivado implementation reports) and core-rail power (board power logs),
                         typed from those files, with their source
  --one-shot   board results at p = 0.12 (``res_d<d>_p0.12_20k.npz``; d = 5: ``res_d5_p0.12_20k_n128.npz``): clock and
               on-chip time per shot at the fixed budget ((start + run + 2) cycles / clock)
  --real-time  board results at p = 0.03 per d: ``d<d>/check_thr.json`` (rounds released as fast as possible: time per
               round = window decode time / d) and ``d<d>/d<d>_rt.npz`` (rounds released by the on-chip timer: clock,
               window start (trigger) and commit times)

Outputs (``--out``): tables/tableS5.csv, figS8/figS8a.csv, figS8/figS8c.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

DEVICE_LUTS = 425_280                               # XCZU48DR


def one_shot_file(dirp, d):
    return Path(dirp) / (f"res_d{d}_p0.12_20k_n128.npz" if d == 5 else f"res_d{d}_p0.12_20k.npz")


def windows(z, shot):
    """(trigger, commit) cycle counts of the windows of one shot, in window order."""
    m = z["r_shot"] == shot
    o = np.argsort(z["r_win"][m], kind="stable")
    trig = z["r_t_trig"][m][o].astype(np.int64)
    done = trig + (z["r_d_take"][m][o] + z["r_d_done"][m][o]).astype(np.int64)
    return trig, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--designs", default=str(C.HERE / "board" / "designs.csv"))
    ap.add_argument("--one-shot", required=True)
    ap.add_argument("--real-time", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows, energy = [], []
    for r in C.read_csv(a.designs):
        d = int(r["d"])
        if r["mode"] == "one_shot":
            z = np.load(one_shot_file(a.one_shot, d))
            clk = float(z["clock_mhz"])
            t = float(((z["r_cyc_start"] + z["r_cyc_run"] + 2) / clk).mean())
            tsrc = "board, time per shot at the fixed budget, 20,000 shots, p = 0.12"
        else:
            z = np.load(Path(a.real_time) / f"d{d}" / f"d{d}_rt.npz")
            clk = float(z["clock_mhz"])
            j = json.loads((Path(a.real_time) / f"d{d}" / "check_thr.json").read_text())
            t = j["timing"]["round_us"]["mean"]
            tsrc = "board, window decode time / d, 2^12 sweeps per window, 1,000 shots, p = 0.03"
        pdec, pidle = float(r["power_decoding_w"]), float(r["power_idle_w"])
        lb = r["power_is_lower_bound"] == "yes"
        rows.append(dict(mode=r["mode"], d=d, design=r["design"], replicas=int(r["replicas"]), layout=r["layout"],
                         luts=int(r["luts"]), lut_percent=100 * int(r["luts"]) / DEVICE_LUTS, clock_mhz=clk,
                         time_us=t, time_unit="per shot" if r["mode"] == "one_shot" else "per round",
                         power_decoding_w=pdec, power_idle_w=pidle, power_is_lower_bound=r["power_is_lower_bound"],
                         source_time=tsrc, source_luts=r["source_luts"], source_power=r["source_power"]))
        energy.append(dict(mode=r["mode"], d=d, time_us=t, power_decoding_w=pdec, energy_j=pdec * t * 1e-6,
                           lower_bound="yes" if lb else "no"))
    C.write_csv(out / "tables" / "tableS5.csv", rows)
    C.write_csv(out / "figS8" / "figS8a.csv", energy)
    for e in energy:
        if (e["mode"], e["d"]) in (("one_shot", 3), ("one_shot", 15), ("real_time", 3), ("real_time", 11)):
            print(f"energy {e['mode']} d = {e['d']}: {e['energy_j'] * 1e3:.4f} mJ")

    # ---- Fig. S8c: timeline of shot 0 at d = 9, rounds every 1 ms
    z = np.load(Path(a.real_time) / "d9" / "d9_rt.npz")
    cpm = float(z["clock_mhz"]) * 1e3                   # cycles per ms
    tr = int(z["m_tround"]) / cpm
    trig, done = windows(z, 0)
    rows = []
    for w in range(len(trig)):
        nxt = trig[w + 1] / cpm if w + 1 < len(trig) else None
        rows.append(dict(window=w + 1, final="yes" if w == len(trig) - 1 else "no", start_ms=trig[w] / cpm,
                         commit_ms=done[w] / cpm, deadline_ms=nxt, slack_ms=(nxt - done[w] / cpm) if nxt is not None else None,
                         round_period_ms=tr))
    C.write_csv(out / "figS8" / "figS8c.csv", rows)
    print("d = 9, shot 0: window times (ms) " + ", ".join(f"{r['commit_ms'] - r['start_ms']:.2f}" for r in rows))
    # smallest slack over all windows, shots and distances (timer runs)
    for d in (3, 5, 7, 9, 11):
        f = Path(a.real_time) / f"d{d}" / f"d{d}_rt.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cpm = float(z["clock_mhz"]) * 1e3
        s = min(min((t[1:] - dn[:-1]) / cpm) for sh in np.unique(z["r_shot"]) for t, dn in [windows(z, sh)])
        print(f"d = {d}: round period {int(z['m_tround']) / cpm:.2f} ms, smallest slack {s:.2f} ms")


if __name__ == "__main__":
    main()
