"""Test vectors for the one-syndrome-per-shot RTL (hardware/sim/tb_pt.sv): generated RTL, generator seeds, syndromes of several consecutive shots
and the golden model's expectations (per move cycle: E, class and label of every replica; per shot: counts, ML and minimum-energy
decisions). The shots run back to back on the same generators, as on the board. The lane thresholds and swap constants are the
runtime tables (thr.mem, theta.mem): set 1 (noise rate --ptab, default p + 0.03, i.e. not the power-on tables of the generation p)
before the first shot, set 2 (--ptab2, default p - 0.02) from shot --switch on (written while idle, then RECALC).

python hardware/sim/gen_pt_case.py --d 3 --NS 4 --nsw 8 --shots 3 --out hardware/sim/work/pt_d3
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hardware" / "gen"))
from nimbus.decoding.one_shot import (PTConfig, build_spec, c0_from_syndrome, seed_states, run_shot, ladder, make_tables,  # noqa: E402
                                  with_tables)
import math  # noqa: E402
import gen_pt  # noqa: E402
from nimbus.decoding.code_capacity import sample_shots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3); ap.add_argument("--p", type=float, default=0.12)
    ap.add_argument("--NS", type=int, default=4); ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=1); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--nsw", type=int, default=8); ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--out", required=True)
    ap.add_argument("--gmin", type=int, default=0); ap.add_argument("--th32", type=int, default=29)
    ap.add_argument("--ptab", type=float, default=None); ap.add_argument("--ptab2", type=float, default=None)
    ap.add_argument("--switch", type=int, default=None)
    a = ap.parse_args()
    a.ptab = a.p + 0.03 if a.ptab is None else a.ptab
    a.ptab2 = a.p - 0.02 if a.ptab2 is None else a.ptab2
    a.switch = a.shots // 2 if a.switch is None else a.switch
    assert a.nsw >= 2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen_pt.main(["--d", str(a.d), "--p", str(a.p), "--NS", str(a.NS), "--bmin", str(a.bmin), "--K", str(a.K), "--B", str(a.B),
                 "--out", str(out)])
    spec = build_spec(a.d, a.p, PTConfig(NS=a.NS, bmin=a.bmin, K=a.K, B=a.B))
    NS, n = a.NS, spec.n
    sets = []
    for pt in (a.ptab, a.ptab2):
        thr, th = make_tables(math.log(3.0 * (1.0 - pt) / pt), ladder(NS, a.bmin), a.B, spec.kmax, spec.cfg.TF)
        sets.append((with_tables(spec, thr, th), thr, th))
    KM, TW = spec.kmax, gen_pt.TW_THETA
    with open(out / "thr.mem", "w") as f:
        for _, thr, _ in sets:
            for t_ in range(NS):
                v = 0
                for k in range(KM, 0, -1):
                    v = (v << a.B) | int(thr[t_, k])
                f.write(f"{v:0{(KM * a.B + 3) // 4}x}\n")
    with open(out / "theta.mem", "w") as f:
        for _, _, th in sets:
            for t_ in range(max(NS - 1, 1)):
                f.write(f"{int(th[t_]) if t_ < NS - 1 else 0:0{(TW + 3) // 4}x}\n")
    EW = gen_pt.clog2(n + 1)
    LW = gen_pt.clog2(NS)
    ga, gb, sw = seed_states(spec, 1, a.seed)
    ga, gb, sw = ga[0], gb[0], sw[0]
    # seed shift files: line t = bit t of every generator (replica i = bit i)
    def wr_bits(fn, arr):            # arr [G, L]
        with open(out / fn, "w") as f:
            for t in range(arr.shape[1]):
                f.write("".join(str(int(arr[i, t])) for i in range(arr.shape[0] - 1, -1, -1)) + "\n")
    N = spec.N
    def padded(g):                    # stream of ceil(p/N) * N bits: leading zeros, then the history oldest first
        L = -(-g.shape[1] // N) * N
        return np.concatenate([np.zeros((g.shape[0], L - g.shape[1]), np.uint8), g], 1)
    sa_s, sb_s = padded(ga), padded(gb)
    wr_bits("seed_a.mem", sa_s)
    wr_bits("seed_b.mem", sb_s)
    swb = np.zeros((max(NS - 1, 1), 128), np.uint8)
    for t in range(NS - 1):
        for k in range(128):
            swb[t, k] = (int(sw[t, k // 32]) >> (k % 32)) & 1
    wr_bits("seed_s.mem", swb)
    e, syn = sample_shots(spec.prob, a.shots, a.seed)
    c0 = c0_from_syndrome(spec, syn)
    heads = np.zeros((NS, 2), np.int64)
    rows, shots = [], []
    for s in range(a.shots):
        sp = sets[0][0] if s < a.switch else sets[1][0]
        r = run_shot(sp, c0[s], a.nsw, ga, gb, heads, sw, a.gmin, a.th32 if a.gmin else 0)
        kb = int(r["stop"]).bit_length() - 1
        rec = r["rec"]
        for cyc in range(rec.shape[0]):
            row = 0
            for i in range(NS - 1, -1, -1):
                row = (row << LW) | int(rec[cyc, i, 3])
            for i in range(NS - 1, -1, -1):
                row = (row << 2) | int(rec[cyc, i, 1])
            for i in range(NS - 1, -1, -1):
                row = (row << EW) | int(rec[cyc, i, 0])
            rows.append(row)
        cnt = r["counts"][max(1, int(r["stop"]) - 1).bit_length()]
        ml = int(np.argmax(cnt))
        shots.append(dict(syn=int("".join(str(int(b)) for b in syn[s][::-1]), 2), cnt=[int(x) for x in cnt], ml=ml,
                          me=int(r["bestC"][np.argmin(r["bestE"])]), emin=int(r["bestE"].min()), cycles=int(rec.shape[0]), stop=int(r["stop"])))
    wbits = NS * (EW + 2 + LW)
    with open(out / "exp_rec.mem", "w") as f:
        for row in rows:
            f.write(f"{row:0{(wbits + 3) // 4}x}\n")
    m = syn.shape[1]
    with open(out / "shots.mem", "w") as f:
        for sh in shots:
            f.write(f"{sh['syn']:0{(m + 3) // 4}x}\n")
    with open(out / "exp_shot.mem", "w") as f:
        for sh in shots:
            v = (sh["emin"] << 116) | (sh["stop"] << 84) | (sh["me"] << 82) | (sh["ml"] << 80) | (sh["cnt"][3] << 60) | (sh["cnt"][2] << 40) \
                | (sh["cnt"][1] << 20) | sh["cnt"][0]
            f.write(f"{v:032x}\n")
    with open(out / "tb_params.svh", "w") as f:
        f.write(f"`define T_NSHOT {a.shots}\n`define T_NSW {a.nsw}\n`define T_ROWS {len(rows)}\n`define T_WB {wbits}\n"
                f"`define T_PA {sa_s.shape[1]}\n`define T_PB {sb_s.shape[1]}\n`define T_N {N}\n"
                f"`define T_GMIN {a.gmin}\n`define T_TH {a.th32 if a.gmin else 0}\n`define T_SWITCH {a.switch}\n")
    (out / "case.json").write_text(json.dumps(dict(vars(a), rows=len(rows), shots_exp=shots), default=str, indent=1))
    print(json.dumps(dict(rows=len(rows), shots=[(s["ml"], s["me"], s["cnt"], s["stop"]) for s in shots])))


if __name__ == "__main__":
    main()
