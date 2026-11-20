"""Test vectors for the real-time RTL (hardware/sim/tb_ptw.sv): generated RTL, per-window generator seeds, window syndromes and the
golden model's expectations (nimbus.decoding.real_time.decode_stream with the hardware schedule of the layout): per sweep, the
energy, class and label of every replica; per window, the commit record (best energy, best class, summary, ML class,
counts of the last block, sigma and frame bits after the commit). Several shots, each streamed through all its windows
(N = 4 d rounds, (C, B) = (d, d): two open windows and the final one).

python hardware/sim/gen_ptw_case.py --d 3 --R 4 --k 1 --nsw 8 --shots 2 --out hardware/sim/work/ptw_d3
"""
import argparse, dataclasses, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hardware" / "gen"))
from nimbus.decoding import real_time as g  # noqa: E402
from nimbus.decoding.one_shot import GFSR_A, GFSR_B, _bits_from, splitmix64_seq  # noqa: E402
import gen_ptw  # noqa: E402
from nimbus.decoding.phenomenological import memory_model  # noqa: E402

M64 = (1 << 64) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3); ap.add_argument("--p", type=float, default=0.04)
    ap.add_argument("--R", type=int, default=4); ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lram", action="store_true"); ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--nsw", type=int, default=8); ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--out", required=True)
    ap.add_argument("--carry", type=int, default=1)       # 1: generators seeded once per shot (the RTL default)
    ap.add_argument("--rs", type=int, default=1); ap.add_argument("--srl", type=int, default=0)
    ap.add_argument("--cs", type=int, default=0); ap.add_argument("--swp", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gargs = ["--d", str(a.d), "--p", str(a.p), "--R", str(a.R), "--k", str(a.k), "--K", str(a.K), "--rs", str(a.rs),
             "--srl", str(a.srl), "--cs", str(a.cs), "--swp", str(a.swp), "--out", str(out)]
    G, info = gen_ptw.main(gargs + (["--lram"] if a.lram else []))
    cfg = dataclasses.replace(G["cfg"], carry=bool(a.carry))
    d, R, T, C = a.d, a.R, G["T"], G["C"]
    N = 4 * d
    Dx, Dz, ox, oz = memory_model(d, N).sample(a.p, a.shots, np.random.default_rng([a.seed, d, 77]))
    res, dbg = g.decode_stream(d, a.p, Dx, Dz, N, C, d, a.nsw, cfg, seed=a.seed, debug=True)
    wins = dbg["windows"]
    sched = dbg["sched"]
    nW = len(sched)
    assert all(T_ == T for _, T_, _ in sched), "the core has one window length (N a multiple of C)"
    wsf = G["ws"][1]
    m, n = wsf.m, wsf.n
    M2 = 2 * m
    NB, LW, EW = info["NB"], info["LW"], info["EW"]
    NP = max(R - 1, 1)
    KB = a.nsw.bit_length() - 1
    sx, sz = g.syndromes(Dx, N, m), g.syndromes(Dz, N, m)
    # ---- tables (identical for both window kinds)
    ROWN = info["ROWN"]
    with open(out / "thr.mem", "w") as f:
        for t in range(R):
            v = 0
            for r in range(ROWN, 0, -1):
                v = (v << cfg.B) | int(wsf.thr[t, r])
            f.write(f"{v:0{(ROWN * cfg.B + 3) // 4}x}\n")
    with open(out / "theta.mem", "w") as f:
        for t in range(NP):
            f.write(f"{int(wsf.theta[t]) if t < R - 1 else 0:05x}\n")
    # ---- per shot and window: seeds, syndromes, expectations
    bank_N = [bl * cfg.B for bl in info["bank_lanes"]]
    pads = {(gn, b): -(-P // bank_N[b]) * bank_N[b] for gn, P in (("a", GFSR_A[0]), ("b", GFSR_B[0])) for b in range(NB)}
    seedf = {(gn, b): open(out / f"seed_{gn}{b}.mem", "w") for gn in "ab" for b in range(NB)}
    fs = open(out / "seed_s.mem", "w")
    fsyn = open(out / "syn.mem", "w")
    frec = open(out / "exp_rec.mem", "w")
    fwin = open(out / "exp_win.mem", "w")
    rows = 0
    WRW = EW + 4 * 20 + (M2 + 2) + 2 + 2 + M2 + 2 + 2
    fr = np.zeros((a.shots, 2), np.int64)
    sig = np.zeros((a.shots, M2), np.int64)
    for s in range(a.shots):
        fr_ml = 0
        gst = g.gen_state(1, R, NB) if a.carry else None
        for w in range(nW):
            e = wins[w]
            t0, Tw, final = e["t0"], e["T"], e["final"]
            ws = e["ws"]
            base = int(g.window_bases(a.seed, [s], t0, Tw, final)[0])
            # lane generator seeds (replica i, bank b): padded history, oldest first, replica i = bit i of each line
            for gn, (P, _) in (("a", GFSR_A), ("b", GFSR_B)):
                for b in range(NB):
                    L = pads[(gn, b)]
                    arr = np.zeros((R, L), np.uint8)
                    for i in range(R):
                        sd = base ^ (((2 * (i * NB + b) + (1 if gn == "a" else 2)) * 0xD1B54A32D192ED03) & M64)
                        arr[i, L - P:] = _bits_from(sd, P)
                    for k in range(L):
                        seedf[(gn, b)].write("".join(str(int(arr[i, k])) for i in range(R - 1, -1, -1)) + "\n")
            swb = np.zeros((NP, 128), np.uint8)
            for t in range(R - 1):
                z = splitmix64_seq(base ^ (((t + 1) * 0xABCDEF0123456789) & M64), 2)
                words = [int(z[0]) & 0xFFFFFFFF, int(z[0]) >> 32, int(z[1]) & 0xFFFFFFFF, int(z[1]) >> 32]
                for k in range(128):
                    swb[t, k] = (words[k // 32] >> (k % 32)) & 1
            for k in range(128):
                fs.write("".join(str(int(swb[t, k])) for t in range(NP - 1, -1, -1)) + "\n")
            # syndromes: bit t M2 + sector m + c
            v = 0
            for t in range(T - 1, -1, -1):
                for sec, ss in ((1, sz), (0, sx)):
                    for c in range(m - 1, -1, -1):
                        v = (v << 1) | int(ss[s, t0 + t, c])
            vp = 0
            for sec, ss in ((1, sz), (0, sx)):
                for c in range(m - 1, -1, -1):
                    vp = (vp << 1) | int(ss[s, N, c])
            fsyn.write(f"{int(final)} {vp:x} {v:x}\n")
            # per-sweep record of every replica
            lit0 = e["lit0"][s:s + 1]
            o = g.run_window(ws, lit0, np.array([base], np.uint64), a.nsw, rec=True, state=gst)
            assert int(o["bestE"][0]) == int(e["out"]["bestE"][s])
            rec = o["rec"]
            for sw_ in range(a.nsw):
                row = 0
                for i in range(R - 1, -1, -1):
                    row = (row << LW) | int(rec[sw_, i, 1])
                for i in range(R - 1, -1, -1):
                    row = (row << 2) | int(rec[sw_, i, 2])
                for i in range(R - 1, -1, -1):
                    row = (row << EW) | int(rec[sw_, i, 0])
                frec.write(f"{row:0{(R * (EW + 2 + LW) + 3) // 4}x}\n")
                rows += 1
            # window record
            bl = e["out"]["best_lit"][s:s + 1]
            mt = g.measurement_literals(ws, bl, 0)[0, C - 1].tolist() + g.measurement_literals(ws, bl, 1)[0, C - 1].tolist()
            nsl = ws.ns if final else C
            par = [int(g.frame_parity(ws, bl, 0, nsl)[0]), int(g.frame_parity(ws, bl, 1, nsl)[0])]
            best_s = par[1] << (M2 + 1) | par[0] << M2 | sum(int(b) << i for i, b in enumerate(mt))
            cnt = [int(x) for x in e["out"]["counts"][s, KB]]
            mlc = int(e["out"]["ml"][s])
            bc = int(e["out"]["bestC"][s])
            fr[s] ^= par
            if not final:
                sg = np.concatenate([e["sigma"][0][s], e["sigma"][1][s]]).astype(np.int64)
                assert (sg == (np.array([int(x) for x in np.concatenate([sx[s, t0 + C - 1], sz[s, t0 + C - 1]])]) ^ mt)).all()
                sig[s] = sg
            else:
                sh = bc ^ mlc
                fr_ml = (int(fr[s, 0]) ^ (sh & 1)) | ((int(fr[s, 1]) ^ (sh >> 1)) << 1)
                assert (int(fr[s, 0]), int(fr[s, 1])) == (int(res["me"][0][s]), int(res["me"][1][s]))
                assert fr_ml == int(res["ml"][0][s]) | (int(res["ml"][1][s]) << 1)
            sgv = sum(int(b) << i for i, b in enumerate(sig[s]))
            frv = int(fr[s, 0]) | (int(fr[s, 1]) << 1)
            # {final, fr_ml, frame, sigma, dec_ml, best_c, best_s, cnt3, cnt2, cnt1, cnt0, best_e}
            v = int(final)
            for x, wd in ((fr_ml, 2), (frv, 2), (sgv, M2), (mlc, 2), (bc, 2), (best_s, M2 + 2), (cnt[3], 20), (cnt[2], 20),
                          (cnt[1], 20), (cnt[0], 20), (int(e["out"]["bestE"][s]), EW)):
                v = (v << wd) | x
            fwin.write(f"{v:0{(WRW + 1 + 3) // 4}x}\n")
    for f_ in list(seedf.values()) + [fs, fsyn, frec, fwin]:
        f_.close()
    with open(out / "tb_params.svh", "w") as f:
        f.write(f"`define T_NSHOT {a.shots}\n`define T_NWIN {nW}\n`define T_NSW {a.nsw}\n`define T_ROWS {rows}\n"
                f"`define T_WB {R * (EW + 2 + LW)}\n`define T_WRW {WRW + 1}\n`define T_WQ {wsf.Wq}\n`define T_WM {wsf.wq}\n"
                f"`define T_MODE {wsf.mode}\n`define T_CARRY {a.carry}\n")
        for b in range(NB):
            f.write(f"`define T_PA{b} {pads[('a', b)]}\n`define T_PB{b} {pads[('b', b)]}\n`define T_N{b} {bank_N[b]}\n")
    (out / "case.json").write_text(json.dumps(dict(vars(a), rows=rows, windows=nW, NB=NB, info=info), default=str, indent=1))
    print(json.dumps(dict(rows=rows, windows=nW, NB=NB, ler_me=[int(x) for x in g.failures(res, ox, oz)["me"]])))


if __name__ == "__main__":
    main()
