"""Board-level test vectors for the real-time design (hardware/sim/tb_ptw_board.sv): the PYNQ driver (hardware/host/ptw_board.py) runs a session
against the register-level mock (hardware/host/ptw_mock.py, golden model inside) through a RecordingMMIO; the recorded bus operations
(writes, compared reads, polls) are replayed by the testbench through AXI4-Lite on the RTL with two unrelated clocks.

Session: tables for --ptab (rank mode of p < 1/28), seed, batch 1 with the round timer at a generous T_ROUND (timing words
compared: trigger-to-take, lag, misses = late = 0); tables for --ptab2 (other rank mode, RECALC), batch 2 at T_ROUND = 0
(throughput; the generator stream runs on); batch 3 reseeded at a tight T_ROUND (the decoder falls behind: decisions compared,
timing words and the miss / late counters printed, not compared). Every result word is compared except the timestamps. The
mock's decisions are also checked against a direct golden run (seeded through real_time, not through the written bits).

python hardware/sim/gen_ptw_board_case.py --d 3 --R 4 --k 1 --nsw 8 --shots 6 --out hardware/sim/work/pbrd_d3
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hardware" / "gen"))
sys.path.insert(0, str(ROOT / "hardware" / "host"))
from nimbus.decoding import real_time as g  # noqa: E402
import gen_ptw  # noqa: E402
import ptw_board as D  # noqa: E402
from ptw_mock import PTWBoardMock, RecordingMMIO  # noqa: E402
from nimbus.decoding.phenomenological import memory_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3); ap.add_argument("--p", type=float, default=0.04)
    ap.add_argument("--R", type=int, default=4); ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lram", type=int, default=0); ap.add_argument("--rs", type=int, default=1)
    ap.add_argument("--cs", type=int, default=0); ap.add_argument("--srl", type=int, default=0)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--nsw", type=int, default=8); ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ptab", type=float, default=0.03); ap.add_argument("--ptab2", type=float, default=0.05)
    ap.add_argument("--tight", type=int, default=1, help="third batch at a tight T_ROUND")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    G, info = gen_ptw.main(["--d", str(a.d), "--p", str(a.p), "--R", str(a.R), "--k", str(a.k), "--K", str(a.K),
                            "--rs", str(a.rs), "--cs", str(a.cs), "--srl", str(a.srl), "--out", str(out)]
                           + (["--lram"] if a.lram else []))
    info = json.loads((out / "ptw_info.json").read_text())
    mock = PTWBoardMock(info, p_gen=a.p)
    mock.allow_tight = True
    rec = RecordingMMIO(mock)
    brd = D.PTWBoard(rec)
    d, N, m = a.d, 4 * a.d, mock.m
    Dx, Dz, ox, oz = memory_model(d, N).sample(a.p, a.shots, np.random.default_rng([a.seed, d, 99]))
    sx, sz = g.syndromes(Dx, N, m), g.syndromes(Dz, N, m)
    syn = np.concatenate([sx, sz], 2).astype(np.uint8)
    S1 = (a.shots + 1) // 2
    cyc = mock.cycles(a.nsw)
    T1 = (cyc + 2 * mock.TR + 60) // d + 1                   # generous: the decoder keeps up
    T3 = max(mock.WPR + 7, cyc // (3 * d))                   # tight: about 3x too fast
    # batch 1: timed, generous
    brd.set_noise(a.ptab)
    tab1 = mock.tables()
    brd.configure(a.nsw, T1)
    r1 = brd.run_batch(syn[:S1], seed=a.seed)
    # batch 2: other tables (RECALC), throughput mode, stream continues
    brd.set_noise(a.ptab2)
    tab2 = mock.tables()
    brd.configure(a.nsw, 0)
    r2 = brd.run_batch(syn[S1:])
    parts = [r1, r2]
    if a.tight:
        brd.configure(a.nsw, T3)
        r3 = brd.run_batch(syn[:2], seed=a.seed + 1)
        parts.append(r3)
    # independent golden check of the mock (seeding path, table switch, stream across batches)
    cfg = mock.cfg
    ref, st = g.decode_batch(d, a.p, sx[:S1], sz[:S1], a.nsw, cfg, seed=a.seed, tables=tab1)
    ref2, st = g.decode_batch(d, a.p, sx[S1:], sz[S1:], a.nsw, cfg, state=st, tables=tab2)
    ref += ref2
    if a.tight:
        ref3, _ = g.decode_batch(d, a.p, sx[:2], sz[:2], a.nsw, cfg, seed=a.seed + 1, tables=tab2)
        ref += ref3
    got = []
    for r in parts:
        for i in range(len(r["shot"])):
            got.append((int(r["best_e"][i]), int(r["best_c"][i]), int(r["dec_ml"][i]), int(r["frame"][i]), int(r["fr_ml"][i]),
                        tuple(int(x) for x in r["counts"][i]), tuple(int(x) for x in r["sigma"][i])))
    exp = [(e["best_e"], e["best_c"], e["dec_ml"], e["frame"], e["fr_ml"], tuple(e["counts"]),
            tuple(int(x) for x in D.sigma_words(e["sigma"]))) for e in ref]
    assert got == exp, f"mock != golden\n{got}\n{exp}"
    with open(out / "ops.mem", "w") as f:
        for op, off, data, mask in rec.ops:
            f.write(f"{op:01x}{off:05x}{data & 0xFFFFFFFF:08x}{mask & 0xFFFFFFFF:08x}\n")
    with open(out / "tb_board_params.svh", "w") as f:
        f.write(f"`define T_NOPS {len(rec.ops)}\n")
    case = dict(vars(a), nops=len(rec.ops), windows=len(got), cyc_win=cyc, T1=T1, T3=T3)
    (out / "case.json").write_text(json.dumps(case, default=str, indent=1))
    print(json.dumps(dict(nops=len(rec.ops), windows=len(got), cyc_win=cyc, T1=T1, T3=T3)))


if __name__ == "__main__":
    main()
