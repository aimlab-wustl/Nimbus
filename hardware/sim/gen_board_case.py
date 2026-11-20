"""Board-level test vectors (hardware/sim/tb_pt_board.sv): the PYNQ driver (hardware/host/pt_board.py) runs a full session against the
register-level mock (hardware/host/pt_mock.py, golden model inside) through a RecordingMMIO; the recorded bus operations (writes,
compared reads, polls) are replayed by the testbench through AXI4-Lite on the RTL with two unrelated clocks.

Session: tables for --ptab (not the generation p), seed, configure, batch 1 (first half of the shots), new tables for --ptab2
(RECALC), batch 2. Every result record is compared word for word (except the timestamp word), including the cycle counts
predicted by the cycle model. The mock's results are also checked against a direct golden run (independent seeding path).

python hardware/sim/gen_board_case.py --d 3 --NS 4 --nsw 8 --shots 8 --out hardware/sim/work/brd_d3
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hardware" / "gen"))
sys.path.insert(0, str(ROOT / "hardware" / "host"))
from nimbus.decoding.one_shot import c0_from_syndrome, seed_states, run_shot  # noqa: E402
import gen_pt  # noqa: E402
import pt_board as D  # noqa: E402
from pt_mock import PTBoardMock, RecordingMMIO  # noqa: E402
from nimbus.decoding.code_capacity import sample_shots  # noqa: E402


def golden_direct(spec, tables_seq, syn, nsw, seed, gmin, th32):
    """Independent reference: generator states from one_shot.seed_states, shots back to back, tables per shot."""
    ga, gb, sw = seed_states(spec, 1, seed)
    ga, gb, sw = ga[0], gb[0], sw[0]
    heads = np.zeros((spec.cfg.NS, 2), np.int64)
    c0 = c0_from_syndrome(spec, syn)
    out = []
    for s in range(len(syn)):
        r = run_shot(tables_seq[s], c0[s], nsw, ga, gb, heads, sw, gmin, th32 if gmin else 0)
        kb = int(r["stop"]).bit_length() - 1
        s1 = int(r["stop"]) - 1                                    # count block of the last sweep (sweeps >= nsw / 2: last block)
        kc = max(1, s1 if s1 < (nsw >> 1) else nsw - 1).bit_length()
        out.append((int(np.argmax(r["counts"][kc])), int(r["bestC"][np.argmin(r["bestE"])]), int(r["stop"]), int(r["bestE"].min()),
                    tuple(int(x) for x in r["counts"][kc])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3); ap.add_argument("--p", type=float, default=0.12)
    ap.add_argument("--NS", type=int, default=4); ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=2); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--nsw", type=int, default=8); ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--gmin", type=int, default=0)
    ap.add_argument("--th32", type=int, default=29)
    ap.add_argument("--ptab", type=float, default=None); ap.add_argument("--ptab2", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.ptab = a.p + 0.03 if a.ptab is None else a.ptab
    a.ptab2 = a.p - 0.02 if a.ptab2 is None else a.ptab2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen_pt.main(["--d", str(a.d), "--p", str(a.p), "--NS", str(a.NS), "--bmin", str(a.bmin), "--K", str(a.K), "--B", str(a.B),
                 "--out", str(out)])
    mock = PTBoardMock(a.d, a.NS, a.K, a.B, p_gen=a.p, bmin=a.bmin)
    rec = RecordingMMIO(mock)
    brd = D.PTBoard(rec)
    spec = mock.spec
    e, syn = sample_shots(spec.prob, a.shots, a.seed)
    S1 = a.shots // 2
    brd.set_noise(a.ptab, bmin=a.bmin)
    sp1 = mock.tables_spec()
    brd.seed(a.seed)
    brd.configure(a.nsw, a.gmin, a.th32)
    r1 = brd.run_batch(syn[:S1])
    brd.set_noise(a.ptab2, bmin=a.bmin)
    sp2 = mock.tables_spec()
    r2 = brd.run_batch(syn[S1:])
    # independent golden check of the mock (seeding path, table switch)
    ref = golden_direct(spec, [sp1] * S1 + [sp2] * (a.shots - S1), syn, a.nsw, a.seed, a.gmin, a.th32)
    got = []
    for r in (r1, r2):
        for i in range(len(r["ml"])):
            got.append((int(r["ml"][i]), int(r["me"][i]), int(r["sweeps"][i]), int(r["emin"][i]),
                        tuple(int(x) for x in r["counts"][i])))
    assert got == ref, f"mock != golden\n{got}\n{ref}"
    with open(out / "ops.mem", "w") as f:
        for op, off, data, mask in rec.ops:
            f.write(f"{op:01x}{off:05x}{data & 0xFFFFFFFF:08x}{mask & 0xFFFFFFFF:08x}\n")
    with open(out / "tb_board_params.svh", "w") as f:
        f.write(f"`define T_NOPS {len(rec.ops)}\n")
    cyc = np.concatenate([r1["cyc_start"] + r1["cyc_run"] + 2, r2["cyc_start"] + r2["cyc_run"] + 2])
    info = dict(vars(a), nops=len(rec.ops), shots_exp=got, cycles=cyc.tolist())
    (out / "case.json").write_text(json.dumps(info, default=str, indent=1))
    print(json.dumps(dict(nops=len(rec.ops), shots=got, cycles=cyc.tolist())))


if __name__ == "__main__":
    main()
