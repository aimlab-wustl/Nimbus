"""Early-stop grid on the board (one syndrome per shot): for one bitstream, rerun a shot file with each (gmin, th32) and compare with the
fixed-budget result of the same shots and seeds. Writes one JSON line per setting (mean / p99 latency, LER of ML, paired
change against the fixed run); the full result of the setting given by --keep is saved for the bit-exact check.

sudo python3 pt_es_grid.py --bit B.bit --shots S.npz --fixed FIXED.npz --p 0.12 --nsw 430 [--pl0 93.75] --out grid.jsonl
"""
import argparse, json, os
import numpy as np
import pt_board as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", required=True); ap.add_argument("--shots", required=True); ap.add_argument("--fixed", required=True)
    ap.add_argument("--p", type=float, required=True); ap.add_argument("--nsw", type=int, required=True)
    ap.add_argument("--pl0", type=float, default=0); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gmin", type=int, nargs="+", default=[16, 32, 64, 128]); ap.add_argument("--th32", type=int, nargs="+", default=[29, 30, 31])
    ap.add_argument("--keep", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.shots)
    syn, fc = z["syn"], z["fail_cls"]
    fx = np.load(a.fixed)
    f_fix = fx["fail_ml"].astype(bool)
    brd = D.PTBoard.from_overlay(a.bit)
    if a.pl0:
        brd.fclk0_mhz = D.PTBoard.set_fclk0(a.pl0)
    mhz = brd.measure_clock(0.5)
    for _ in range(20):
        m2 = brd.measure_clock(0.5)
        if abs(m2 - mhz) <= 1e-3 * m2:
            break
        mhz = m2
    brd.set_noise(a.p, bmin=0.1)
    ar = np.arange(len(syn))
    for g in a.gmin:
        if g >= a.nsw:
            continue
        for th in a.th32:
            brd.seed(a.seed)
            brd.configure(a.nsw, g, th)
            res = brd.run(syn)
            f = fc[ar, res["ml"]].astype(bool)
            lat = (res["cyc_start"] + res["cyc_run"] + 2) / mhz
            dl = f.astype(float) - f_fix
            rec = dict(nsw=a.nsw, gmin=g, th32=th, clock=mhz, shots=len(syn), ler=float(f.mean()), ler_fixed=float(f_fix.mean()),
                       d_ler=float(dl.mean()), se=float(dl.std(ddof=1) / np.sqrt(len(dl))), lat_mean=float(lat.mean()),
                       lat_p50=float(np.median(lat)), lat_p99=float(np.quantile(lat, 0.99)), lat_max=float(lat.max()),
                       stopped=float(np.mean(res["stopped"])))
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
            if a.keep == f"{g},{th}":
                D.save_results(os.path.splitext(a.out)[0] + f"_g{g}_t{th}.npz", res, brd.meta(), dict(clock_mhz=mhz, syn=syn,
                               fail_ml=f), csv=False)


if __name__ == "__main__":
    main()
