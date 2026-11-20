"""Core-rail (VCCINT, PMBus VCCp0V85) power of a one-syndrome-per-shot bitstream while decoding and at idle.

A sampler process (its own interpreter: the MMIO loop holds the GIL) reads the rail every 0.1 s for 16 s; the main
process repeats batches from 8 s to 12 s. Idle = median after 13 s; decoding = median of the readings > idle + 0.3 W (the
sensor steps are 0.2-0.4 W). The host writes syndromes between batches, so at d <= 5 (1-2 us shots) the decoder is
busy for a small part of each reading and the decoding value is a lower bound.

usage (board, as root with the PYNQ venv): python3 pt_power.py BIT SHOTS.npz P NSW [PL0_MHZ]
"""
import sys, time, subprocess, numpy as np
sys.path.insert(0, ".")
if sys.argv[1] == "sample":
    import pynq
    rails = pynq.get_rails()
    r = rails["VCCp0V85"]
    t0 = time.time()
    while time.time() - t0 < float(sys.argv[2]):
        print("%.2f %.3f %.4f %.3f" % (time.time() - t0, r.power.value, r.voltage.value, r.current.value), flush=True)
        time.sleep(0.1)
    sys.exit()
import pt_board
bit, shots, p, nsw = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
brd = pt_board.PTBoard.from_overlay(bit)
if len(sys.argv) > 5:
    pt_board.PTBoard.set_fclk0(float(sys.argv[5]))
time.sleep(2)
prev = brd.measure_clock(0.5)
for _ in range(20):
    mhz = brd.measure_clock(0.5)
    if abs(mhz - prev) <= 1e-3 * mhz:
        break
    prev = mhz
fmax = float(brd.info.get("dec_fmax_mhz", 0) or 0)
assert not fmax or mhz <= fmax, (mhz, fmax)
print("clock %.1f MHz" % mhz)
syn = np.load(shots)["syn"]
brd.set_noise(p); brd.seed(1); brd.configure(nsw)
brd.run(syn[:brd.max_shots])
s = subprocess.Popen([sys.executable, __file__, "sample", "16"], stdout=subprocess.PIPE, text=True)
time.sleep(8); t0 = time.time(); n = 0
while time.time() - t0 < 4:
    brd.run_batch(syn[:brd.max_shots]); n += 1
x = np.array([[float(v) for v in l.split()] for l in s.communicate()[0].splitlines() if l.strip()])
idle = np.median(x[x[:, 0] > 13, 1]); hi = x[x[:, 1] > idle + 0.3]
print(bit, "batches", n, "idle %.2f W" % idle, "decoding %.2f W (median of %d readings), %.4f V %.2f A; all %s" %
      (np.median(hi[:, 1]) if len(hi) else float("nan"), len(hi), np.median(hi[:, 2]) if len(hi) else 0, np.median(hi[:, 3]) if len(hi) else 0, sorted(set(np.round(x[:, 1], 1)))))
