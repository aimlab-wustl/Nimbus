"""Core-rail (VCCINT, PMBus VCCp0V85) power of a real-time bitstream while decoding and at idle (as pt_power.py).

A sampler process (its own interpreter: the MMIO loop holds the GIL) reads the rail every 0.1 s for 16 s; the main process
repeats throughput batches (T_ROUND = 0, one generator stream, seeded once) from 8 s to 12 s. Idle = median after 13 s;
decoding = median of the readings > idle + 0.3 W (the sensor steps are 0.2-0.4 W). The decoder is busy for all but the
host's write / read time of each batch (milliseconds per batch of windows of 0.1-5 ms each), so the plateau is the
decoding power. A second plateau with the real-time round timer (--tround-us) gives the average power at the platform's
round time (mostly idle between windows).

usage (board, as root with the PYNQ venv): python3 ptw_power.py BIT SHOTS.npz P NSW [PL0_MHZ] [--tround-us US] [--batch S]
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
import ptw_board
args = [a for a in sys.argv[1:]]
opt = {}
for key in ("--tround-us", "--batch"):
    if key in args:
        i = args.index(key); opt[key] = float(args[i + 1]); del args[i:i + 2]
bit, shots, p, nsw = args[0], args[1], float(args[2]), int(args[3])
brd = ptw_board.PTWBoard.from_overlay(bit)
if len(args) > 4:
    ptw_board.PTWBoard.set_fclk0(float(args[4]))
time.sleep(2)
mhz = brd.settle_clock()
fmax = float(brd.info.get("dec_fmax_mhz", 0) or 0)
assert not fmax or mhz <= fmax, (mhz, fmax)
print("clock %.1f MHz" % mhz)
syn = np.load(shots)["syn"]
nb = int(opt.get("--batch", 0)) or len(syn)
syn = syn[:min(nb, brd.max_shots)]
tround = int(round(opt.get("--tround-us", 0) * mhz))
brd.set_noise(p); brd.seed(1); brd.configure(nsw, tround)
r0 = brd.run_batch(syn)
per_batch = r0["batch_cycles"] / mhz * 1e-6
print("batch of %d shots: %.3f s on chip" % (len(syn), per_batch))
s = subprocess.Popen([sys.executable, __file__, "sample", "16"], stdout=subprocess.PIPE, text=True)
time.sleep(8); t0 = time.time(); n = 0; busy = 0
while time.time() - t0 < 4 or n == 0:
    busy += brd.run_batch(syn)["batch_cycles"]; n += 1
wall = time.time() - t0
x = np.array([[float(v) for v in l.split()] for l in s.communicate()[0].splitlines() if l.strip()])
idle = np.median(x[x[:, 0] > 13, 1]); hi = x[x[:, 1] > idle + 0.3]
print(bit, "T_round %d cycles" % tround, "batches", n, "duty %.3f" % (busy / mhz * 1e-6 / wall), "idle %.2f W" % idle,
      "decoding %.2f W (median of %d readings), %.4f V %.2f A; all %s" %
      (np.median(hi[:, 1]) if len(hi) else float("nan"), len(hi), np.median(hi[:, 2]) if len(hi) else 0,
       np.median(hi[:, 3]) if len(hi) else 0, sorted(set(np.round(x[:, 1], 1)))))
