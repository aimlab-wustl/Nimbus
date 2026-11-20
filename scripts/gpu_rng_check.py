#!/usr/bin/env python
"""Cross-check the CUDA RNG/schedule/acceptance headers against the Python references.

What it does
------------
1. Builds one source from src/nimbus/backends/kernels/philox.cuh + common.cuh (inlined as a code
   string; local #include lines are resolved) plus small test kernels.
2. Executes it and compares with ``nimbus.core.rng`` / ``nimbus.annealing.schedule`` /
   ``nimbus.annealing.policy``:
     * Random123 known-answer vectors for Philox4x32-10 (exact),
     * Philox words on random counters and keys (exact),
     * solver draws (group index r, 52-bit uniform u) on random (seed, attempt, replica, instance, R),
       including attempts >= 2**32 and R in {1, 2, 3, 7, 2**31, 2**32-1} (exact),
     * generic stream words / uniforms / integers (exact),
     * schedule temperatures tau_n of every kind (ulp difference vs the CPU libm reference),
     * log, log1p, exp and the threshold xi = -(tau/2) log u (ulp difference),
     * acceptance decisions; a mismatch counts as "explained" only if v lies between the CPU and
       device thresholds (inputs placed exactly on the CPU threshold are included on purpose).
3. Prints a JSON report (also written with --out). Exit status: 0 = every exact check passed and every
   ulp difference <= --max-ulp and every acceptance mismatch is explained; 1 = a check failed;
   2 = not executed (no CuPy / no device / no compiler / no NVRTC library).

Modes
-----
default        GPU through CuPy ``RawModule`` (NVRTC). Needs a CUDA device and CuPy.
--compile-only NVRTC compilation through ctypes only (no device, no CuPy), e.g. on a machine without a GPU, with
               the pip wheel ``nvidia-cuda-nvrtc-cu12``; reports the log and the kernels in the PTX.
--host         compiles the headers as plain host C++ (their non-CUDA branch) with --cxx (default $CXX or
               g++; e.g. --cxx "python -m ziglang c++") and runs the same checks on the CPU. This checks the
               header logic; device intrinsics and device libm are only covered by the GPU mode.

Usage (on a machine with a CUDA GPU and CuPy):
    python scripts/gpu_rng_check.py --n-random 1048576 --out gpu_rng_check.json
    python scripts/gpu_rng_check.py --compile-only --arch compute_90
    python scripts/gpu_rng_check.py --host --cxx g++
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA = "nimbus.gpu_rng_check/1"
REPO = Path(__file__).resolve().parents[1]

try:  # the package is normally installed (pip install -e .); fall back to the source tree
    import nimbus  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from nimbus.annealing import policy as _policy  # noqa: E402
from nimbus.annealing import schedule as _schedule  # noqa: E402
from nimbus.core import rng as _rng  # noqa: E402

KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]

# Each body is shared by the CUDA kernels (grid-stride loop) and the host harness (plain loop).
_BODIES = {
    "philox": """
        const tpi_u32x4 y = tpi_philox4x32_10(ctr[4 * j], ctr[4 * j + 1], ctr[4 * j + 2], ctr[4 * j + 3],
                                              key[2 * j], key[2 * j + 1]);
        out[4 * j] = y.x0; out[4 * j + 1] = y.x1; out[4 * j + 2] = y.x2; out[4 * j + 3] = y.x3;""",
    "draw": """
        const tpi_draw d = tpi_solver_draw(seed[j], attempt[j], replica[j], instance[j], ngroups[j]);
        r[j] = d.r; u[j] = d.u;""",
    "stream": """
        word[j] = tpi_stream_word(prm[0], prm[1], (unsigned long long)j, prm[2], prm[3], prm[4]);
        unif[j] = tpi_stream_uniform(prm[0], prm[1], (unsigned long long)j, prm[2], prm[3], prm[4]);
        integer[j] = tpi_stream_integer(prm[0], prm[1], (unsigned long long)j, prm[5], prm[2], prm[3], prm[4]);""",
    "schedule": """
        out[j] = tpi_schedule_tau(kind, sp[0], sp[1], sp[2], sp[3], n_total, idx[j]);""",
    "math": """
        out[j] = which == 0 ? log(x[j]) : (which == 1 ? log1p(x[j]) : exp(x[j]));""",
    "accept": """
        xi[j] = tpi_metropolis_threshold(tau[j], u[j]);
        acc[j] = tpi_metropolis_accept(v[j], tau[j], u[j]) ? 1 : 0;""",
}

CUDA_KERNELS = r"""
#define TPI_GRID_LOOP(j, n) for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < (n); \
                                 j += (long long)blockDim.x * gridDim.x)
extern "C" __global__ void k_philox(const unsigned int* ctr, const unsigned int* key, unsigned int* out, long long n) {
    TPI_GRID_LOOP(j, n) { @philox@ }
}
extern "C" __global__ void k_solver_draw(const unsigned int* seed, const unsigned long long* attempt,
                                         const unsigned int* replica, const unsigned int* instance,
                                         const unsigned int* ngroups, unsigned int* r, double* u, long long n) {
    TPI_GRID_LOOP(j, n) { @draw@ }
}
extern "C" __global__ void k_stream(const unsigned int* prm, unsigned int* word, double* unif, unsigned int* integer,
                                    long long n) {
    TPI_GRID_LOOP(j, n) { @stream@ }
}
extern "C" __global__ void k_schedule(int kind, const double* sp, long long n_total, const long long* idx, double* out,
                                      long long n) {
    TPI_GRID_LOOP(j, n) { @schedule@ }
}
extern "C" __global__ void k_math(const double* x, double* out, int which, long long n) {
    TPI_GRID_LOOP(j, n) { @math@ }
}
extern "C" __global__ void k_accept(const double* v, const double* tau, const double* u, double* xi,
                                    unsigned char* acc, long long n) {
    TPI_GRID_LOOP(j, n) { @accept@ }
}
"""

HOST_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
template <class T> static T* rd(FILE* f, long long n) {
    T* p = (T*)malloc(sizeof(T) * (size_t)(n > 0 ? n : 1));
    if (n > 0 && fread(p, sizeof(T), (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(3); }
    return p;
}
template <class T> static T* al(long long n) { return (T*)malloc(sizeof(T) * (size_t)(n > 0 ? n : 1)); }
template <class T> static void wr(FILE* f, const T* p, long long n) { fwrite(p, sizeof(T), (size_t)n, f); }
int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: harness <op> in out\n"); return 2; }
    FILE* in = fopen(argv[2], "rb"); FILE* fo = fopen(argv[3], "wb");
    if (!in || !fo) return 4;
    long long n = 0;
    if (fread(&n, 8, 1, in) != 1) return 3;
    const char* op = argv[1];
    if (!strcmp(op, "philox")) {
        unsigned int* ctr = rd<unsigned int>(in, 4 * n); unsigned int* key = rd<unsigned int>(in, 2 * n);
        unsigned int* out = al<unsigned int>(4 * n);
        for (long long j = 0; j < n; ++j) { @philox@ }
        wr(fo, out, 4 * n);
    } else if (!strcmp(op, "draw")) {
        unsigned int* seed = rd<unsigned int>(in, n); unsigned long long* attempt = rd<unsigned long long>(in, n);
        unsigned int* replica = rd<unsigned int>(in, n); unsigned int* instance = rd<unsigned int>(in, n);
        unsigned int* ngroups = rd<unsigned int>(in, n);
        unsigned int* r = al<unsigned int>(n); double* u = al<double>(n);
        for (long long j = 0; j < n; ++j) { @draw@ }
        wr(fo, r, n); wr(fo, u, n);
    } else if (!strcmp(op, "stream")) {
        unsigned int* prm = rd<unsigned int>(in, 6);
        unsigned int* word = al<unsigned int>(n); double* unif = al<double>(n); unsigned int* integer = al<unsigned int>(n);
        for (long long j = 0; j < n; ++j) { @stream@ }
        wr(fo, word, n); wr(fo, unif, n); wr(fo, integer, n);
    } else if (!strcmp(op, "schedule")) {
        int* kp = rd<int>(in, 1); int kind = kp[0]; double* sp = rd<double>(in, 4);
        long long* np_ = rd<long long>(in, 1); long long n_total = np_[0]; long long* idx = rd<long long>(in, n);
        double* out = al<double>(n);
        for (long long j = 0; j < n; ++j) { @schedule@ }
        wr(fo, out, n);
    } else if (!strcmp(op, "math")) {
        int* wp = rd<int>(in, 1); int which = wp[0]; double* x = rd<double>(in, n); double* out = al<double>(n);
        for (long long j = 0; j < n; ++j) { @math@ }
        wr(fo, out, n);
    } else if (!strcmp(op, "accept")) {
        double* v = rd<double>(in, n); double* tau = rd<double>(in, n); double* u = rd<double>(in, n);
        double* xi = al<double>(n); unsigned char* acc = al<unsigned char>(n);
        for (long long j = 0; j < n; ++j) { @accept@ }
        wr(fo, xi, n); wr(fo, acc, n);
    } else { fprintf(stderr, "unknown op\n"); return 2; }
    fclose(fo); fclose(in);
    return 0;
}
"""

KERNEL_NAMES = ("k_philox", "k_solver_draw", "k_stream", "k_schedule", "k_math", "k_accept")


# --------------------------------------------------------------------------------------------
# Source assembly
# --------------------------------------------------------------------------------------------

def default_kernels_dir() -> Path:
    import nimbus.backends as _b
    return Path(_b.__file__).resolve().parent / "kernels"


def inline_header(name: str, kernels_dir: Path, _seen=None) -> str:
    """Contents of a header with local #include "..." lines replaced by the included files (once each)."""
    seen = set() if _seen is None else _seen
    if name in seen:
        return ""
    seen.add(name)
    text = (Path(kernels_dir) / name).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
        if m:
            out.append(f"// ---- inlined {m.group(1)} ----")
            out.append(inline_header(m.group(1), kernels_dir, seen))
        else:
            out.append(line)
    return "\n".join(out)


def headers_source(kernels_dir: Path) -> str:
    seen: set = set()
    return "\n".join([inline_header("philox.cuh", kernels_dir, seen), inline_header("common.cuh", kernels_dir, seen)])


def _fill(template: str) -> str:
    for k, body in _BODIES.items():
        template = template.replace(f"@{k}@", body)
    return template


def build_source(kernels_dir: Path) -> str:
    """CUDA source: inlined headers + test kernels."""
    return headers_source(kernels_dir) + "\n" + _fill(CUDA_KERNELS)


def build_host_source(kernels_dir: Path) -> str:
    """Host C++ source: inlined headers (host branch) + file-driven harness."""
    return headers_source(kernels_dir) + "\n" + _fill(HOST_HARNESS)


# --------------------------------------------------------------------------------------------
# Executors: same operations on the GPU (CuPy) or on the host (compiled harness)
# --------------------------------------------------------------------------------------------

class NotExecuted(Exception):
    pass


class CupyExecutor:
    mode = "gpu"

    def __init__(self, kernels_dir):
        try:
            import cupy as cp
        except Exception as exc:  # noqa: BLE001
            raise NotExecuted(f"CuPy not importable: {exc!r}") from None
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:  # noqa: BLE001
            raise NotExecuted(f"no usable CUDA device: {exc!r}") from None
        if ndev < 1:
            raise NotExecuted("no CUDA device visible")
        self.cp = cp
        t0 = time.perf_counter()
        mod = cp.RawModule(code=build_source(kernels_dir), options=("--std=c++14",))
        self.k = {k: mod.get_function(k) for k in KERNEL_NAMES}
        self.compile_seconds = time.perf_counter() - t0
        self.environment = cupy_environment(cp)

    def _launch(self, name, n, *args):
        block = 256
        grid = int(min(max(1, (n + block - 1) // block), 65535))
        self.k[name]((grid,), (block,), args + (np.int64(n),))
        self.cp.cuda.Device().synchronize()

    def philox(self, ctr, key):
        cp = self.cp
        out = cp.zeros((ctr.shape[0], 4), dtype=cp.uint32)
        self._launch("k_philox", ctr.shape[0], cp.asarray(ctr), cp.asarray(key), out)
        return cp.asnumpy(out)

    def draw(self, seed, attempt, replica, instance, ngroups):
        cp = self.cp
        n = seed.shape[0]
        r = cp.zeros(n, dtype=cp.uint32)
        u = cp.zeros(n, dtype=cp.float64)
        self._launch("k_solver_draw", n, cp.asarray(seed), cp.asarray(attempt), cp.asarray(replica),
                     cp.asarray(instance), cp.asarray(ngroups), r, u)
        return cp.asnumpy(r), cp.asnumpy(u)

    def stream(self, prm, n):
        cp = self.cp
        w = cp.zeros(n, dtype=cp.uint32)
        u = cp.zeros(n, dtype=cp.float64)
        i = cp.zeros(n, dtype=cp.uint32)
        self._launch("k_stream", n, cp.asarray(np.asarray(prm, dtype=np.uint32)), w, u, i)
        return cp.asnumpy(w), cp.asnumpy(u), cp.asnumpy(i)

    def schedule(self, kind, p, n_total, idx):
        cp = self.cp
        out = cp.zeros(idx.shape[0], dtype=cp.float64)
        self._launch("k_schedule", idx.shape[0], np.int32(kind), cp.asarray(np.asarray(p, dtype=np.float64)),
                     np.int64(n_total), cp.asarray(idx), out)
        return cp.asnumpy(out)

    def math(self, x, which):
        cp = self.cp
        out = cp.zeros(x.shape[0], dtype=cp.float64)
        self._launch("k_math", x.shape[0], cp.asarray(x), out, np.int32(which))
        return cp.asnumpy(out)

    def accept(self, v, tau, u):
        cp = self.cp
        xi = cp.zeros(v.shape[0], dtype=cp.float64)
        acc = cp.zeros(v.shape[0], dtype=cp.uint8)
        self._launch("k_accept", v.shape[0], cp.asarray(v), cp.asarray(tau), cp.asarray(u), xi, acc)
        return cp.asnumpy(xi), cp.asnumpy(acc).astype(bool)


class HostExecutor:
    mode = "host"

    def __init__(self, kernels_dir, cxx):
        self.tmp = tempfile.TemporaryDirectory(prefix="nimbus_hostcheck_")
        d = Path(self.tmp.name)
        src = d / "harness.cpp"
        src.write_text(build_host_source(kernels_dir), encoding="utf-8")
        exe = d / ("harness.exe" if os.name == "nt" else "harness")
        cmd = shlex.split(cxx, posix=(os.name != "nt")) + ["-std=c++14", "-O2", "-ffp-contract=off", str(src), "-o",
                                                            str(exe)]
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NotExecuted(f"host compiler {cxx!r} not runnable: {exc!r}") from None
        self.compile_seconds = time.perf_counter() - t0
        own = [ln for ln in (res.stdout + res.stderr).splitlines() if "harness.cpp" in ln]
        if res.returncode != 0 or not exe.exists():
            raise RuntimeError(f"host compilation failed ({' '.join(cmd)}):\n{res.stderr[-4000:]}")
        self.exe = exe
        self.environment = {"cxx": cxx, "compile_command": cmd, "compiler_messages_for_harness": own[:50]}
        try:
            ver = subprocess.run(shlex.split(cxx, posix=(os.name != "nt")) + ["--version"], capture_output=True,
                                 text=True, timeout=60)
            self.environment["cxx_version"] = (ver.stdout or ver.stderr).splitlines()[0] if (ver.stdout or ver.stderr) else None
        except Exception:  # noqa: BLE001
            pass

    def _run(self, op, n, arrays, outs):
        d = Path(self.tmp.name)
        fin, fout = d / "in.bin", d / "out.bin"
        with open(fin, "wb") as f:
            f.write(np.int64(n).tobytes())
            for a in arrays:
                f.write(np.ascontiguousarray(a).tobytes())
        res = subprocess.run([str(self.exe), op, str(fin), str(fout)], capture_output=True, text=True, timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(f"host harness op {op} failed: {res.stderr}")
        buf = fout.read_bytes()
        off, result = 0, []
        for dt, cnt in outs:
            a = np.frombuffer(buf, dtype=dt, count=cnt, offset=off).copy()
            off += a.nbytes
            result.append(a)
        return result

    def philox(self, ctr, key):
        n = ctr.shape[0]
        (w,) = self._run("philox", n, [ctr.astype(np.uint32), key.astype(np.uint32)], [(np.uint32, 4 * n)])
        return w.reshape(n, 4)

    def draw(self, seed, attempt, replica, instance, ngroups):
        n = seed.shape[0]
        return tuple(self._run("draw", n, [seed.astype(np.uint32), attempt.astype(np.uint64), replica.astype(np.uint32),
                                           instance.astype(np.uint32), ngroups.astype(np.uint32)],
                               [(np.uint32, n), (np.float64, n)]))

    def stream(self, prm, n):
        return tuple(self._run("stream", n, [np.asarray(prm, dtype=np.uint32)],
                               [(np.uint32, n), (np.float64, n), (np.uint32, n)]))

    def schedule(self, kind, p, n_total, idx):
        n = idx.shape[0]
        (o,) = self._run("schedule", n, [np.int32(kind), np.asarray(p, dtype=np.float64), np.int64(n_total),
                                         idx.astype(np.int64)], [(np.float64, n)])
        return o

    def math(self, x, which):
        n = x.shape[0]
        (o,) = self._run("math", n, [np.int32(which), x.astype(np.float64)], [(np.float64, n)])
        return o

    def accept(self, v, tau, u):
        n = v.shape[0]
        xi, acc = self._run("accept", n, [v, tau, u], [(np.float64, n), (np.uint8, n)])
        return xi, acc.astype(bool)


# --------------------------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------------------------

def ulp_diff(a, b) -> np.ndarray:
    """|ordinal(a) - ordinal(b)| for float64 arrays; NaN vs NaN = 0, NaN vs number = 2**62."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    ia, ib = a.view(np.int64), b.view(np.int64)
    lo = np.int64(-(2 ** 63))
    with np.errstate(over="ignore"):
        oa = np.where(ia < 0, lo - ia, ia)
        ob = np.where(ib < 0, lo - ib, ib)
        d = oa - ob
        d = np.where(d < 0, -d, d)
    d = np.where(d < 0, np.int64(2 ** 62), d)  # wrapped extremes
    na, nb = np.isnan(a), np.isnan(b)
    d = np.where(na & nb, 0, d)
    d = np.where(na ^ nb, np.int64(2 ** 62), d)
    return d


def exact_record(dev, ref, n=None):
    dev = np.asarray(dev)
    ref = np.asarray(ref)
    if dev.dtype.kind == "f":
        neq = np.ascontiguousarray(dev).view(np.uint64) != np.ascontiguousarray(ref, dtype=np.float64).view(np.uint64)
    else:
        neq = dev.astype(np.uint64) != ref.astype(np.uint64)
    bad = np.flatnonzero(neq.ravel())
    rec = {"n": int(dev.size if n is None else n), "mismatches": int(bad.size), "pass": bool(bad.size == 0)}
    if bad.size:
        j = int(bad[0])
        rec["first_mismatch"] = {"index": j, "device": repr(dev.ravel()[j].item()), "ref": repr(ref.ravel()[j].item())}
    return rec


def ulp_record(dev, ref, max_ulp):
    d = ulp_diff(dev, ref)
    hist = {str(k): int(np.sum(d == k)) for k in range(0, 4)}
    hist[">3"] = int(np.sum(d > 3))
    m = int(d.max()) if d.size else 0
    rec = {"n": int(d.size), "max_ulp": m, "n_differ": int(np.sum(d > 0)), "ulp_histogram": hist,
           "pass": bool(m <= max_ulp)}
    if m:
        j = int(np.argmax(d))
        rec["worst"] = {"index": j, "device": repr(float(np.asarray(dev).ravel()[j])),
                        "ref": repr(float(np.asarray(ref).ravel()[j]))}
    return rec


def test_schedules():
    S = _schedule
    return {
        "zero": S.zero(1000),
        "constant": S.constant(0.37, 1000),
        "geometric": S.geometric(10.0, 0.01, 1000),
        "geometric_B1": S.geometric(5.0, 0.5, 1),
        "geometric_long": S.geometric(1e3, 1e-4, 10 ** 10),
        "fn_endpoints": S.fn_endpoints(10.0, 0.05, 0.1, 1000),
        "fn_endpoints_long": S.fn_endpoints(123.4, 1e-3, 3.0, 10 ** 9),
        "fn_endpoints_flat": S.fn_endpoints(2.0, 2.0, 0.1, 50),
        "fn_endpoints_B1": S.fn_endpoints(2.0, 0.1, 0.1, 1),
        "fn_legacy": S.fn_legacy(0.5, 100.0, 1.0, 0.01, 10 ** 6),
        "fn_legacy_B1": S.fn_legacy(2.5e-3, 1e4, 0.5, 3.7, 1),
    }


# --------------------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------------------

def run_checks(ex, args) -> dict:
    checks: dict = {}
    g = np.random.default_rng(args.seed)
    N = int(args.n_random)

    # -- KAT --------------------------------------------------------------------------------------
    ctr = np.array([k[0] for k in KAT], dtype=np.uint32)
    key = np.array([k[1] for k in KAT], dtype=np.uint32)
    exp = np.array([k[2] for k in KAT], dtype=np.uint32)
    checks["philox_kat"] = exact_record(ex.philox(ctr, key), exp)
    ref_kat = np.array([_rng.philox4x32_scalar(*c, *k) for c, k, _ in KAT], dtype=np.uint32)
    checks["philox_kat"]["reference_matches_kat"] = bool(np.array_equal(ref_kat, exp))

    # -- random counters and keys ---------------------------------------------------------------
    ctr = g.integers(0, 2 ** 32, size=(N, 4), dtype=np.uint64).astype(np.uint32)
    keys = g.integers(0, 2 ** 32, size=(16, 2), dtype=np.uint64).astype(np.uint32)
    block = (N + 15) // 16
    key = np.repeat(keys, block, axis=0)[:N]
    dev = ex.philox(ctr, key)
    ref = np.empty_like(dev)
    for b in range(16):
        sl = slice(b * block, min((b + 1) * block, N))
        if sl.start >= N:
            break
        ref[sl] = np.stack(_rng.philox4x32(ctr[sl, 0], ctr[sl, 1], ctr[sl, 2], ctr[sl, 3], int(keys[b, 0]),
                                           int(keys[b, 1])), axis=1)
    checks["philox_random"] = exact_record(dev, ref, n=N)

    # -- solver draws ---------------------------------------------------------------------------------
    nseeds = 16
    seeds = g.integers(0, 2 ** 32, size=nseeds, dtype=np.uint64)
    seed = np.repeat(seeds, (N + nseeds - 1) // nseeds)[:N].astype(np.uint32)
    attempt = g.integers(0, 2 ** 62, size=N, dtype=np.uint64)
    attempt[: N // 4] = g.integers(0, 2 ** 32, size=N // 4, dtype=np.uint64)
    attempt[:5] = [0, 1, 2 ** 32 - 1, 2 ** 32, 2 ** 33 + 17]
    replica = g.integers(0, 2 ** 32, size=N, dtype=np.uint64).astype(np.uint32)
    instance = g.integers(0, 2 ** 32, size=N, dtype=np.uint64).astype(np.uint32)
    ngroups = g.integers(1, 2 ** 32, size=N, dtype=np.uint64)
    edge = np.array([1, 2, 3, 7, 2 ** 31, 2 ** 32 - 1], dtype=np.uint64)
    m = min(N, edge.size * 100)
    ngroups[:m] = np.tile(edge, 100)[:m]
    ngroups[N // 2: N // 2 + N // 4] = g.integers(1, 5000, size=N // 4, dtype=np.uint64)
    ngroups = ngroups.astype(np.uint32)
    dr, du = ex.draw(seed, attempt, replica, instance, ngroups)
    ref_r = np.empty(N, dtype=np.int64)
    ref_u = np.empty(N, dtype=np.float64)
    for s in np.unique(seed):
        idx = np.flatnonzero(seed == s)
        x0, x1, x2, x3 = _rng.philox4x32(attempt[idx] & np.uint64(0xFFFFFFFF), attempt[idx] >> np.uint64(32),
                                         replica[idx], instance[idx], int(s), _rng.SOLVER)
        x0, x1 = x0.astype(np.uint64), x1.astype(np.uint64)
        R = ngroups[idx].astype(np.uint64)
        ref_r[idx] = ((x0 * R + ((x1 * R) >> np.uint64(32))) >> np.uint64(32)).astype(np.int64)
        ref_u[idx] = _rng.uniform52_array(x2, x3)
    # scalar reference spot check (guards the vectorized reconstruction above)
    spot = np.unique(np.concatenate([np.arange(min(N, 2000)), g.integers(0, N, 2000)]))
    spot_ok = all(_rng.solver_draw(int(seed[j]), int(attempt[j]), int(replica[j]), int(instance[j]), int(ngroups[j]))
                  == (int(ref_r[j]), float(ref_u[j])) for j in spot)
    checks["solver_draw_group"] = exact_record(dr, ref_r, n=N)
    checks["solver_draw_group"]["scalar_reference_spot_check"] = bool(spot_ok)
    checks["solver_draw_uniform"] = exact_record(du, ref_u, n=N)
    checks["solver_draw_uniform"]["all_inside_open_interval"] = bool(np.all((du > 0) & (du < 1)))

    # -- generic streams ---------------------------------------------------------------------------
    stream_cases = [(0, _rng.INIT, 0, 0, 0, 10), (987654321, _rng.CHANNEL, 0, 5, 2, 2 ** 32 - 1),
                    (2 ** 32 - 1, _rng.PILOT, 7, 2 ** 32 - 1, 11, 3), (42, _rng.DATA, 1, 0, 0, 1)]
    ns = max(1000, N // 16)
    srec = {"n_per_case": ns, "cases": len(stream_cases), "mismatches": 0, "pass": True}
    for (sd, st, c1, c2, c3, high) in stream_cases:
        w, u, i = ex.stream([sd, st, c1, c2, c3, high], ns)
        for dev_arr, ref_arr in ((w, _rng.stream_words(sd, st, ns, c2=c2, c3=c3, c1=c1)),
                                 (u, _rng.stream_uniforms(sd, st, ns, c2=c2, c3=c3, c1=c1)),
                                 (i, _rng.stream_integers(sd, st, ns, high, c2=c2, c3=c3, c1=c1))):
            rec = exact_record(dev_arr, ref_arr)
            srec["mismatches"] += rec["mismatches"]
            if not rec["pass"]:
                srec["pass"] = False
                srec.setdefault("first_failure", {"case": [sd, st, c1, c2, c3, high], **rec})
    checks["streams"] = srec

    # -- schedules -------------------------------------------------------------------------------------
    sched = {}
    nt = max(1000, N // 16)
    for name, s in test_schedules().items():
        kind, p0, p1, p2, p3, B = s.kernel_params()
        idx = np.unique(np.concatenate([[0, B - 1], np.arange(min(B, 2000)),
                                        g.integers(0, B, size=nt)])).astype(np.int64)
        dev = ex.schedule(kind, (p0, p1, p2, p3), B, idx)
        ref = np.array([s.tau(int(k)) for k in idx])
        sched[name] = ulp_record(dev, ref, args.max_ulp)
        sched[name]["kind"] = s.name
    checks["schedule_tau"] = sched

    # -- math functions, threshold and acceptance -------------------------------------------------------
    u = du[: min(N, 1 << 20)]
    xs = {"log": (0, np.concatenate([u, np.exp(g.uniform(-700, 0, 1 << 16))]), math.log),
          "log1p": (1, np.concatenate([np.exp(g.uniform(-40, 3, 1 << 16)), g.uniform(0, 10, 1 << 16)]), math.log1p),
          "exp": (2, g.uniform(-700, 700, 1 << 17), math.exp)}
    mrec = {}
    for name, (which, x, f) in xs.items():
        mrec[name] = ulp_record(ex.math(x, which), np.array([f(t) for t in x]), args.max_ulp)
    checks["libm"] = mrec

    na = u.size
    tau = np.exp(g.uniform(-8, 6, na))
    tau[: na // 10] = 0.0
    ref_xi = np.array([_policy.metropolis_threshold(t, x) for t, x in zip(tau, u)])
    v = g.normal(0, 1, na) * np.maximum(tau, 1.0)
    v[na // 10: na // 5] = 0.0
    near = slice(na // 5, na // 5 + na // 10)
    v[near] = ref_xi[near]  # exactly on the CPU threshold: most sensitive to ulp differences
    dxi, dacc = ex.accept(v, tau, u)
    checks["threshold_xi"] = ulp_record(dxi, ref_xi, args.max_ulp)
    ref_acc = np.array([_policy.metropolis_accept(a, t, x) for a, t, x in zip(v, tau, u)])
    mis = np.flatnonzero(dacc != ref_acc)
    lo, hi = np.minimum(dxi, ref_xi), np.maximum(dxi, ref_xi)
    explained = (dxi[mis] != ref_xi[mis]) & (v[mis] >= lo[mis]) & (v[mis] <= hi[mis]) & (tau[mis] > 0)
    checks["acceptance"] = {"n": int(na), "mismatches": int(mis.size),
                            "explained_by_threshold_ulp": int(np.sum(explained)),
                            "unexplained": int(np.sum(~explained)), "pass": bool(np.all(explained)),
                            "zero_temperature_mismatches": int(np.sum(tau[mis] == 0)),
                            "note": "inputs placed exactly on the CPU threshold are included on purpose"}

    exact_names = ("philox_kat", "philox_random", "solver_draw_group", "solver_draw_uniform", "streams")
    exact_ok = (all(checks[k]["pass"] for k in exact_names) and checks["philox_kat"]["reference_matches_kat"]
                and checks["solver_draw_group"]["scalar_reference_spot_check"]
                and checks["solver_draw_uniform"]["all_inside_open_interval"])
    ulp_ok = (all(r["pass"] for r in sched.values()) and all(r["pass"] for r in mrec.values())
              and checks["threshold_xi"]["pass"])
    summary = {
        "exact_checks_pass": bool(exact_ok),
        "ulp_checks_pass": bool(ulp_ok),
        "acceptance_pass": bool(checks["acceptance"]["pass"]),
        "max_ulp": {"schedule_tau": max(r["max_ulp"] for r in sched.values()),
                    "schedule_tau_by_case": {k: r["max_ulp"] for k, r in sched.items()},
                    "log": mrec["log"]["max_ulp"], "log1p": mrec["log1p"]["max_ulp"], "exp": mrec["exp"]["max_ulp"],
                    "threshold_xi": checks["threshold_xi"]["max_ulp"]},
        "max_ulp_allowed": int(args.max_ulp),
    }
    status = "passed" if (exact_ok and ulp_ok and checks["acceptance"]["pass"]) else "failed"
    return {"checks": checks, "summary": summary, "status": status}


def run_executor(args, make) -> dict:
    report = {"schema": SCHEMA, "mode": "host" if args.host else "gpu", "status": "not_executed",
              "environment": environment(), "n_random": int(args.n_random), "seed": int(args.seed)}
    try:
        ex = make()
    except NotExecuted as exc:
        report["reason"] = str(exc)
        return report
    except Exception as exc:  # noqa: BLE001 - a compile error is a failed check, not a skip
        report["status"] = "failed"
        report["reason"] = f"compilation failed: {exc}"
        return report
    report["environment"].update(ex.environment)
    report["compile_seconds"] = ex.compile_seconds
    t0 = time.perf_counter()
    report.update(run_checks(ex, args))
    report["check_seconds"] = time.perf_counter() - t0
    return report


# --------------------------------------------------------------------------------------------
# NVRTC compile-only (ctypes; no device, no CuPy)
# --------------------------------------------------------------------------------------------

_DLL_DIRS = []  # keep os.add_dll_directory handles alive for the process lifetime


def find_nvrtc(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("NIMBUS_NVRTC"):
        cands.append(os.environ["NIMBUS_NVRTC"])
    pats = ["nvrtc64_*.dll", "libnvrtc.so*"]
    roots = []
    try:
        import nvidia.cuda_nvrtc as _nv  # pip wheel nvidia-cuda-nvrtc-cu12
        roots += [os.path.join(p, sub) for p in _nv.__path__ for sub in ("bin", "lib")]
    except Exception:  # noqa: BLE001
        pass
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            roots += [os.path.join(os.environ[var], sub) for sub in ("bin", "lib64", "lib")]
    roots += ["/usr/local/cuda/lib64"]
    for r in roots:
        for p in pats:
            cands += sorted(glob.glob(os.path.join(r, p)), reverse=True)
    lib = ctypes.util.find_library("nvrtc")
    if lib:
        cands.append(lib)
    names = [os.path.basename(c) for c in cands]
    return [c for c, nm in zip(cands, names) if "builtins" not in nm and ".alt." not in nm]


def compile_only(args) -> dict:
    report = {"schema": SCHEMA, "mode": "compile_only", "status": "not_executed", "environment": environment()}
    src = build_source(Path(args.kernels_dir))
    lib = None
    tried = []
    for cand in find_nvrtc(args.nvrtc):
        tried.append(cand)
        try:
            if os.name == "nt" and os.path.dirname(cand):
                _DLL_DIRS.append(os.add_dll_directory(os.path.dirname(cand)))
            lib = ctypes.CDLL(cand)
            report["nvrtc_library"] = cand
            break
        except OSError:
            continue
    if lib is None:
        report["reason"] = f"NVRTC library not found (tried {tried}); pass --nvrtc or set NIMBUS_NVRTC"
        return report
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    report["nvrtc_version"] = f"{major.value}.{minor.value}"
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"nimbus_rng_check.cu", 0, None, None)
    if rc:
        report.update(status="failed", reason=f"nvrtcCreateProgram: {lib.nvrtcGetErrorString(rc).decode()}")
        return report
    opts = [f"--gpu-architecture={args.arch}".encode(), b"--std=c++14"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    t0 = time.perf_counter()
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    report["compile_seconds"] = time.perf_counter() - t0
    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(prog, buf)
    report["log"] = buf.value.decode(errors="replace")
    report["options"] = [o.decode() for o in opts]
    if rc == 0:
        lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
        ptx = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetPTX(prog, ptx)
        text = ptx.value.decode(errors="replace")
        report["ptx_bytes"] = len(text)
        report["kernels_in_ptx"] = [k for k in KERNEL_NAMES if f".entry {k}" in text]
        report["status"] = "passed" if len(report["kernels_in_ptx"]) == len(KERNEL_NAMES) else "failed"
        if args.ptx_out:
            Path(args.ptx_out).write_text(text, encoding="utf-8")
    else:
        report["status"] = "failed"
        report["reason"] = f"nvrtcCompileProgram: {lib.nvrtcGetErrorString(rc).decode()}"
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return report


# --------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------

def environment() -> dict:
    env = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__,
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        env["git_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        env["git_dirty"] = bool(dirty)
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    return env


def cupy_environment(cp) -> dict:
    out = {"cupy": cp.__version__}
    try:
        out["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        out["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get("name", b"")
        out["device"] = name.decode() if isinstance(name, bytes) else str(name)
        out["compute_capability"] = f"{props.get('major')}.{props.get('minor')}"
        out["nvrtc"] = ".".join(str(x) for x in cp.cuda.nvrtc.getVersion())
    except Exception as exc:  # noqa: BLE001
        out["environment_error"] = repr(exc)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=1 << 20, help="random counters per exact check (default 2**20)")
    ap.add_argument("--seed", type=int, default=20260930, help="numpy seed for the test inputs")
    ap.add_argument("--max-ulp", type=int, default=4, help="largest accepted ulp difference for libm-based values")
    ap.add_argument("--kernels-dir", default=None, help="directory with philox.cuh and common.cuh")
    ap.add_argument("--out", default=None, help="also write the JSON report to this file")
    ap.add_argument("--compile-only", action="store_true", help="only compile with NVRTC via ctypes (no GPU needed)")
    ap.add_argument("--arch", default="compute_90", help="NVRTC virtual architecture for --compile-only (NVIDIA H100: compute_90)")
    ap.add_argument("--nvrtc", default=None, help="path to the NVRTC shared library for --compile-only")
    ap.add_argument("--ptx-out", default=None, help="write the PTX of --compile-only to this file")
    ap.add_argument("--host", action="store_true", help="run the checks on the host-compiled headers (no GPU)")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"), help="host C++ compiler command for --host")
    args = ap.parse_args(argv)
    if args.kernels_dir is None:
        args.kernels_dir = str(default_kernels_dir())
    if args.compile_only:
        report = compile_only(args)
    elif args.host:
        report = run_executor(args, lambda: HostExecutor(args.kernels_dir, args.cxx))
    else:
        report = run_executor(args, lambda: CupyExecutor(args.kernels_dir))
    text = json.dumps(report, indent=2, sort_keys=False, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return {"passed": 0, "failed": 1}.get(report["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
