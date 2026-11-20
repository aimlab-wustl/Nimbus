"""Numba RNG / schedule / acceptance helpers versus the pure-Python references (bit for bit)."""

import math

import numpy as np
import pytest
from numba import njit

from nimbus.annealing import policy, schedule
from nimbus.backends import _numba_common as nc
from nimbus.core import rng

pytestmark = pytest.mark.quick

# Random123 known-answer vectors for philox4x32_10 (kat_vectors): (counter, key, expected).
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]
N_RANDOM = 200_000


@njit(cache=False)
def _draws_rows(seed, attempt, replica, instance, ngroups, out_r, out_u):
    for j in range(seed.shape[0]):
        r, u = nc.solver_draw(seed[j], attempt[j], replica[j], instance[j], ngroups[j])
        out_r[j] = r
        out_u[j] = u


@njit(cache=False)
def _stream_rows(seed, stream, n, c1, c2, c3, high, words, unif, ints, bits):
    for j in range(n):
        words[j] = nc.stream_word(seed, stream, j, c1, c2, c3)
        bits[j] = nc.stream_bit(seed, stream, j, c1, c2, c3)
        unif[j] = nc.stream_uniform(seed, stream, j, c1, c2, c3)
        ints[j] = nc.stream_integer(seed, stream, j, high, c1, c2, c3)


@njit(cache=False)
def _accept_rows(v, tau, u, xi, acc):
    for j in range(v.shape[0]):
        xi[j] = nc.metropolis_threshold(tau[j], u[j])
        acc[j] = nc.metropolis_accept(v[j], tau[j], u[j])


def _random_u32(g, shape):
    return g.integers(0, 2 ** 32, size=shape, dtype=np.uint64).astype(np.uint32)


@pytest.mark.parametrize("ctr,key,expected", KAT)
def test_kat_numba_and_reference(ctr, key, expected):
    assert rng.philox4x32_scalar(*ctr, *key) == expected
    got = nc.philox4x32_10(*ctr, *key)
    assert tuple(int(x) for x in got) == expected
    out = np.zeros((1, 4), dtype=np.uint32)
    nc.philox4x32_10_rows(np.array([ctr], dtype=np.uint32), np.array([key], dtype=np.uint32), out)
    assert tuple(int(x) for x in out[0]) == expected


def test_philox_random_counters_bitwise():
    g = np.random.default_rng(20260930)
    ctr = _random_u32(g, (N_RANDOM, 4))
    # a few keys, each shared by a block of counters (the vectorized reference takes a scalar key)
    keys = _random_u32(g, (8, 2))
    key = np.repeat(keys, N_RANDOM // 8, axis=0)
    out = np.zeros_like(ctr)
    nc.philox4x32_10_rows(ctr, key, out)
    block = N_RANDOM // 8
    for b in range(8):
        sl = slice(b * block, (b + 1) * block)
        ref = rng.philox4x32(ctr[sl, 0], ctr[sl, 1], ctr[sl, 2], ctr[sl, 3], int(keys[b, 0]), int(keys[b, 1]))
        for lane in range(4):
            np.testing.assert_array_equal(out[sl, lane], ref[lane])
    # the scalar reference agrees with the vectorized one on a subset
    for j in range(0, N_RANDOM, N_RANDOM // 500):
        assert tuple(int(x) for x in out[j]) == rng.philox4x32_scalar(*(int(x) for x in ctr[j]),
                                                                      *(int(x) for x in key[j]))


def test_solver_draw_random_counters():
    g = np.random.default_rng(7)
    n = N_RANDOM
    seed = g.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.int64)
    attempt = g.integers(0, 2 ** 62, n, dtype=np.int64)
    attempt[: n // 4] = g.integers(0, 2 ** 32, n // 4)  # mostly-low attempts, hi word 0
    attempt[:5] = [0, 1, 2 ** 32 - 1, 2 ** 32, 2 ** 33 + 17]
    replica = g.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.int64)
    instance = g.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.int64)
    ngroups = g.integers(1, 2 ** 32, n, dtype=np.uint64).astype(np.int64)
    edge = np.array([1, 2, 3, 7, 2 ** 31, 2 ** 32 - 1], dtype=np.int64)
    ngroups[: edge.size * 50] = np.tile(edge, 50)
    ngroups[n // 2: n // 2 + n // 4] = g.integers(1, 5000, n // 4)
    out_r = np.empty(n, dtype=np.int64)
    out_u = np.empty(n, dtype=np.float64)
    _draws_rows(seed, attempt, replica, instance, ngroups, out_r, out_u)
    # reference, vectorized through the reference building blocks (seed varies per row -> by-seed groups
    # are impractical, so call the scalar Philox and the reference mapping functions)
    ref_r = np.empty(n, dtype=np.int64)
    ref_u = np.empty(n, dtype=np.float64)
    for j in range(n):
        ref_r[j], ref_u[j] = rng.solver_draw(int(seed[j]), int(attempt[j]), int(replica[j]), int(instance[j]),
                                             int(ngroups[j]))
    np.testing.assert_array_equal(out_r, ref_r)
    assert np.array_equal(out_u.view(np.uint64), ref_u.view(np.uint64))
    assert np.all((out_r >= 0) & (out_r < ngroups))
    assert np.all((out_u > 0) & (out_u < 1))


def test_solver_draws_into_matches_vectorized_reference():
    seed, replica, instance, R = 12345, 3, 9, 977
    a0, n = 2 ** 32 - 1000, 5000  # crosses the attempt hi-word boundary
    out_r = np.empty(n, dtype=np.int64)
    out_u = np.empty(n, dtype=np.float64)
    nc.solver_draws_into(seed, a0, replica, instance, R, out_r, out_u)
    ref_r, ref_u = rng.solver_draws(seed, np.arange(a0, a0 + n, dtype=np.uint64), replica, instance, R)
    np.testing.assert_array_equal(out_r, ref_r)
    assert np.array_equal(out_u.view(np.uint64), ref_u.view(np.uint64))


def test_group_and_uniform_extremes():
    m = 0xFFFFFFFF
    assert nc.group_index(m, m, 2 ** 32 - 1) == rng.group_index(m, m, 2 ** 32 - 1) == 2 ** 32 - 2
    assert nc.group_index(0, 0, 5) == 0
    assert nc.group_index(m, m, 1) == 0
    assert nc.uniform52(0, 0) == rng.uniform52(0, 0) == 0.5 * 2.0 ** -52
    assert nc.uniform52(m, m) == rng.uniform52(m, m) == 1.0 - 2.0 ** -53


@pytest.mark.parametrize("seed,stream,c1,c2,c3,high", [
    (0, rng.INIT, 0, 0, 0, 10), (987654321, rng.CHANNEL, 0, 5, 2, 2 ** 32 - 1),
    (2 ** 32 - 1, rng.PILOT, 7, 2 ** 32 - 1, 11, 3), (42, rng.DATA, 1, 0, 0, 1),
])
def test_stream_helpers_match_reference(seed, stream, c1, c2, c3, high):
    n = 20_001
    words = np.empty(n, dtype=np.uint32)
    unif = np.empty(n, dtype=np.float64)
    ints = np.empty(n, dtype=np.int64)
    bits = np.empty(n, dtype=np.uint8)
    _stream_rows(seed, stream, n, c1, c2, c3, high, words, unif, ints, bits)
    np.testing.assert_array_equal(words, rng.stream_words(seed, stream, n, c2=c2, c3=c3, c1=c1))
    np.testing.assert_array_equal(bits, rng.stream_bits(seed, stream, n, c2=c2, c3=c3, c1=c1))
    ref_u = rng.stream_uniforms(seed, stream, n, c2=c2, c3=c3, c1=c1)
    assert np.array_equal(unif.view(np.uint64), ref_u.view(np.uint64))
    np.testing.assert_array_equal(ints, rng.stream_integers(seed, stream, n, high, c2=c2, c3=c3, c1=c1))


def _schedules():
    return [
        schedule.zero(1000),
        schedule.constant(0.37, 1000),
        schedule.geometric(10.0, 0.01, 1000),
        schedule.geometric(3.0, 3.0, 7),
        schedule.geometric(5.0, 0.5, 1),
        schedule.fn_endpoints(10.0, 0.05, 0.1, 1000),
        schedule.fn_endpoints(123.4, 1e-3, 3.0, 10 ** 9),
        schedule.fn_endpoints(2.0, 2.0, 0.1, 50),
        schedule.fn_endpoints(2.0, 0.1, 0.1, 1),
        schedule.fn_legacy(0.5, 100.0, 1.0, 0.01, 10 ** 6),
        schedule.fn_legacy(2.5e-3, 1e4, 0.5, 3.7, 1),
    ]


def test_schedule_tau_bitwise_vs_reference():
    g = np.random.default_rng(3)
    for s in _schedules():
        kind, p0, p1, p2, p3, B = s.kernel_params()
        n = np.unique(np.concatenate([[0, B - 1], np.arange(min(B, 2000)),
                                      g.integers(0, B, 50_000)])).astype(np.int64)
        out = np.empty(n.shape[0], dtype=np.float64)
        nc.schedule_taus_into(kind, p0, p1, p2, p3, B, n, out)
        ref = np.array([s.tau(int(k)) for k in n])
        mismatch = int(np.sum(out.view(np.uint64) != ref.view(np.uint64)))
        assert mismatch == 0, f"{s.name}: {mismatch} taus differ from Schedule.tau"
        assert nc.schedule_tau(kind, p0, p1, p2, p3, B, int(n[-1])) == ref[-1]
        # the vectorized numpy path may use SIMD transcendental functions: allow 2 ulp
        vec = s.taus(n)
        ulp = np.abs(vec.view(np.int64) - ref.view(np.int64))
        assert int(ulp.max()) <= 2, f"{s.name}: numpy taus differ by {int(ulp.max())} ulp"


def test_schedule_tau_unknown_kind_raises():
    with pytest.raises(ValueError):
        nc.schedule_tau(9, 1.0, 0.0, 0.0, 0.0, 10, 3)


def test_acceptance_bitwise_vs_reference():
    g = np.random.default_rng(11)
    n = 100_000
    tau = np.exp(g.uniform(-8, 6, n))
    tau[: n // 10] = 0.0
    _, u = rng.solver_draws(5, np.arange(n, dtype=np.uint64), 0, 0, 3)
    ref_xi = np.array([policy.metropolis_threshold(t, x) for t, x in zip(tau, u)])
    v = g.normal(0, 1, n) * np.maximum(tau, 1.0)
    v[n // 10: n // 5] = 0.0  # neutral / self-loop inputs
    near = slice(n // 5, n // 5 + 10_000)  # inputs sitting exactly on the threshold
    v[near] = ref_xi[near]
    xi = np.empty(n)
    acc = np.empty(n, dtype=np.bool_)
    _accept_rows(v, tau, u, xi, acc)
    assert np.array_equal(xi.view(np.uint64), ref_xi.view(np.uint64))
    ref_acc = np.array([policy.metropolis_accept(a, t, x) for a, t, x in zip(v, tau, u)])
    np.testing.assert_array_equal(acc, ref_acc)
    assert np.all(acc[n // 10: n // 5])  # v = 0 always accepted (self-loops), at any tau
    assert np.all(acc[near])  # v == xi is accepted (<=)
    # tau == 0: accept iff v <= 0
    z = tau == 0
    np.testing.assert_array_equal(acc[z], v[z] <= 0)


def test_libm_functions_agree_with_python_math():
    g = np.random.default_rng(5)
    x = np.concatenate([g.uniform(1e-300, 1.0, 50_000), np.exp(g.uniform(-700, 0, 50_000))])
    out = np.empty_like(x)
    nc.log_into(x, out)
    assert np.array_equal(out.view(np.uint64), np.array([math.log(t) for t in x]).view(np.uint64))
    y = np.exp(g.uniform(-40, 3, 50_000))
    out = np.empty_like(y)
    nc.log1p_into(y, out)
    assert np.array_equal(out.view(np.uint64), np.array([math.log1p(t) for t in y]).view(np.uint64))
    z = g.uniform(-700, 700, 50_000)
    out = np.empty_like(z)
    nc.exp_into(z, out)
    assert np.array_equal(out.view(np.uint64), np.array([math.exp(t) for t in z]).view(np.uint64))


# ---------------------------------------------------------------- CUDA headers (no GPU needed)

def _load_check_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_rng_check.py"
    spec = importlib.util.spec_from_file_location("gpu_rng_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cuda_headers_have_no_system_includes_in_the_device_branch():
    import re
    chk = _load_check_script()
    kdir = chk.default_kernels_dir()
    for name in ("philox.cuh", "common.cuh"):
        text = (kdir / name).read_text(encoding="utf-8")
        m = re.search(r"#if defined\(__CUDACC_RTC__\) \|\| defined\(__CUDACC__\)(.*?)#else", text, re.S)
        assert m, f"{name}: device/host guard missing"
        assert "#include" not in m.group(1), f"{name}: include in the device branch"
        outside = text.replace(m.group(0), "")
        assert all(inc in ('#include "philox.cuh"', "#include <math.h>")
                   for inc in re.findall(r"#include\s*[<\"][^>\"]+[>\"]", outside))
    src = chk.build_source(kdir)
    assert '#include "' not in src.replace('// ---- inlined', '')
    assert src.count("TPI_FN tpi_u32x4 tpi_philox4x32_10(") == 1  # each header inlined once


def test_ulp_diff_helper():
    chk = _load_check_script()
    one = np.array([1.0])
    assert chk.ulp_diff(one, np.nextafter(one, 2.0))[0] == 1
    assert chk.ulp_diff(np.array([-0.0]), np.array([0.0]))[0] == 0
    tiny = np.array([5e-324])
    assert chk.ulp_diff(tiny, -tiny)[0] == 2
    assert chk.ulp_diff(np.array([np.nan]), np.array([np.nan]))[0] == 0
    assert chk.ulp_diff(np.array([np.nan]), one)[0] == 2 ** 62


def test_cuda_headers_compile_with_nvrtc_if_available():
    chk = _load_check_script()
    if not chk.find_nvrtc():
        pytest.skip("NVRTC library not available (pip wheel nvidia-cuda-nvrtc-cu12 or CUDA toolkit); not executed")
    import argparse
    args = argparse.Namespace(kernels_dir=str(chk.default_kernels_dir()), nvrtc=None, arch="compute_90", ptx_out=None)
    rep = chk.compile_only(args)
    if rep["status"] == "not_executed":
        pytest.skip(rep.get("reason", "NVRTC not loadable"))
    assert rep["status"] == "passed", rep.get("log") or rep.get("reason")
    assert rep["log"].strip() == ""


def _host_cxx():
    import os
    import shutil
    import sys
    if os.environ.get("NIMBUS_TEST_CXX"):
        return os.environ["NIMBUS_TEST_CXX"]
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return c
    try:
        import ziglang  # noqa: F401
        return f"{sys.executable} -m ziglang c++"
    except ImportError:
        return None


def test_cuda_headers_host_execution_matches_reference_if_compiler_available():
    chk = _load_check_script()
    cxx = _host_cxx()
    if cxx is None:
        pytest.skip("no host C++ compiler (g++, clang++, ziglang or NIMBUS_TEST_CXX); not executed")
    import argparse
    args = argparse.Namespace(kernels_dir=str(chk.default_kernels_dir()), n_random=1 << 14, seed=1, max_ulp=4,
                              host=True)
    rep = chk.run_executor(args, lambda: chk.HostExecutor(args.kernels_dir, cxx))
    if rep["status"] == "not_executed":
        pytest.skip(rep["reason"])
    assert rep["summary"]["exact_checks_pass"], rep["checks"]
    assert rep["status"] == "passed", rep["summary"]
