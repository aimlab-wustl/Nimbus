// philox.cuh -- Philox4x32-10 (Random123-compatible) and the nimbus key/counter/draw mapping.
//
// Header-only. Compiles under NVRTC without any system include (only builtin types and CUDA device
// intrinsics), under nvcc, and as plain host C++ (for host-side cross-checks; then portable C++
// replaces the intrinsics). Reproduces nimbus.core.rng bit for bit:
//
//   key     = (seed, stream)                    streams: SOLVER=1 INIT=2 CHANNEL=3 PILOT=4 DIAG=5 DATA=6 BASELINE=7
//   counter = (attempt & 0xffffffff, attempt >> 32, replica, instance)
//   one Philox call per attempt -> (x0, x1, x2, x3)
//   group   r = (x0*R + ((x1*R) >> 32)) >> 32  in exact unsigned 64-bit arithmetic, R < 2^32
//   uniform u = (((x2 << 20) | (x3 >> 12)) + 0.5) * 2^-52   (exact, strictly inside (0, 1))
//
// Generic streams (rng.stream_words layout): word w = lane (w % 4) of the block with counter
// (w / 4 masked to 32 bits, c1, c2, c3); uniform/integer j uses words (2j, 2j+1).
//
// Only 'unsigned int' (32-bit) and 'unsigned long long' (64-bit) are used, as on every CUDA target.

#ifndef NIMBUS_PHILOX_CUH
#define NIMBUS_PHILOX_CUH

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define TPI_FN __device__ __forceinline__
#define TPI_MULHI32(a, b) __umulhi((a), (b))
#else
#define TPI_FN static inline
#define TPI_MULHI32(a, b) ((unsigned int)(((unsigned long long)(a) * (unsigned long long)(b)) >> 32))
#endif

#define TPI_PHILOX_M0 0xD2511F53u
#define TPI_PHILOX_M1 0xCD9E8D57u
#define TPI_PHILOX_W0 0x9E3779B9u
#define TPI_PHILOX_W1 0xBB67AE85u

#define TPI_STREAM_SOLVER 1u
#define TPI_STREAM_INIT 2u
#define TPI_STREAM_CHANNEL 3u
#define TPI_STREAM_PILOT 4u
#define TPI_STREAM_DIAG 5u
#define TPI_STREAM_DATA 6u
#define TPI_STREAM_BASELINE 7u

// 2^-52 written as its exact decimal expansion (no hex-float literal needed).
#define TPI_TWO_M52 2.220446049250313080847263336181640625e-16

struct tpi_u32x4 {
    unsigned int x0, x1, x2, x3;
};

struct tpi_draw {
    unsigned int r;  // group index in [0, R)
    double u;        // uniform in (0, 1)
};

// Philox4x32 with 10 rounds; the key is bumped by the Weyl constants before rounds 1..9.
TPI_FN tpi_u32x4 tpi_philox4x32_10(unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
                                   unsigned int k0, unsigned int k1) {
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        if (i > 0) {
            k0 += TPI_PHILOX_W0;  // unsigned 32-bit wraparound == & 0xffffffff
            k1 += TPI_PHILOX_W1;
        }
        const unsigned int hi0 = TPI_MULHI32(TPI_PHILOX_M0, c0);
        const unsigned int lo0 = TPI_PHILOX_M0 * c0;
        const unsigned int hi1 = TPI_MULHI32(TPI_PHILOX_M1, c2);
        const unsigned int lo1 = TPI_PHILOX_M1 * c2;
        const unsigned int y0 = hi1 ^ c1 ^ k0;
        const unsigned int y2 = hi0 ^ c3 ^ k1;
        c0 = y0;
        c1 = lo1;
        c2 = y2;
        c3 = lo0;
    }
    tpi_u32x4 out;
    out.x0 = c0;
    out.x1 = c1;
    out.x2 = c2;
    out.x3 = c3;
    return out;
}

// floor(((x0 << 32) | x1) * R / 2^64), exact for R < 2^32 (no overflow: x0*R <= (2^32-1)^2).
TPI_FN unsigned int tpi_group_index(unsigned int x0, unsigned int x1, unsigned int n_groups) {
    const unsigned long long R = (unsigned long long)n_groups;
    const unsigned long long t = (unsigned long long)x0 * R + (((unsigned long long)x1 * R) >> 32);
    return (unsigned int)(t >> 32);
}

// 52-bit open-interval uniform; k < 2^52 so (double)k, k + 0.5 and the scaling are all exact.
TPI_FN double tpi_uniform52(unsigned int x2, unsigned int x3) {
    const unsigned long long k = ((unsigned long long)x2 << 20) | (unsigned long long)(x3 >> 12);
    return ((double)k + 0.5) * TPI_TWO_M52;
}

// Solver-stream draw of one attempt of one replica (key = (seed, SOLVER)).
TPI_FN tpi_draw tpi_solver_draw(unsigned int seed, unsigned long long attempt, unsigned int replica,
                                unsigned int instance, unsigned int n_groups) {
    const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)(attempt & 0xffffffffull), (unsigned int)(attempt >> 32),
                                          replica, instance, seed, TPI_STREAM_SOLVER);
    tpi_draw d;
    d.r = tpi_group_index(x.x0, x.x1, n_groups);
    d.u = tpi_uniform52(x.x2, x.x3);
    return d;
}

TPI_FN unsigned int tpi_lane(const tpi_u32x4 x, unsigned int lane) {
    return lane == 0u ? x.x0 : (lane == 1u ? x.x1 : (lane == 2u ? x.x2 : x.x3));
}

// Word w of rng.stream_words(seed, stream, ..., c2=c2, c3=c3, c1=c1).
TPI_FN unsigned int tpi_stream_word(unsigned int seed, unsigned int stream, unsigned long long w, unsigned int c1,
                                    unsigned int c2, unsigned int c3) {
    const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)((w >> 2) & 0xffffffffull), c1, c2, c3, seed, stream);
    return tpi_lane(x, (unsigned int)(w & 3ull));
}

// Uniform j of rng.stream_uniforms (words 2j, 2j+1 = block j/2, lanes (0,1) or (2,3)).
TPI_FN double tpi_stream_uniform(unsigned int seed, unsigned int stream, unsigned long long j, unsigned int c1,
                                 unsigned int c2, unsigned int c3) {
    const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)((j >> 1) & 0xffffffffull), c1, c2, c3, seed, stream);
    return (j & 1ull) ? tpi_uniform52(x.x2, x.x3) : tpi_uniform52(x.x0, x.x1);
}

// Integer j of rng.stream_integers (uniform in [0, high), multiply-high rule on words 2j, 2j+1).
TPI_FN unsigned int tpi_stream_integer(unsigned int seed, unsigned int stream, unsigned long long j,
                                       unsigned int high, unsigned int c1, unsigned int c2, unsigned int c3) {
    const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)((j >> 1) & 0xffffffffull), c1, c2, c3, seed, stream);
    return (j & 1ull) ? tpi_group_index(x.x2, x.x3, high) : tpi_group_index(x.x0, x.x1, high);
}

#endif  // NIMBUS_PHILOX_CUH
