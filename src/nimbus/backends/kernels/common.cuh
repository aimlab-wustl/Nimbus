// common.cuh -- attempt-indexed schedule temperatures and the Metropolis threshold acceptance.
//
// Header-only, NVRTC-compilable without system includes (exp/log/log1p and the __d*_rn intrinsics
// are CUDA builtins under NVRTC). Also compiles as host C++ for cross-checks.
//
// schedule (nimbus.annealing.schedule), closed form from (kind, p0..p3, B, n):
//   frac = 0 if B == 1 else n / (B - 1)
//   ZERO 0: 0   CONSTANT 1: p0   GEOMETRIC 2: exp(p0 + (p1 - p0) frac)
//   FN_ENDPOINT 3: p0 / log1p(p1 + (p2 - p1) frac)   FN_LEGACY 4: p0 / log1p((p2 + n p3) / p1)
// acceptance (Methods, Stochastic transition dynamics and FN annealing, Methods, Stochastic transition dynamics and FN annealing):
//   xi = -(tau/2) log(u); accept iff v <= xi for tau > 0, and iff v <= 0 for tau == 0
//   (v is the transition-neuron input; Delta E = 2 v).
//
// Arithmetic uses the round-to-nearest intrinsics __dadd_rn/__dsub_rn/__dmul_rn/__ddiv_rn, which the
// compiler never contracts into FMA, so every basic operation is the same IEEE operation as on the
// CPU. Only the library functions exp/log1p/log may differ from the host libm (CUDA documents a
// maximum error of 1-2 ulp); scripts/gpu_rng_check.py measures that difference.

#ifndef NIMBUS_COMMON_CUH
#define NIMBUS_COMMON_CUH

#include "philox.cuh"

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define TPI_DADD(a, b) __dadd_rn((a), (b))
#define TPI_DSUB(a, b) __dsub_rn((a), (b))
#define TPI_DMUL(a, b) __dmul_rn((a), (b))
#define TPI_DDIV(a, b) __ddiv_rn((a), (b))
#define TPI_QNAN() __longlong_as_double(0x7ff8000000000000ll)
#else
#include <math.h>
#define TPI_DADD(a, b) ((a) + (b))
#define TPI_DSUB(a, b) ((a) - (b))
#define TPI_DMUL(a, b) ((a) * (b))
#define TPI_DDIV(a, b) ((a) / (b))
#define TPI_QNAN() (NAN)
#endif

#define TPI_SCHED_ZERO 0
#define TPI_SCHED_CONSTANT 1
#define TPI_SCHED_GEOMETRIC 2
#define TPI_SCHED_FN_ENDPOINT 3
#define TPI_SCHED_FN_LEGACY 4

// tau_n for attempt n of a schedule with B = n_total attempts. Unknown kinds return a quiet NaN
// (every acceptance comparison with NaN is false); the host validates kinds before launching.
TPI_FN double tpi_schedule_tau(int kind, double p0, double p1, double p2, double p3, long long n_total, long long n) {
    const double frac = (n_total == 1) ? 0.0 : TPI_DDIV((double)n, (double)(n_total - 1));
    switch (kind) {
        case TPI_SCHED_ZERO:
            return 0.0;
        case TPI_SCHED_CONSTANT:
            return p0;
        case TPI_SCHED_GEOMETRIC:
            return exp(TPI_DADD(p0, TPI_DMUL(TPI_DSUB(p1, p0), frac)));
        case TPI_SCHED_FN_ENDPOINT:
            return TPI_DDIV(p0, log1p(TPI_DADD(p1, TPI_DMUL(TPI_DSUB(p2, p1), frac))));
        case TPI_SCHED_FN_LEGACY:
            return TPI_DDIV(p0, log1p(TPI_DDIV(TPI_DADD(p2, TPI_DMUL((double)n, p3)), p1)));
        default:
            return TPI_QNAN();
    }
}

// xi = -(tau / 2) * log(u). tau / 2 equals tau * 0.5 exactly; negation is exact.
TPI_FN double tpi_metropolis_threshold(double tau, double u) {
    return TPI_DMUL(-TPI_DMUL(tau, 0.5), log(u));
}

// Metropolis threshold acceptance of a selected transition with neuron input v.
TPI_FN bool tpi_metropolis_accept(double v, double tau, double u) {
    if (tau == 0.0) return v <= 0.0;
    return v <= tpi_metropolis_threshold(tau, u);
}

#endif  // NIMBUS_COMMON_CUH
