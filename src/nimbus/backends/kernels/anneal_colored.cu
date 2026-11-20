// anneal_colored.cu -- GPU kernel of the `colored` proposal process.
//
// SEPARATE from anneal.cu on purpose: the select/shared/sweep/parallel_trial/nfold kernel is kept unchanged; this file is a new code path and shares only the header-only helpers (philox.cuh, common.cuh).
//
// One thread block (TPI_CNT threads, 32 = one warp by default) owns ONE replica. A step visits one colour
// class (fixed cyclic order of the plan's position order); the groups of the class are mutually independent
// (disjoint route clause sets, no group controls a route of another), so thread i tests and commits group
// order[p + i] on its own, in parallel, with no communication until the end of the class:
//   v      = sequential sum over the group's weighted routes in route order (alpha from the pre-transition signs)
//   u      = Philox key (seed, COLOR = 12), counter (step, r, replica_id, instance_id), words x0, x1 -> uniform52
//   accept = v <= -(tau/2) log u  (tau == 0: v <= 0), tau = schedule(attempt clock at the first position of the class)
//   commit = mark pass (T = 2T, sign kept) then toggle pass, exactly the mark latch of anneal.cu
// One barrier per class: it publishes the commits and the per-group results (accepted, changed, v, dE per slot,
// double-buffered in shared memory). After it EVERY thread redundantly folds the class results into E, the counters
// and the best/feasible/target bookkeeping in ascending position order (the CPU's order: bit-identical floating
// point sums), so no second barrier or broadcast is needed. Clause signs T live in shared memory for the run.
// The step index of the Philox counter and the clock are 64-bit here, step < 2^32 is checked on the host.

#include "common.cuh"

#ifndef TPI_CNT
#define TPI_CNT 32
#endif
#ifndef TPI_MAX_SLOTS
#define TPI_MAX_SLOTS 4
#endif
#if TPI_CNT == 32
#define TPI_CSYNC() __syncwarp()
#else
#define TPI_CSYNC() __syncthreads()
#endif

#define TPI_C_ATTEMPTS 0
#define TPI_C_ACCEPTED 1
#define TPI_C_ACCEPTED_CHANGED 2
#define TPI_C_SELF_LOOPS 3
#define TPI_C_REJECTED 4
#define TPI_C_UPHILL_ACCEPTED 5
#define TPI_C_TARGET_HIT_ATTEMPT 6
#define TPI_C_STEPS 7
#define TPI_C_CAND 8
#define TPI_C_FROZEN 9
#define TPI_N_COUNTERS 10
#define TPI_STREAM_COLOR 12u

extern "C" __global__ void tpi_color_anneal(
    const long long* __restrict__ route_ptr, const long long* __restrict__ route_wend, const int2* __restrict__ rec,
    const int* __restrict__ spin_clause, int n_spins, int n_clauses, int n_groups, int n_slots, int score_slot,
    int violation_slot, const double* __restrict__ J,
    signed char* __restrict__ T, double* __restrict__ energies, double* __restrict__ best_energy,
    signed char* __restrict__ best_spins, long long* __restrict__ best_attempt,
    double* __restrict__ feasible_best_energy, signed char* __restrict__ feasible_best_spins,
    long long* __restrict__ feasible_best_attempt, long long* __restrict__ counters,
    long long* __restrict__ next_attempt, const double* __restrict__ targets, unsigned char* __restrict__ stopped,
    const int* __restrict__ replica_binding, const unsigned int* __restrict__ replica_id,
    const unsigned int* __restrict__ instance_id, const double* __restrict__ scale,
    unsigned int seed, int sched_kind, double p0, double p1, double p2, double p3, long long n_total,
    long long attempt_stop, int stop_on_target, int n_rep,
    const int* __restrict__ order, const int* __restrict__ class_ptr, const int* __restrict__ class_of_pos,
    int n_colors, int max_class) {
    extern __shared__ __align__(16) unsigned char tpi_smem[];
    const int tid = threadIdx.x;
    const long long rep = (long long)blockIdx.x;
    if (rep >= n_rep) return;
    if (stopped[rep]) return;

    double* sdE = (double*)tpi_smem;                              // [2][max_class][TPI_MAX_SLOTS]
    double* sv = sdE + (size_t)2 * max_class * TPI_MAX_SLOTS;     // [2][max_class]
    unsigned char* sfl = (unsigned char*)(sv + (size_t)2 * max_class);  // [2][max_class] bit0 accepted, bit1 changed
    signed char* Tr = (signed char*)(sfl + (((size_t)2 * max_class + 15) / 16) * 16);
    const size_t M1 = (size_t)n_clauses + 1;
    signed char* Tg = T + (size_t)rep * M1;
    for (size_t k = tid; k < M1; k += TPI_CNT) Tr[k] = Tg[k];
    TPI_CSYNC();
#define TR(k) Tr[(size_t)(k)]
    auto alpha_of = [&](int y) -> int {
        const int tc = (int)TR(y >> 1);
        return ((y & 1) ? -tc : tc) > 0;
    };

    const int b = replica_binding[rep];
    const double* Jb = J + (size_t)b * n_slots * n_clauses;
    const unsigned int rid = replica_id[rep], iid = instance_id[rep];
    const double* sc = scale + (size_t)rep * n_slots;
    const double sc0 = sc[0];
    const double tgt = targets[rep];
    double E[TPI_MAX_SLOTS];
#pragma unroll
    for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl) E[sl] = (sl < n_slots) ? energies[(size_t)rep * n_slots + sl] : 0.0;
    long long cnt[TPI_N_COUNTERS];
#pragma unroll
    for (int c = 0; c < TPI_N_COUNTERS; ++c) cnt[c] = counters[(size_t)rep * TPI_N_COUNTERS + c];
    double best_e = best_energy[rep], fb_e = feasible_best_energy[rep];
    long long best_a = best_attempt[rep], fb_a = feasible_best_attempt[rep];
    bool stop_now = false;
    long long n = next_attempt[rep];
    const long long R = (long long)n_groups;
    long long cyc = n / R;
    int p = (int)(n - cyc * R);
    int par = 0;

    while (n < attempt_stop) {
        const int k = class_of_pos[p];
        const int lo = class_ptr[k], hi = class_ptr[k + 1];
        const long long a_start = n - (long long)(p - lo);
        const unsigned long long step = (unsigned long long)cyc * (unsigned long long)n_colors + (unsigned long long)k;
        const double tau = tpi_schedule_tau(sched_kind, p0, p1, p2, p3, n_total, a_start);
        long long rem = attempt_stop - n;
        const int n_do = (int)((long long)(hi - p) < rem ? (long long)(hi - p) : rem);
        double* dEb = sdE + (size_t)par * max_class * TPI_MAX_SLOTS;
        double* vb = sv + (size_t)par * max_class;
        unsigned char* fl = sfl + (size_t)par * max_class;
        // ---- every group of the class in parallel: test, and commit if accepted ----
        for (int i = tid; i < n_do; i += TPI_CNT) {
            const unsigned int r = (unsigned int)order[p + i];
            const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)(step & 0xffffffffull), r, rid, iid, seed, TPI_STREAM_COLOR);
            const double u = tpi_uniform52(x.x0, x.x1);
            const long long rb0 = route_ptr[r], rb1 = route_ptr[r + 1], rbw = route_wend[r];
            double v = 0.0;
            for (long long t = rb0; t < rbw; ++t) {
                const int2 q = rec[t];
                if (alpha_of(q.y)) {
                    const int kk = q.x & 0x7fffffff;
                    v = TPI_DADD(v, TPI_DMUL(Jb[kk], (double)TR(kk)));
                }
            }
            const double xi = tpi_metropolis_threshold(tau, u);
            const bool accepted = (tau == 0.0) ? (v <= 0.0) : (v <= xi);
            unsigned char f = 0;
            double dEp[TPI_MAX_SLOTS];
#pragma unroll
            for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl) dEp[sl] = 0.0;
            if (accepted) {
                f = 1;
                bool spin = false;
                for (long long t = rb0; t < rb1; ++t) {  // mark pass: sigma latched in place, sign kept
                    const int2 q = rec[t];
                    if (alpha_of(q.y)) {
                        const int kk = q.x & 0x7fffffff;
                        TR(kk) = (signed char)(2 * (int)TR(kk));
                    }
                }
                for (long long t = rb0; t < rb1; ++t) {  // commit pass: toggle every marked clause
                    const int x2 = rec[t].x;
                    const int kk = x2 & 0x7fffffff;
                    const int tv = (int)TR(kk);
                    if (tv == 2 || tv == -2) {
                        const signed char told = (signed char)(tv / 2);
#pragma unroll
                        for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl)
                            if (sl < n_slots)
                                dEp[sl] = TPI_DADD(dEp[sl], TPI_DMUL(2.0, TPI_DMUL(Jb[(size_t)sl * n_clauses + kk], (double)told)));
                        TR(kk) = (signed char)(-told);
                        spin |= (x2 < 0);
                    }
                }
                if (spin) f |= 2;
            }
            fl[i] = f;
            vb[i] = v;
#pragma unroll
            for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl) dEb[(size_t)i * TPI_MAX_SLOTS + sl] = dEp[sl];
        }
        TPI_CSYNC();  // commits and class results visible to every thread
        // ---- fold the class (ascending position order), redundantly in every thread ----
        int acc_n = 0, chg_n = 0, slf_n = 0, up_n = 0;
        for (int i = 0; i < n_do; ++i) {
            const unsigned char f = fl[i];
            if (!(f & 1)) continue;
            const double v = vb[i];
            ++acc_n;
            E[0] = TPI_DADD(E[0], TPI_DMUL(2.0, v));
#pragma unroll
            for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl)
                if (sl < n_slots) E[sl] = TPI_DADD(E[sl], dEb[(size_t)i * TPI_MAX_SLOTS + sl]);
            if (f & 2) {
                ++chg_n;
                if (v > 1e-12 * sc0) ++up_n;
            } else {
                ++slf_n;
            }
        }
        const int p_old = p;
        n += n_do;
        p += n_do;
        if (p == (int)R) {
            p = 0;
            ++cyc;
        }
        par ^= 1;
        cnt[TPI_C_ATTEMPTS] += n_do;
        cnt[TPI_C_STEPS] += (p_old == lo) ? 1 : 0;  // a step is started when the visit begins at the class's first position
        cnt[TPI_C_CAND] += n_do;
        cnt[TPI_C_ACCEPTED] += acc_n;
        cnt[TPI_C_REJECTED] += n_do - acc_n;
        cnt[TPI_C_ACCEPTED_CHANGED] += chg_n;
        cnt[TPI_C_SELF_LOOPS] += slf_n;
        cnt[TPI_C_UPHILL_ACCEPTED] += up_n;
        if (chg_n > 0) {  // best / feasible best / target, once per class at the clock value where it ends
            bool copied = false;
            double e_viol = 0.0, e_score = E[0];
#pragma unroll
            for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl) {
                if (sl == violation_slot) e_viol = E[sl];
                if (sl == score_slot) e_score = E[sl];
            }
            if (E[0] < best_e - 1e-12 * sc0) {
                best_e = E[0];
                best_a = n;
                for (int i = tid; i < n_spins; i += TPI_CNT) best_spins[(size_t)rep * n_spins + i] = TR(spin_clause[i]);
                copied = true;
            }
            const bool feasible = (violation_slot < 0) || (fabs(e_viol) <= 1e-9 * sc[violation_slot]);
            if (feasible) {
                const double es = e_score;
                if (fb_a < 0 || es < fb_e - 1e-12 * sc[score_slot]) {
                    fb_e = es;
                    fb_a = n;
                    for (int i = tid; i < n_spins; i += TPI_CNT)
                        feasible_best_spins[(size_t)rep * n_spins + i] = TR(spin_clause[i]);
                    copied = true;
                }
                if (!(tgt != tgt) && cnt[TPI_C_TARGET_HIT_ATTEMPT] < 0 && es <= tgt + 1e-9 * sc[score_slot]) {
                    cnt[TPI_C_TARGET_HIT_ATTEMPT] = n;
                    if (stop_on_target) stop_now = true;
                }
            }
            if (copied) TPI_CSYNC();  // readers of T finished before the next class toggles it
            if (stop_now) break;
        }
    }
    if (tid == 0) {
#pragma unroll
        for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl)
            if (sl < n_slots) energies[(size_t)rep * n_slots + sl] = E[sl];
#pragma unroll
        for (int c = 0; c < TPI_N_COUNTERS; ++c) counters[(size_t)rep * TPI_N_COUNTERS + c] = cnt[c];
        best_energy[rep] = best_e;
        best_attempt[rep] = best_a;
        feasible_best_energy[rep] = fb_e;
        feasible_best_attempt[rep] = fb_a;
        next_attempt[rep] = n;
        if (stop_now) stopped[rep] = 1;
    }
    TPI_CSYNC();
    for (size_t k = tid; k < M1; k += TPI_CNT) Tg[k] = Tr[k];
#undef TR
}
