// anneal.cu -- GPU architecture backend kernels: one source for every variant, strategy and process.
//
// A "replica group" of TPI_LANES threads owns one replica:
//   TPI_LANES == 1           variant "thread": one thread per replica (blockDim.x replicas per block)
//   TPI_LANES == 32          variant "warp":   one warp per replica (blockDim.x / 32 replicas per block)
//   TPI_LANES == blockDim.x  variant "block":  one thread block per replica (64..1024 threads)
// The lanes of a replica stride over the routes of a group: lane l visits route positions
// route_ptr[r] + l + j * TPI_LANES, j = 0, 1, ...
//
// ATTEMPT PROCESSES (proc_kind 0 select, 1 shared, 2 sweep), per attempt n of [attempt_start, attempt_stop):
//   1. the group: select: SOLVER words x0, x1; shared: SHARED stream (n, cohort, instance) words x0, x1;
//      sweep: perm_{n // R}[n % R] (Fisher-Yates of the SWEEP stream, redrawn by lane 0 at each new
//      sweep into perm_buf); the acceptance uniform is always the replica's SOLVER words x2, x3. Every
//      lane computes the same draw itself (identical, deterministic);
//   2. recompute: pass 1 over the weighted routes, alpha = [pol * T[ctrl] > 0] from the PRE-transition
//      signs (constant switches point at the dummy controller T[M] = +1), lane partial v += J0[k] T[k]
//      for alpha = 1 in the lane's route order, then a fixed xor-butterfly over the lanes (the block
//      variant: per warp, then warp 0 over the warp sums), bit-identical in every lane;
//      incremental: v = (double)V[r] * 2^-e_b from the maintained fixed-point inputs (below);
//   3. accept iff v <= xi (tau > 0), v <= 0 (tau == 0);
//   4. accepted: commit (below), attempt count n + 1.
// STEP PROCESSES (proc_kind 3 parallel_trial, 4 nfold), per step starting at clock n < attempt_stop:
//   phase A (lanes over groups r = lane, lane + L, ...): candidate test (state-changing: some
//   spin-singleton route of r open, sr lists), v_r (incremental: from V; recompute: one lane sums the
//   group's weighted routes sequentially in route order) into vbuf; parallel_trial: fired iff
//   v_r <= 0 or (tau > 0 and v_r <= -(tau/2) log u_{n,r}), u from the PT stream, fired bits built by
//   warp ballots (thread variant: in a register) into fbuf; nfold: a_r = 1 / exp(-2 v_r / tau) / 0 into
//   abuf (-1 for non-candidates); candidates counted by a lane reduction;
//   phase B (lane 0, broadcast): parallel_trial: |F| by popcount, arbiter index (PT stream, counter
//   (n, 0xFFFFFFFF, replica, instance)) -> the j-th fired group in ascending r; nfold: sequential
//   prefix sums in ascending r (the CPU's exact order and expressions), first S_r > u_sel A (else the
//   last candidate with a non-empty interval), W = 1 + floor(log(u_w) / log1p(-A/R)), overshoot and
//   freezing rules of the proposal processes; NFOLD stream counter = the replica's step index;
//   then commit, attempt count n + W.
// COMMIT (every process): latch sigma over ALL routes of r from the PRE-transition signs before any
//   toggle -- TPI_LATCH_MARK == 0: per-lane 64-bit register of alpha bits (max routes per group <=
//   64 L); == 1: a mark pass T[k] = 2 T[k] (sign kept, so concurrent controller reads still see the
//   pre-transition sign) -- barrier; incremental only: stage 1/2 on V from the marked state; barrier;
//   commit pass T_k <- -T_k (spin singletons included: no spin array); changed iff a toggled clause
//   is a spin singleton; E[0] += 2 v; E[slot >= 1] += lane-reduced 2 J[slot][k] T_old[k]; best /
//   feasible-best / target bookkeeping computed redundantly by every lane, written by lane 0, readout
//   T[spin_clause[i]] copied by all lanes.
//
// Maintained inputs (TPI_INCREMENTAL): V[r] = sum over weighted routes of Jc[k] alpha T[k] as a 64-bit
// fixed-point integer, Jc[k] = round(J[b][0][k] 2^e_b), |V| < 2^61. Integer atomicAdd is associative, so
// V is identical in every order: deterministic and drift-free. Stage 1: for every marked clause k and
// weighted route (r', k): V[r'] += Jc[k] alpha_old (-2 T_old[k]); stage 2: for every toggled g and
// weighted route (r', k') controlled by g: V[r'] += Jc[k'] T_new[k'] (1 - 2 alpha_old).
//
// Clause signs T: device rows of M~ + 1 int8 (dummy last), replica-major (row per replica), or
// replica-minor (TPI_T_MINOR, thread variant only: element k of replica rep at T[k * n_rep + rep], so the
// replicas of a warp that draw the same group -- the shared process -- read 32 consecutive bytes); in
// shared memory for the chunk when TPI_T_SHARED (major: one row per replica; minor: k * rpb + slot).
// Route record (int2, coalesced): x = clause | (is_spin << 31), y = (controller << 1) | (pol < 0).

#include "common.cuh"

#ifndef TPI_LANES
#define TPI_LANES 1
#endif
#ifndef TPI_MAX_SLOTS
#define TPI_MAX_SLOTS 4
#endif
#ifndef TPI_T_SHARED
#define TPI_T_SHARED 0
#endif
#ifndef TPI_LATCH_MARK
#define TPI_LATCH_MARK 0
#endif
#ifndef TPI_INCREMENTAL
#define TPI_INCREMENTAL 0
#endif
#ifndef TPI_V_SHARED
#define TPI_V_SHARED 0
#endif
#ifndef TPI_T_MINOR
#define TPI_T_MINOR 0
#endif
#ifndef TPI_PERSISTENT
#define TPI_PERSISTENT 0
#endif
#if TPI_INCREMENTAL && !TPI_LATCH_MARK
#error "the incremental strategy needs the mark latch (stage 1/2 read the marked, pre-transition state)"
#endif
#if TPI_T_MINOR && TPI_LANES != 1
#error "the replica-minor clause layout is for the thread variant"
#endif

// counter layout, must match nimbus.backends.gpu.DEVICE_COUNTERS
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

#define TPI_PROC_SELECT 0
#define TPI_PROC_SHARED 1
#define TPI_PROC_SWEEP 2
#define TPI_PROC_PT 3
#define TPI_PROC_NFOLD 4
#define TPI_STREAM_SHARED 8u
#define TPI_STREAM_PT 9u
#define TPI_STREAM_NFOLD 10u
#define TPI_STREAM_SWEEP 11u

#if TPI_LANES == 1
#define TPI_SYNC() ((void)0)
#elif TPI_LANES == 32
#define TPI_SYNC() __syncwarp()
#else
#define TPI_SYNC() __syncthreads()
#endif

// Sum of one double per lane, identical (bit for bit) in every lane. red: block-variant scratch.
__device__ __forceinline__ double tpi_lane_sum(double x, double* red) {
#if TPI_LANES == 1
    (void)red;
    return x;
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = TPI_DADD(x, __shfl_xor_sync(0xffffffffu, x, o));
#if TPI_LANES > 32
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = x;
    __syncthreads();
    if (w == 0) {
        double y = (l < (TPI_LANES >> 5)) ? red[l] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) y = TPI_DADD(y, __shfl_xor_sync(0xffffffffu, y, o));
        if (l == 0) red[32] = y;
    }
    __syncthreads();
    x = red[32];
#else
    (void)red;
#endif
    return x;
#endif
}

// OR of one predicate per lane, identical in every lane (the block form is also a barrier).
__device__ __forceinline__ bool tpi_lane_any(bool p) {
#if TPI_LANES == 1
    return p;
#elif TPI_LANES == 32
    return __any_sync(0xffffffffu, p) != 0;
#else
    return __syncthreads_or(p ? 1 : 0) != 0;
#endif
}

// Lane 0's value in every lane (block form: through shared memory; two barriers).
__device__ __forceinline__ long long tpi_bcast_ll(long long x, long long* slot) {
#if TPI_LANES == 1
    (void)slot;
    return x;
#elif TPI_LANES == 32
    (void)slot;
    return __shfl_sync(0xffffffffu, x, 0);
#else
    if (threadIdx.x == 0) *slot = x;
    __syncthreads();
    x = *slot;
    __syncthreads();
    return x;
#endif
}

// v[r'] += d on the maintained fixed-point inputs (two's-complement wrap: only the final value must be in range)
__device__ __forceinline__ void tpi_vadd(long long* p, unsigned long long d) {
#if TPI_LANES == 1
    *p = (long long)((unsigned long long)*p + d);
#else
    atomicAdd((unsigned long long*)p, d);
#endif
}

// V[rep][r] = sum over the weighted routes of r of Jc[k] * alpha * T[k] (exact integer sum), one thread
// per (replica, group); used at init/load and when the adaptive strategy switches to incremental.
// t_minor: clause k of replica rep at T[k * n_rep + rep], else T[rep * (M + 1) + k].
extern "C" __global__ void tpi_rebuild_v(const long long* __restrict__ route_ptr, const long long* __restrict__ route_wend,
                                         const int2* __restrict__ rec, const long long* __restrict__ Jc,
                                         const signed char* __restrict__ T, long long* __restrict__ vint,
                                         const int* __restrict__ replica_binding, int n_clauses, int n_groups, int n_rep,
                                         int t_minor) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n_rep * n_groups) return;
    const long long rep = i / n_groups;
    const int r = (int)(i % n_groups);
    const size_t st = t_minor ? (size_t)n_rep : (size_t)1;
    const signed char* Tr = t_minor ? T + rep : T + (size_t)rep * ((size_t)n_clauses + 1);
    const long long* Jcb = Jc + (size_t)replica_binding[rep] * n_clauses;
    unsigned long long s = 0ull;
    for (long long p = route_ptr[r]; p < route_wend[r]; ++p) {
        const int2 q = rec[p];
        const int tc = (int)Tr[(size_t)(q.y >> 1) * st];
        if (((q.y & 1) ? -tc : tc) > 0) {
            const int k = q.x & 0x7fffffff;
            s += (unsigned long long)((long long)Tr[(size_t)k * st] * Jcb[k]);
        }
    }
    vint[i] = (long long)s;
}

extern "C" __global__ void tpi_anneal_recompute(
    // ---- device-resident topology (read-only, shared by every replica) ----
    const long long* __restrict__ route_ptr, const long long* __restrict__ route_wend,
    const long long* __restrict__ jw_off, const int2* __restrict__ rec,
    const int* __restrict__ spin_clause,  // [n_spins]: singleton clause of spin i (readout)
    int n_spins, int n_clauses, int n_groups, int n_slots, int score_slot, int violation_slot,
    // ---- per-binding weights ----
    const double* __restrict__ J, const double* __restrict__ Jw, int use_jw, long long jw_stride,
    // ---- per-replica state (read-write); T rows have n_clauses + 1 entries (dummy last) ----
    signed char* __restrict__ T, double* __restrict__ energies,
    double* __restrict__ best_energy, signed char* __restrict__ best_spins, long long* __restrict__ best_attempt,
    double* __restrict__ feasible_best_energy, signed char* __restrict__ feasible_best_spins,
    long long* __restrict__ feasible_best_attempt,
    long long* __restrict__ counters, long long* __restrict__ next_attempt,
    const double* __restrict__ targets, unsigned char* __restrict__ stopped,
    const int* __restrict__ replica_binding, const unsigned int* __restrict__ replica_id,
    const unsigned int* __restrict__ instance_id, const double* __restrict__ scale,  // [n_rep, n_slots]
    // ---- trace buffer (validation only; trace_cap == 0 disables it) ----
    long long* __restrict__ trace_attempt, int* __restrict__ trace_replica, int* __restrict__ trace_group,
    double* __restrict__ trace_u, double* __restrict__ trace_v, double* __restrict__ trace_xi,
    unsigned char* __restrict__ trace_accepted, unsigned char* __restrict__ trace_changed, long long trace_cap,
    unsigned long long* __restrict__ trace_count,
    // ---- run parameters ----
    unsigned int seed, int sched_kind, double p0, double p1, double p2, double p3, long long n_total,
    long long attempt_start, long long attempt_stop, int stop_on_target, int n_rep, long long t_smem_stride,
    // ---- incremental strategy (unused by recompute): maintained fixed-point inputs and dependency lists ----
    long long* __restrict__ vint, const long long* __restrict__ Jc, const double* __restrict__ vscale,
    const long long* __restrict__ cw_ptr, const int2* __restrict__ cw_rec,  // clause -> its weighted routes {r', y}
    const long long* __restrict__ cc_ptr, const int2* __restrict__ cc_rec,  // controller -> weighted routes {r', k'|neg<<31}
    long long v_smem_base,
    // ---- proposal process ----
    int proc_kind, const unsigned int* __restrict__ cohort, int pt_exclude, int nfold_attempts_clock,
    int* __restrict__ perm_buf,                                         // [n_rep][R] sweep permutations
    const long long* __restrict__ sr_ptr, const int* __restrict__ sr_y,  // group -> its singleton routes' y
    double* __restrict__ vbuf, double* __restrict__ abuf, unsigned int* __restrict__ fbuf, int fwords,
    // ---- persistent (queue) scheduling: a global atomic cursor over [0, n_rep), zeroed before launch.
    // TPI_PERSISTENT == 0: unused (every worker owns exactly its static blockIdx-derived replica). ----
    long long* __restrict__ work_counter) {
    extern __shared__ __align__(16) unsigned char tpi_smem[];
#if TPI_LANES > 32
    __shared__ double red[33];
    __shared__ long long bslot;
#else
    double* red = nullptr;
    long long* bslot_p = nullptr;
#endif
#if TPI_LANES > 32
    long long* bslot_p = &bslot;
#endif
    const int lane = (TPI_LANES == 1) ? 0 : (int)(threadIdx.x % TPI_LANES);
    const int rib = (int)(threadIdx.x / TPI_LANES);  // replica slot within the block
    const int rpb = (int)(blockDim.x / TPI_LANES);
    (void)attempt_start;
    // Persistent (queue) scheduling: this worker (one replica group of TPI_LANES lanes) repeatedly claims
    // the next unfinished replica index from a global atomic cursor instead of owning a fixed
    // blockIdx-derived replica for the whole launch, so a replica that stops early within this window
    // (budget end, stop_on_target hit, or an nfold freeze) frees the worker for the next replica instead
    // of leaving it idle until every other worker finishes. The claim is uniform across the lanes of a
    // replica group (tpi_bcast_ll is the same primitive already used for phase-B decisions below). This
    // only changes WHICH worker executes a replica and WHEN; the transaction executed for a given replica
    // index is identical to the static mapping (same seed, replica_id/instance_id, same per-attempt code
    // below), so results are bit-identical to TPI_PERSISTENT == 0.
#if TPI_PERSISTENT
    long long rep;
    {
        long long nxt = 0;
        if (lane == 0) nxt = (long long)atomicAdd((unsigned long long*)work_counter, 1ull);
        rep = tpi_bcast_ll(nxt, bslot_p);
    }
    for (;;) {
    if (rep >= n_rep) return;
#else
    (void)work_counter;
    const long long rep = (long long)blockIdx.x * rpb + rib;
    if (rep >= n_rep) return;
#endif
    if (!stopped[rep]) {

    const size_t M1 = (size_t)n_clauses + 1;
    (void)M1;
#if TPI_T_MINOR
    const size_t tg_st = (size_t)n_rep;
    signed char* Tg = T + rep;
#else
    const size_t tg_st = 1;
    signed char* Tg = T + (size_t)rep * M1;
#endif
#if TPI_T_SHARED
#if TPI_T_MINOR
    const size_t tst = (size_t)rpb;
    signed char* Tr = (signed char*)tpi_smem + rib;
#else
    const size_t tst = 1;
    signed char* Tr = (signed char*)tpi_smem + (size_t)rib * (size_t)t_smem_stride;
#endif
    for (size_t k = lane; k < M1; k += TPI_LANES) Tr[k * tst] = Tg[k * tg_st];
    TPI_SYNC();
#else
    (void)t_smem_stride;
    const size_t tst = tg_st;
    signed char* Tr = Tg;
#endif
#define TR(k) Tr[(size_t)(k) * tst]
    auto alpha_of = [&](int y) -> int {
        const int tc = (int)TR(y >> 1);
        return ((y & 1) ? -tc : tc) > 0;  // sign test: also correct on a marked (doubled) controller
    };

    const int b = replica_binding[rep];
    const double* Jb = J + (size_t)b * n_slots * n_clauses;
    const unsigned int rid = replica_id[rep], iid = instance_id[rep], cid = cohort[rep];
    const double* sc = scale + (size_t)rep * n_slots;
    const double sc0 = sc[0];
    const double tgt = targets[rep];
#if TPI_INCREMENTAL
    long long* Vg = vint + (size_t)rep * n_groups;
#if TPI_V_SHARED
    long long* Vr = (long long*)(tpi_smem + v_smem_base) + (size_t)rib * n_groups;
    for (int g = lane; g < n_groups; g += TPI_LANES) Vr[g] = Vg[g];
    TPI_SYNC();
#else
    long long* Vr = Vg;
#endif
    const long long* Jcb = Jc + (size_t)b * n_clauses;
    const double vsc = vscale[b];
    (void)Jw; (void)use_jw; (void)jw_stride; (void)jw_off;
#else
    const double* Jwb = Jw + (size_t)b * jw_stride;
    (void)vint; (void)Jc; (void)vscale; (void)cw_ptr; (void)cw_rec; (void)cc_ptr; (void)cc_rec; (void)v_smem_base;
#endif
    // per-replica scalars in registers (every loop over slots/counters is unrolled to a constant bound)
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

    // ---- commit of group r (latched sigma, optional stage 1/2, toggles, energies); returns "changed".
    // Mask latch: amask/t0/j0 continue the latch of a pass 1 (or start fresh: 0, rb0 + lane, 0). ----
    auto commit = [&](unsigned int r, double v, unsigned long long amask, long long t0, int j0) -> bool {
        const long long rb0 = route_ptr[r], rb1 = route_ptr[r + 1];
        double dEp[TPI_MAX_SLOTS];
#pragma unroll
        for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl) dEp[sl] = 0.0;
        bool spin = false;
        long long t;
#if !TPI_LATCH_MARK
        int j = j0;
        for (t = t0; t < rb1; t += TPI_LANES) {  // latch the remaining routes (still the pre-transition state)
            amask |= (unsigned long long)alpha_of(rec[t].y) << j;
            ++j;
        }
        TPI_SYNC();  // every lane's sigma is latched before any toggle
        j = 0;
        for (t = rb0 + lane; t < rb1; t += TPI_LANES, ++j) {
            if ((amask >> j) & 1ull) {
                const int x = rec[t].x;
                const int k = x & 0x7fffffff;
                const signed char told = TR(k);
#pragma unroll
                for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl)
                    if (sl < n_slots)
                        dEp[sl] = TPI_DADD(dEp[sl], TPI_DMUL(2.0, TPI_DMUL(Jb[(size_t)sl * n_clauses + k], (double)told)));
                TR(k) = (signed char)(-told);  // T <- (1 - 2 sigma) T, singletons included
                spin |= (x < 0);
            }
        }
#else
        (void)amask; (void)t0; (void)j0;
        for (t = rb0 + lane; t < rb1; t += TPI_LANES) {  // mark pass: sigma recorded in place (sign kept)
            const int2 q = rec[t];
            if (alpha_of(q.y)) {
                const int k = q.x & 0x7fffffff;
                TR(k) = (signed char)(2 * (int)TR(k));
            }
        }
        TPI_SYNC();  // every mark is placed before any toggle
#if TPI_INCREMENTAL
        for (t = rb0 + lane; t < rb1; t += TPI_LANES) {  // stage 1 + 2 from the marked (pre-transition) state
            const int k = rec[t].x & 0x7fffffff;
            const int tv = (int)TR(k);
            if (tv != 2 && tv != -2) continue;
            const int told = tv / 2;
            const long long jk = Jcb[k];
            if (jk != 0) {
                const unsigned long long d1 = (unsigned long long)(-2LL * told) * (unsigned long long)jk;
                for (long long q = cw_ptr[k]; q < cw_ptr[k + 1]; ++q) {
                    const int2 e = cw_rec[q];
                    if (alpha_of(e.y)) tpi_vadd(&Vr[e.x], d1);
                }
            }
            for (long long q = cc_ptr[k]; q < cc_ptr[k + 1]; ++q) {
                const int2 e = cc_rec[q];
                const int k2 = e.y & 0x7fffffff;
                const int a_old = (((e.y < 0) ? -told : told) > 0);
                const int t2 = (int)TR(k2);
                const int tnew = (t2 == 2 || t2 == -2) ? -(t2 / 2) : t2;
                const long long j2 = Jcb[k2];
                if (j2 != 0)  // (alpha_new - alpha_old) = 1 - 2 alpha_old
                    tpi_vadd(&Vr[e.x], (unsigned long long)((long long)(tnew * (1 - 2 * a_old))) * (unsigned long long)j2);
            }
        }
        TPI_SYNC();  // every update read the marked state before the commit pass changes it
#endif
        for (t = rb0 + lane; t < rb1; t += TPI_LANES) {  // commit pass: toggle every marked clause
            const int x = rec[t].x;
            const int k = x & 0x7fffffff;
            const int tv = (int)TR(k);
            if (tv == 2 || tv == -2) {
                const signed char told = (signed char)(tv / 2);
#pragma unroll
                for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl)
                    if (sl < n_slots)
                        dEp[sl] = TPI_DADD(dEp[sl], TPI_DMUL(2.0, TPI_DMUL(Jb[(size_t)sl * n_clauses + k], (double)told)));
                TR(k) = (signed char)(-told);
                spin |= (x < 0);
            }
        }
#endif
        const bool changed = tpi_lane_any(spin);
        TPI_SYNC();  // toggles visible to every lane (readout, next attempt)
        E[0] = TPI_DADD(E[0], TPI_DMUL(2.0, v));
#pragma unroll
        for (int sl = 1; sl < TPI_MAX_SLOTS; ++sl)
            if (sl < n_slots) E[sl] = TPI_DADD(E[sl], tpi_lane_sum(dEp[sl], red));
        return changed;
    };

    // ---- bookkeeping after a commit reached attempt count `count`; returns true to stop the replica ----
    auto book = [&](bool changed, double v, long long count) -> bool {
        if (!changed) {
            cnt[TPI_C_SELF_LOOPS]++;
            return false;
        }
        cnt[TPI_C_ACCEPTED_CHANGED]++;
        if (v > 1e-12 * sc0) cnt[TPI_C_UPHILL_ACCEPTED]++;
        double e_viol = 0.0, e_score = E[0];
#pragma unroll
        for (int sl = 0; sl < TPI_MAX_SLOTS; ++sl) {
            if (sl == violation_slot) e_viol = E[sl];
            if (sl == score_slot) e_score = E[sl];
        }
        if (E[0] < best_e - 1e-12 * sc0) {
            best_e = E[0];
            best_a = count;
            for (int i = lane; i < n_spins; i += TPI_LANES) best_spins[(size_t)rep * n_spins + i] = TR(spin_clause[i]);
        }
        const bool feasible = (violation_slot < 0) || (fabs(e_viol) <= 1e-9 * sc[violation_slot]);
        if (feasible) {
            const double es = e_score;
            if (fb_a < 0 || es < fb_e - 1e-12 * sc[score_slot]) {
                fb_e = es;
                fb_a = count;
                for (int i = lane; i < n_spins; i += TPI_LANES)
                    feasible_best_spins[(size_t)rep * n_spins + i] = TR(spin_clause[i]);
            }
            if (!(tgt != tgt) && cnt[TPI_C_TARGET_HIT_ATTEMPT] < 0 && es <= tgt + 1e-9 * sc[score_slot]) {
                cnt[TPI_C_TARGET_HIT_ATTEMPT] = count;
                if (stop_on_target) return true;
            }
        }
        return false;
    };

    const unsigned int R = (unsigned int)n_groups;
    if (proc_kind <= TPI_PROC_SWEEP) {
        // ========================== attempt processes: select / shared / sweep ==========================
        int* perm = perm_buf + (size_t)rep * n_groups;
        long long cur_sweep = -1;
        for (; n < attempt_stop; ++n) {
            const tpi_draw d = tpi_solver_draw(seed, (unsigned long long)n, rid, iid, R);
            unsigned int r = d.r;
            if (proc_kind == TPI_PROC_SHARED) {
                const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)((unsigned long long)n & 0xffffffffull),
                                                      (unsigned int)((unsigned long long)n >> 32), cid, iid, seed,
                                                      TPI_STREAM_SHARED);
                r = tpi_group_index(x.x0, x.x1, R);
            } else if (proc_kind == TPI_PROC_SWEEP) {
                const long long s = n / (long long)R;
                if (s != cur_sweep) {  // Fisher-Yates of sweep s (lane 0), words (2t, 2t+1) of the SWEEP stream
                    // Barrier BEFORE lane 0 rewrites perm: with several warps per replica ("block" variant) the other
                    // lanes may still be reading perm[last] of the previous sweep; a corrupted read gives lanes of one
                    // replica different groups -> divergent barriers -> cudaErrorIllegalInstruction (observed in the
                    // sweep process).
                    TPI_SYNC();
                    if (lane == 0) {
                        for (unsigned int i = 0; i < R; ++i) perm[i] = (int)i;
                        tpi_u32x4 x;
                        for (unsigned int t = 0; t + 1 < R; ++t) {
                            if ((t & 1u) == 0u)
                                x = tpi_philox4x32_10((unsigned int)s, t >> 1, cid, iid, seed, TPI_STREAM_SWEEP);
                            const unsigned int i = R - 1u - t;
                            const unsigned int jj = (t & 1u) ? tpi_group_index(x.x2, x.x3, i + 1u)
                                                             : tpi_group_index(x.x0, x.x1, i + 1u);
                            const int tmp = perm[i];
                            perm[i] = perm[jj];
                            perm[jj] = tmp;
                        }
                    }
                    cur_sweep = s;
#if TPI_LANES > 32
                    __syncthreads();
#else
                    TPI_SYNC();
#endif
                }
                r = (unsigned int)perm[n - s * (long long)R];
            }
            const double tau = tpi_schedule_tau(sched_kind, p0, p1, p2, p3, n_total, n);
            const long long rb0 = route_ptr[r];
            unsigned long long amask = 0ull;
            int j = 0;
            long long t = rb0 + lane;
#if TPI_INCREMENTAL
            const double v = TPI_DMUL((double)Vr[r], vsc);  // maintained input, identical in every lane
#else
            const long long rbw = route_wend[r];
            const long long jo = use_jw ? jw_off[r] : 0;
            double vp = 0.0;  // pass 1: weighted routes, pre-transition alphas, lane-partial v
            for (; t < rbw; t += TPI_LANES) {
                const int2 q = rec[t];
                const int al = alpha_of(q.y);
#if !TPI_LATCH_MARK
                amask |= (unsigned long long)al << j;
                ++j;
#endif
                if (al) {
                    const int k = q.x & 0x7fffffff;
                    const double Jk = use_jw ? Jwb[jo + t] : Jb[k];
                    vp = TPI_DADD(vp, TPI_DMUL(Jk, (double)TR(k)));
                }
            }
            const double v = tpi_lane_sum(vp, red);
#endif
            const double xi = tpi_metropolis_threshold(tau, d.u);
            const bool accepted = (tau == 0.0) ? (v <= 0.0) : (v <= xi);
            cnt[TPI_C_ATTEMPTS]++;
            cnt[TPI_C_STEPS]++;
            cnt[TPI_C_CAND]++;
            long long my_ti = -1;
            if (trace_cap > 0 && lane == 0) {
                const unsigned long long ti = atomicAdd(trace_count, 1ull);
                if (ti < (unsigned long long)trace_cap) {
                    my_ti = (long long)ti;
                    trace_attempt[my_ti] = n;
                    trace_replica[my_ti] = (int)rep;
                    trace_group[my_ti] = (int)r;
                    trace_u[my_ti] = d.u;
                    trace_v[my_ti] = v;
                    trace_xi[my_ti] = xi;
                    trace_accepted[my_ti] = accepted ? 1 : 0;
                    trace_changed[my_ti] = 0;
                }
            }
            if (!accepted) {
                cnt[TPI_C_REJECTED]++;
                continue;
            }
            cnt[TPI_C_ACCEPTED]++;
#if TPI_INCREMENTAL
            t = rb0 + lane;
#endif
            const bool changed = commit(r, v, amask, t, j);
            if (my_ti >= 0) trace_changed[my_ti] = changed ? 1 : 0;
            if (book(changed, v, n + 1)) {
                stop_now = true;
                ++n;
                break;
            }
        }
    } else {
        // ========================== step processes: parallel_trial / nfold ==========================
        const bool is_pt = proc_kind == TPI_PROC_PT;
        const bool need_sl = (!is_pt) || pt_exclude;
        double* vb = vbuf + (size_t)rep * n_groups;
        double* ab = abuf + (size_t)rep * n_groups;
        unsigned int* fb = fbuf + (size_t)rep * fwords;
        long long kstep = cnt[TPI_C_STEPS];
        while (n < attempt_stop) {
            const double tau = tpi_schedule_tau(sched_kind, p0, p1, p2, p3, n_total, n);
            // ---- phase A: every group, lanes in parallel (uniform trip count: base over 0, L, 2L, ...) ----
            double n_eval_p = 0.0;
#if TPI_LANES == 1
            unsigned int word = 0u;
#endif
            for (unsigned int base = 0; base < R; base += TPI_LANES) {
                const unsigned int r = base + (unsigned int)lane;
                bool fired = false;
                if (r < R) {
                    bool cand = true;
                    if (need_sl) {
                        cand = false;
                        for (long long q = sr_ptr[r]; q < sr_ptr[r + 1]; ++q)
                            if (alpha_of(sr_y[q])) {
                                cand = true;
                                break;
                            }
                    }
                    double v = 0.0;
                    if (cand) {
                        n_eval_p += 1.0;
#if TPI_INCREMENTAL
                        v = TPI_DMUL((double)Vr[r], vsc);
#else
                        for (long long p = route_ptr[r]; p < route_wend[r]; ++p) {  // sequential route order
                            const int2 q = rec[p];
                            if (alpha_of(q.y)) {
                                const int k = q.x & 0x7fffffff;
                                v = TPI_DADD(v, TPI_DMUL(Jb[k], (double)TR(k)));
                            }
                        }
#endif
                        vb[r] = v;
                    }
                    if (is_pt) {
                        if (cand) {
                            if (v <= 0.0) {
                                fired = true;
                            } else if (tau > 0.0) {
                                const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)n, r, rid, iid, seed, TPI_STREAM_PT);
                                fired = v <= tpi_metropolis_threshold(tau, tpi_uniform52(x.x0, x.x1));
                            }
                        }
                    } else {
                        double a = -1.0;  // non-candidate marker
                        if (cand) a = (v <= 0.0) ? 1.0 : ((tau > 0.0) ? exp(TPI_DDIV(TPI_DMUL(-2.0, v), tau)) : 0.0);
                        ab[r] = a;
                    }
                }
                if (is_pt) {
#if TPI_LANES == 1
                    if (fired) word |= 1u << (r & 31u);
                    if ((r & 31u) == 31u || r + 1u == R) {
                        fb[r >> 5] = word;
                        word = 0u;
                    }
#else
                    const unsigned int bal = __ballot_sync(0xffffffffu, fired);
                    if ((lane & 31) == 0) {
                        const unsigned int w = (base + (unsigned int)(lane & ~31)) >> 5;
                        if ((int)w < fwords) fb[w] = bal;
                    }
#endif
                }
            }
            const long long n_eval = (long long)tpi_lane_sum(n_eval_p, red);
#if TPI_LANES > 32
            __syncthreads();  // phase A writes (vbuf/abuf/fbuf) visible to lane 0
#else
            TPI_SYNC();
#endif
            // ---- phase B: lane 0 decides (outcome code, group, W), broadcast ----
            // outcome: 0 commit, 1 nothing fired, 2 frozen, 3 budget overshoot
            long long dec_out = 0, dec_r = -1, dec_w = 1;
            if (lane == 0) {
                if (is_pt) {
                    long long nf = 0;
                    for (int w = 0; w < fwords; ++w) nf += __popc(fb[w]);
                    if (nf == 0) {
                        dec_out = 1;
                    } else {
                        const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)n, 0xFFFFFFFFu, rid, iid, seed, TPI_STREAM_PT);
                        long long jsel = (long long)tpi_group_index(x.x0, x.x1, (unsigned int)nf);
                        for (int w = 0; w < fwords; ++w) {
                            unsigned int bits = fb[w];
                            const int c = __popc(bits);
                            if (jsel >= c) {
                                jsel -= c;
                                continue;
                            }
                            for (; jsel > 0; --jsel) bits &= bits - 1u;  // drop the lowest set bits
                            dec_r = (long long)w * 32 + (__ffs(bits) - 1);
                            break;
                        }
                    }
                } else {
                    double A = 0.0;
                    long long m = 0;
                    for (unsigned int r = 0; r < R; ++r) {
                        const double a = ab[r];
                        if (a < 0.0) continue;
                        A = TPI_DADD(A, a);
                        ++m;
                    }
                    if (m == 0 || A == 0.0) {
                        dec_out = 2;
                    } else {
                        const tpi_u32x4 x = tpi_philox4x32_10((unsigned int)kstep, 0u, rid, iid, seed, TPI_STREAM_NFOLD);
                        const double u_sel = tpi_uniform52(x.x0, x.x1), u_w = tpi_uniform52(x.x2, x.x3);
                        const double target = TPI_DMUL(u_sel, A);
                        double S = 0.0, S_prev = 0.0;
                        long long last_inc = -1, first = -1;
                        for (unsigned int r = 0; r < R; ++r) {  // the same sequential sums as the first pass
                            const double a = ab[r];
                            if (a < 0.0) continue;
                            S = TPI_DADD(S, a);
                            if (first < 0) first = r;
                            if (S > target) {
                                dec_r = r;
                                break;
                            }
                            if (S > S_prev || last_inc < 0) last_inc = r;  // (first candidate: index 0 fallback)
                            S_prev = S;
                        }
                        if (dec_r < 0) dec_r = last_inc;
                        if (nfold_attempts_clock) {
                            const double pr = TPI_DDIV(A, (double)R);
                            if (!(pr >= 1.0)) {
                                const double q = TPI_DDIV(log(u_w), log1p(-pr));
                                if (q >= (double)(n_total - n)) {
                                    dec_out = 3;
                                    dec_w = n_total - n;
                                } else {
                                    dec_w = (long long)floor(q) + 1;
                                }
                            }
                        }
                    }
                }
            }
            dec_out = tpi_bcast_ll(dec_out, bslot_p);
            dec_r = tpi_bcast_ll(dec_r, bslot_p);
            dec_w = tpi_bcast_ll(dec_w, bslot_p);
            ++kstep;
            cnt[TPI_C_STEPS]++;
            cnt[TPI_C_CAND] += n_eval;
            if (dec_out == 2 || dec_out == 3) {
                cnt[TPI_C_ATTEMPTS] += n_total - n;
                cnt[TPI_C_REJECTED] += n_total - n;
                if (dec_out == 2) cnt[TPI_C_FROZEN] = n;
                n = n_total;
                break;
            }
            cnt[TPI_C_ATTEMPTS] += dec_w;
            if (dec_out == 1) {
                cnt[TPI_C_REJECTED]++;
                n += 1;
                continue;
            }
            const unsigned int r = (unsigned int)dec_r;
            const double v = vb[r];
            const bool changed = commit(r, v, 0ull, route_ptr[r] + lane, 0);
            cnt[TPI_C_ACCEPTED]++;
            cnt[TPI_C_REJECTED] += dec_w - 1;
            n += dec_w;
            if (book(changed, v, n)) {
                stop_now = true;
                break;
            }
        }
    }

    if (lane == 0) {
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
#if TPI_T_SHARED
    TPI_SYNC();
    for (size_t k = lane; k < M1; k += TPI_LANES) Tg[k * tg_st] = Tr[k * tst];
#endif
#if TPI_INCREMENTAL && TPI_V_SHARED
    TPI_SYNC();
    for (int g = lane; g < n_groups; g += TPI_LANES) Vg[g] = Vr[g];
#endif
    }  // !stopped[rep]
#if TPI_PERSISTENT
    {
        long long nxt = 0;
        if (lane == 0) nxt = (long long)atomicAdd((unsigned long long*)work_counter, 1ull);
        rep = tpi_bcast_ll(nxt, bslot_p);
    }
    }  // for (;;)
#endif
#undef TR
}
