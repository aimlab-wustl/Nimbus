# Internals

The data layouts and the semantics that every backend implements. Read this to write a backend, to check results at
the level of single transactions, or to map the code onto the equations of the paper.

## Energy convention

E(s) = E_const - sum_k J_k T_k(s), with T_k(s) = prod_{i in S_k} s_i the sign of stored clause k. A problem
polynomial E = c_0 + sum_t c_t prod s has J_t = -c_t (`SpinPolynomial.clause_weights`). The input of the transition
neuron of move r is v_r, and accepting r changes the energy by dE = 2 v_r. Spins are +1 (selected, true) or -1.

## Compiled model

`nimbus.compiler.compile_model(structure, library, identities=...)` returns an immutable `CompiledModel`
(`nimbus.core.model`); weights are bound separately (`bind_weights`), so one topology serves many weight sets.

| Quantity | Arrays | Meaning |
|---|---|---|
| stored clauses | `clause_ptr`, `clause_spins` (CSR, sorted supports), `clause_role` | every product the machine stores, with roles OBJECTIVE (weighted), GATE (zero weight, supplies a switch), READOUT (the singleton {i} of spin i), SCORE (baseline score slots only) |
| moves | `n_groups` = R, `group_label` | one transition neuron per move |
| connections | `route_ptr` (CSR by move), `route_wend`, `route_clause` | the clauses a move reads and toggles; weighted connections first in each move |
| switches | `route_ctrl`, `route_pol` | alpha = 1 if `route_ctrl` is -1, else alpha = (1 + pol T[ctrl]) / 2: the connection is open iff the controlling stored clause has the sign pol |
| spin readout | `readout_clause_of_spin` | every spin is a stored singleton clause; spins are read from these clauses |
| dependencies | `clause_wroute_*`, `ctrl_route_*`, `ctrl_wroute_*` | for event-driven updating: the weighted connections of each clause, and the connections each clause controls |
| move descriptors | `act_*`, `cond_*` | the spin map of each move; used only by the oracle backend and by verification |

The compiler derives, for every move and every stored clause the move affects, whether the move toggles the clause
always, never, or exactly when one stored product has a given sign. A required product that is not yet stored is
added as a gate clause, until closure. Products that are equal on the feasible set (through the parity identities
of the problem) share one stored clause. `model.describe()` and the manifest report the counts: objective clauses M,
added gate clauses M_add, readout-only clauses M_read, moves R, connections and memory.

`WeightBinding` holds `J[slot, clause]` and `const[slot]`. Slot 0 (`run`) drives acceptance; penalty runs add the
slots `objective` and `violation` (zero iff feasible).

## Transaction semantics

Iteration n (0-based) of one replica, in the reference `select` process:

1. Draw the move r_n (uniform over the R moves) and a uniform u_n in (0, 1) from the replica's solver stream; take
   tau_n from the schedule.
2. Read the input v_r = sum over the weighted connections of r of J_k alpha_k T_k, from the current clause state
   (recomputed, or maintained by events: the strategies give identical values).
3. Accept iff v_r <= -(tau_n / 2) ln u_n (tau_n > 0), or v_r <= 0 (tau_n = 0).
4. If accepted: latch sigma_k = alpha_k over all connections of r (weighted, gate and readout) from the state
   before the move, then toggle T_k <- -T_k for every latched clause. This is the whole state update: the spins
   change because their singleton clauses toggle. The move changed the state iff a singleton toggled; otherwise it
   was a self-loop.
5. Event-driven updating (emulation, `incremental` strategy): every input that reads a toggled clause is corrected
   with the old switches, then every switch controlled by a toggled clause is re-evaluated and the inputs it gates
   are corrected with the new signs. The result equals a fresh recomputation.
6. Update the energies of every slot (E += 2 sum over latched clauses of J_k T_k before the toggle), the best and
   feasible-best states and the counters.

Rejections and self-loops advance the clock; there is no resampling. Attempt counts in results are n + 1 for the
transaction of iteration n, 0 for the start state and -1 for never. Capacity batching loads the batch of the
proposed move (inputs initialized from the current clause state) and leaves the proposal law and the random draws
unchanged. The other proposal processes ([settings.md](../guides/settings.md#proposal-process)) change only step 1 and the
clock.

## Random streams

Counter-based Philox4x32-10 (Random123-compatible; `nimbus.core.rng` is the reference, the Numba and CUDA versions
reproduce it bit for bit). Key = (seed, stream), counter = (iteration & 0xffffffff, iteration >> 32, replica id,
instance id). One call per iteration gives four 32-bit words:

* move: r = floor(((x0 << 32) | x1) R / 2^64);
* uniform: u = (((x2 << 20) | (x3 >> 12)) + 0.5) 2^-52, strictly inside (0, 1).

| Stream | Id | Use |
|---|---|---|
| `SOLVER` | 1 | move and acceptance draws |
| `INIT` | 2 | feasible starting states |
| `CHANNEL` | 3 | channel noise |
| `PILOT` | 4 | calibration pilots |
| `DIAG` | 5 | diagnostic sampling (never on a solver path) |
| `DATA` | 6 | generated instances |
| `BASELINE` | 7 | baseline-internal randomness |
| `SHARED`, `PT`, `NFOLD`, `SWEEP`, `COLOR` | 8-12 | the proposal processes `shared`, `parallel_trial`, `nfold`, `sweep`, `colored` |

Because every draw is a function of (seed, stream, iteration, replica, instance), a replica's trajectory does not
depend on the backend, the number of threads, the other replicas or where a run was cut into chunks, and a
checkpoint needs only the clause state and the iteration index. The surface-code decoders in `nimbus.decoding` use
the generators of the hardware instead (documented in their modules).

## Backend interface

```python
from nimbus.backends import available_backends, get_backend

print(available_backends())
```

A backend implements `prepare(model, strategy=, layout=, threads=, process=, ...) -> plan`,
`init_state(plan, bindings, spins0, replica_id=, instance_id=, is_feasible=, targets=)`,
`run(plan, state, schedule, n_attempts_total=, start=, stop=, seed=, recording=, stop_on_target=)`,
`export_state` / `load_state` (the `ReplicaStates` snapshot: clause signs, maintained inputs, energies, best
states, counters) and `evaluate_inputs` (inputs without a commit, for pilots and checks). New backends register with
`nimbus.backends.register_backend`. The tests check the emulation, the CPU backend and the oracle against each other
transaction by transaction.
