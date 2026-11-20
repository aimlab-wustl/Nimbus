<!-- Hover definitions, appended to every page (pymdownx.snippets auto_append). Keep in step with docs/concepts/glossary.md. -->
*[move neuron]: Inner-layer neuron of one move. Its input is the energy change of the move; when it fires, the clauses on its active connections toggle.
*[move neurons]: Inner-layer neurons, one per move. The input of each is the energy change of its move; when one fires, the clauses on its active connections toggle.
*[transition neuron]: Another name for a move neuron: the inner-layer neuron of one move.
*[transition neurons]: Another name for move neurons: the inner-layer neurons, one per move.
*[clause neuron]: Outer-layer neuron that stores the sign of one product of spins (a clause). The spins themselves are one-spin clauses.
*[clause neurons]: Outer-layer neurons, each storing the sign of one product of spins (a clause). The spins themselves are one-spin clauses.
*[gate clause]: A stored product of spins whose sign sets switches. The compiler adds it when a switch needs a product that is not yet stored.
*[gate clauses]: Stored products of spins whose signs set switches. The compiler adds them when a switch needs a product that is not yet stored.
*[switch-control clause]: A stored product of spins whose sign sets switches (a gate clause).
*[switch-control clauses]: Stored products of spins whose signs set switches (gate clauses).
*[auxiliary clause]: A product of spins that the compiler adds only to control switches (a gate clause); its energy weight is zero.
*[auxiliary clauses]: Products of spins that the compiler adds only to control switches (gate clauses); their energy weight is zero.
*[move library]: The set of moves the machine can propose for a problem. Every move maps each feasible state to another feasible state.
*[move libraries]: Sets of moves the machine can propose. Every move maps each feasible state to another feasible state.
*[self-loop]: An accepted move that changes nothing in the current state, because all its switches are off.
*[self-loops]: Accepted moves that change nothing in the current state, because all their switches are off.
*[iterations to solution]: The expected number of iterations to reach the target with probability 0.99 by independent repetitions: ITS(B) = B ln(0.01) / ln(1 - p).
*[ITS]: Iterations to solution: the expected number of iterations to reach the target with probability 0.99 by independent repetitions.
*[parallel tempering]: Replicas at different temperatures that exchange their states, so that hot replicas help cold ones escape local minima.
*[Fowler-Nordheim]: The inverse-logarithmic annealing schedule tau_n = tau0 / ln(1 + q_n), named after Fowler-Nordheim tunnelling; the schedule of every result in the paper.
*[backbone]: A set of moves that connects all feasible states, so that every feasible state can be reached.
*[shortcut]: A move added to a backbone that joins in one transition states the backbone connects only through several, bypassing energy barriers.
*[shortcuts]: Moves added to a backbone that join in one transition states the backbone connects only through several, bypassing energy barriers.
*[clause output]: The sign, +1 or -1, of a stored clause (a product of spins).
*[clause outputs]: The signs, +1 or -1, of the stored clauses (products of spins).
*[spin transition]: A change from one feasible state to another.
*[spin transitions]: Changes from one feasible state to another.
