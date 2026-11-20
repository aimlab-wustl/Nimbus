"""Problem adapter interface.

A problem adapter owns input loading, the spin convention, its original objective, constraints,
feasible initialization, independent objective checks and encoding/decoding. It never owns an
annealing loop: every backend executes the compiled architecture produced from
`structure()` + `transition_library()` by `nimbus.compiler.compile_model`.

A new objective under an existing constraint family is *not* new constraint coverage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

Support = tuple  # sorted tuple of spin indices (odd multiplicity), see core.algebra


@dataclass(frozen=True)
class SpinPolynomial:
    """E(s) = const + sum_t c_t prod_{i in support_t} s_i, in the problem's own energy units.

    Under the manuscript convention E = E_const - sum_k J_k T_k, the clause weight is J = -c.
    This object is the problem's independent polynomial (not the compiler's output); it is used for
    direct energy checks and to build the compiler's ObjectiveStructure.
    """

    n_spins: int
    supports: tuple  # tuple[Support, ...], canonical, unique
    coeffs: np.ndarray  # float64[n_terms], c_t
    const: float

    def evaluate(self, spins) -> np.ndarray:
        """Energy of one state (1-D) or many states (2-D, states x N)."""
        s = np.asarray(spins, dtype=np.int64)
        single = s.ndim == 1
        if single:
            s = s[None, :]
        e = np.full(s.shape[0], float(self.const))
        for sup, c in zip(self.supports, self.coeffs):
            if len(sup):
                e += c * np.prod(s[:, list(sup)], axis=1)
            else:
                e += c
        return e[0] if single else e

    def clause_weights(self) -> np.ndarray:
        """J_t = -c_t for E = E_const - sum J T."""
        return -np.asarray(self.coeffs, dtype=np.float64)


@dataclass
class PenaltyFormulation:
    """Unconstrained formulation used by penalty / second-order baselines.

    run: the penalized polynomial driving acceptance (slot 'run');
    objective: original objective of the physical spins (slot 'objective');
    violation: penalty part, zero iff the physical spins are feasible (slot 'violation');
    n_physical: number of physical spins (auxiliaries, if any, follow them);
    ledger: strengths, auxiliary definitions, bounds used, and any reduction record.
    """

    run: SpinPolynomial
    objective: SpinPolynomial
    violation: SpinPolynomial
    n_physical: int
    ledger: dict = field(default_factory=dict)


class Problem(ABC):
    """Base class for problem adapters. Subclasses set `family`, `name`, `n_spins`."""

    family: str = "unspecified"
    name: str = "unnamed"
    n_spins: int = 0
    #: number of weight bindings (e.g. several weight sets on one constraint structure) sharing one topology
    n_instances: int = 1

    # ---- objective and constraints -------------------------------------------------------------
    @abstractmethod
    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        """The problem's own spin-product objective for one instance (weights may vary by instance,
        supports may not)."""

    def structure(self):
        """Compiler input: the objective supports (structural, weight-free)."""
        from nimbus.core.model import ObjectiveStructure

        poly = self.polynomial(0)
        return ObjectiveStructure(n_spins=self.n_spins, terms=list(poly.supports))

    def term_weights(self, instance: int = 0) -> np.ndarray:
        """Clause weights J_t (manuscript sign) in the order of `structure().terms`."""
        return self.polynomial(instance).clause_weights()

    def const(self, instance: int = 0) -> float:
        return float(self.polynomial(instance).const)

    def identities(self) -> Sequence[tuple]:
        """Parity identities prod_{i in S} s_i = sign valid on the feasible set (for gate reuse)."""
        return ()

    @abstractmethod
    def is_feasible(self, spins) -> bool:
        """Independent constraint check of one state."""

    # ---- independent evaluation and conversion -------------------------------------------------
    def energy(self, spins, instance: int = 0) -> float:
        """Direct evaluation of the problem's polynomial (not via compiled clauses)."""
        return float(self.polynomial(instance).evaluate(np.asarray(spins)))

    @abstractmethod
    def native_objective(self, spins, instance: int = 0) -> float:
        """Problem-native objective (assignment cost, negative log-likelihood, ...), computed
        independently of the spin polynomial. Validation only; never an acceptance delta."""

    def energy_to_native(self, energy: float, instance: int = 0) -> float:
        """Affine map from polynomial energy to native units (identity unless overridden)."""
        return float(energy)

    @abstractmethod
    def decode(self, spins) -> dict:
        """Native solution description (tour, codeword bits, sequence, selected set...)."""

    # ---- initialization, transitions -----------------------------------------------------------
    @abstractmethod
    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        """n feasible states (int8, n x N) from the INIT stream, independent of observations and of
        any transmitted/hidden truth."""

    @abstractmethod
    def library_names(self) -> list:
        """Names accepted by `transition_library`."""

    @abstractmethod
    def transition_library(self, name: str, **options):
        """A verified nimbus.transitions.library.TransitionLibrary."""

    def readout_default(self):
        """Readout request passed to the compiler by default (None, 'all' or spin list)."""
        return None

    def enumerate_feasible(self) -> np.ndarray | None:
        """All feasible states for tiny instances (int8, |Omega| x N), else None."""
        return None

    # ---- baselines -----------------------------------------------------------------------------
    def penalty_formulation(self, instance: int = 0, **strengths) -> PenaltyFormulation:
        raise NotImplementedError(f"{type(self).__name__} has no penalty formulation")

    def metadata(self) -> dict[str, Any]:
        return {"family": self.family, "name": self.name, "n_spins": int(self.n_spins),
                "n_instances": int(self.n_instances)}
