"""Architecture backends.

``get_backend(name)`` returns a backend by name: ``"oracle"`` (direct recomputation, validation
only), ``"emulation"`` (event-driven neuromorphic emulation), ``"cpu"`` and ``"gpu"`` (optimized
backends, imported on first use). Importing this package does not import Numba or CuPy.
"""

from nimbus.backends.base import (
    COUNTER_NAMES,
    PROCESS_COUNTER_NAMES,
    PROCESSES,
    ArchitectureValidationError,
    Backend,
    BackendPlan,
    ChunkResult,
    ProcessSpec,
    ReplicaStates,
    RunResult,
    StepRecord,
    available_backends,
    get_backend,
    register_backend,
)

__all__ = [
    "COUNTER_NAMES", "PROCESS_COUNTER_NAMES", "PROCESSES", "ProcessSpec", "ArchitectureValidationError", "Backend", "BackendPlan", "ChunkResult", "ReplicaStates",
    "RunResult", "StepRecord", "available_backends", "get_backend", "register_backend",
]
