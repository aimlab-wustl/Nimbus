"""Common compiler of the transition-programmable neuromorphic Ising architecture.

``compile_model(structure, library, readout=..., identities=..., limits=..., score_structures=...)``
turns an objective structure and a verified transition library into a ``CompiledModel``: stored
clauses (objective, gate, readout, score), retained connections Pi with their switches alpha, the
groups' spin actions and the dependency routes. ``preflight`` estimates resources without
materializing.
"""

from nimbus.compiler.compile import COMPILER_VERSION, CompileLimits, CompileOptions, compile_model
from nimbus.compiler.preflight import estimate_model_bytes, preflight, row_exchange_B
from nimbus.compiler.table_fit import AffineFitter, domain_identities

__all__ = ["compile_model", "CompileLimits", "CompileOptions", "COMPILER_VERSION", "preflight",
           "estimate_model_bytes", "row_exchange_B", "AffineFitter", "domain_identities"]
