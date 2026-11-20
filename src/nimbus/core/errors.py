"""Exception types shared across the package."""


class NimbusError(Exception):
    """Base class for package errors."""


class FormatError(NimbusError, ValueError):
    """Malformed or unsupported input file. Message should name the file and line."""


class ConfigError(NimbusError, ValueError):
    """Invalid, unknown or inconsistent configuration fields."""


class UnsupportedTransition(NimbusError):
    """A transformation cannot be compiled into constant or single-product switches, or fails verification."""


class UnsupportedProblem(NimbusError):
    """Input is outside the supported restrictions of its constraint family."""


class ResourceLimitExceeded(NimbusError):
    """A compile/run limit was exceeded. `report` carries the counts reached (never a truncated architecture)."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = dict(report or {})


class BackendUnavailable(NimbusError):
    """The requested backend/device cannot run here. Never silently replaced by another backend."""


class InfeasibleState(NimbusError):
    """A state violates the problem constraints where feasibility is required."""


class IncompatibleCheckpoint(NimbusError):
    """Checkpoint does not match the model/weights/configuration being resumed."""
