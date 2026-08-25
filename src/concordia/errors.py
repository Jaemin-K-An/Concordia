class ConcordiaError(Exception):
    """Base class for explicit domain errors."""


class ValidationError(ConcordiaError):
    """Raised when an input violates a fail-fast invariant."""


class InfeasibleAssignment(ConcordiaError):
    """Raised when no recommendation satisfies all hard constraints."""


class SimulatorUnavailable(ConcordiaError):
    """Raised when a requested external simulator is not installed."""


class SolverUnavailable(ConcordiaError):
    """Raised when an explicitly requested optimization solver is unavailable."""
