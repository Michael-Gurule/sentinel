"""Typed errors raised instead of returning silent sentinel values."""


class SentinelError(Exception):
    """Base class for all SENTINEL errors."""


class InsufficientMeasurementsError(SentinelError, ValueError):
    """Too few measurements to determine the requested unknowns."""


class GeometryError(SentinelError, ValueError):
    """Sensor/emitter geometry makes the problem singular or unobservable."""


class NotPositiveDefiniteError(SentinelError, ValueError):
    """A matrix required to be a covariance is not positive definite."""
