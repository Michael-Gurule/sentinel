"""Typed errors raised instead of returning silent locant values."""


class LocantError(Exception):
    """Base class for all Locant errors."""


class InsufficientMeasurementsError(LocantError, ValueError):
    """Too few measurements to determine the requested unknowns."""


class GeometryError(LocantError, ValueError):
    """Sensor/emitter geometry makes the problem singular or unobservable."""


class NotPositiveDefiniteError(LocantError, ValueError):
    """A matrix required to be a covariance is not positive definite."""
