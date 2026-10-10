"""Structured logging on the standard library.

Library modules log *events* with fields and never configure handlers:

    log_event(logger, logging.WARNING, "rf_fix_rejected", chi2=70628.3, dof=2)

Applications (the CLI) call :func:`configure_logging` once, choosing
human-readable ``key=value`` lines or one JSON object per line for machine
consumption (``--log-json``).
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

_FIELDS = "locant_fields"


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log ``event`` with structured ``fields`` (rendered by the formatters below)."""
    if logger.isEnabledFor(level):
        logger.log(level, event, extra={_FIELDS: fields}, stacklevel=2)


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, _FIELDS, None)
    return dict(fields) if isinstance(fields, dict) else {}


def _plain(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str | int | bool) or value is None:
        return value
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return value.tolist()
    return str(value)


class JSONFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, event, then fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update({k: _plain(v) for k, v in _fields(record).items()})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class KeyValueFormatter(logging.Formatter):
    """``HH:MM:SS LEVEL logger event key=value ...`` for terminals."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        fields = " ".join(f"{k}={_plain(v)}" for k, v in _fields(record).items())
        line = f"{stamp} {record.levelname:<7} {record.name} {record.getMessage()}"
        line = f"{line} {fields}" if fields else line
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(
    level: str | int = "INFO", json_lines: bool = False, stream: TextIO | None = None
) -> None:
    """Route ``locant`` loggers to ``stream`` (default stderr).

    Idempotent: replaces handlers installed by a previous call.
    """
    logger = logging.getLogger("locant")
    for handler in list(logger.handlers):
        if getattr(handler, "_locant", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter() if json_lines else KeyValueFormatter())
    handler._locant = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False
