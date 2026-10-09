"""Single source of truth for the OPIR event taxonomy.

Datasets, models, artifacts, and the pipeline all import these names, so class
order cannot drift between them (audit defect C4).
"""

from typing import Final

EVENT_CLASSES: Final[tuple[str, ...]] = (
    "launch",
    "explosion",
    "fire",
    "aircraft",
    "background",
)
"""Class names in label / model-output order."""

BACKGROUND: Final[int] = EVENT_CLASSES.index("background")
"""Label index of the no-event class."""
