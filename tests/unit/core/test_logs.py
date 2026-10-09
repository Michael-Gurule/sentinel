import io
import json
import logging

import numpy as np

from sentinel.core.logs import configure_logging, log_event


def test_json_lines_carry_event_and_fields():
    stream = io.StringIO()
    configure_logging("DEBUG", json_lines=True, stream=stream)
    log_event(
        logging.getLogger("sentinel.test"),
        logging.WARNING,
        "rf_fix_rejected",
        chi2=np.float64(70628.31234567),
        dof=2,
        shape=np.zeros(2),
    )
    record = json.loads(stream.getvalue())
    assert record["event"] == "rf_fix_rejected"
    assert record["level"] == "warning"
    assert record["logger"] == "sentinel.test"
    assert record["chi2"] == 70628.312346
    assert record["dof"] == 2
    assert record["shape"] == [0.0, 0.0]


def test_key_value_lines_and_level_filter():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    configure_logging("INFO", stream=stream)  # idempotent: one handler
    logger = logging.getLogger("sentinel.test")
    log_event(logger, logging.DEBUG, "hidden", x=1)
    log_event(logger, logging.INFO, "frame_processed", tracks=3)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("sentinel.test frame_processed tracks=3")
    logger.info("plain message")
    assert stream.getvalue().splitlines()[-1].endswith("plain message")
