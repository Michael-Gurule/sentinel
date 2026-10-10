"""Every number in the docs comes from the experiment reports.

``reports/metrics.json`` and the generated tables in the README and docs must
match what the reports produce; rerun ``make metrics`` after an experiment.
"""

import json
import shutil

import pytest
from experiments import metrics


def test_docs_and_metrics_match_the_reports():
    assert metrics.stale() == [], "run `make metrics`"


def test_every_document_block_names_a_known_table():
    seen = set()
    for name in metrics.DOCUMENTS:
        path = metrics.ROOT / name
        if path.exists():
            seen |= set(metrics._BLOCK_NAMES.findall(path.read_text()))
    assert seen
    assert seen <= set(metrics.TABLES)


def test_render_replaces_only_marked_blocks():
    data = metrics.collect()
    text = "intro\n<!-- metrics:latency -->\nstale\n<!-- /metrics:latency -->\nend\n"
    rendered = metrics.render(text, data)
    assert rendered.startswith("intro\n<!-- metrics:latency -->\n| Latency |")
    assert rendered.endswith("<!-- /metrics:latency -->\nend\n")
    assert metrics.render(rendered, data) == rendered
    empty = "<!-- metrics:latency --><!-- /metrics:latency -->"
    assert "| Latency |" in metrics.render(empty, data)  # empty blocks are filled too
    with pytest.raises(KeyError, match="nope"):
        metrics.render("<!-- metrics:nope -->\nx\n<!-- /metrics:nope -->", data)


def test_quick_mode_reports_are_rejected(tmp_path):
    for path in metrics.REPORTS.values():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(metrics.ROOT / path, tmp_path / path)
    e5 = tmp_path / metrics.REPORTS["e5"]
    e5.write_text(json.dumps({**json.loads(e5.read_text()), "quick": True}))
    with pytest.raises(ValueError, match="quick"):
        metrics.collect(tmp_path)
