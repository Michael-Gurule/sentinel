"""The package layering in docs/architecture.md, enforced.

Each package may import only the packages listed for it. A new dependency
must be added here deliberately (and to the architecture doc), which keeps
the low-level algorithm packages free of pipeline, CLI, or experiment code.
"""

import ast
from collections import defaultdict
from pathlib import Path

import locant

ROOT = Path(locant.__file__).parent

ALLOWED: dict[str, set[str]] = {
    # Foundation
    "core": set(),
    "taxonomy": set(),
    # Algorithms
    "geolocation": {"core"},
    "detection": {"core"},
    "tracking": {"core"},
    "eval": {"core"},
    # Algorithms built on algorithms (classification shares the detector's
    # noise estimator so both stages use the same noise reference)
    "classification": {"core", "detection", "taxonomy"},
    "fusion": {"core", "geolocation", "tracking"},
    # Simulation and data
    "sim": {"core", "geolocation"},
    "data": {"core", "sim", "taxonomy"},
    # Orchestration
    "pipeline": {
        "core",
        "detection",
        "classification",
        "geolocation",
        "tracking",
        "fusion",
        "eval",
        "sim",
    },
    # Interfaces
    "runs": {"locant"},
    "cli": {"locant", "core", "classification", "data", "pipeline", "runs", "sim"},
    "__main__": {"cli"},
    "__init__": set(),
}


def _package(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    return parts[0].removesuffix(".py")


def import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for path in ROOT.rglob("*.py"):
        source = _package(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                if parts[0] != "locant":
                    continue
                target = parts[1] if len(parts) > 1 else "locant"
                if target != source:
                    graph[source].add(target)
    return graph


def test_every_package_is_declared():
    packages = {_package(p) for p in ROOT.rglob("*.py")}
    assert packages <= set(ALLOWED), f"undeclared packages: {packages - set(ALLOWED)}"


def test_imports_follow_the_layering():
    violations = {
        source: sorted(targets - ALLOWED[source])
        for source, targets in import_graph().items()
        if targets - ALLOWED[source]
    }
    assert not violations, f"imports outside the declared layering: {violations}"


def test_algorithm_packages_do_not_depend_on_orchestration():
    upper = {"pipeline", "cli", "runs", "data", "sim"}
    for package in ("core", "geolocation", "detection", "tracking", "eval", "fusion"):
        assert not ALLOWED[package] & upper
