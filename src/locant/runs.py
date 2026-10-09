"""A file-based run registry.

Every pipeline run and experiment gets a directory ``<root>/<run id>/`` with a
``run.json`` record: what ran (name, kind, command line), with what
(configuration and its hash, seed, git commit and whether the tree was dirty,
package versions), when and for how long, how it ended, its headline metrics,
and the files it produced. Records are plain JSON so they can be diffed,
grepped, and reviewed; ADR 0002 explains why this replaces MLflow here.

    registry = RunRegistry("runs")
    with registry.start("multi_int", config, seed=0, kind="pipeline") as run:
        ...
        run.log_metrics(gospa_mean_m=989.2)
        run.add_artifact(run.directory / "tracks.json")
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import locant

RECORD = "run.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def config_hash(config: Any) -> str:
    """SHA-256 of the canonical JSON form of ``config``."""
    canonical = json.dumps(_jsonable(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    versions["locant"] = locant.__version__
    for package in ("numpy", "scipy", "torch", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclass
class RunRecord:
    id: str
    name: str
    kind: str
    status: str
    created: str
    config: dict[str, Any]
    config_sha256: str
    seed: int | None = None
    finished: str | None = None
    duration_s: float | None = None
    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


class ActiveRun:
    """A run in progress; its record is written on start and on finish."""

    def __init__(self, record: RunRecord, directory: Path) -> None:
        self.record = record
        self.directory = directory

    @property
    def id(self) -> str:
        return self.record.id

    def log_metrics(
        self, metrics: Mapping[str, Any] | None = None, **more: Any
    ) -> None:
        self.record.metrics.update(_jsonable({**(metrics or {}), **more}))
        self._write()

    def add_artifact(self, path: str | Path) -> Path:
        """Record a produced file (stored relative to the run directory when inside it)."""
        p = Path(path)
        try:
            name = str(p.resolve().relative_to(self.directory.resolve()))
        except ValueError:
            name = str(p)
        if name not in self.record.artifacts:
            self.record.artifacts.append(name)
            self._write()
        return p

    def _write(self) -> None:
        (self.directory / RECORD).write_text(
            json.dumps(asdict(self.record), indent=2, sort_keys=True) + "\n"
        )


class RunRegistry:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def _new_id(self, name: str, digest: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = f"{stamp}-{name}-{digest[:8]}"
        candidate, n = base, 1
        while (self.root / candidate).exists():
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    @contextmanager
    def start(
        self,
        name: str,
        config: Any,
        seed: int | None = None,
        kind: str = "run",
    ) -> Iterator[ActiveRun]:
        """Open a run; it is marked ``completed``, or ``failed`` with the
        error if the block raises (the exception propagates)."""
        digest = config_hash(config)
        run_id = self._new_id(name, digest)
        directory = self.root / run_id
        directory.mkdir(parents=True)
        record = RunRecord(
            id=run_id,
            name=name,
            kind=kind,
            status="running",
            created=datetime.now(UTC).isoformat(timespec="microseconds"),
            config=_jsonable(config),
            config_sha256=digest,
            seed=seed,
            git=git_state(),
            environment=environment(),
            argv=list(sys.argv),
        )
        run = ActiveRun(record, directory)
        run._write()
        started = time.perf_counter()
        try:
            yield run
        except BaseException as exc:
            record.status, record.error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        else:
            record.status = "completed"
        finally:
            record.finished = datetime.now(UTC).isoformat(timespec="seconds")
            record.duration_s = round(time.perf_counter() - started, 3)
            run._write()

    def records(self) -> list[RunRecord]:
        """All runs, oldest first."""
        records = [self._read(path) for path in sorted(self.root.glob(f"*/{RECORD}"))]
        return sorted(records, key=lambda r: (r.created, r.id))

    def load(self, run_id: str) -> RunRecord:
        """A run by id or unique id prefix.

        Raises:
            KeyError: no run, or an ambiguous prefix.
        """
        exact = self.root / run_id / RECORD
        if exact.exists():
            return self._read(exact)
        matches = list(self.root.glob(f"{run_id}*/{RECORD}"))
        if len(matches) != 1:
            raise KeyError(
                f"{'no' if not matches else 'ambiguous'} run matching {run_id!r}"
            )
        return self._read(matches[0])

    @staticmethod
    def _read(path: Path) -> RunRecord:
        return RunRecord(**json.loads(path.read_text()))
