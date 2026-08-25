from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional

from concordia.errors import ValidationError


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def capture_source_state() -> tuple[str, bool]:
    """Capture the source revision before an experiment starts writing artifacts."""
    return _git_commit(), _git_dirty()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _invalid_reasons(value: Any, path: str = "metrics") -> list[str]:
    reasons: list[str] = []
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        if value.get("complete") is False:
            reasons.append(f"incomplete run at {path}")
        for key, child in value.items():
            reasons.extend(_invalid_reasons(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reasons.extend(_invalid_reasons(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        reasons.append(f"non-finite value at {path}")
    return reasons


def _json_safe(value: Any) -> Any:
    """Preserve an invalid artifact while replacing forbidden non-finite JSON values."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ExperimentRegistry:
    """Immutable run registry with validity, environment, and content hashes."""

    def __init__(self, root: str = "artifacts/runs") -> None:
        self.root = Path(root)

    def create(
        self,
        config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        simulator_version: Optional[str] = None,
        input_paths: tuple[str, ...] = (),
        external_output_paths: tuple[str, ...] = (),
        explicit_invalid_reasons: tuple[str, ...] = (),
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        source_commit: Optional[str] = None,
        source_dirty: Optional[bool] = None,
    ) -> Path:
        started = started_at or datetime.now(timezone.utc)
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        short_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        timestamp = started.strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{timestamp}-{short_digest}"
        run_dir = self.root / run_id
        if run_dir.exists():
            raise ValidationError(f"run id collision: {run_id}")
        run_dir.mkdir(parents=True)

        raw_metrics = dict(metrics)
        decisions = raw_metrics.pop("decision_log", [])
        if decisions and not isinstance(decisions, list):
            raise ValidationError("decision_log must be a list")
        reasons = list(explicit_invalid_reasons)
        reasons.extend(_invalid_reasons(raw_metrics))
        metrics_without_decisions = _json_safe(raw_metrics)
        declared_seeds = config.get("seeds", [config.get("seed")])
        if not isinstance(declared_seeds, list) or any(seed is None for seed in declared_seeds):
            reasons.append("missing or malformed seed declaration")

        input_hashes = {}
        for raw_path in input_paths:
            path = Path(raw_path)
            if not path.is_file():
                reasons.append(f"missing input: {raw_path}")
            else:
                input_hashes[raw_path] = _digest(path.read_bytes())

        config_bytes = _json_bytes(config)
        metrics_bytes = _json_bytes(metrics_without_decisions)
        (run_dir / "config.json").write_bytes(config_bytes)
        (run_dir / "metrics.json").write_bytes(metrics_bytes)
        output_hashes = {
            "config.json": _digest(config_bytes),
            "metrics.json": _digest(metrics_bytes),
        }
        if decisions:
            decision_bytes = "".join(
                json.dumps(decision, sort_keys=True, allow_nan=False) + "\n"
                for decision in decisions
            ).encode("utf-8")
            (run_dir / "decisions.jsonl").write_bytes(decision_bytes)
            output_hashes["decisions.jsonl"] = _digest(decision_bytes)

        for raw_path in external_output_paths:
            path = Path(raw_path)
            if not path.is_file():
                reasons.append(f"missing output: {raw_path}")
            else:
                output_hashes[raw_path] = _digest(path.read_bytes())

        ended = ended_at or datetime.now(timezone.utc)
        if ended < started:
            reasons.append("run end timestamp precedes start timestamp")
        manifest = {
            "run_id": run_id,
            "status": "invalid" if reasons else "valid",
            "invalid_reasons": sorted(set(reasons)),
            "start_timestamp_utc": started.isoformat(),
            "end_timestamp_utc": ended.isoformat(),
            "duration_seconds": (ended - started).total_seconds(),
            "git_commit": source_commit if source_commit is not None else _git_commit(),
            "git_dirty": source_dirty if source_dirty is not None else _git_dirty(),
            "python": sys.version,
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor() or "not-reported",
                "logical_cpu_count": os.cpu_count(),
                "platform": platform.platform(),
            },
            "dependencies": {
                package: _version(package)
                for package in ("numpy", "networkx", "PyYAML", "scipy", "traci")
            },
            "solver_version": f"scipy-{_version('scipy')}/HiGHS",
            "rl_library_version": "not-installed-not-used",
            "simulator_version": simulator_version or "analytical-no-sumo",
            "seeds": declared_seeds,
            "input_hashes": {"canonical_config": _digest(config_bytes), **input_hashes},
            "output_hashes": output_hashes,
        }
        (run_dir / "manifest.json").write_bytes(_json_bytes(manifest))
        return run_dir
