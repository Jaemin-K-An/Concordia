from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from concordia.errors import ValidationError


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


class ExperimentRegistry:
    def __init__(self, root: str = "artifacts/runs") -> None:
        self.root = Path(root)

    def create(
        self,
        config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        simulator_version: Optional[str] = None,
    ) -> Path:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{timestamp}-{digest}"
        run_dir = self.root / run_id
        if run_dir.exists():
            raise ValidationError(f"run id collision: {run_id}")
        run_dir.mkdir(parents=True)
        dependencies: Dict[str, str] = {}
        for package in ("numpy", "networkx", "PyYAML"):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = "not-installed"
        manifest = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
            "simulator_version": simulator_version or "analytical-no-sumo",
            "seeds": config.get("seeds", [config.get("seed")]),
        }
        metrics_without_decisions = dict(metrics)
        decisions = metrics_without_decisions.pop("decision_log", [])
        if decisions and not isinstance(decisions, list):
            raise ValidationError("decision_log must be a list")
        for filename, value in (
            ("config.json", config),
            ("metrics.json", metrics_without_decisions),
            ("manifest.json", manifest),
        ):
            with (run_dir / filename).open("x", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        if decisions:
            with (run_dir / "decisions.jsonl").open("x", encoding="utf-8") as handle:
                for decision in decisions:
                    handle.write(json.dumps(decision, sort_keys=True, allow_nan=False) + "\n")
        return run_dir
