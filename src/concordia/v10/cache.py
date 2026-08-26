from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


class RolloutCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(value: Mapping[str, object]) -> str:
        required = {
            "state_id", "action_id", "stage", "replica",
            "horizon_seconds", "simulator_parameter_hash",
        }
        if not required.issubset(value):
            raise ValueError("rollout cache key is incomplete")
        payload = {name: value[name] for name in sorted(required)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def load(self, value: Mapping[str, object]) -> dict | None:
        path = self.directory / f"{self.key(value)}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def store(self, value: Mapping[str, object], result: Mapping[str, object]) -> Path:
        if bool(value.get("final_realized_evaluation", False)):
            raise ValueError("final realized evaluation must never be cached")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.key(value)}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

