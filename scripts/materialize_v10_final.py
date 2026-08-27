#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from concordia.v10.integrity import (
    assert_file_hashes,
    canonical_json_sha256,
    first_primes_at_or_above,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts/v10/freeze_manifest.json"
SEEDS = ROOT / "artifacts/v10/final_seed_manifest.json"
FINAL_RAW = ROOT / "artifacts/studies/v10_micro_holdout/raw_metrics.json"


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def run() -> Path:
    if not FREEZE.is_file():
        raise RuntimeError("v10 freeze manifest is absent")
    if SEEDS.exists() or FINAL_RAW.exists():
        raise RuntimeError("v10 final holdout was already materialized")
    freeze = json.loads(FREEZE.read_text())
    assert_file_hashes(freeze["frozen_file_hashes"], ROOT)
    assert_file_hashes(freeze["implementation_file_hashes"], ROOT)
    local_commit = _git("rev-parse", "HEAD")
    remote_line = _git("ls-remote", "origin", "refs/heads/main")
    remote_commit = remote_line.split()[0] if remote_line else None
    if remote_commit != local_commit:
        raise RuntimeError("local freeze commit is not present at remote main")
    if _git("log", "-1", "--pretty=%s") != "Freeze CONCORDIA v10 policy":
        raise RuntimeError("remote main does not point to the required v10 freeze commit")
    protocol = yaml.safe_load((ROOT / "configs/v10/final_protocol.yaml").read_text())
    seeds = first_primes_at_or_above(5003, 25)
    digest = canonical_json_sha256(seeds)
    if digest != protocol["seed_list_canonical_json_sha256"]:
        raise RuntimeError("v10 final seed commitment mismatch")
    conditions = yaml.safe_load(
        (ROOT / protocol["condition_templates_source"]).read_text()
    )["condition_templates"]
    state_ids = [
        f"V10F::{condition['id']}::s{seed}"
        for seed in seeds
        for condition in conditions
    ]
    value = {
        "study": "CONCORDIA_v10_single_use_untouched_microscopic_holdout",
        "freeze_commit": local_commit,
        "remote_main_commit_verified": remote_commit,
        "seed_generation_rule": protocol["seed_generation_rule"],
        "seed_families": seeds,
        "seed_list_canonical_json_sha256": digest,
        "condition_template_ids": [condition["id"] for condition in conditions],
        "state_ids": state_ids,
        "state_count": len(state_ids),
        "materialized_after_freeze": True,
        "realized_outcomes_present": False,
    }
    SEEDS.parent.mkdir(parents=True, exist_ok=True)
    SEEDS.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(SEEDS)
    return SEEDS


if __name__ == "__main__":
    run()
