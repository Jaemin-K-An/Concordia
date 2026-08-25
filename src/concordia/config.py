from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from concordia.errors import ValidationError


def load_config(path: str) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"config file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValidationError("experiment config must be a mapping")
    required = {
        "scenario",
        "demand_scale",
        "population",
        "policy",
        "navigation_penetration",
        "utility_epsilon",
        "seeds",
    }
    missing = required - set(config)
    if missing:
        raise ValidationError(f"missing config fields: {sorted(missing)}")
    if config["scenario"] not in {"two_route", "braess", "ring", "merge", "signalized"}:
        raise ValidationError("unsupported scenario")
    if config["policy"] not in {"exact", "greedy_vde"}:
        raise ValidationError("policy must be 'exact' or 'greedy_vde'")
    if float(config["demand_scale"]) <= 0:
        raise ValidationError("demand_scale must be positive")
    if not 0 <= float(config["navigation_penetration"]) <= 1:
        raise ValidationError("navigation_penetration must be in [0, 1]")
    if float(config["utility_epsilon"]) < 0:
        raise ValidationError("utility_epsilon cannot be negative")
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValidationError("seeds must be a non-empty list")
    if any(not isinstance(seed, int) or seed < 0 for seed in config["seeds"]):
        raise ValidationError("all seeds must be non-negative integers")
    return config
