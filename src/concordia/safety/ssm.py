from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from concordia.errors import ValidationError


@dataclass(frozen=True)
class SSMConflict:
    ego: str
    foe: str
    begin: float
    end: float
    min_ttc: Optional[float]
    min_pet: Optional[float]
    max_drac: Optional[float]
    max_mdrac: Optional[float]
    conflict_type: str = "unknown"


def _measure(conflict: ET.Element, tag: str) -> Optional[float]:
    node = conflict.find(f".//{tag}")
    if node is None:
        return None
    raw = node.get("value")
    if raw is None:
        return None
    if raw.strip().upper() in {"", "NA", "N/A", "NAN"}:
        return None
    value = float(raw)
    return value if math.isfinite(value) and value >= 0 else None


def _first_measure(conflict: ET.Element, *tags: str) -> Optional[float]:
    for tag in tags:
        value = _measure(conflict, tag)
        if value is not None:
            return value
    return None


def parse_sumo_ssm(path: str) -> List[SSMConflict]:
    """Parse SUMO SSM conflict XML without silently fabricating absent measures."""
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"SSM file does not exist: {source}")
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValidationError(f"malformed SSM XML: {source}") from exc
    conflicts = []
    for node in root.findall(".//conflict"):
        try:
            conflicts.append(
                SSMConflict(
                    ego=node.attrib["ego"],
                    foe=node.attrib["foe"],
                    begin=float(node.attrib["begin"]),
                    end=float(node.attrib["end"]),
                    min_ttc=_measure(node, "minTTC"),
                    min_pet=_first_measure(node, "minPET", "PET"),
                    max_drac=_measure(node, "maxDRAC"),
                    max_mdrac=_measure(node, "maxMDRAC"),
                    conflict_type=(
                        node.get("type") or node.get("conflictType") or "unknown"
                    ),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValidationError("SSM conflict is missing required identifiers/times") from exc
    return conflicts


def summarize_ssm_conflict_types(conflicts: List[SSMConflict]) -> Dict[str, int]:
    """Preserve SUMO's raw type while reporting requested merge/lane-change counts."""
    counts = {"lane_change": 0, "merge": 0, "other_or_unknown": 0}
    for conflict in conflicts:
        normalized = conflict.conflict_type.lower().replace("-", "_")
        if "lane" in normalized and "chang" in normalized:
            counts["lane_change"] += 1
        elif "merg" in normalized:
            counts["merge"] += 1
        else:
            counts["other_or_unknown"] += 1
    return counts
