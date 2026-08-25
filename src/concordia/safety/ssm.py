from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


def _measure(conflict: ET.Element, tag: str) -> Optional[float]:
    node = conflict.find(f".//{tag}")
    if node is None:
        return None
    raw = node.get("value")
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) and value >= 0 else None


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
                    min_pet=_measure(node, "minPET"),
                    max_drac=_measure(node, "maxDRAC"),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValidationError("SSM conflict is missing required identifiers/times") from exc
    return conflicts
