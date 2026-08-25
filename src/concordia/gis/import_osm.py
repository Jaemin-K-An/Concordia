from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from concordia.errors import ValidationError
from concordia.gis.provenance import DatasetProvenance, sha256_file
from concordia.models import EdgeData, EdgeKey
from concordia.network import RoadNetwork


_DEFAULT_SPEED_KPH = {
    "motorway": 90.0,
    "trunk": 70.0,
    "primary": 60.0,
    "secondary": 50.0,
    "tertiary": 40.0,
    "residential": 30.0,
    "service": 20.0,
    "unclassified": 30.0,
}
_NON_DRIVABLE = {"footway", "cycleway", "path", "pedestrian", "steps", "bridleway"}


@dataclass(frozen=True)
class ImportedOSMNetwork:
    network: RoadNetwork
    coordinates: Mapping[str, Tuple[float, float]]
    geometries: Mapping[EdgeKey, Tuple[Tuple[float, float], ...]]
    osm_way_ids: Mapping[EdgeKey, str]
    provenance: DatasetProvenance


def _haversine_km(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def _numeric(tag: str, default: float) -> float:
    match = re.search(r"\d+(?:\.\d+)?", tag or "")
    return float(match.group()) if match else default


def import_osm_xml(
    path: str,
    source_url: str,
    retrieval_date: str,
    demand_provenance: str = "synthetic demand on real topology",
) -> ImportedOSMNetwork:
    """Import drivable OSM ways while retaining every supplied geometry vertex."""
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"OSM XML does not exist: {source}")
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValidationError(f"malformed OSM XML: {source}") from exc
    coordinates = {
        node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
        if {"id", "lon", "lat"}.issubset(node.attrib)
    }
    network = RoadNetwork(source.stem)
    geometries: Dict[EdgeKey, Tuple[Tuple[float, float], ...]] = {}
    way_ids: Dict[EdgeKey, str] = {}
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        highway = tags.get("highway")
        if not highway or highway in _NON_DRIVABLE or tags.get("access") in {"no", "private"}:
            continue
        nodes = [node.attrib.get("ref", "") for node in way.findall("nd")]
        if len(nodes) < 2 or any(node not in coordinates for node in nodes):
            continue
        speed_kph = _numeric(tags.get("maxspeed", ""), _DEFAULT_SPEED_KPH.get(highway, 30.0))
        lanes = max(1, round(_numeric(tags.get("lanes", ""), 1.0)))
        capacity = lanes * 1800.0
        oneway = tags.get("oneway", "").lower()
        pairs = list(zip(nodes, nodes[1:]))
        if oneway == "-1":
            pairs = [(target, origin) for origin, target in reversed(pairs)]
            bidirectional = False
        else:
            bidirectional = oneway not in {"yes", "true", "1"} and highway != "motorway"
        for origin, target in pairs:
            length = _haversine_km(coordinates[origin], coordinates[target])
            if length <= 0:
                continue
            edge = (origin, target)
            data = EdgeData(
                free_flow_time=length / speed_kph * 60.0,
                capacity=capacity,
                length=length,
                variability=0.0,
                risk=0.0,
                complexity=0.0,
            )
            network.add_edge(origin, target, data)
            geometries[edge] = (coordinates[origin], coordinates[target])
            way_ids[edge] = way.attrib.get("id", "unknown")
            if bidirectional:
                reverse = (target, origin)
                network.add_edge(target, origin, data)
                geometries[reverse] = tuple(reversed(geometries[edge]))
                way_ids[reverse] = way.attrib.get("id", "unknown")
    if not network.edges:
        raise ValidationError("OSM extract contains no usable drivable edges")
    used_nodes = {node for edge in network.edges for node in edge}
    provenance = DatasetProvenance(
        source_url=source_url,
        retrieval_date=retrieval_date,
        license="OpenStreetMap contributors, ODbL 1.0",
        checksum_sha256=sha256_file(path),
        crs="EPSG:4326",
        units={"length": "km", "free_flow_time": "minutes", "capacity": "vehicles/hour"},
        demand_provenance=demand_provenance,
    )
    return ImportedOSMNetwork(
        network=network,
        coordinates={node: coordinates[node] for node in used_nodes},
        geometries=geometries,
        osm_way_ids=way_ids,
        provenance=provenance,
    )

