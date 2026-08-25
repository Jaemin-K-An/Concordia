from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from concordia.errors import ValidationError
from concordia.models import EdgeKey
from concordia.network import RoadNetwork


def export_edge_geojson(
    network: RoadNetwork,
    coordinates: Mapping[str, Tuple[float, float]],
    output: str,
    metrics: Optional[Mapping[EdgeKey, Mapping[str, float]]] = None,
    geometries: Optional[Mapping[EdgeKey, Sequence[Tuple[float, float]]]] = None,
    provenance_source: str = "concordia_synthetic_network",
    crs: str = "EPSG:4326",
) -> Path:
    """Reproducibly export a QGIS-readable edge layer and provenance manifest."""
    if crs != "EPSG:4326":
        raise ValidationError("GeoJSON coordinates must be exported as EPSG:4326")
    missing = set(network.graph.nodes) - set(coordinates)
    if missing:
        raise ValidationError(f"missing coordinates for nodes: {sorted(missing)}")
    metrics = metrics or {}
    features = []
    for source, target in sorted(network.edges):
        data = network.edge_data((source, target))
        properties = {
            "source": source,
            "target": target,
            "free_flow_min": data.free_flow_time,
            "capacity_vph": data.capacity,
            "risk_index": data.risk,
            "complexity": data.complexity,
            "legal": data.legal,
            **{key: float(value) for key, value in metrics.get((source, target), {}).items()},
        }
        features.append(
            {
                "type": "Feature",
                "id": f"{source}->{target}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        list(point)
                        for point in (
                            geometries[(source, target)]
                            if geometries and (source, target) in geometries
                            else (coordinates[source], coordinates[target])
                        )
                    ],
                },
                "properties": properties,
            }
        )
    collection = {"type": "FeatureCollection", "name": network.name, "features": features}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(collection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.write_text(payload, encoding="utf-8")
    manifest = {
        "source": provenance_source,
        "retrieval_date": date.today().isoformat(),
        "license": "MIT",
        "checksum_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "preprocessing": "concordia.gis.export.export_edge_geojson",
        "crs": crs,
        "units": {"free_flow_min": "minutes", "capacity_vph": "vehicles/hour"},
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
