#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from concordia.gis import (
    audit_topology,
    export_edge_geojson,
    import_osm_xml,
    largest_weak_component,
)


def main() -> int:
    source = "data/raw/gangnam_intersection.osm"
    imported = import_osm_xml(
        source,
        source_url=(
            "https://api.openstreetmap.org/api/0.6/map?"
            "bbox=127.0250,37.4970,127.0300,37.5015"
        ),
        retrieval_date="2026-08-25",
    )
    network, coordinates, geometries = largest_weak_component(
        imported.network,
        imported.coordinates,
        imported.geometries,
    )
    audit = audit_topology(network, "1906756262", "5376448907", 3)
    if not audit.valid:
        raise SystemExit(f"real-topology audit failed: {audit.reasons}")
    output = export_edge_geojson(
        network,
        coordinates,
        "data/processed/gangnam_edges.geojson",
        geometries=geometries,
        provenance_source=imported.provenance.source_url,
    )
    audit_path = Path("artifacts/studies/real_topology/topology_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                **audit.__dict__,
                "demand_provenance": imported.provenance.demand_provenance,
                "geojson": str(output),
                "raw_checksum_sha256": imported.provenance.checksum_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
