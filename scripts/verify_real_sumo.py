#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from concordia.simulation import SumoAdapter


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = ROOT / "data" / "raw" / "gangnam_intersection.osm"
    destination = ROOT / "artifacts" / "studies" / "real_topology" / "sumo_conversion.json"
    netconvert = SumoAdapter.resolve_binary("netconvert")
    if netconvert is None:
        raise SystemExit("netconvert is unavailable")
    with tempfile.TemporaryDirectory(prefix="concordia-real-net-") as temporary:
        network = Path(temporary) / "gangnam.net.xml"
        completed = subprocess.run(
            [
                netconvert,
                "--osm-files",
                str(source),
                "--output-file",
                str(network),
                "--geometry.remove",
                "true",
                "--ramps.guess",
                "true",
                "--junctions.join",
                "true",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not network.is_file():
            raise SystemExit(f"real OSM netconvert failed: {completed.stderr[-1000:]}")
        network_bytes = network.read_bytes()
        edge_count = network_bytes.count(b'<edge id="')
        junction_count = network_bytes.count(b'<junction id="')
        payload = {
            "complete": True,
            "source": str(source.relative_to(ROOT)),
            "netconvert_version": SumoAdapter.simulator_version(netconvert),
            "generated_network_sha256": hashlib.sha256(network_bytes).hexdigest(),
            "generated_network_bytes": len(network_bytes),
            "sumo_edge_element_count_including_internal": edge_count,
            "sumo_junction_element_count_including_internal": junction_count,
            "demand_provenance": "synthetic demand on real topology",
            "simulation_executed": False,
            "claim_boundary": (
                "conversion/topology verification only; no calibrated OD or real-topology "
                "traffic-mechanism result"
            ),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
