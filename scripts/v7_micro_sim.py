from __future__ import annotations

import math
import subprocess
from pathlib import Path

from concordia.simulation import SumoAdapter
from v6_micro_sim import build_v6_network, run_v6_pair


def build_v7_network(
    directory: Path, topology: str, perturbation: str
) -> tuple[Path, dict]:
    """Build a v7 network while preserving the frozen v6 simulator.

    The asymmetric case contains three physically legal source-destination paths.  The paired
    treatment still compares the pre-registered main and alternate executable candidates; the
    third path is a transfer/topology motif and is never silently treated as an intervention.
    """
    if topology != "asymmetric":
        return build_v6_network(directory, topology, perturbation)
    target = directory / f"{topology}-{perturbation}"
    target.mkdir(parents=True, exist_ok=True)
    nodes = target / "network.nod.xml"
    edges = target / "network.edg.xml"
    network = target / "network.net.xml"
    nodes.write_text(
        "<nodes>\n"
        ' <node id="s" x="-100" y="0"/><node id="o" x="0" y="0"/>\n'
        ' <node id="m1" x="180" y="0"/><node id="m2" x="390" y="0"/>\n'
        ' <node id="m3" x="620" y="0"/><node id="j" x="850" y="0"/>\n'
        ' <node id="d" x="1050" y="0"/>\n'
        ' <node id="a1" x="220" y="-260"/><node id="a2" x="610" y="-260"/>\n'
        ' <node id="b1" x="250" y="190"/><node id="b2" x="560" y="310"/>\n'
        "</nodes>\n",
        encoding="utf-8",
    )
    factor = {"none": 1.0, "weak": 0.92, "medium": 0.82, "strong": 0.70}[
        perturbation
    ]
    main_speed = 9.5 * factor
    alternate_speed = 15.5 * (1.0 if perturbation in {"none", "weak"} else 0.92)
    edges.write_text(
        "<edges>\n"
        ' <edge id="in" from="s" to="o" numLanes="2" speed="20"/>\n'
        ' <edge id="m0" from="o" to="m1" numLanes="2" speed="20"/>\n'
        ' <edge id="m1" from="m1" to="m2" numLanes="2" speed="18"/>\n'
        ' <edge id="m2" from="m2" to="m3" numLanes="1" speed="16"/>\n'
        f' <edge id="m3" from="m3" to="j" numLanes="1" speed="{main_speed:.3f}"/>\n'
        f' <edge id="a0" from="o" to="a1" numLanes="1" speed="{alternate_speed:.3f}"/>\n'
        f' <edge id="a1" from="a1" to="a2" numLanes="1" speed="{alternate_speed:.3f}"/>\n'
        f' <edge id="a2" from="a2" to="j" numLanes="1" speed="{alternate_speed:.3f}"/>\n'
        ' <edge id="b0" from="o" to="b1" numLanes="1" speed="13"/>\n'
        ' <edge id="b1" from="b1" to="b2" numLanes="1" speed="12"/>\n'
        ' <edge id="b2" from="b2" to="j" numLanes="1" speed="13"/>\n'
        ' <edge id="out" from="j" to="d" numLanes="1" speed="20"/>\n'
        "</edges>\n",
        encoding="utf-8",
    )
    binary = SumoAdapter.resolve_binary("netconvert")
    if binary is None:
        raise RuntimeError("netconvert is unavailable")
    completed = subprocess.run(
        [
            binary,
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(network),
            "--no-turnarounds",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not network.is_file():
        raise RuntimeError(f"v7 asymmetric netconvert failed: {completed.stderr[-1000:]}")
    alternate_length = (
        math.sqrt(220**2 + 260**2)
        + 390.0
        + math.sqrt(240**2 + 260**2)
    )
    return network, {
        "route_overlap": 2.0 / 9.0,
        "alternative_capacity_ratio": 0.5,
        "bottleneck_centrality": 2.0 / 12.0,
        "route_length_ratio": alternate_length / 850.0,
        "main_capacity": 1800.0,
        "perturbation_strength": {
            "none": 0.0,
            "weak": 1.0,
            "medium": 2.0,
            "strong": 3.0,
        }[perturbation],
        "physical_route_alternatives": 3,
        "treatment_candidate_routes": 2,
    }


__all__ = ["build_v7_network", "run_v6_pair"]
