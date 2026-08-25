from __future__ import annotations

from typing import Tuple

from concordia.models import EdgeData
from concordia.network import RoadNetwork


def two_route() -> Tuple[RoadNetwork, Tuple[str, str], float]:
    """Two alternatives: a fast-low-capacity road and a slower reliable bypass."""
    network = RoadNetwork("two_route")
    network.add_edge("O", "A", EdgeData(5.0, 600.0, variability=2.0, risk=0.03, complexity=0.2))
    network.add_edge("A", "D", EdgeData(5.0, 600.0, variability=2.0, risk=0.03, complexity=0.2))
    network.add_edge("O", "B", EdgeData(7.0, 1500.0, variability=0.5, risk=0.03, complexity=0.1))
    network.add_edge("B", "D", EdgeData(7.0, 1500.0, variability=0.5, risk=0.03, complexity=0.1))
    return network, ("O", "D"), 1000.0


def braess(with_connector: bool = True) -> Tuple[RoadNetwork, Tuple[str, str], float]:
    """Classical Braess construction in minutes with demand 4,000 veh/h.

    Variable links approximate x/100 via t(x)=0.01+0.01x. Constant links cost 45.
    The connector induces the familiar 80-minute UE versus 65 minutes without it.
    """
    network = RoadNetwork("braess_connected" if with_connector else "braess_base")
    variable = EdgeData(0.01, 100.0, alpha=100.0, beta=1.0)
    constant = EdgeData(45.0, 10000.0, alpha=0.0, beta=1.0)
    network.add_edge("S", "A", variable)
    network.add_edge("A", "T", constant)
    network.add_edge("S", "B", constant)
    network.add_edge("B", "T", variable)
    if with_connector:
        network.add_edge("A", "B", EdgeData(0.01, 10000.0, alpha=0.0, beta=1.0))
    return network, ("S", "T"), 4000.0


def ring_road() -> Tuple[RoadNetwork, Tuple[str, str], float]:
    network = RoadNetwork("ring_road")
    nodes = ["N0", "N1", "N2", "N3", "N4", "N5"]
    for index, source in enumerate(nodes):
        target = nodes[(index + 1) % len(nodes)]
        network.add_edge(source, target, EdgeData(1.0, 900.0, risk=0.02))
        network.add_edge(target, source, EdgeData(1.0, 900.0, risk=0.02))
    return network, ("N0", "N3"), 1400.0


def merge_bottleneck() -> Tuple[RoadNetwork, Tuple[str, str], float]:
    network = RoadNetwork("merge_bottleneck")
    network.add_edge("O", "M1", EdgeData(3.0, 900.0, complexity=0.4))
    network.add_edge("O", "M2", EdgeData(4.0, 900.0, complexity=0.2))
    network.add_edge("M1", "B", EdgeData(2.0, 650.0, risk=0.08, complexity=0.8))
    network.add_edge("M2", "B", EdgeData(2.0, 650.0, risk=0.06, complexity=0.6))
    network.add_edge("B", "D", EdgeData(3.0, 1000.0, risk=0.07, complexity=0.5))
    network.add_edge("M2", "X", EdgeData(5.0, 1000.0, risk=0.02, complexity=0.1))
    network.add_edge("X", "D", EdgeData(5.0, 1000.0, risk=0.02, complexity=0.1))
    return network, ("O", "D"), 1200.0


def signalized_intersection() -> Tuple[RoadNetwork, Tuple[str, str], float]:
    network = RoadNetwork("signalized_intersection")
    network.add_edge("O", "I", EdgeData(4.0, 800.0, variability=3.0, risk=0.05, complexity=0.7))
    network.add_edge("I", "D", EdgeData(4.0, 800.0, variability=3.0, risk=0.05, complexity=0.7))
    network.add_edge("O", "R1", EdgeData(6.0, 1000.0, variability=0.5, risk=0.02, complexity=0.2))
    network.add_edge("R1", "R2", EdgeData(3.0, 1000.0, variability=0.5, risk=0.02, complexity=0.2))
    network.add_edge("R2", "D", EdgeData(3.0, 1000.0, variability=0.5, risk=0.02, complexity=0.2))
    return network, ("O", "D"), 900.0
