#!/usr/bin/env python3
from __future__ import annotations

import importlib.util

from concordia.simulation import SumoAdapter


def main() -> int:
    executable = SumoAdapter.resolve_binary("sumo")
    netconvert = SumoAdapter.resolve_binary("netconvert")
    traci = importlib.util.find_spec("traci") is not None
    print(f"sumo_executable={executable or 'missing'}")
    print(f"traci={'available' if traci else 'missing'}")
    print(f"netconvert_executable={netconvert or 'missing'}")
    if executable and traci:
        print(f"version={SumoAdapter.simulator_version()}")
    return 0 if executable and netconvert and traci else 1


if __name__ == "__main__":
    raise SystemExit(main())
