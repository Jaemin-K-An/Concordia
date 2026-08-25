#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil


def main() -> int:
    executable = shutil.which("sumo")
    traci = importlib.util.find_spec("traci") is not None
    print(f"sumo_executable={executable or 'missing'}")
    print(f"traci={'available' if traci else 'missing'}")
    return 0 if executable and traci else 1


if __name__ == "__main__":
    raise SystemExit(main())
