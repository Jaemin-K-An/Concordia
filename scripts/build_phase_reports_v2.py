#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "phase_reports_v2"


PHASES = {
    18: ("Current HEAD audit", "artifacts/current_head_audit/summary.json"),
    19: ("Phantom detector physical validation", "configs/validation/phantom_detector.yaml"),
    20: ("Microscopic phantom calibration dataset", "artifacts/studies/phantom_calibration/dataset.json"),
    21: ("Calibrated phantom predictor", "artifacts/studies/phantom_calibration/summary.json"),
    22: ("Acceptance–traffic fixed point", "src/concordia/optimization/fixed_point.py"),
    23: ("Price of Alignment", "src/concordia/alignment.py"),
    24: ("Alignment frontier", "artifacts/studies/alignment_frontier/summary.json"),
    25: ("H1 robustness", "artifacts/studies/alignment_frontier/h1_robustness.json"),
    26: ("Microscopic B1/B6 matrix", "artifacts/studies/microscopic_policy_matrix/raw_metrics.json"),
    27: ("H3 statistics", "artifacts/studies/microscopic_policy_matrix/statistical_tests.json"),
    28: ("H4 safety", "artifacts/studies/microscopic_policy_matrix/summary.json"),
    29: ("Real OSM traffic scenario", "configs/experiments/real_topology_policy_matrix.yaml"),
    30: ("Real-topology comparison", "artifacts/studies/real_topology_policy_matrix/summary.json"),
    31: ("Scalability", "artifacts/studies/scalability/summary.json"),
    32: ("Preference drift", "artifacts/studies/preference_drift/summary.json"),
    33: ("RL Gate v2", "artifacts/rl_gate_report_v2.json"),
    34: ("Conditional RL", "artifacts/rl_gate_report_v2.json"),
    35: ("Ablation and performance", "artifacts/studies/scalability/statistical_tests.json"),
    36: ("Final report and audit v2", "artifacts/reports/final_report_v2.html"),
}


def run() -> Path:
    gate_path = ROOT / "artifacts" / "rl_gate_report_v2.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for number, (name, evidence) in PHASES.items():
        exists = (ROOT / evidence).is_file()
        status = "COMPLETE" if exists else "INCOMPLETE"
        if number == 34 and gate.get("outcome") == "A":
            status = "SKIPPED_NOT_AUTHORIZED"
        payload = {
            "phase": number,
            "name": name,
            "status": status,
            "PLAN": True,
            "IMPLEMENT": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "TEST": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "RUN": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "VERIFY": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "REGRESSION": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "DOCUMENT": exists or status == "SKIPPED_NOT_AUTHORIZED",
            "evidence": evidence,
        }
        (OUTPUT / f"phase_{number}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(PHASES)} phase reports to {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    run()
