#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from concordia.v10.integrity import assert_file_hashes


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "artifacts/studies/v10_racing_validation"
FINAL_DIR = ROOT / "artifacts/studies/v10_micro_holdout"
AUDIT_DIR = ROOT / "artifacts/studies/v10_final_audit"
FREEZE = ROOT / "artifacts/v10/freeze_manifest.json"
SEEDS = ROOT / "artifacts/v10/final_seed_manifest.json"

REQUIRED_COMMITS = [
    "Preregister CONCORDIA v10 multi-fidelity racing",
    "Build CONCORDIA v10 racing engine",
    "Validate CONCORDIA v10 racing policy",
    "Freeze CONCORDIA v10 policy",
    "Record untouched CONCORDIA v10 microscopic holdout",
]


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _commit_sequence() -> tuple[list[str], bool]:
    subjects = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%s"], cwd=ROOT, text=True
    ).splitlines()
    positions = []
    for required in REQUIRED_COMMITS:
        try:
            positions.append(subjects.index(required))
        except ValueError:
            return subjects, False
    return subjects, positions == sorted(positions) and len(set(positions)) == len(positions)


def _leakage_hits(final_ids: list[str]) -> list[dict[str, str]]:
    hits = []
    final_id_set = set(final_ids)
    for path in sorted(VALIDATION_DIR.glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "V10F::" not in text:
            continue
        for token in text.replace('"', " ").split():
            state_id = token.rstrip(",")
            if state_id in final_id_set:
                hits.append(
                    {"path": str(path.relative_to(ROOT)), "state_id": state_id}
                )
    return hits


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def run() -> Path:
    required_files = [
        FREEZE,
        SEEDS,
        VALIDATION_DIR / "development_repair_3_summary.json",
        VALIDATION_DIR / "development_repair_3_baseline_comparison.json",
        VALIDATION_DIR / "development_repair_3_ablations.json",
        VALIDATION_DIR / "validation_summary.json",
        FINAL_DIR / "raw_metrics.json",
        FINAL_DIR / "summary.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"v10 final audit inputs are incomplete: {missing}")

    freeze = _read(FREEZE)
    seed_manifest = _read(SEEDS)
    development = _read(VALIDATION_DIR / "development_repair_3_summary.json")
    validation = _read(VALIDATION_DIR / "validation_summary.json")
    baselines = _read(
        VALIDATION_DIR / "development_repair_3_baseline_comparison.json"
    )
    ablations = _read(VALIDATION_DIR / "development_repair_3_ablations.json")
    final = _read(FINAL_DIR / "summary.json")
    final_raw = _read(FINAL_DIR / "raw_metrics.json")

    assert isinstance(freeze, dict)
    assert isinstance(seed_manifest, dict)
    assert isinstance(development, dict)
    assert isinstance(validation, dict)
    assert isinstance(baselines, dict)
    assert isinstance(ablations, dict)
    assert isinstance(final, dict)
    assert isinstance(final_raw, list)
    assert_file_hashes(freeze["frozen_file_hashes"], ROOT)
    assert_file_hashes(freeze["implementation_file_hashes"], ROOT)

    _subjects, commit_order_valid = _commit_sequence()
    final_ids = [str(value) for value in seed_manifest["state_ids"]]
    leakage_hits = _leakage_hits(final_ids)
    raw_ids = [str(record["state_id"]) for record in final_raw]
    manifest_matches_raw = raw_ids == final_ids
    materialized_after_freeze = bool(
        seed_manifest["materialized_after_freeze"]
        and seed_manifest["freeze_commit"]
        == seed_manifest["remote_main_commit_verified"]
        and commit_order_valid
    )
    final_targets = final["primary_targets"]
    integrity_passed = bool(
        commit_order_valid
        and materialized_after_freeze
        and not leakage_hits
        and manifest_matches_raw
        and len(final_raw) == 500
        and final["decision_evaluation_seed_overlap"] == 0
        and not final["final_realized_outcome_cache_used"]
        and not final["rl_used"]
    )
    checks = [
        {
            "requirement": "Preregistration committed before development",
            "value": commit_order_valid,
            "passed": commit_order_valid,
        },
        {
            "requirement": "Final materialized after remote freeze",
            "value": materialized_after_freeze,
            "passed": materialized_after_freeze,
        },
        {
            "requirement": "Final IDs absent from development artifacts",
            "value": len(leakage_hits),
            "passed": not leakage_hits,
        },
        {
            "requirement": "Decision/evaluation seed overlap",
            "value": final["decision_evaluation_seed_overlap"],
            "passed": final["decision_evaluation_seed_overlap"] == 0,
        },
        {
            "requirement": "Validation precision >= 0.85",
            "value": validation["precision"],
            "passed": validation["precision"] >= 0.85,
        },
        {
            "requirement": "Validation safety violations = 0",
            "value": validation["safety_violation_count"],
            "passed": validation["safety_violation_count"] == 0,
        },
        {
            "requirement": "Validation interventions >= 30",
            "value": validation["intervention_count"],
            "passed": validation["intervention_count"] >= 30,
        },
        {
            "requirement": "Validation coverage >= 0.10",
            "value": validation["coverage"],
            "passed": validation["coverage"] >= 0.10,
        },
        {
            "requirement": "Final precision >= 0.80",
            "value": final["precision"],
            "passed": final_targets["precision_at_least_0_80"],
        },
        {
            "requirement": "Final coverage >= 0.10",
            "value": final["coverage"],
            "passed": final_targets["coverage_at_least_0_10"],
        },
        {
            "requirement": "Final interventions >= 40",
            "value": final["intervention_count"],
            "passed": final_targets["interventions_at_least_40"],
        },
        {
            "requirement": "Final safety violations = 0",
            "value": final["safety_violation_count"],
            "passed": final_targets["safety_violations_zero"],
        },
    ]
    report = {
        "study": "CONCORDIA_v10_final_audit",
        "final_outcome": final["final_outcome"],
        "integrity_passed": integrity_passed,
        "checks": checks,
        "passed_check_count": sum(bool(check["passed"]) for check in checks),
        "check_count": len(checks),
        "final_id_leakage_hits": leakage_hits,
        "final_manifest_matches_raw_order": manifest_matches_raw,
        "development_stage_survival": {
            "stage_1": development["stage_1_oracle_survival"],
            "stage_2": development["stage_2_oracle_survival"],
            "stage_3": development["stage_3_oracle_survival"],
            "verification": development["verification_oracle_survival"],
        },
        "validation": validation,
        "final": final,
        "baselines": baselines,
        "ablations": ablations,
        "osm_transfer": "External-topology transfer remains future work.",
        "rl_decision": "NO",
    }
    summary_path = AUDIT_DIR / "summary.json"
    _write_json(summary_path, report)

    full = ablations["full_v10"]
    single = ablations["no_replication"]
    no_lcb = ablations["no_robust_lcb"]
    b6 = baselines["B6"]
    rows = [
        ("Preregistration committed before development?", _yes_no(commit_order_valid)),
        ("Final materialized after freeze?", _yes_no(materialized_after_freeze)),
        ("Decision/evaluation seed overlap", str(final["decision_evaluation_seed_overlap"])),
        ("Stage 1 oracle survival", f"{development['stage_1_oracle_survival']:.2%}"),
        ("Stage 2 oracle survival", f"{development['stage_2_oracle_survival']:.2%}"),
        ("Stage 3 oracle survival", f"{development['stage_3_oracle_survival']:.2%}"),
        ("Validation precision >=85%?", f"{validation['precision']:.2%} ({_yes_no(validation['precision'] >= 0.85)})"),
        ("Validation safety violations =0?", f"{validation['safety_violation_count']} ({_yes_no(validation['safety_violation_count'] == 0)})"),
        ("Validation interventions >=30?", f"{validation['intervention_count']} ({_yes_no(validation['intervention_count'] >= 30)})"),
        ("Final precision >=80%?", f"{final['precision']:.2%} ({_yes_no(final_targets['precision_at_least_0_80'])})"),
        ("Final coverage >=10%?", f"{final['coverage']:.2%} ({_yes_no(final_targets['coverage_at_least_0_10'])})"),
        ("Final interventions >=40?", f"{final['intervention_count']} ({_yes_no(final_targets['interventions_at_least_40'])})"),
        ("Final safety violations =0?", f"{final['safety_violation_count']} ({_yes_no(final_targets['safety_violations_zero'])})"),
        ("Population TTT gain", f"{final['population_mean_relative_ttt_gain']:.4%}"),
        ("B6 comparison", f"precision {b6['precision']:.2%}; population gain {b6['population_mean_relative_ttt_gain']:.4%}"),
        ("Single-rollout ablation", f"precision {single['precision']:.2%} vs full {full['precision']:.2%}"),
        ("Robust-LCB ablation", f"precision {no_lcb['precision']:.2%} vs full {full['precision']:.2%}"),
        ("Final outcome", str(final["final_outcome"])),
    ]
    outcome = str(final["final_outcome"])
    if outcome == "S+":
        statement = "**Outcome S+ — Robust Multi-Fidelity CONCORDIA strongly supported in the prespecified microscopic SUMO domain.**"
    elif outcome == "S":
        statement = "**Outcome S — Multi-Fidelity CONCORDIA supported in the prespecified microscopic SUMO domain.**"
    elif outcome == "P":
        statement = "**Outcome P — Partial support only in the prespecified microscopic SUMO domain.**"
    else:
        statement = "**Outcome F — The prespecified microscopic SUMO success criteria were not supported.**"
    markdown = [
        "# FINAL_AUDIT_V10",
        "",
        statement,
        "",
        "| Metric | Result |",
        "| --- | --- |",
        *[f"| {name} | {value} |" for name, value in rows],
        "",
        "## Integrity",
        "",
        f"- Freeze and implementation hashes valid: **{_yes_no(integrity_passed)}**",
        f"- Final manifest/raw order match: **{_yes_no(manifest_matches_raw)}**",
        f"- Final ID leakage hits: **{len(leakage_hits)}**",
        "- Final realized-outcome cache used: **NO**",
        "- RL used: **NO**",
        "",
        "## Scope",
        "",
        "The primary result is restricted to the prespecified synthetic microscopic SUMO domain. External-topology transfer remains future work.",
        "",
    ]
    report_path = ROOT / "FINAL_AUDIT_V10.md"
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not integrity_passed:
        raise RuntimeError("v10 final integrity audit failed")
    return report_path


if __name__ == "__main__":
    run()
