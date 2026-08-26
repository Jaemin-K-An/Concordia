#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from collections import Counter

import numpy as np

from v8_common import ROOT, write_json, write_svg_line_chart
from v8_frozen import verify_frozen


OUTPUT = ROOT / "artifacts/studies/v8_failure_analysis"


def _taxonomy(rows, decisions):
    counts = Counter()
    examples = []
    for row, decision in zip(rows, decisions):
        success = bool(row["outcomes"]["safe_micro_success"])
        unsafe = float(row["outcomes"]["tau_s"]) > 0.25
        regret = float(row["outcomes"]["max_regret"]) > 0.08
        traffic_failure = float(row["outcomes"]["tau_t_relative"]) <= 0.01
        if decision["intervene"] and unsafe:
            name = "selected_unsafe"
        elif decision["intervene"] and regret:
            name = "selected_regret_violation"
        elif decision["intervene"] and traffic_failure:
            name = "selected_nonbeneficial"
        elif decision["intervene"] and not success:
            name = "selected_other_failure"
        elif not decision["intervene"] and success:
            name = "missed_safe_success"
        elif decision["reason"] == "unsafe_probability_veto" and unsafe:
            name = "correct_unsafe_veto"
        elif decision["reason"] == "unsafe_probability_veto":
            name = "conservative_safety_veto"
        elif decision["reason"] == "traffic_rank_below_cutoff":
            name = "low_rank_abstention"
        else:
            name = "other_abstention"
        counts[name] += 1
        if len([item for item in examples if item["taxonomy"] == name]) < 8:
            examples.append({
                "taxonomy": name,
                "pair_id": row["pair_id"],
                "reason": decision["reason"],
                "traffic_score": decision["traffic_uplift_score"],
                "unsafe_probability": decision["unsafe_probability"],
                "realized_uplift": row["outcomes"]["tau_t_relative"],
                "safety_effect": row["outcomes"]["tau_s"],
                "regret": row["outcomes"]["max_regret"],
            })
    return {"counts": counts, "examples": examples}


def _mechanism(rows, decisions):
    candidates = sorted(
        [
            (row, decision)
            for row, decision in zip(rows, decisions)
            if float(row["outcomes"]["tau_s"]) > 0.25
        ],
        key=lambda pair: (-pair[1]["traffic_uplift_score"], -float(pair[0]["outcomes"]["tau_s"])),
    )[:20]
    profiles = []
    for index, (row, decision) in enumerate(candidates):
        feature = row["v8_features"]
        base_safety = row["counterfactual_B1"]["safety"]
        adaptive_safety = row["counterfactual_adaptive"]["safety"]
        profiles.append({
            "candidate_rank": index + 1,
            "pair_id": row["pair_id"],
            "topology": row["condition"]["topology"],
            "traffic_uplift_score": decision["traffic_uplift_score"],
            "unsafe_probability": decision["unsafe_probability"],
            "route_flow_redistribution_vph_predecision_expectation": feature["proposed_rerouted_flow_vph"],
            "conflict_zone_exposure_delta_predecision_expectation": feature["conflict_zone_exposure_delta"],
            "lane_change_demand_delta_predecision_expectation": feature["lane_change_demand_delta"],
            "bottleneck_load_delta_predecision_expectation": feature["bottleneck_load_delta"],
            "drac_cvar_b1": base_safety["cvar_drac_95"],
            "drac_cvar_adaptive": adaptive_safety["cvar_drac_95"],
            "drac_cvar_delta": adaptive_safety["cvar_drac_95"] - base_safety["cvar_drac_95"],
            "ttc_conflict_delta": adaptive_safety["ttc_conflicts"] - base_safety["ttc_conflicts"],
            "high_closing_speed_conflict_delta": adaptive_safety["high_closing_speed_conflicts"] - base_safety["high_closing_speed_conflicts"],
            "x_t_map_basis": "predecision density/flow and paired aggregate conflict evidence; no unobserved trajectory is invented",
        })
    series = {
        "DRAC delta": [(item["candidate_rank"], item["drac_cvar_delta"]) for item in profiles],
        "Conflict exposure": [(item["candidate_rank"], item["conflict_zone_exposure_delta_predecision_expectation"]) for item in profiles],
        "Lane-change demand": [(item["candidate_rank"], item["lane_change_demand_delta_predecision_expectation"]) for item in profiles],
        "Bottleneck load": [(item["candidate_rank"], item["bottleneck_load_delta_predecision_expectation"]) for item in profiles],
    }
    write_svg_line_chart(OUTPUT / "unsafe_candidate_mechanisms.svg", series, "Unsafe candidate mechanism diagnostics")
    # Compact evidence-grounded x-t proxy map: candidates x registered precursor intensity.
    width, height, left, top = 900, 500, 220, 70
    fields = [
        ("route flow redistribution", "route_flow_redistribution_vph_predecision_expectation"),
        ("conflict exposure", "conflict_zone_exposure_delta_predecision_expectation"),
        ("lane-change demand", "lane_change_demand_delta_predecision_expectation"),
        ("bottleneck load", "bottleneck_load_delta_predecision_expectation"),
        ("paired DRAC delta", "drac_cvar_delta"),
    ]
    matrix = np.asarray([[float(item[key]) for key_name, key in fields] for item in profiles], dtype=float)
    if len(matrix):
        scale = np.maximum(np.nanmax(np.abs(matrix), axis=0), 1e-9)
        normalized = np.clip(matrix / scale, -1.0, 1.0)
    else:
        normalized = np.zeros((0, len(fields)))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#07111F"/>',
        '<text x="30" y="36" fill="#F4F8FF" font-family="sans-serif" font-size="22">Unsafe-candidate x–t proxy evidence map</text>',
    ]
    cell_w = (width - left - 30) / max(len(profiles), 1)
    cell_h = (height - top - 30) / len(fields)
    for row_index, (label, _) in enumerate(fields):
        svg.append(f'<text x="12" y="{top+(row_index+0.65)*cell_h:.1f}" fill="#CBD5E1" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
        for column in range(len(profiles)):
            value = normalized[column, row_index]
            red = int(40 + 190 * max(value, 0.0))
            blue = int(50 + 180 * max(-value, 0.0))
            svg.append(f'<rect x="{left+column*cell_w:.1f}" y="{top+row_index*cell_h:.1f}" width="{max(cell_w-1,1):.1f}" height="{cell_h-1:.1f}" fill="rgb({red},70,{blue})"/>')
    svg.append('</svg>')
    (OUTPUT / "unsafe_candidate_xt_proxy.svg").write_text("\n".join(svg) + "\n")
    return profiles


def run():
    verify_frozen()
    final_rows = json.loads((ROOT / "artifacts/studies/v8_micro_holdout/raw_metrics.json").read_text())
    final_decisions = json.loads((ROOT / "artifacts/studies/v8_micro_holdout/decisions.json").read_text())
    osm_rows = json.loads((ROOT / "artifacts/studies/v8_real_topology/paired_feature_outcome_rows.json").read_text())
    osm_decisions = json.loads((ROOT / "artifacts/studies/v8_real_topology/decision_log.json").read_text())
    profiles = _mechanism(final_rows, final_decisions)
    summary = {
        "complete": True,
        "microscopic_final": _taxonomy(final_rows, final_decisions),
        "real_topology": _taxonomy(osm_rows, osm_decisions),
        "unsafe_mechanism_profile_count": len(profiles),
        "mechanism_evidence_is_paired_or_predecision_proxy": True,
        "rl_used": False,
    }
    write_json(OUTPUT / "mechanism_profiles.json", profiles)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    run()
