from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from concordia.errors import ValidationError


def _load_json(path: str) -> Dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}


def _valid_registry_records(runs_root: str) -> List[dict]:
    records = []
    for metrics_path in sorted(Path(runs_root).glob("*/metrics.json")):
        manifest_path = metrics_path.parent / "manifest.json"
        manifest = _load_json(str(manifest_path))
        if manifest.get("status", "valid") != "valid":
            continue
        records.append(
            {
                "run_id": metrics_path.parent.name,
                "manifest": manifest,
                **_load_json(str(metrics_path)),
            }
        )
    return records


def _placeholder(axis: Any, title: str, reason: str) -> None:
    axis.set_title(title)
    axis.axis("off")
    axis.text(0.5, 0.55, "NOT TESTED", ha="center", va="center", fontsize=18, weight="bold")
    axis.text(0.5, 0.38, reason, ha="center", va="center", fontsize=9, wrap=True)


def _generate_figures(research: Mapping[str, Any], figure_root: Path) -> Dict[str, str]:
    matplotlib_cache = Path(".matplotlib-cache").resolve()
    xdg_cache = Path(".xdg-cache").resolve()
    matplotlib_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValidationError("matplotlib is required to build research figures") from exc
    figure_root.mkdir(parents=True, exist_ok=True)
    focused = list(research.get("focused", []))
    screening = list(research.get("screening", []))
    if not focused or not screening:
        raise ValidationError("research summary lacks focused or screening rows")
    names: Dict[str, str] = {}

    def save(name: str, figure: Any) -> None:
        path = figure_root / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        names[name] = str(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    for policy in ("B0", "B1", "B2", "B4", "B6"):
        scales = sorted({float(row["demand_scale"]) for row in focused})
        means = [
            np.mean(
                [
                    row["policies"][policy]["total_travel_time_vehicle_minutes_per_hour"]
                    for row in focused
                    if float(row["demand_scale"]) == scale
                ]
            )
            for scale in scales
        ]
        axis.plot(scales, means, marker="o", label=policy)
    axis.set(xlabel="Demand scale", ylabel="TTT (vehicle-min/hour)", title="Synthetic analytical TTT vs demand")
    axis.legend(ncol=3)
    save("ttt_vs_demand", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    _placeholder(
        axis,
        "Phantom-jam probability vs demand",
        "Only an unpaired microscopic ring smoke run exists; no policy-by-demand probability curve.",
    )
    save("phantom_probability_vs_demand", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    regrets = [row["policies"]["B6"]["max_regret"] for row in focused]
    axis.hist(regrets, bins=min(12, len(regrets)), color="#31688e")
    axis.set(xlabel="Maximum user regret", ylabel="Focused runs", title="B6 regret distribution")
    save("regret_distribution", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    scenarios = sorted({row["scenario"] for row in focused})
    acceptance = [
        np.mean(
            [row["policies"]["B6"].get("acceptance_rate", 0.0) for row in focused if row["scenario"] == scenario]
        )
        for scenario in scenarios
    ]
    axis.bar(scenarios, acceptance, color="#35b779")
    axis.set(ylim=(0, 1), ylabel="Accepted / offered", title="Synthetic acceptance rate")
    save("acceptance_rate", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for policy in ("B1", "B4", "B6"):
        scales = sorted({float(row["demand_scale"]) for row in focused})
        entropy = [
            np.mean([row["policies"][policy]["route_entropy"] for row in focused if float(row["demand_scale"]) == scale])
            for scale in scales
        ]
        axis.plot(scales, entropy, marker="o", label=policy)
    axis.set(xlabel="Demand scale", ylabel="Route entropy", title="Route entropy vs demand")
    axis.legend()
    save("route_entropy", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    risk = [
        np.mean([row["policies"]["B6"]["route_surrogate_risk_cvar95"] for row in focused if row["scenario"] == scenario])
        for scenario in scenarios
    ]
    axis.bar(scenarios, risk, color="#fde725")
    axis.set(ylabel="Route-level surrogate risk CVaR95", title="Analytical safety surrogate (not crash risk)")
    save("safety_cvar", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    consumed = [row["policies"]["B6"]["preference_slack_consumed"] for row in focused]
    axis.hist(consumed, bins=min(12, len(consumed)), color="#443983")
    axis.set(xlabel="Aggregate Preference Slack", ylabel="Focused runs", title="Preference Slack consumption")
    save("preference_slack", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    levels = ["none", "low", "high", "long_tail"]
    opportunity = [
        np.mean([row["beneficial_diversion_opportunities"] for row in screening if row["heterogeneity"] == level])
        for level in levels
    ]
    axis.bar(levels, opportunity, color="#21918c")
    axis.set(ylabel="Mean opportunities/population", title="Heterogeneity vs beneficial diversion")
    save("heterogeneity_diversion", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    values = [
        np.mean([row["policies"][policy]["total_travel_time_vehicle_minutes_per_hour"] for row in focused])
        for policy in ("B4", "B6")
    ]
    axis.bar(["B4 greedy", "B6 MPC"], values, color=["#3b528b", "#5ec962"])
    axis.text(0.5, 0.95, "RL not shown unless authorized by the mandatory gate", transform=axis.transAxes, ha="center", va="top")
    axis.set(ylabel="Mean TTT (vehicle-min/hour)", title="Optimizer comparison; RL conditional")
    save("rl_vs_optimizer", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    sizes = [
        len(row["policies"]["B6"]["assignments"]) * int(row["candidate_route_count"])
        for row in focused
    ]
    latency = [row["policies"]["B6"]["latency_seconds"] for row in focused]
    axis.scatter(sizes, latency, alpha=0.7)
    axis.set(xlabel="Users × candidate routes", ylabel="End-to-end B6 seconds", title="Latency scaling at correctness scale")
    save("latency_scaling", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    scales = sorted({float(row["demand_scale"]) for row in screening})
    heat = np.asarray(
        [
            [
                np.mean(
                    [
                        row["beneficial_diversion_opportunities"]
                        for row in screening
                        if row["heterogeneity"] == level and float(row["demand_scale"]) == scale
                    ]
                )
                for scale in scales
            ]
            for level in levels
        ]
    )
    image = axis.imshow(heat, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(scales)), scales)
    axis.set_yticks(range(len(levels)), levels)
    axis.set(xlabel="Demand scale", ylabel="Heterogeneity", title="Beneficial-diversion phase diagram")
    figure.colorbar(image, ax=axis, label="Mean opportunities")
    save("phase_diagram", figure)
    return names


def build_report(runs_root: str, output: str) -> Path:
    records = _valid_registry_records(runs_root)
    is_project_registry = Path(runs_root).resolve() == Path("artifacts/runs").resolve()
    research = (
        _load_json("artifacts/studies/analytical_matrix/summary.json")
        if is_project_registry
        else {}
    )
    if not research:
        if not records:
            raise ValidationError(f"no experiment runs found under {runs_root}")
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(record['run_id'])}</td>"
            f"<td>{html.escape(str(record.get('scenario', 'unknown')))}</td>"
            f"<td>{record.get('proposed_ttt', {}).get('mean', float('nan')):.3f}</td>"
            "</tr>"
            for record in records
        )
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "<!doctype html><html><body><h1>Synthetic analytical results</h1>"
            "<p>Safety values are surrogate exposure, not collision probability.</p>"
            f"<table><tr><th>Run</th><th>Scenario</th><th>Proposed TTT</th></tr>{rows}</table>"
            "</body></html>\n",
            encoding="utf-8",
        )
        return destination
    sumo = _load_json("artifacts/studies/sumo_ring/summary.json")
    topology = _load_json("artifacts/studies/real_topology/topology_audit.json")
    gate = _load_json("artifacts/rl_gate_report.json")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figures = _generate_figures(research, Path("artifacts/figures"))
    hypotheses = research["hypotheses"]
    figure_html = "\n".join(
        f'<figure><img src="../figures/{Path(path).name}" alt="{html.escape(name)}"><figcaption>{html.escape(name.replace("_", " "))}</figcaption></figure>'
        for name, path in figures.items()
    )
    focused_count = research["focused_row_count"]
    simulation_scope = html.escape(str(sumo.get("claim_boundary", "microscopic evidence unavailable")))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CONCORDIA research report</title>
<style>body{{font:16px/1.55 system-ui;max-width:1180px;margin:2rem auto;padding:0 1.2rem;color:#17202a}}
h1,h2{{line-height:1.2}}.warning{{background:#fff3cd;border-left:5px solid #f0ad4e;padding:1rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}}figure{{margin:0;border:1px solid #ddd;padding:.5rem}}img{{width:100%}}code{{background:#f4f6f7;padding:.1rem .25rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd1d1;padding:.5rem;text-align:left}}</style>
</head><body><h1>CONCORDIA Adaptive Navigation — reproducible research report</h1>
<p class="warning"><strong>Claim boundary.</strong> The B0–B6 matrix is synthetic analytical BPR evidence. SUMO evidence is a microscopic synthetic smoke study. Safety metrics are surrogate conflict indicators, not crash probabilities. Demand on the real OSM topology is synthetic.</p>
<h2>1. Research question</h2><p>Can truthful, voluntarily accepted route recommendations use heterogeneous private preference slack to reduce network externality without violating declared regret and safety-surrogate constraints?</p>
<h2>2. Mathematical model and system</h2><p>The controller observes explicit count/density/flow/speed state, predicts horizon route attributes, solves a constrained first-action MPC plan, offers a route, samples a separately parameterized acceptance model, executes only accepted routes, and observes again. Exact enumeration remains a small-instance oracle; B5 is a HiGHS linearized MIP baseline.</p>
<h2>3. Experimental design</h2><p>{research['screening_row_count']} screening cases selected the focused factors; {focused_count} paired focused rows compare B0–B6 across two-route, merge, signalized, and analytical ring families. Seeds are matched. The research registry contains config, commit/dirty state, versions, timestamps, hardware, and hashes.</p>
<h2>4. Baselines</h2><table><tr><th>ID</th><th>Definition</th></tr>{''.join(f'<tr><td>{key}</td><td>{html.escape(value)}</td></tr>' for key,value in research['policy_definitions'].items())}</table>
<h2>5. Hypothesis results</h2><table><tr><th>Hypothesis</th><th>Status</th><th>Primary evidence</th></tr>
<tr><td>H1</td><td>{hypotheses['H1']['status']}</td><td>Beneficial low-regret diversion opportunities</td></tr>
<tr><td>H2</td><td>{hypotheses['H2']['status']}</td><td>B1−B6 paired TTT CI [{hypotheses['H2']['bootstrap_ci95_low']:.3f}, {hypotheses['H2']['bootstrap_ci95_high']:.3f}]</td></tr>
<tr><td>H3</td><td>NOT TESTED</td><td>No matched microscopic adaptive-vs-ETA probability matrix</td></tr>
<tr><td>H4</td><td>PARTIAL</td><td>Microscopic SSM/trajectory smoke only; no matched policy non-inferiority</td></tr>
<tr><td>H5</td><td>{hypotheses['H5']['status']}</td><td>In-sample exploratory variance/tail association</td></tr>
<tr><td>H6</td><td>{hypotheses['H6']['status']}</td><td>Descriptive feedback stability only</td></tr>
<tr><td>H7</td><td>NOT TESTED</td><td>RL gate did not authorize RL</td></tr></table>
<h2>6. Safety and phantom-jam evidence</h2><p>{simulation_scope}</p><p>The detector found {len(sumo.get('phantom_events', []))} candidate wave events in one synthetic ring run. This does not establish prevention. SSM output and full trajectory-derived distributions are stored, including TTC, DRAC, hard braking and CVaR.</p>
<h2>7. Preference heterogeneity and user outcomes</h2><p>H1 status: {hypotheses['H1']['status']}. All B6 focused rows satisfy the configured regret bound: {hypotheses['H2']['all_b6_regret_constraints_met']}.</p>
<h2>8. RL gate</h2><p><strong>{html.escape(str(gate.get('decision', 'Gate evidence unavailable')))}</strong> Untested larger-scale and nonstationary settings remain limitations rather than justification to insert RL.</p>
<h2>9. Ablation, counterfactuals, and failure cases</h2><p>B0/B1/B2/B4/B6 separate static routing, ETA response, preference-only choice, externality-aware greedy assignment, and feedback-aware MPC. This is a policy-layer ablation, not a complete component-wise causal ablation. B5 exposes linearization failure cases; B6 enumeration is explicitly restricted to correctness scale. H3, matched microscopic H4, full preference-drift traffic evaluation, and calibrated real demand remain unresolved.</p>
<h2>10. Real topology</h2><p>OSM topology valid: {topology.get('valid', False)}; nodes: {topology.get('node_count', 'n/a')}; edges: {topology.get('edge_count', 'n/a')}; alternate routes: {topology.get('alternative_route_count', 'n/a')}. Geometry and ODbL provenance are preserved. Demand is synthetic on real topology.</p>
<h2>11. Figures</h2><div class="grid">{figure_html}</div>
<h2>12. Reproducibility</h2><p>Run <code>make lint test benchmark simulation-test research rl-gate rl-evaluate report</code>. Heavy microscopic full matrices are not silently substituted by analytical output. Valid registry runs included: {len(records)}.</p>
<h2>13. Conclusion</h2><p>CONCORDIA demonstrates a working truthful, acceptance-separated, constrained closed loop and synthetic analytical low-regret coordination opportunities. It does <em>not</em> yet establish statistically meaningful microscopic phantom-jam prevention or real-world transfer. RL outcome: {html.escape(str(gate.get('outcome', 'not evaluated')))}.</p>
</body></html>\n"""
    destination.write_text(document, encoding="utf-8")
    return destination
