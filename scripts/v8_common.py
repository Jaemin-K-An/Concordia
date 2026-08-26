from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_self_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_svg_line_chart(path: Path, series: dict[str, list[tuple[float, float]]], title: str) -> None:
    width, height = 900, 520
    margin = 70
    points = [point for values in series.values() for point in values]
    x_values = [point[0] for point in points] or [0.0, 1.0]
    y_values = [point[1] for point in points] or [0.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(0.0, min(y_values)), max(1e-9, max(y_values))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = ["#00D4FF", "#FFB020", "#FF4D6D", "#7AE582", "#A78BFA", "#FFFFFF"]
    def xy(point):
        x = margin + (point[0] - x_min) / (x_max - x_min) * (width - 2 * margin)
        y = height - margin - (point[1] - y_min) / (y_max - y_min) * (height - 2 * margin)
        return f"{x:.2f},{y:.2f}"
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07111F"/>',
        f'<text x="{margin}" y="38" fill="#F4F8FF" font-family="sans-serif" font-size="22">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#718096"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#718096"/>',
    ]
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coords = " ".join(xy(point) for point in values)
        content.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        content.append(f'<text x="{width-250}" y="{40+22*index}" fill="{color}" font-family="sans-serif" font-size="14">{name}</text>')
    content.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n")
