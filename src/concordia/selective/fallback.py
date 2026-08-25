from __future__ import annotations

from typing import Mapping


def baseline_fallback(current_routes: Mapping[str, str]) -> dict[str, str]:
    """Return an isolated copy; abstention never mutates any current route."""
    return dict(current_routes)
