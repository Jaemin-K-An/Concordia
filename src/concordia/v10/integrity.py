from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hashes(paths: Iterable[Path], root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    return {
        str(Path(path).resolve().relative_to(root)): sha256_file(Path(path))
        for path in sorted(map(Path, paths), key=lambda value: str(value))
    }


def assert_file_hashes(expected: Mapping[str, str], root: Path) -> None:
    mismatches = {}
    for relative, digest in expected.items():
        path = Path(root) / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != digest:
            mismatches[relative] = {"expected": digest, "actual": actual}
    if mismatches:
        raise RuntimeError(f"frozen v10 hash mismatch: {mismatches}")


def first_primes_at_or_above(start: int, count: int) -> list[int]:
    if start < 2 or count < 1:
        raise ValueError("prime seed request must start at two and have positive count")

    def is_prime(value: int) -> bool:
        divisor = 2
        while divisor * divisor <= value:
            if value % divisor == 0:
                return False
            divisor += 1
        return True

    values = []
    candidate = start
    while len(values) < count:
        if is_prime(candidate):
            values.append(candidate)
        candidate += 1
    return values


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
