from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from concordia.errors import ValidationError


@dataclass(frozen=True)
class DatasetProvenance:
    source_url: str
    retrieval_date: str
    license: str
    checksum_sha256: str
    crs: str
    units: Dict[str, str]
    demand_provenance: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.source_url,
                self.retrieval_date,
                self.license,
                self.checksum_sha256,
                self.crs,
                self.demand_provenance,
            )
        ):
            raise ValidationError("dataset provenance fields cannot be empty")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def sha256_file(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"cannot checksum missing dataset: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

