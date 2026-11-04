"""Required exclusion BED for admixture fine mapping (GRCh38, BED coordinates)."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import config

MASK_CONTRACT = "exclude_bed_intervals_at_vcf_pos_all_variant_types_v1"


def mask_path() -> Path:
    """Reuse the reference staged for LAI; never silently run without a mask."""
    override = os.environ.get("AOU_FIG4_HARDMASK_BED")
    if override:
        path = Path(override).expanduser()
    else:
        path = Path(config.ADMIXTURE_WORK_DIR).expanduser() / "workspace_inputs" / "hardmask.hg38.v4.over99.bed"
    if not path.is_file() or path.stat().st_size == 0:
        if not override and os.environ.get("GOOGLE_PROJECT"):
            source, filename = config.ADMIXTURE_HARDMASK_GS.rsplit("/", 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[figure2_sources][mask] staging {config.ADMIXTURE_HARDMASK_GS} -> {path.parent}", flush=True)
            subprocess.run([
                "gcloud", "storage", "rsync", source, str(path.parent),
                "--exclude", f"^(?!{re.escape(filename)}$).*",
                "--billing-project", os.environ["GOOGLE_PROJECT"],
            ], check=True)
            staged = path.parent / filename
            if staged != path and staged.is_file():
                staged.replace(path)
        if path.is_file() and path.stat().st_size:
            return path
        raise FileNotFoundError(
            f"Required admixture hardmask is missing/empty: {path}. Stage "
            f"{config.ADMIXTURE_HARDMASK_GS} on Workbench, or set AOU_FIG4_HARDMASK_BED."
        )
    return path


@lru_cache(maxsize=2)
def _parse(content: bytes) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    intervals = {}
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", "track", "browser")):
            continue
        fields = line.split()
        try:
            chrom, start, end = fields[0].removeprefix("chr"), int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError("invalid interval")
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed hardmask BED at line {number}") from exc
        intervals.setdefault(chrom, []).append((start, end))
    if not intervals:
        raise ValueError("Hardmask BED has no intervals")
    merged = {}
    for chrom, spans in intervals.items():
        starts, ends = [], []
        for start, end in sorted(spans):
            if ends and start <= ends[-1]:
                ends[-1] = max(end, ends[-1])
            else:
                starts.append(start)
                ends.append(end)
        merged[chrom] = (np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64))
    return merged


class Hardmask:
    def __init__(self):
        path = mask_path()
        content = path.read_bytes()  # Hash bytes, not Git revision or file timestamps.
        self.intervals = _parse(content)
        self.parameters = {
            "hardmask_contract": MASK_CONTRACT,
            "hardmask_sha256": hashlib.sha256(content).hexdigest(),
            "hardmask_size_bytes": len(content),
        }
        self.provenance = {**self.parameters, "path": str(path.resolve()),
                           "configured_source": config.ADMIXTURE_HARDMASK_GS,
                           "coordinate_rule": "BED start <= VCF POS - 1 < BED end",
                           "variant_scope": "all types, POS anchor; not full SV span"}

    def contains(self, chrom, positions):
        """Vectorized negative-mask lookup; missing chromosomes have no exclusions."""
        pos = np.asarray(positions)
        if pos.dtype.kind not in "iu" or np.any(pos < 1):
            raise ValueError("Hardmask lookup requires positive integer VCF positions")
        spans = self.intervals.get(str(chrom).removeprefix("chr"))
        if spans is None:
            return np.zeros(pos.shape, dtype=bool)
        starts, ends = spans
        p0 = pos - 1
        index = np.searchsorted(starts, p0, side="right") - 1
        return (index >= 0) & (p0 < ends[np.maximum(index, 0)])
