"""Audit cached LAI markers and PLSR features against the fine-map BED, on disk."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import admixture_inputs, admixture_lai, admixture_mask, admixture_plrs
from . import figure2_1025 as corrected
from .common import config
from .common.genome import CHROM_SIZES


def run():
    """Called inside analysis_context; only read existing controlled products."""
    hardmask = admixture_mask.Hardmask()
    root = corrected.analysis_root() / "masking"
    table, manifest = root / "upstream_mask_audit.tsv", root / "upstream_mask_audit.json"
    contract = {
        "audit_contract": "cached_lai_and_plrs_positions_vs_negative_bed_v1",
        **hardmask.parameters,
        "telomere_pad_bp": admixture_lai.TELOMERE_PAD_BP,
        "lai_manifest_sha256": corrected._sha256_file(corrected._stage_manifest_path("lai")),
        "plrs_manifest_sha256": corrected._sha256_file(corrected._stage_manifest_path("plrs_models")),
    }
    # Recheck source bytes on every audit, including unchanged stage manifests.
    sources = {}
    for stage in ("lai", "plrs_models"):
        saved = json.loads(corrected._stage_manifest_path(stage).read_text(encoding="utf-8"))
        sources[stage] = saved

    def verify(stage, path):
        saved = sources[stage]
        if not corrected._stage_current(stage, saved["contract"], [path]):
            raise ValueError(f"Mask audit source is missing, changed or corrupt: {path}")

    def scan(chrom):
        path = corrected.lai_dir() / "chromosome_scans" / f"chr{chrom}.eur_fraction.npz"
        verify("lai", path)
        with np.load(path, allow_pickle=False) as data:
            pos = data["pos"]
            excluded = hardmask.contains(chrom, pos)
        trim = ((pos >= admixture_lai.TELOMERE_PAD_BP)
                & (pos <= CHROM_SIZES[chrom] - admixture_lai.TELOMERE_PAD_BP))
        return [
            dict(stage="lai_markers", key=f"chr{chrom}", path=str(path),
                 n_positions=len(pos), n_in_hardmask=int(excluded.sum())),
            dict(stage="outlier_calling_markers", key=f"chr{chrom}", path=str(path),
                 n_positions=int(trim.sum()), n_in_hardmask=int(excluded[trim].sum())),
        ]

    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows = [row for result in pool.map(scan, range(1, 23)) for row in result]
    summary = admixture_plrs.analysis_dir() / "plrs_models_summary.tsv"
    verify("plrs_models", summary)
    models = pd.read_csv(summary, sep="\t", dtype={"region_id": str})
    if models.empty or models["region_id"].isna().any() or models["region_id"].duplicated().any():
        raise ValueError(f"Missing or duplicated PLSR regions in {summary}")

    def model(region_id):
        path = admixture_plrs.model_path(region_id)
        verify("plrs_models", path)
        payload = admixture_inputs.pickle_load(path)
        pos = np.asarray(payload["feature_positions"])
        if payload["region"]["region_id"] != region_id or len(pos) != len(payload["feature_ids"]):
            raise ValueError(f"Inconsistent PLSR feature metadata: {path}")
        excluded = hardmask.contains(payload["region"]["chrom"], pos)
        return dict(stage="plrs_features", key=region_id, path=str(path),
                    n_positions=len(pos), n_in_hardmask=int(excluded.sum()))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows.extend(pool.map(model, models["region_id"].astype(str)))
    frame = pd.DataFrame(rows)
    frame["status"] = np.where(frame["n_in_hardmask"].eq(0), "clear", "overlap")
    result = {
        "status": "overlap" if frame["n_in_hardmask"].gt(0).any() else "clear",
        "mask": hardmask.provenance, "contract": contract,
        "lai_source": config.LAI_MASKED_GS_DIR,
        "interpretation": "Overlap means cached upstream positions do not match this exclusion BED; no upstream refit is performed.",
        "plrs_prediction": "Only trained feature IDs are extracted from discovery/replication VCFs.",
        "plrs_target_weighting": "Midpoint marker widths span gaps; masked bases are not subtracted.",
        "mask_creation_provenance": "Not present in this repository; a clear audit checks positions, not FLARE input preparation.",
        "threads": threads,
    }
    corrected._atomic_tsv(frame, table)
    corrected._atomic_json(manifest, result)
    print(f"[figure2_sources][mask-audit] {result['status']}; rows={len(frame)}; saved={table}; "
          f"provenance={manifest}", flush=True)
    return {"status": result["status"], "table": str(table), "manifest": str(manifest)}
