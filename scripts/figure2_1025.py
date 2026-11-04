"""Manifest-driven Figure 2 rerun using the canonical 1,025-person LAI panel.

This module deliberately keeps the established admixture reference statistical routines
as the implementation of the published method while giving the corrected run
its own paths, contracts, sample manifests, and cache validation.  Raw inputs
are shared read-only; analysis caches retain ``analysis/figure5_v2``
or ``figures/figure2``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import (
    admixture_finemap,
    admixture_inputs,
    admixture_lai,
    admixture_phewas,
    admixture_plrs,
    figure2_data,
    figure2_sv,
)
from .common import config
from .common import samples as sample_set

FIGURE_ID = "figure2"
RUN_LABEL = "Figure_2_1025"
PIPELINE_CONTRACT = "figure5_v2_unrelated1025_manifested_v1"
OLD_REGION_MIDPOINT = 63_793_287
DISPLAY_CHROM = 9
MIN_ELIGIBLE_MODEL_SNVS = 10
MIN_OBSERVED_MODEL_FRACTION = 0.10
EXPECTED_BASE_COUNTS = {"discovery": 9_276, "replication": 28_552}

COHORT_AUDIT_REMOTES = {
    "discovery": {
        "flagged": config.workspace_bucket_path(
            "phewas_sv_v5/relatedness_flagged_in_cohort.tsv"
        ),
        "relatedness": config.workspace_bucket_path("phewas_sv_v5/relatedness.tsv"),
    },
    "replication": {
        "flagged": config.workspace_bucket_path(
            "phewas_acaf_replication/relatedness_flagged_samples.tsv"
        ),
        "relatedness": config.workspace_bucket_path(
            "phewas_acaf_replication/relatedness.tsv"
        ),
    },
}

_COHORT_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analysis_root() -> Path:
    return config.analysis_dir(FIGURE_ID)


def lai_dir() -> Path:
    path = analysis_root() / "lai"
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_dir() -> Path:
    path = Path(
        os.environ.get("AOU_FIG5_1025_WORK_DIR", str(Path.home() / "figure5_v2_1025"))
    ).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_dir() -> Path:
    path = analysis_root() / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_sha(ids: set[str] | list[str] | tuple[str, ...]) -> str:
    return _sha256_bytes(("\n".join(sorted(map(str, ids))) + "\n").encode("utf-8"))


def training_ids() -> tuple[str, ...]:
    values = sample_set.load_aou_sample_list(download=True)
    if values is None:
        raise FileNotFoundError(
            "The canonical 1,025-person sample list is unavailable: "
            f"{sample_set.AOU_SAMPLE_LIST_GS}"
        )
    ids = tuple(sorted(set(map(str, values)).difference(sample_set.load_excluded_sample_ids())))
    if len(ids) != 1_025:
        raise ValueError(f"Expected exactly 1,025 LAI samples; found {len(ids):,}")
    if sample_set.load_excluded_sample_ids().intersection(ids):
        raise ValueError(
            "The canonical LAI manifest still contains an excluded related ID"
        )
    return ids


def _id_column(columns: Any) -> str:
    return next(
        (
            name
            for name in ("person_id", "sample_id", "research_id", "s", "IID")
            if name in columns
        ),
        next(iter(columns)),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _output_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _stage_manifest_path(stage: str) -> Path:
    return manifest_dir() / f"{stage}_1025.json"


def _stage_current(stage: str, contract: dict[str, Any], outputs: list[Path]) -> bool:
    path = _stage_manifest_path(stage)
    if not path.is_file() or any(not output.is_file() for output in outputs):
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    saved_contract = dict(saved.get("contract", {}))
    if "excluded_ids" in saved_contract:
        saved_contract["exclusion_manifest_sha256"] = _sample_sha(
            saved_contract.pop("excluded_ids")
        )
    if saved_contract != contract:
        return False
    recorded = {entry.get("path"): entry for entry in saved.get("outputs", [])}
    for output in outputs:
        entry = recorded.get(str(output))
        if entry is None or output.stat().st_size != int(entry.get("size_bytes", -1)):
            return False
        if _sha256_file(output) != entry.get("sha256"):
            return False
    if saved_contract != saved.get("contract"):
        saved["contract"] = saved_contract
        _atomic_json(path, saved)
    return True


def _write_stage_manifest(
    stage: str, contract: dict[str, Any], outputs: list[Path]
) -> None:
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage {stage} did not produce: {missing}")
    _atomic_json(
        _stage_manifest_path(stage),
        {
            "pipeline_contract": PIPELINE_CONTRACT,
            "stage": stage,
            "completed_utc": _utc_now(),
            "contract": contract,
            "outputs": [_output_fingerprint(path) for path in outputs],
        },
    )


def _selected_samples(vcf_path: str | os.PathLike, ids: tuple[str, ...]) -> list[str]:
    VCF = admixture_lai._cyvcf2()
    header = VCF(str(vcf_path))
    try:
        allowed = set(ids)
        selected = [str(sample) for sample in header.samples if str(sample) in allowed]
    finally:
        header.close()
    if len(selected) != 1_025 or set(selected) != set(ids):
        raise ValueError(
            f"{vcf_path} contains {len(selected):,}/1,025 required LAI samples; "
            "inspect the VCF and sample manifest in Workbench."
        )
    return selected


def _scan_chromosome_1025(
    vcf_path: str | os.PathLike, eur_idx: int, *, ids: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, int]:
    VCF = admixture_lai._cyvcf2()
    selected = _selected_samples(vcf_path, ids)
    vcf = VCF(str(vcf_path), samples=selected)
    positions: list[int] = []
    fractions: list[float] = []
    try:
        for record in vcf:
            an1, an2 = record.format("AN1"), record.format("AN2")
            if an1 is None or an2 is None:
                continue
            calls = np.concatenate(
                [np.asarray(an1).reshape(-1), np.asarray(an2).reshape(-1)]
            )
            valid = calls >= 0
            if valid.any():
                positions.append(int(record.POS))
                fractions.append(float(np.mean(calls[valid] == eur_idx)))
    finally:
        vcf.close()
    return np.asarray(positions, dtype=np.int64), np.asarray(fractions), len(selected)


def _strict_display_region(regions: pd.DataFrame, requested: str = "") -> pd.Series:
    del requested
    matches = regions[
        regions["chrom"].astype(int).eq(DISPLAY_CHROM)
        & (pd.to_numeric(regions["start"]) <= OLD_REGION_MIDPOINT)
        & (pd.to_numeric(regions["end"]) >= OLD_REGION_MIDPOINT)
    ].copy()
    if matches.empty:
        raise RuntimeError(
            "Corrected 1,025-sample LAI scan has no outlier region containing "
            f"chr{DISPLAY_CHROM}:{OLD_REGION_MIDPOINT:,}; refusing to substitute another locus"
        )
    matches["abs_peak_z"] = pd.to_numeric(matches["peak_z"], errors="coerce").abs()
    return matches.sort_values("abs_peak_z", ascending=False).iloc[0].drop("abs_peak_z")


def _extract_training_region_1025(original, *args, **kwargs) -> dict[str, Any]:
    extracted = original(*args, **kwargs)
    n_snvs = len(extracted.get("feature_ids", []))
    if n_snvs < MIN_ELIGIBLE_MODEL_SNVS:
        region = args[1] if len(args) > 1 else kwargs.get("region", {})
        region_id = region.get("region_id", "unknown")
        raise ValueError(
            f"Only {n_snvs} eligible biallelic MAF-filtered SNVs in {region_id}; "
            f"at least {MIN_ELIGIBLE_MODEL_SNVS} are required"
        )
    return extracted


def _load_lai_markers_1025(
    vcf_path: Path,
    chrom: int,
    window_start: int,
    window_end: int,
    *,
    ids: tuple[str, ...],
):
    from cyvcf2 import VCF

    mapping = admixture_lai.parse_ancestry_header(vcf_path)
    eur_index, afr_index, _, _ = admixture_lai.ancestry_indices(mapping)
    selected = _selected_samples(vcf_path, ids)
    vcf = VCF(str(vcf_path), samples=selected)
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from {vcf_path}")
    positions, hap1, hap2 = [], [], []
    start = max(1, int(window_start) - figure2_sv.LAI_NEAREST_PAD_BP)
    end = int(window_end) + figure2_sv.LAI_NEAREST_PAD_BP
    try:
        for record in vcf(f"{token}:{start}-{end}"):
            an1, an2 = record.format("AN1"), record.format("AN2")
            if an1 is None or an2 is None:
                continue
            positions.append(int(record.POS))
            hap1.append(np.asarray(an1).reshape(-1).astype(np.int16))
            hap2.append(np.asarray(an2).reshape(-1).astype(np.int16))
    finally:
        vcf.close()
    if not positions:
        raise ValueError(
            f"No LAI markers found around chr{chrom}:{window_start}-{window_end}"
        )
    return np.asarray(positions), np.stack(hap1), np.stack(hap2), eur_index, afr_index


def _discover_svs_1025(
    vcf_path: Path,
    chrom: int,
    window_start: int,
    window_end: int,
    *,
    ids: tuple[str, ...],
):
    from cyvcf2 import VCF

    selected = _selected_samples(vcf_path, ids)
    vcf = VCF(str(vcf_path), samples=selected)
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from {vcf_path}")
    rows: list[dict[str, Any]] = []
    carriers_by_variant: dict[str, list[tuple[str, int]]] = {}
    try:
        for record in vcf(f"{token}:{window_start}-{window_end}"):
            if len(record.ALT or []) != 1:
                continue
            sv = figure2_sv.classify_sv(
                record.REF,
                record.ALT[0],
                info_type=figure2_sv._record_info(record, "SVTYPE"),
                info_length=figure2_sv._record_info(record, "SVLEN"),
            )
            if sv is None:
                continue
            sv_type, sv_length = sv
            carriers, n_unphased = figure2_sv.phased_carrier_haplotypes(
                selected, record.genotypes
            )
            alleles = np.asarray(record.genotypes, dtype=object)[:, :2]
            numeric = np.asarray(alleles, dtype=int)
            called = numeric >= 0
            called_alleles = int(called.sum())
            alt_alleles = int((numeric[called] > 0).sum())
            variant_id = f"chr{chrom}:{int(record.POS)}:{sv_type}:{sv_length}"
            rows.append(
                {
                    "chrom": int(chrom),
                    "pos": int(record.POS),
                    "sv_type": sv_type,
                    "sv_length_bp": int(sv_length),
                    "variant_id": variant_id,
                    "alt_allele_count": alt_alleles,
                    "called_allele_count": called_alleles,
                    "alt_allele_frequency": (
                        alt_alleles / called_alleles if called_alleles else float("nan")
                    ),
                    "n_carrier_haplotypes": len(carriers),
                    "n_unphased_heterozygotes": int(n_unphased),
                }
            )
            carriers_by_variant[variant_id] = carriers
    finally:
        vcf.close()
    return rows, carriers_by_variant


def _vcf_sample_manifest_1025(
    vcf_path: Path, *, ids: tuple[str, ...]
) -> dict[str, Any]:
    selected = _selected_samples(vcf_path, ids)
    return {
        "n_samples": len(selected),
        "sample_sha256": _sample_sha(selected),
        "size_bytes": vcf_path.stat().st_size,
    }


def _current_python_runtime() -> Path:
    """PheTK must already be installed by the enclosing ``uv run``."""
    try:
        import phetk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PheTK is absent from the active uv environment. Run with "
            "requirements-figure2.txt; no child environment will be created."
        ) from exc
    return Path(sys.executable)


@contextlib.contextmanager
def analysis_context(ids: tuple[str, ...] | None = None) -> Iterator[None]:
    """Redirect established routines into the isolated Figure 2 v2 contract."""
    ids = ids or training_ids()
    digest = _sample_sha(ids)[:16]
    original_extract = admixture_plrs.extract_training_region
    model_contract = f"figure2_plrs_unrelated1025_min10_{digest}_v1"
    cohort_audits = {}
    for label in ("discovery", "replication"):
        audit_path = analysis_root() / "cohorts" / f"{label}_cohort_audit_1025.json"
        if audit_path.is_file():
            cohort_audits[label] = json.loads(audit_path.read_text(encoding="utf-8"))
    if set(cohort_audits) == {"discovery", "replication"}:
        cohort_token = (
            cohort_audits["discovery"]["sample_sha256"][:12]
            + "_"
            + cohort_audits["replication"]["sample_sha256"][:12]
        )
        discovery_token = cohort_audits["discovery"]["sample_sha256"][:12]
    else:
        cohort_token = digest
        discovery_token = digest
    phewas_contract = f"figure2_phetk_logit_20pc_qc_unrelated_{cohort_token}_v1"
    finemap_contract = f"figure2_susie_100kb_20pc_qc_unrelated_{discovery_token}_v1"
    sv_contract = f"figure2_sv_overall_nearest_lai_unrelated1025_{digest}_v1"
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(config, "ADMIXTURE_WORK_DIR", str(work_dir())))
        stack.enter_context(patch.object(admixture_lai, "analysis_dir", lai_dir))
        stack.enter_context(
            patch.object(
                admixture_lai,
                "scan_chromosome",
                lambda path, eur: _scan_chromosome_1025(path, eur, ids=ids),
            )
        )
        stack.enter_context(
            patch.object(admixture_lai, "choose_display_region", _strict_display_region)
        )
        stack.enter_context(
            patch.object(admixture_plrs, "MODEL_CONTRACT", model_contract)
        )
        stack.enter_context(
            patch.object(
                admixture_plrs,
                "extract_training_region",
                lambda *args, **kwargs: _extract_training_region_1025(
                    original_extract, *args, **kwargs
                ),
            )
        )
        stack.enter_context(patch.object(admixture_plrs, "_cohort_ids", cohort_ids))
        stack.enter_context(
            patch.object(admixture_phewas, "ANALYSIS_CONTRACT", phewas_contract)
        )
        stack.enter_context(
            patch.object(admixture_phewas, "load_cohort", load_cohort_1025)
        )
        stack.enter_context(
            patch.object(
                admixture_phewas, "_ensure_phetk_runtime", _current_python_runtime
            )
        )
        stack.enter_context(
            patch.object(admixture_finemap, "FINEMAP_CONTRACT", finemap_contract)
        )
        stack.enter_context(
            patch.object(figure2_sv, "SV_ANCESTRY_CONTRACT", sv_contract)
        )
        stack.enter_context(
            patch.object(
                figure2_sv,
                "_load_lai_markers",
                lambda path, chrom, start, end: _load_lai_markers_1025(
                    path, chrom, start, end, ids=ids
                ),
            )
        )
        stack.enter_context(
            patch.object(
                figure2_sv,
                "_discover_svs",
                lambda path, chrom, start, end: _discover_svs_1025(
                    path, chrom, start, end, ids=ids
                ),
            )
        )
        stack.enter_context(
            patch.object(
                figure2_sv,
                "_sample_manifest",
                lambda path: _vcf_sample_manifest_1025(path, ids=ids),
            )
        )
        yield


def _stage_audit_sources(label: str) -> dict[str, Path]:
    remotes = COHORT_AUDIT_REMOTES[label]
    outputs = {}
    for kind, remote in remotes.items():
        local = admixture_inputs.stage_file(
            remote,
            Path(label) / "relatedness_audit" / Path(remote).name,
            force=False,
            required=True,
        )
        assert local is not None
        outputs[kind] = local
    return outputs


def _relatedness_internal_pairs(path: Path, retained: set[str]) -> int:
    pairs = pd.read_csv(path, sep="\t", dtype=str)
    left = next((name for name in ("i.s", "i", "sample_id_1") if name in pairs), None)
    right = next((name for name in ("j.s", "j", "sample_id_2") if name in pairs), None)
    if left is None or right is None:
        raise ValueError(f"Relatedness table lacks pair ID columns: {path}")
    return int(
        pairs[left]
        .astype(str)
        .isin(retained)
        .mul(pairs[right].astype(str).isin(retained))
        .sum()
    )


def load_cohort_1025(label: str, *, force: bool = False):
    """Load the agreed unrelated discovery/replication base plus exactly 20 PCs."""
    if label not in EXPECTED_BASE_COUNTS:
        raise ValueError(f"Unknown Figure 2 v2 cohort: {label}")
    if label in _COHORT_CACHE and not force:
        covariates, counts = _COHORT_CACHE[label]
        return covariates.copy(), counts.copy()
    sources = admixture_inputs.stage_cohort_sources(label, force=False)
    audits = _stage_audit_sources(label)
    covariates = pd.read_csv(sources["covariates"], sep="\t")
    covariates = covariates.rename(
        columns={_id_column(covariates.columns): "person_id"}
    )
    covariates["person_id"] = covariates["person_id"].astype(str)
    if covariates["person_id"].duplicated().any():
        raise ValueError(f"{label} covariates_base contains duplicate person IDs")
    if len(covariates) != EXPECTED_BASE_COUNTS[label]:
        raise ValueError(
            f"{label} covariates_base drifted: expected {EXPECTED_BASE_COUNTS[label]:,}, "
            f"found {len(covariates):,}"
        )
    flagged = pd.read_csv(audits["flagged"], sep="\t", dtype=str)
    flagged_ids = set(flagged[_id_column(flagged.columns)].dropna().astype(str))
    base_ids = set(covariates["person_id"])
    matching_flag_overlap = base_ids.intersection(flagged_ids)
    removals = matching_flag_overlap.union(base_ids.intersection(sample_set.load_excluded_sample_ids()))
    covariates = covariates[~covariates["person_id"].isin(removals)].copy()
    retained_pre_pca = set(covariates["person_id"])
    internal_pairs = _relatedness_internal_pairs(
        audits["relatedness"], retained_pre_pca
    )
    if internal_pairs:
        raise RuntimeError(
            f"{label} has {internal_pairs:,} relatedness pairs with both members retained; "
            "the agreed base cohort contract is not satisfied"
        )
    if "age_at_last_ehr_event_squared" not in covariates:
        age = pd.to_numeric(covariates["age_at_last_ehr_event"], errors="coerce")
        covariates["age_at_last_ehr_event_squared"] = age**2
    compression = "gzip" if str(sources["pca"]).endswith((".gz", ".bgz")) else "infer"
    pca = admixture_phewas.harmonize_pca(
        pd.read_csv(sources["pca"], sep="\t", compression=compression), n_pcs=20
    )
    covariates = covariates.merge(
        pca, on="person_id", how="inner", validate="one_to_one"
    )
    covariates["sex_at_birth"] = admixture_phewas._normalize_sex(
        covariates["sex_at_birth"]
    )
    for column in admixture_phewas.COVARIATE_COLUMNS:
        covariates[column] = pd.to_numeric(covariates[column], errors="coerce")
    before_complete = len(covariates)
    covariates = covariates.dropna(
        subset=admixture_phewas.COVARIATE_COLUMNS
    ).drop_duplicates("person_id")
    final_ids = set(covariates["person_id"])
    if final_ids.intersection(flagged_ids) or final_ids.intersection(
        sample_set.load_excluded_sample_ids()
    ):
        raise AssertionError(f"{label} final cohort contains a forbidden sample")
    if _relatedness_internal_pairs(audits["relatedness"], final_ids):
        raise AssertionError(f"{label} final cohort contains an internal related pair")
    cohort_dir = analysis_root() / "cohorts"
    id_path = cohort_dir / f"{label}_analysis_samples_1025.tsv"
    _atomic_tsv(pd.DataFrame({"person_id": sorted(final_ids)}), id_path)
    audit_payload = {
        "pipeline_contract": PIPELINE_CONTRACT,
        "cohort": label,
        "source_covariate_rows": EXPECTED_BASE_COUNTS[label],
        "matching_flagged_ids": len(flagged_ids),
        "matching_flag_overlap_before_filter": len(matching_flag_overlap),
        "exclusion_manifest_sha256": _sample_sha(sample_set.load_excluded_sample_ids()),
        "retained_before_pca": len(retained_pre_pca),
        "retained_after_pca_and_complete_covariates": len(final_ids),
        "removed_for_pca_or_covariate_missingness": before_complete - len(final_ids),
        "n_pcs": 20,
        "internal_related_pairs": 0,
        "sample_sha256": _sample_sha(final_ids),
        "sources": {key: str(value) for key, value in {**sources, **audits}.items()},
    }
    _atomic_json(cohort_dir / f"{label}_cohort_audit_1025.json", audit_payload)
    counts = pd.read_csv(
        sources["phenotypes"],
        sep="\t",
        usecols=lambda column: column in {"person_id", "phecode", "count"},
        dtype={"person_id": str, "phecode": str},
    )
    if not {"person_id", "phecode", "count"}.issubset(counts):
        raise ValueError(f"{label} phenotype count table lacks required columns")
    counts = counts[counts["person_id"].astype(str).isin(final_ids)].copy()
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0)
    indexed = covariates.set_index("person_id")
    _COHORT_CACHE[label] = (indexed.copy(), counts.copy())
    print(
        f"[figure2_1025][cohort] {label}: base={EXPECTED_BASE_COUNTS[label]:,}; "
        f"explicit/flag removals={len(removals):,}; final complete-case n={len(final_ids):,}; "
        "internal related pairs=0; PCs=20",
        flush=True,
    )
    return indexed, counts


def cohort_ids(covariates_path: Path) -> list[str]:
    """Return IDs only from the audited base manifest used by PLRS prediction."""
    label = (
        "replication" if "replication" in str(covariates_path).lower() else "discovery"
    )
    covariates, _ = load_cohort_1025(label)
    return covariates.index.astype(str).tolist()


def _lai_outputs() -> list[Path]:
    root = lai_dir()
    return [
        root / "outlier_regions.tsv",
        root / "genome_scan_plot.tsv.gz",
        root / "region_zoom.tsv.gz",
        root / "scan_stats.json",
        root / "display_region.json",
        root / "karyogram_segments.tsv.gz",
        root / "karyogram_masks.tsv.gz",
        *[
            root / "chromosome_scans" / f"chr{chrom}.eur_fraction.npz"
            for chrom in range(1, 23)
        ],
    ]


def run_lai(ids: tuple[str, ...], *, force: bool = False) -> dict[str, Any]:
    contract = {
        "pipeline": PIPELINE_CONTRACT,
        "sample_n": len(ids),
        "sample_sha256": _sample_sha(ids),
        "exclusion_manifest_sha256": _sample_sha(sample_set.load_excluded_sample_ids()),
        "z_threshold": admixture_lai.Z_THRESHOLD,
        "telomere_pad_bp": admixture_lai.TELOMERE_PAD_BP,
        "region_gap_bp": admixture_lai.REGION_GAP_BP,
        "valley_min_markers": admixture_lai.VALLEY_MIN_MARKERS,
        "centromere_max_overlap": admixture_lai.CENTROMERE_MAX_OVERLAP,
        "required_display_locus": f"chr{DISPLAY_CHROM}:{OLD_REGION_MIDPOINT}",
    }
    outputs = _lai_outputs()
    current = not force and _stage_current("lai", contract, outputs)
    if current:
        print("[figure2_1025][stage1] LAI cache verified", flush=True)
        return {"status": "cached", "outputs": outputs}
    result = admixture_lai.run(force=True)
    _write_stage_manifest("lai", contract, outputs)
    return {"status": "rebuilt", **result}


def _cohort_contract(label: str) -> dict[str, Any]:
    audit = json.loads(
        (analysis_root() / "cohorts" / f"{label}_cohort_audit_1025.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "sample_n": audit["retained_after_pca_and_complete_covariates"],
        "sample_sha256": audit["sample_sha256"],
        "n_pcs": 20,
        "internal_related_pairs": 0,
    }


def _require_main_hit(meta: pd.DataFrame) -> pd.Series:
    region = json.loads((lai_dir() / "display_region.json").read_text(encoding="utf-8"))
    significant = meta[
        meta["region_id"].astype(str).eq(str(region["region_id"]))
        & meta["meta_significant"].astype(str).str.lower().isin({"true", "1", "1.0"})
    ].copy()
    significant["p_meta"] = pd.to_numeric(significant["p_meta"], errors="coerce")
    significant = significant.dropna(subset=["p_meta"]).sort_values("p_meta")
    if significant.empty:
        raise RuntimeError(
            "The corrected region exists, but it has no meta-significant phenotype; "
            "the workflow will not substitute a different region or phenotype"
        )
    hit = significant.iloc[0]
    _atomic_json(
        analysis_root() / "main_hit_1025.json",
        {
            "region_id": str(hit["region_id"]),
            "phecode": str(hit["phecode"]),
            "p_meta": float(hit["p_meta"]),
            "selection": "lowest p_meta among meta-significant",
        },
    )
    return hit


def run_all(*, force: bool = False, render: bool = True) -> dict[str, Any]:
    """Run or resume every corrected stage, then render Figure 2_1025."""
    ids = training_ids()
    with analysis_context(ids):
        lai_result = run_lai(ids, force=force)
        discovery_cov, _ = load_cohort_1025("discovery", force=force)
        replication_cov, _ = load_cohort_1025("replication", force=force)
        cohort_contracts = {
            "discovery": _cohort_contract("discovery"),
            "replication": _cohort_contract("replication"),
        }
        cohort_token = (
            cohort_contracts["discovery"]["sample_sha256"][:12]
            + "_"
            + cohort_contracts["replication"]["sample_sha256"][:12]
        )
        admixture_phewas.ANALYSIS_CONTRACT = (
            f"figure2_phetk_logit_20pc_qc_unrelated_{cohort_token}_v1"
        )
        admixture_finemap.FINEMAP_CONTRACT = (
            "figure2_susie_100kb_20pc_qc_unrelated_"
            f"{cohort_contracts['discovery']['sample_sha256'][:12]}_v1"
        )

        summary_path = admixture_plrs.analysis_dir() / "plrs_models_summary.tsv"
        holdout_path = admixture_plrs.analysis_dir() / "plrs_holdout_predictions.tsv.gz"
        try:
            cached_summary = (
                pd.read_csv(summary_path, sep="\t", dtype={"region_id": str})
                if summary_path.is_file()
                else pd.DataFrame(columns=["region_id"])
            )
        except (OSError, ValueError, pd.errors.EmptyDataError):
            cached_summary = pd.DataFrame(columns=["region_id"])
        if "region_id" not in cached_summary:
            cached_summary = pd.DataFrame(columns=["region_id"])
        model_paths = [
            admixture_plrs.model_path(region_id)
            for region_id in cached_summary["region_id"].astype(str)
        ]
        model_outputs = [summary_path, holdout_path, *model_paths]
        model_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "lai_samples": _sample_sha(ids),
            "min_eligible_snvs": MIN_ELIGIBLE_MODEL_SNVS,
            "min_maf": admixture_plrs.MIN_MAF,
            "test_fraction": admixture_plrs.TEST_FRACTION,
            "random_state": admixture_plrs.RANDOM_STATE,
            "r2_min": admixture_plrs.R2_MIN,
        }
        models_current = not force and _stage_current(
            "plrs_models", model_contract, model_outputs
        )
        if not models_current:
            summary = admixture_plrs.fit_plrs_models(force=True)
            if summary.empty:
                raise RuntimeError("No corrected PLRS models could be fitted")
            if not holdout_path.is_file():
                raise FileNotFoundError(
                    "PLRS fitting did not produce hold-out predictions"
                )
            model_paths = [
                admixture_plrs.model_path(region_id)
                for region_id in summary["region_id"].astype(str)
            ]
            model_outputs = [summary_path, holdout_path, *model_paths]
            _write_stage_manifest("plrs_models", model_contract, model_outputs)
        else:
            summary = pd.read_csv(summary_path, sep="\t", dtype={"region_id": str})
            print("[figure2_1025][stage2] PLRS model cache verified", flush=True)

        discovery_prediction = (
            admixture_plrs.analysis_dir() / "eur_dosage_predictions.discovery.tsv.gz"
        )
        prediction_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "cohort": cohort_contracts["discovery"],
            "model_manifest_sha256": _sha256_file(_stage_manifest_path("plrs_models")),
            "min_observed_model_fraction": MIN_OBSERVED_MODEL_FRACTION,
        }
        if force or not _stage_current(
            "discovery_predictions", prediction_contract, [discovery_prediction]
        ):
            with patch.object(
                admixture_plrs,
                "MIN_OBSERVED_FEATURE_FRACTION",
                MIN_OBSERVED_MODEL_FRACTION,
            ):
                admixture_plrs.predict_discovery(summary, force=True)
            _write_stage_manifest(
                "discovery_predictions", prediction_contract, [discovery_prediction]
            )

        discovery_output = admixture_phewas.analysis_dir() / "phewas.discovery.tsv.gz"
        discovery_assoc_manifest = discovery_output.with_suffix(
            discovery_output.suffix + ".manifest.json"
        )
        discovery_phewas_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "analysis_contract": admixture_phewas.ANALYSIS_CONTRACT,
            "cohort": cohort_contracts["discovery"],
            "prediction_sha256": _sha256_file(discovery_prediction),
            "method": admixture_phewas.PHETK_METHOD,
            "min_cases": admixture_phewas.MIN_CASES,
            "min_phecode_count": admixture_phewas.MIN_PHECODE_COUNT,
        }
        discovery_phewas_current = not force and _stage_current(
            "discovery_phewas",
            discovery_phewas_contract,
            [discovery_output, discovery_assoc_manifest],
        )
        discovery = admixture_phewas.run_discovery(force=not discovery_phewas_current)
        _write_stage_manifest(
            "discovery_phewas",
            discovery_phewas_contract,
            [discovery_output, discovery_assoc_manifest],
        )
        hits = admixture_phewas.discovery_hits(discovery)
        if hits.empty:
            raise RuntimeError(
                f"No discovery association passed p < {admixture_phewas.DISCOVERY_P_THRESHOLD:.0e}; "
                "replication and Figure 2_1025 cannot proceed"
            )
        region_ids = hits["region_id"].astype(str).drop_duplicates().tolist()
        replication_prediction = (
            admixture_plrs.analysis_dir() / "eur_dosage_predictions.replication.tsv.gz"
        )
        replication_prediction_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "cohort": cohort_contracts["replication"],
            "model_manifest_sha256": _sha256_file(_stage_manifest_path("plrs_models")),
            "discovery_regions": sorted(region_ids),
            "min_observed_model_fraction": MIN_OBSERVED_MODEL_FRACTION,
        }
        replication_prediction_current = not force and _stage_current(
            "replication_predictions",
            replication_prediction_contract,
            [replication_prediction],
        )
        if not replication_prediction_current:
            with patch.object(
                admixture_plrs,
                "MIN_OBSERVED_FEATURE_FRACTION",
                MIN_OBSERVED_MODEL_FRACTION,
            ):
                replication_prediction = admixture_plrs.predict_replication(
                    summary, region_ids=region_ids, force=True
                )
            _write_stage_manifest(
                "replication_predictions",
                replication_prediction_contract,
                [replication_prediction],
            )
        replication_output = admixture_phewas.analysis_dir() / "phewas.replication.tsv.gz"
        meta_output = admixture_phewas.analysis_dir() / "meta_analysis_results.tsv"
        replication_assoc_manifest = replication_output.with_suffix(
            replication_output.suffix + ".manifest.json"
        )
        from . import figure2_1025_tables

        gene_table_path = (
            admixture_finemap.analysis_dir() / "refgene_transcripts.tsv.gz"
        )
        admixture_finemap._gene_table(force=False)
        phecode_source_paths = figure2_1025_tables.snv_phewas_source_paths()
        replication_phewas_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "analysis_contract": admixture_phewas.ANALYSIS_CONTRACT,
            "cohort": cohort_contracts["replication"],
            "prediction_sha256": _sha256_file(replication_prediction),
            "discovery_association_sha256": _sha256_file(discovery_output),
            "meta_annotation_columns": figure2_1025_tables.META_ANNOTATION_COLUMNS,
            "meta_gene_source_contract": (
                figure2_1025_tables.META_GENE_SOURCE_CONTRACT
            ),
            "meta_phecode_source_contract": (
                figure2_1025_tables.META_PHECODE_SOURCE_CONTRACT
            ),
            "meta_gene_table_sha256": (
                _sha256_file(gene_table_path) if gene_table_path.is_file() else None
            ),
            "meta_phecode_source_sha256": {
                path.name: _sha256_file(path) for path in phecode_source_paths
            },
        }
        replication_phewas_current = not force and _stage_current(
            "replication_phewas",
            replication_phewas_contract,
            [replication_output, replication_assoc_manifest, meta_output],
        )
        phewas_outputs = admixture_phewas.run_replication(
            discovery,
            replication_prediction,
            force=not replication_phewas_current,
        )
        if not replication_phewas_current:
            figure2_1025_tables.annotate_meta_analysis_file()
        _write_stage_manifest(
            "replication_phewas",
            replication_phewas_contract,
            [replication_output, replication_assoc_manifest, meta_output],
        )
        meta = pd.read_csv(phewas_outputs["meta"], sep="\t", dtype={"phecode": str})
        main_hit = _require_main_hit(meta)
        finemap_table = (
            admixture_finemap.analysis_dir()
            / f"susie_{admixture_finemap.hit_tag(main_hit)}.tsv.gz"
        )
        finemap_summary = admixture_finemap.analysis_dir() / "finemap_summary.tsv"
        gene_table = admixture_finemap.analysis_dir() / "refgene_transcripts.tsv.gz"
        finemap_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "analysis_contract": admixture_finemap.FINEMAP_CONTRACT,
            "discovery_cohort": cohort_contracts["discovery"],
            "region_id": str(main_hit["region_id"]),
            "phecode": str(main_hit["phecode"]),
            "window_pad_bp": admixture_finemap.WINDOW_PAD_BP,
            **admixture_finemap.filter_parameters(),
        }
        finemap_current = not force and _stage_current(
            "finemap",
            finemap_contract,
            [finemap_table, finemap_summary, gene_table],
        )
        if finemap_current:
            finemap_outputs = {
                "summary": finemap_summary,
                "main_table": finemap_table,
                "genes": gene_table,
            }
        else:
            finemap_outputs = admixture_finemap.run(force=True)
            _write_stage_manifest(
                "finemap",
                finemap_contract,
                [finemap_table, finemap_summary, gene_table],
            )
        region = json.loads(
            (lai_dir() / "display_region.json").read_text(encoding="utf-8")
        )
        window_start, window_end = figure2_data.locus_window(region)
        sv_output = figure2_sv.cache_path()
        sv_manifest = figure2_sv.manifest_path()
        sv_contract = {
            "pipeline": PIPELINE_CONTRACT,
            "sample_sha256": _sample_sha(ids),
            "region_id": str(region["region_id"]),
            "window_start": window_start,
            "window_end": window_end,
            "method": "overall_nearest_marker",
            "sv_min_length_bp": figure2_sv.SV_MIN_LENGTH_BP,
        }
        sv_current = not force and _stage_current(
            "regional_svs", sv_contract, [sv_output, sv_manifest]
        )
        svs = figure2_sv.ensure_cache(
            region, window_start, window_end, force=not sv_current
        )
        _write_stage_manifest("regional_svs", sv_contract, [sv_output, sv_manifest])
        figure2_data.prepare_locus_outputs(force=force)

        table_paths = figure2_1025_tables.write_tables()

        from . import figure2_loci

        nominal_loci = figure2_loci.run(force=force, render=render, ids=ids)

        figure = None
        supplementary_phewas = None
        if render:
            from . import figure4_meta, figure2_1025_plots
            from . import figure2_discovery_phewas as admixture_discovery_phewas

            full_phewas_annotations = figure4_meta._stage_gcs_file(
                Path(config.PHEWAS_FILTERED_TSV),
                config.PHEWAS_FILTERED_GS,
                force=False,
            )
            supplementary_phewas = admixture_discovery_phewas.render_all(
                discovery_output,
                [full_phewas_annotations, *phecode_source_paths],
                force=force,
            )

            figure = figure2_1025_plots.build_figure2_1025(
                clean_run_data=False, clean_run_plots=True, ids=ids
            )
        return {
            "status": "ready",
            "lai": lai_result,
            "discovery_n": len(discovery_cov),
            "replication_n": len(replication_cov),
            "main_region": str(main_hit["region_id"]),
            "main_phecode": str(main_hit["phecode"]),
            "main_meta_p": float(main_hit["p_meta"]),
            "n_regional_svs": len(svs),
            "finemap": finemap_outputs,
            "nominal_loci": nominal_loci,
            "tables": table_paths,
            "figure": figure,
            "supplementary_phewas": supplementary_phewas,
        }


def build_figure_only(*, clean_run_plots: bool = True):
    """Render from verified v2 caches without rerunning controlled-data stages."""
    ids = training_ids()
    with analysis_context(ids):
        from . import figure2_1025_plots

        return figure2_1025_plots.build_figure2_1025(
            clean_run_data=False, clean_run_plots=clean_run_plots, ids=ids
        )
