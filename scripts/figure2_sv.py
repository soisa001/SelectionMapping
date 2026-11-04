"""Regional SV discovery and cohort-wide local-ancestry assignment.

Figure 2 uses the phased 1,025-person SNV/SV panel to identify sequence-resolved
insertions/deletions and the FLARE ``AN1``/``AN2`` calls for the matching cohort
to estimate cohort-wide EUR local ancestry at each SV coordinate. Because an
SV need not be a FLARE marker, its plotted ancestry is the all-haplotype EUR
fraction at the nearest LAI marker.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import admixture_inputs, admixture_lai
from .common import config

SV_MIN_LENGTH_BP = 50
LAI_NEAREST_PAD_BP = 1_000_000
SV_ANCESTRY_CONTRACT = "figure5_sv_overall_lai_nearest_marker_1k_v4"
SV_COLUMNS = [
    "contract", "region_id", "chrom", "pos", "sv_type", "sv_length_bp",
    "variant_id", "alt_allele_count", "called_allele_count",
    "alt_allele_frequency", "overall_eur_frac", "n_carrier_haplotypes",
    "n_unphased_heterozygotes",
    "n_evaluable_haplotypes", "n_eur_haplotypes",
    "n_afr_haplotypes", "n_other_haplotypes", "ancestry_method",
    "nearest_lai_marker", "nearest_lai_distance_bp",
    "left_lai_marker", "right_lai_marker",
]


def cache_path() -> Path:
    return admixture_lai.analysis_dir() / "display_region_sv_ancestry.tsv.gz"


def manifest_path() -> Path:
    return cache_path().with_suffix(".manifest.json")


def sv_source_path(chrom: int) -> Path:
    """Return the mounted phased SNV/SV source for the 1,025-person LAI set."""
    token = f"chr{int(chrom)}"
    return Path(config.FIGURE2_SV_VCF_PATTERN.format(chrom=token)).expanduser()


def stage_sv_source_vcf(chrom: int, *, force: bool = False) -> Path:
    """Resolve the indexed phased SNV/SV chromosome VCF used for carriers."""
    mounted = sv_source_path(chrom)
    mounted_index = Path(str(mounted) + ".tbi")
    if not force and mounted.exists() and mounted_index.exists():
        return mounted
    token = f"chr{int(chrom)}"
    remote = config.FIGURE2_SV_VCF_GS_PATTERN.format(chrom=token)
    name = Path(remote).name
    local = admixture_inputs.stage_file(
        remote,
        Path("sv_source_1k") / name,
        force=force,
        required=True,
    )
    assert local is not None
    admixture_inputs.stage_file(
        remote + ".tbi",
        Path("sv_source_1k") / (name + ".tbi"),
        force=force,
        required=True,
    )
    return local


def classify_sv(
    ref: str,
    alt: str,
    *,
    info_type: str | None = None,
    info_length: float | None = None,
) -> tuple[str, int] | None:
    """Classify INS/DEL using INFO when present, otherwise sequence length."""
    normalized_type = str(info_type or "").strip().upper()
    if normalized_type in {"INS", "DEL"}:
        length = abs(int(float(info_length))) if info_length is not None else 0
        if length <= 0:
            length = abs(len(str(alt)) - len(str(ref)))
        if length >= SV_MIN_LENGTH_BP:
            return normalized_type, length
    difference = len(str(alt)) - len(str(ref))
    if difference >= SV_MIN_LENGTH_BP:
        return "INS", difference
    if difference <= -SV_MIN_LENGTH_BP:
        return "DEL", abs(difference)
    return None


def phased_carrier_haplotypes(
    samples: list[str] | np.ndarray,
    genotypes: list[list[Any]],
) -> tuple[list[tuple[str, int]], int]:
    """Return ``(sample, hap=1/2)`` ALT carriers and unphased exclusions."""
    carriers: list[tuple[str, int]] = []
    unphased_heterozygotes = 0
    for sample, genotype in zip(samples, genotypes):
        if len(genotype) < 2:
            continue
        alleles = (int(genotype[0]), int(genotype[1]))
        phased = bool(genotype[2]) if len(genotype) > 2 else False
        if min(alleles) < 0:
            continue
        if not phased and alleles[0] != alleles[1]:
            unphased_heterozygotes += 1
            continue
        for hap_index, allele in enumerate(alleles, start=1):
            if allele > 0:
                carriers.append((str(sample), hap_index))
    return carriers, unphased_heterozygotes


def nearest_marker_overall_ancestry(
    sv_pos: int,
    marker_positions: np.ndarray,
    ancestry_hap1: np.ndarray,
    ancestry_hap2: np.ndarray,
    eur_index: int,
    afr_index: int | None = None,
) -> dict[str, int | float]:
    """Return the cohort-wide ancestry fraction at the nearest LAI marker."""
    positions = np.asarray(marker_positions, dtype=np.int64)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("No LAI markers are available for nearest-marker assignment")
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    hap1 = np.asarray(ancestry_hap1)[order]
    hap2 = np.asarray(ancestry_hap2)[order]
    insertion = int(np.searchsorted(positions, int(sv_pos), side="left"))
    left_index = max(0, insertion - 1)
    right_index = min(len(positions) - 1, insertion)
    if positions[right_index] == int(sv_pos):
        left_index = right_index
    left_pos = int(positions[left_index])
    right_pos = int(positions[right_index])
    if abs(int(sv_pos) - left_pos) <= abs(right_pos - int(sv_pos)):
        nearest_index = left_index
    else:
        nearest_index = right_index
    nearest_pos = int(positions[nearest_index])
    ancestry_calls = np.concatenate([
        np.asarray(hap1[nearest_index]).reshape(-1),
        np.asarray(hap2[nearest_index]).reshape(-1),
    ]).astype(int)
    ancestry_calls = ancestry_calls[ancestry_calls >= 0]
    n_evaluable = len(ancestry_calls)
    n_eur = sum(call == int(eur_index) for call in ancestry_calls)
    n_afr = (
        sum(call == int(afr_index) for call in ancestry_calls)
        if afr_index is not None else 0
    )
    n_other = n_evaluable - n_eur - n_afr if afr_index is not None else 0
    return {
        "overall_eur_frac": (
            float(n_eur / n_evaluable) if n_evaluable else float("nan")
        ),
        "n_evaluable_haplotypes": n_evaluable,
        "n_eur_haplotypes": n_eur,
        "n_afr_haplotypes": n_afr,
        "n_other_haplotypes": n_other,
        "ancestry_method": "overall_nearest_marker",
        "nearest_lai_marker": nearest_pos,
        "nearest_lai_distance_bp": abs(nearest_pos - int(sv_pos)),
        "left_lai_marker": left_pos,
        "right_lai_marker": right_pos,
    }


def match_finemap_svs(
    regional_svs: pd.DataFrame,
    finemap: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact SV association/effect and min-p LD metrics for plotting."""
    output = regional_svs.copy()
    metric_columns = [
        "finemap_variant_id", "exact_finemap_match", "beta_score", "p_score",
        "neg_log10_p", "r2_to_min_p",
    ]
    if output.empty:
        for column in metric_columns:
            output[column] = pd.Series(dtype="object")
        return output
    candidates = finemap.copy()
    candidates["pos"] = pd.to_numeric(candidates.get("pos"), errors="coerce")
    is_sv = candidates.get("is_sv", pd.Series(False, index=candidates.index))
    candidates = candidates[
        is_sv.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    ].copy()
    candidates["sv_type"] = candidates.get(
        "sv_type", pd.Series("", index=candidates.index)
    ).astype(str).str.upper()
    candidates["sv_length_bp"] = pd.to_numeric(
        candidates.get("sv_length_bp"), errors="coerce"
    )
    rows = []
    for sv in output.to_dict("records"):
        matches = candidates[
            candidates["pos"].eq(int(sv["pos"]))
            & candidates["sv_type"].eq(str(sv["sv_type"]).upper())
            & candidates["sv_length_bp"].eq(int(sv["sv_length_bp"]))
        ].copy()
        matches["p_score"] = pd.to_numeric(matches.get("p_score"), errors="coerce")
        matches = matches.sort_values("p_score", na_position="last")
        match = None if matches.empty else matches.iloc[0]
        sv.update({
            "finemap_variant_id": "" if match is None else str(match["variant_id"]),
            "exact_finemap_match": match is not None,
            **{
                column: float("nan") if match is None else pd.to_numeric(
                    match.get(column), errors="coerce"
                )
                for column in ("beta_score", "p_score", "neg_log10_p", "r2_to_min_p")
            },
        })
        rows.append(sv)
    return pd.DataFrame(rows)


def _record_info(record: Any, key: str) -> Any:
    try:
        return record.INFO.get(key)
    except (KeyError, TypeError):
        return None


def _discover_svs(
    vcf_path: Path,
    chrom: int,
    window_start: int,
    window_end: int,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, int]]]]:
    from cyvcf2 import VCF

    vcf = VCF(str(vcf_path))
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from {vcf_path}")
    samples = list(vcf.samples)
    rows: list[dict[str, Any]] = []
    carriers_by_variant: dict[str, list[tuple[str, int]]] = {}
    try:
        for record in vcf(f"{token}:{window_start}-{window_end}"):
            if len(record.ALT or []) != 1:
                continue
            sv = classify_sv(
                record.REF,
                record.ALT[0],
                info_type=_record_info(record, "SVTYPE"),
                info_length=_record_info(record, "SVLEN"),
            )
            if sv is None:
                continue
            sv_type, sv_length = sv
            carriers, n_unphased = phased_carrier_haplotypes(
                samples, record.genotypes
            )
            alleles = np.asarray(record.genotypes, dtype=object)[:, :2]
            called = np.asarray(alleles, dtype=int) >= 0
            called_alleles = int(called.sum())
            alt_alleles = int((np.asarray(alleles, dtype=int)[called] > 0).sum())
            variant_id = f"chr{chrom}:{int(record.POS)}:{sv_type}:{sv_length}"
            rows.append({
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
            })
            carriers_by_variant[variant_id] = carriers
    finally:
        vcf.close()
    return rows, carriers_by_variant


def _load_lai_markers(
    vcf_path: Path,
    chrom: int,
    window_start: int,
    window_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    from cyvcf2 import VCF

    ancestry_mapping = admixture_lai.parse_ancestry_header(vcf_path)
    eur_index, afr_index, _, _ = admixture_lai.ancestry_indices(ancestry_mapping)
    vcf = VCF(str(vcf_path))
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from {vcf_path}")
    positions: list[int] = []
    hap1: list[np.ndarray] = []
    hap2: list[np.ndarray] = []
    query_start = max(1, int(window_start) - LAI_NEAREST_PAD_BP)
    query_end = int(window_end) + LAI_NEAREST_PAD_BP
    try:
        for record in vcf(f"{token}:{query_start}-{query_end}"):
            an1, an2 = record.format("AN1"), record.format("AN2")
            if an1 is None or an2 is None:
                continue
            positions.append(int(record.POS))
            hap1.append(np.asarray(an1).reshape(-1).astype(np.int16))
            hap2.append(np.asarray(an2).reshape(-1).astype(np.int16))
    finally:
        vcf.close()
    if not positions:
        raise ValueError(f"No LAI markers found around chr{chrom}:{window_start}-{window_end}")
    return (
        np.asarray(positions, dtype=np.int64),
        np.stack(hap1),
        np.stack(hap2),
        eur_index,
        afr_index,
    )


def _cache_matches(region: dict, window_start: int, window_end: int) -> bool:
    path, manifest = cache_path(), manifest_path()
    if not path.exists() or not manifest.exists():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("contract") == SV_ANCESTRY_CONTRACT
        and str(metadata.get("region_id")) == str(region["region_id"])
        and int(metadata.get("window_start", -1)) == int(window_start)
        and int(metadata.get("window_end", -1)) == int(window_end)
        and int(metadata.get("sv_source_n_samples", 0)) > 0
        and int(metadata.get("lai_n_samples", 0)) > 0
        and bool(metadata.get("sv_source_sample_sha256"))
        and bool(metadata.get("lai_sample_sha256"))
        and path.stat().st_size > 0
    )


def _sample_manifest(vcf_path: Path) -> dict[str, int | str]:
    from cyvcf2 import VCF

    vcf = VCF(str(vcf_path))
    try:
        samples = [str(sample) for sample in vcf.samples]
    finally:
        vcf.close()
    digest = hashlib.sha256("\n".join(samples).encode("utf-8")).hexdigest()
    return {
        "n_samples": len(samples),
        "sample_sha256": digest,
        "size_bytes": vcf_path.stat().st_size,
    }


def load_cache() -> pd.DataFrame:
    path = cache_path()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=SV_COLUMNS)
    return pd.read_csv(path, sep="\t")


def ensure_cache(
    region: dict,
    window_start: int,
    window_end: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build or reuse cohort-wide nearest-marker SV-position LAI."""
    if not force and _cache_matches(region, window_start, window_end):
        return load_cache()
    chrom = int(region["chrom"])
    source_vcf = stage_sv_source_vcf(chrom, force=False)
    lai_vcf = admixture_lai.stage_lai_chromosome(chrom, force=False)
    sv_rows, _ = _discover_svs(
        source_vcf, chrom, window_start, window_end
    )
    if sv_rows:
        lai = _load_lai_markers(
            lai_vcf, chrom, window_start, window_end
        )
        positions, hap1, hap2, eur_index, afr_index = lai
        for row in sv_rows:
            row.update(nearest_marker_overall_ancestry(
                int(row["pos"]),
                positions,
                hap1,
                hap2,
                eur_index,
                afr_index,
            ))
    frame = pd.DataFrame(sv_rows)
    for column in SV_COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame["contract"] = SV_ANCESTRY_CONTRACT
    frame["region_id"] = str(region["region_id"])
    frame = frame[SV_COLUMNS]
    output = cache_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    temporary.replace(output)
    metadata = {
        "contract": SV_ANCESTRY_CONTRACT,
        "region_id": str(region["region_id"]),
        "chrom": chrom,
        "window_start": int(window_start),
        "window_end": int(window_end),
        "n_svs": len(frame),
        "sv_min_length_bp": SV_MIN_LENGTH_BP,
        "lai_nearest_pad_bp": LAI_NEAREST_PAD_BP,
        "sv_source_vcf": str(source_vcf),
        "lai_vcf": str(lai_vcf),
    }
    source_manifest = _sample_manifest(source_vcf)
    lai_manifest = _sample_manifest(lai_vcf)
    metadata.update({
        "sv_source_n_samples": source_manifest["n_samples"],
        "sv_source_sample_sha256": source_manifest["sample_sha256"],
        "sv_source_size_bytes": source_manifest["size_bytes"],
        "lai_n_samples": lai_manifest["n_samples"],
        "lai_sample_sha256": lai_manifest["sample_sha256"],
        "lai_size_bytes": lai_manifest["size_bytes"],
    })
    manifest_tmp = manifest_path().with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path())
    print(
        f"[figure2_sources][sv] {len(frame)} regional SVs; cohort-wide nearest-marker LAI "
        f"written to {output}",
        flush=True,
    )
    return frame
