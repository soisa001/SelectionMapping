"""Supplementary Figure S9: discovery-hit MAF versus effect size.

The source notebook built this panel from the genic, converged, MAC >= 20
discovery results forwarded at p <= 1e-3.  This module makes that contract
explicit, colors SV-linked hits separately, and refuses to render unless every
expected discovery hit has a finite minor allele frequency.

The row unit is one unique ``variant_tag`` x ``phecode`` association.  The
default expected count (446) comes from the controlled-data audit of the exact
source-notebook ``p <= 1e-3`` contract; the manuscript's earlier 447 was stale.
Participant-level cohort files are read only to reconstruct variant MAC when
the discovery results do not already contain ``mac_disc``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

# Workbench notebooks set the inline backend even when this module is run in
# a minimal headless uv environment that does not include matplotlib-inline.
# Override it before importing scripts.common, whose package initializer imports
# Matplotlib through compose.py.
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from .common import config, style

FIGURE_ID = "s9_phewas"
FIGURE_STEM = "Figure_S9_MAF_Effect"
DISCOVERY_P_THRESHOLD = 1e-3
MAC_MIN = 20
EXPECTED_HIT_COUNT = 446
EXPECTED_SV_REGION_COUNT = 17
SV_LABEL_SEED = 20260820
SOURCE_SCHEMA_VERSION = 2

CATEGORY_ORDER = ("discovery hit", "SV-linked hit", "meta-significant")
CATEGORY_COLORS = {
    "discovery hit": "#A7A7A7",
    "SV-linked hit": "#007C83",
    "meta-significant": "#B94F3A",
}
SOURCE_DATA_COLUMNS = (
    "variant_tag",
    "phecode",
    "phecode_string",
    "p",
    "beta",
    "se",
    "maf",
    "mac_disc",
    "is_sv_linked",
    "sv_annotation",
    "is_meta_significant",
    "category",
    "abs_beta",
)


class S9ContractError(ValueError):
    """Raised when an input would produce an incomplete or ambiguous S9."""

    def __init__(self, message: str, *, audit: dict[str, object] | None = None):
        super().__init__(message)
        self.audit = audit or {}


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )


def _nonempty(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return values.notna() & ~text.isin({"", ".", "NA", "None", "nan", "<NA>"})


def _has_gene(values: pd.Series) -> pd.Series:
    return _nonempty(values)


def _normalize_chromosome(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"^chr", "", regex=True)
    return text.str.replace(r"\.0$", "", regex=True)


def _sentinel_variant_tags(
    sentinels: pd.DataFrame, *, prefer_linked_lead: bool = False
) -> pd.Series:
    variant = _first_column(sentinels, ("variant_tag", "snp", "SNP", "variant_id", "tag"))
    if variant and not prefer_linked_lead:
        return sentinels[variant].astype(str).str.strip()

    # These are complete schema alternatives, in precedence order.  In the
    # archived *.sigp.linked.tsv, generic ``pos`` can coexist with the lead's
    # ``max_IHS_position``; the source notebook explicitly uses the latter.
    schemas = [
        (("chrom", "chr"), "max_IHS_position", "IHS_REF", "IHS_ALT"),
        (("lead_chr",), "lead_pos", "lead_ref", "lead_alt"),
    ]
    if not prefer_linked_lead:
        schemas.append((("chrom", "chr"), "pos", "ref", "alt"))
    selected: tuple[str, str, str, str] | None = None
    for chrom_options, pos, ref, alt in schemas:
        chrom = _first_column(sentinels, chrom_options)
        if chrom and all(column in sentinels.columns for column in (pos, ref, alt)):
            selected = chrom, pos, ref, alt
            break
    if selected is None and variant:
        return sentinels[variant].astype(str).str.strip()
    if selected is None:
        raise S9ContractError("Sentinel table cannot construct a source-notebook variant_tag")
    chrom, pos, ref, alt = selected
    positions = pd.to_numeric(sentinels[pos], errors="coerce")
    if positions.isna().any():
        raise S9ContractError("Sentinel position column contains non-numeric values")
    return (
        _normalize_chromosome(sentinels[chrom])
        + "_"
        + positions.astype(np.int64).astype(str)
        + "_"
        + sentinels[ref].astype(str).str.strip()
        + "_"
        + sentinels[alt].astype(str).str.strip()
    )


def _linked_rows(sentinels: pd.DataFrame) -> pd.Series:
    """Identify rows represented as SV-linked in old and source schemas."""
    linked = pd.Series(False, index=sentinels.index)
    for column in (
        "has_linked_SV",
        "has_linked_sv",
        "_has_linked_sv",
        "lead_is_SV",
        "lead_is_sv",
    ):
        if column in sentinels.columns:
            linked |= _truthy(sentinels[column])
    for column in ("SV_n_linked", "n_linked_svs", "n_linked_sv"):
        if column in sentinels.columns:
            linked |= pd.to_numeric(sentinels[column], errors="coerce").fillna(0).gt(0)
    # The archived source notebook defines its 17-region set exactly as rows
    # where Selection Scan SV_TYPE is populated.  The linked TSV preserves
    # this field and is the staged, text-based equivalent used by this runner.
    for column in ("SV_TYPE", "sv_type"):
        if column in sentinels.columns:
            linked |= _nonempty(sentinels[column])
    for column in ("SV_CHR", "sv_chr", "linked_sv_chr"):
        if column in sentinels.columns:
            linked |= _nonempty(sentinels[column])
    return linked


def _sentinel_contract(
    sentinels: pd.DataFrame, *, prefer_linked_lead: bool = False
) -> pd.DataFrame:
    out = sentinels.copy()
    out["variant_tag"] = _sentinel_variant_tags(
        out, prefer_linked_lead=prefer_linked_lead
    )

    gene = _first_column(out, ("gene_in_window", "gene", "genes_in_region"))
    out["is_genic"] = _has_gene(out[gene]) if gene else True

    out["is_sv_linked"] = _linked_rows(out)

    return (
        out.groupby("variant_tag", as_index=False)
        .agg(is_genic=("is_genic", "any"), is_sv_linked=("is_sv_linked", "any"))
    )


def _clean_scalar(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", ".", "NA", "None", "nan", "<NA>"}:
        return None
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _row_scalar(row: pd.Series, names: Iterable[str]) -> str | None:
    for name in names:
        if name in row.index:
            value = _clean_scalar(row[name])
            if value is not None:
                return value
    return None


def _reference_sv_annotation(row: pd.Series) -> str:
    """Create a compact, source-backed label for one linked SV region."""
    sv_id = _row_scalar(
        row, ("SV_ID", "sv_id", "linked_sv_id", "SV_rs", "SV_variant_tag")
    )
    chrom = _row_scalar(row, ("SV_CHR", "sv_chr", "linked_sv_chr", "chrom", "chr"))
    position = _row_scalar(row, ("SV_POS", "sv_pos", "linked_sv_pos"))
    sv_type = _row_scalar(row, ("SV_TYPE", "sv_type"))
    sv_length = _row_scalar(
        row,
        (
            "SV_LEN",
            "sv_len",
            "linked_sv_len",
            "SVLEN",
            "svlen",
            "SV_LENGTH",
            "sv_length",
            "linked_sv_length",
            "SV_LENGTH_BP",
            "sv_length_bp",
            "SV_SIZE",
            "sv_size",
            "SV_SIZE_BP",
            "sv_size_bp",
        ),
    )
    sv_start = _row_scalar(
        row, ("SV_START", "sv_start", "linked_sv_start")
    )
    sv_end = _row_scalar(
        row, ("SV_END", "sv_end", "linked_sv_end", "SV_STOP", "sv_stop")
    )
    sv_ref = _row_scalar(row, ("SV_REF", "sv_ref", "linked_sv_ref"))
    sv_alt = _row_scalar(row, ("SV_ALT", "sv_alt", "linked_sv_alt"))
    start = _row_scalar(row, ("start", "region_start"))
    stop = _row_scalar(row, ("stop", "end", "region_end"))

    length_bp: int | None = None
    if sv_length is not None:
        try:
            length_bp = round(abs(float(sv_length.replace(",", ""))))
        except ValueError:
            length_bp = None
    if length_bp in (None, 0) and sv_start is not None and sv_end is not None:
        try:
            length_bp = round(
                abs(
                    float(sv_end.replace(",", ""))
                    - float(sv_start.replace(",", ""))
                )
            )
        except ValueError:
            length_bp = None
    if (
        length_bp in (None, 0)
        and sv_ref is not None
        and sv_alt is not None
        and not (sv_alt.startswith("<") and sv_alt.endswith(">"))
    ):
        length_bp = abs(len(sv_alt) - len(sv_ref)) or None

    if chrom is not None and position is not None:
        identity = f"chr{chrom.removeprefix('chr')}:{position}"
    elif chrom is not None and start is not None and stop is not None:
        identity = f"chr{chrom.removeprefix('chr')}:{start}-{stop}"
    elif sv_id is not None:
        identity = sv_id
    else:
        identity = "linked region"
    display_type = sv_type.upper() if sv_type is not None else None
    if length_bp is not None and display_type is not None:
        suffix = f" ({length_bp:,} bp {display_type})"
    elif display_type is not None:
        suffix = f" ({display_type})"
    elif length_bp is not None:
        suffix = f" ({length_bp:,} bp SV)"
    else:
        suffix = ""
    return f"{identity}{suffix}"


def _large_allele_sv_annotation(tag: str) -> str | None:
    parts = tag.split("_")
    if len(parts) < 4:
        return None
    chrom, position, ref = parts[:3]
    alt = "_".join(parts[3:])
    length_difference = len(alt) - len(ref)
    if abs(length_difference) < 50:
        return None
    sv_type = "INS" if length_difference > 0 else "DEL"
    return (
        f"chr{chrom.removeprefix('chr')}:{position} "
        f"({abs(length_difference):,} bp {sv_type})"
    )


def _sv_linked_hit_mask(
    variant_tags: pd.Series, linked_sentinels: pd.DataFrame
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Reproduce the source notebook's SV-lead and linked-region rules."""
    linked_reference_rows = _linked_rows(linked_sentinels)
    reference_rows = linked_sentinels.loc[linked_reference_rows].copy()
    reference_rows["_lead_tag"] = _sentinel_variant_tags(
        linked_sentinels, prefer_linked_lead=True
    ).loc[linked_reference_rows]
    reference_rows["_sv_annotation"] = reference_rows.apply(
        _reference_sv_annotation, axis=1
    )
    linked_leads = set(reference_rows["_lead_tag"])
    tags = variant_tags.astype(str).str.strip()
    exact_lead = tags.isin(linked_leads)

    # A tag encoding an insertion/deletion of at least 50 bp is itself an SV.
    annotations = tags.map(_large_allele_sv_annotation).astype("string")
    large_allele = annotations.notna()
    interval = pd.Series(False, index=tags.index)

    for lead_tag, annotation in reference_rows[
        ["_lead_tag", "_sv_annotation"]
    ].itertuples(index=False, name=None):
        needs_annotation = tags.eq(lead_tag) & annotations.isna()
        annotations.loc[needs_annotation] = annotation

    chrom_column = _first_column(linked_sentinels, ("chrom", "chr"))
    start_column = _first_column(linked_sentinels, ("start", "region_start"))
    stop_column = _first_column(linked_sentinels, ("stop", "end", "region_end"))
    if chrom_column and start_column and stop_column:
        tag_parts = tags.str.split("_", n=2, expand=True)
        if tag_parts.shape[1] >= 2:
            hit_chrom = _normalize_chromosome(tag_parts[0])
            hit_position = pd.to_numeric(tag_parts[1], errors="coerce")

            region_rows = reference_rows.copy()
            region_rows["_chrom"] = _normalize_chromosome(region_rows[chrom_column])
            region_rows["_start"] = pd.to_numeric(
                region_rows[start_column], errors="coerce"
            )
            region_rows["_stop"] = pd.to_numeric(
                region_rows[stop_column], errors="coerce"
            )
            region_rows = region_rows.dropna(subset=["_start", "_stop"])
            for region_chrom, region_start, region_stop, annotation in region_rows[
                ["_chrom", "_start", "_stop", "_sv_annotation"]
            ].itertuples(index=False, name=None):
                region_match = (
                    hit_chrom.eq(region_chrom)
                    & hit_position.ge(min(region_start, region_stop))
                    & hit_position.le(max(region_start, region_stop))
                )
                interval |= region_match
                annotations.loc[region_match & annotations.isna()] = annotation

    linked = exact_lead | large_allele | interval
    fallback = linked & annotations.isna()
    annotations.loc[fallback] = tags.loc[fallback].map(
        lambda tag: "chr"
        + tag.replace("chr", "", 1).replace("_", ":", 1).split("_", 1)[0]
    )
    audit = {
        "reference_rows": len(linked_sentinels),
        "linked_reference_regions": int(linked_reference_rows.sum()),
        "exact_lead_association_rows": int(exact_lead.sum()),
        "large_allele_association_rows": int(large_allele.sum()),
        "interval_association_rows": int(interval.sum()),
        "sv_linked_association_rows": int(linked.sum()),
        "annotated_sv_association_rows": int((linked & annotations.notna()).sum()),
    }
    return linked, annotations, audit


def _safe_tag(tag: str) -> str:
    if len(tag) <= 200:
        return tag
    chrom, pos, ref, alt = tag.split("_", 3)
    return f"{chrom}_{pos}_{ref[:100]}_{alt[:100]}"


def compute_variant_mac(cohort_dir: Path, variant_tags: Iterable[str]) -> pd.DataFrame:
    """Compute minor allele count from source-notebook cohort TSVs."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for tag in sorted(set(map(str, variant_tags))):
        cohort_path = cohort_dir / f"{_safe_tag(tag)}_cohort.tsv"
        if not cohort_path.exists():
            missing.append(tag)
            continue
        genotype = pd.read_csv(cohort_path, sep="\t", usecols=["genotype"])["genotype"]
        genotype = pd.to_numeric(genotype, errors="coerce")
        if genotype.isna().any():
            raise S9ContractError(f"Non-numeric genotype in {cohort_path}")
        n_alt = int(genotype.sum())
        n_total = 2 * len(genotype)
        rows.append(
            {
                "variant_tag": tag,
                "mac_disc": min(n_alt, n_total - n_alt),
                "n_genotyped_disc": len(genotype),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise S9ContractError(
            f"Missing cohort TSVs for {len(missing)} discovery variants; first: {preview}"
        )
    return pd.DataFrame(rows)


def _meta_keys(meta: pd.DataFrame) -> set[tuple[str, str]]:
    variant = _first_column(meta, ("variant_tag", "snp", "SNP", "variant_id", "tag"))
    phecode = _first_column(meta, ("phecode", "Phecode", "phenotype"))
    if not variant or not phecode:
        raise S9ContractError("Meta table needs variant_tag and phecode columns")
    flag = _first_column(meta, ("meta_significant", "is_meta_significant"))
    keep = _truthy(meta[flag]) if flag else pd.Series(True, index=meta.index)
    return set(
        zip(
            meta.loc[keep, variant].astype(str).str.strip(),
            meta.loc[keep, phecode].astype(str).str.strip(),
        )
    )


def prepare_s9_data(
    discovery: pd.DataFrame,
    meta: pd.DataFrame,
    sentinels: pd.DataFrame,
    *,
    sv_links: pd.DataFrame | None = None,
    discovery_p_threshold: float = DISCOVERY_P_THRESHOLD,
    mac_min: int = MAC_MIN,
    expected_hit_count: int | None = EXPECTED_HIT_COUNT,
    expected_sv_region_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the notebook S9 contract and return plot rows plus an audit.

    MAF is filled row-by-row from ``mac_disc / (2 * tested N)`` whenever an
    otherwise valid MAF column contains a missing value.  This is deliberately
    stricter than Matplotlib, which silently omits non-finite coordinates.
    """
    frame = discovery.copy()
    variant = _first_column(frame, ("variant_tag", "snp", "SNP", "variant_id", "tag"))
    phecode = _first_column(frame, ("phecode", "Phecode", "phenotype"))
    phecode_string = _first_column(
        frame,
        (
            "phecode_string",
            "phecode_name",
            "phecode_description",
            "description",
            "meaning",
            "phenotype_name",
        ),
    )
    p_col = _first_column(frame, ("p", "P", "pval", "pvalue", "p_value", "p_ind"))
    beta_col = _first_column(frame, ("beta", "beta_ind", "Beta", "BETA", "b", "log_odds"))
    se_col = _first_column(frame, ("se", "se_ind", "SE", "std_error"))
    required = {
        "variant_tag": variant,
        "phecode": phecode,
        "phecode_string": phecode_string,
        "p": p_col,
        "beta": beta_col,
        "se": se_col,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise S9ContractError("Discovery table is missing: " + ", ".join(missing))

    frame = frame.rename(columns={
        variant: "variant_tag",
        phecode: "phecode",
        phecode_string: "phecode_string",
        p_col: "p",
        beta_col: "beta",
        se_col: "se",
    })
    frame["variant_tag"] = frame["variant_tag"].astype(str).str.strip()
    frame["phecode"] = frame["phecode"].astype(str).str.strip()
    frame["phecode_string"] = frame["phecode_string"].astype(str).str.strip()
    for column in ("p", "beta", "se"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sentinel_contract = _sentinel_contract(sentinels)
    genic_tags = set(sentinel_contract.loc[sentinel_contract["is_genic"], "variant_tag"])
    frame = frame.loc[frame["variant_tag"].isin(genic_tags)].copy()
    rows_after_genic = len(frame)

    converged = _first_column(frame, ("converged", "converged_d", "convergence"))
    converged_mask = _truthy(frame[converged]) if converged else pd.Series(True, index=frame.index)
    finite_mask = (
        np.isfinite(frame["p"])
        & np.isfinite(frame["beta"])
        & np.isfinite(frame["se"])
        & frame["se"].gt(0)
    )

    mac_col = _first_column(frame, ("mac_disc", "mac", "MAC"))
    cohort_mac = (
        pd.to_numeric(frame["mac_disc_cohort"], errors="coerce")
        if "mac_disc_cohort" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    if mac_col:
        frame["mac_disc"] = pd.to_numeric(frame[mac_col], errors="coerce")
        frame["mac_disc"] = frame["mac_disc"].where(
            np.isfinite(frame["mac_disc"]), cohort_mac
        )
    else:
        frame["mac_disc"] = cohort_mac
    mac_mask = frame["mac_disc"].ge(mac_min)

    frame = frame.loc[converged_mask & finite_mask & mac_mask].copy()
    rows_after_qc = len(frame)
    frame = frame.loc[frame["p"].le(discovery_p_threshold)].copy()
    discovery_hit_rows = len(frame)

    invalid_names = ~_nonempty(frame["phecode_string"])
    if invalid_names.any():
        examples = frame.loc[
            invalid_names, ["variant_tag", "phecode", "phecode_string"]
        ].head(5)
        raise S9ContractError(
            "Discovery hits are missing phenotype string names: "
            + examples.to_dict(orient="records").__repr__()
        )

    duplicate_mask = frame.duplicated(["variant_tag", "phecode"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["variant_tag", "phecode"]].head(5)
        raise S9ContractError(
            "Discovery hits are not unique variant_tag x phecode rows: "
            + examples.to_dict(orient="records").__repr__()
        )

    maf_col = _first_column(frame, ("maf", "MAF", "minor_allele_frequency", "af", "AF", "EAF", "eaf"))
    if maf_col:
        maf = pd.to_numeric(frame[maf_col], errors="coerce")
        maf = np.minimum(maf, 1.0 - maf)
    else:
        maf = pd.Series(np.nan, index=frame.index, dtype=float)

    tested_n = None
    n_cases = _first_column(frame, ("n_cases", "n_cases_d", "cases"))
    n_controls = _first_column(frame, ("n_controls", "n_controls_d", "controls"))
    if n_cases and n_controls:
        tested_n = pd.to_numeric(frame[n_cases], errors="coerce") + pd.to_numeric(
            frame[n_controls], errors="coerce"
        )
    elif "n_genotyped_disc" in frame.columns and "mac_disc_cohort" not in frame.columns:
        tested_n = pd.to_numeric(frame["n_genotyped_disc"], errors="coerce")

    invalid_maf = ~np.isfinite(maf) | maf.le(0) | maf.gt(0.5)
    recovered_from_tested_n = pd.Series(False, index=frame.index)
    if tested_n is not None:
        fallback = frame["mac_disc"] / (2.0 * tested_n)
        can_recover = invalid_maf & np.isfinite(fallback) & fallback.gt(0) & fallback.le(0.5)
        maf.loc[can_recover] = fallback.loc[can_recover]
        recovered_from_tested_n.loc[can_recover] = True

    recovered_from_cohort = pd.Series(False, index=frame.index)
    if "n_genotyped_disc" in frame.columns and "mac_disc_cohort" in frame.columns:
        cohort_n = pd.to_numeric(frame["n_genotyped_disc"], errors="coerce")
        cohort_fallback = cohort_mac / (2.0 * cohort_n)
        remaining = ~np.isfinite(maf) | maf.le(0) | maf.gt(0.5)
        can_recover = (
            remaining
            & np.isfinite(cohort_fallback)
            & cohort_fallback.gt(0)
            & cohort_fallback.le(0.5)
        )
        maf.loc[can_recover] = cohort_fallback.loc[can_recover]
        recovered_from_cohort.loc[can_recover] = True
    frame["maf"] = maf

    still_invalid = ~np.isfinite(frame["maf"]) | frame["maf"].le(0) | frame["maf"].gt(0.5)
    audit: dict[str, object] = {
        "row_unit": "unique variant_tag x phecode association",
        "discovery_p_threshold": discovery_p_threshold,
        "mac_min": mac_min,
        "expected_hit_count": expected_hit_count,
        "expected_sv_region_count": expected_sv_region_count,
        "counts": {
            "input_rows": len(discovery),
            "rows_after_genic_filter": int(rows_after_genic),
            "rows_after_qc": int(rows_after_qc),
            "discovery_hit_rows_before_maf_validation": int(discovery_hit_rows),
            "maf_recovered_from_mac": int(
                recovered_from_tested_n.sum() + recovered_from_cohort.sum()
            ),
            "maf_recovered_from_tested_n": int(recovered_from_tested_n.sum()),
            "maf_recovered_from_cohort": int(recovered_from_cohort.sum()),
            "invalid_maf_rows": int(still_invalid.sum()),
        },
    }
    if still_invalid.any():
        audit["invalid_maf_examples"] = frame.loc[
            still_invalid, ["variant_tag", "phecode", "mac_disc"]
        ].head(10).to_dict(orient="records")
        raise S9ContractError(
            f"S9 has {int(still_invalid.sum())} discovery hits with invalid MAF; refusing silent omission",
            audit=audit,
        )

    meta_keys = _meta_keys(meta)
    keys = list(zip(frame["variant_tag"], frame["phecode"]))
    frame["is_meta_significant"] = [key in meta_keys for key in keys]
    (
        frame["is_sv_linked"],
        frame["sv_annotation"],
        sv_linkage_audit,
    ) = _sv_linked_hit_mask(
        frame["variant_tag"], sv_links if sv_links is not None else sentinels
    )
    audit["sv_linkage"] = sv_linkage_audit
    linked_annotations = frame.loc[frame["is_sv_linked"], "sv_annotation"].fillna("")
    missing_sv_size = ~linked_annotations.str.contains(
        r"\([0-9][0-9,]* bp [^)]+\)$", regex=True
    )
    if missing_sv_size.any():
        audit["sv_linkage"]["missing_size_annotation_rows"] = int(
            missing_sv_size.sum()
        )
        raise S9ContractError(
            "S9 SV-linked rows need chromosome/position and size/type annotations; "
            "the linked table must provide SV length, endpoints, or REF/ALT alleles",
            audit=audit,
        )
    frame["category"] = "discovery hit"
    frame.loc[frame["is_sv_linked"], "category"] = "SV-linked hit"
    # Meta-significance has visual precedence, making legend groups exclusive.
    frame.loc[frame["is_meta_significant"], "category"] = "meta-significant"
    frame["abs_beta"] = frame["beta"].abs()

    category_counts = {
        category: int(frame["category"].eq(category).sum())
        for category in CATEGORY_ORDER
    }
    audit["counts"].update(
        {
            "plot_rows": len(frame),
            "unique_variants": int(frame["variant_tag"].nunique()),
            "unique_phecodes": int(frame["phecode"].nunique()),
            "meta_significant_rows": int(frame["is_meta_significant"].sum()),
            "sv_linked_rows_including_meta": int(frame["is_sv_linked"].sum()),
        }
    )
    audit["category_counts"] = category_counts

    if sum(category_counts.values()) != len(frame):
        raise S9ContractError("S9 category masks are not exhaustive and exclusive", audit=audit)
    if (
        expected_sv_region_count is not None
        and sv_linkage_audit["linked_reference_regions"] != expected_sv_region_count
    ):
        raise S9ContractError(
            "S9 expected "
            f"{expected_sv_region_count} SV-linked reference regions but found "
            f"{sv_linkage_audit['linked_reference_regions']}",
            audit=audit,
        )
    if (
        expected_sv_region_count is not None
        and sv_linkage_audit["sv_linked_association_rows"] == 0
    ):
        raise S9ContractError(
            "S9 found the SV-linked reference regions but matched zero discovery-hit "
            "associations; check lead-tag and interval coordinates",
            audit=audit,
        )
    if expected_hit_count is not None and len(frame) != expected_hit_count:
        near_threshold = discovery.copy()
        near_p = _first_column(near_threshold, ("p", "P", "pval", "pvalue", "p_value", "p_ind"))
        if near_p:
            near_threshold[near_p] = pd.to_numeric(near_threshold[near_p], errors="coerce")
            audit["closest_threshold_rows"] = (
                near_threshold.assign(_distance=(near_threshold[near_p] - discovery_p_threshold).abs())
                .nsmallest(10, "_distance")
                [[column for column in (variant, phecode, near_p) if column]]
                .to_dict(orient="records")
            )
        raise S9ContractError(
            f"S9 expected {expected_hit_count} discovery-hit rows but found {len(frame)}",
            audit=audit,
        )

    return frame[list(SOURCE_DATA_COLUMNS)].reset_index(drop=True), audit


def plot_s9(
    plot_data: pd.DataFrame,
    *,
    marker_scale: float = 1.0,
    font_scale: float = 1.0,
    legend_frame: bool = True,
    annotation_boxes: bool = True,
):
    """Render the title-free, letter-landscape S9 panel.

    ``marker_scale`` changes marker area only and ``font_scale`` changes all
    plot text. Supplementary Figure S9 keeps the historical defaults while
    Figure 4 v5 can use larger points and text without changing the rows or
    axes. ``annotation_boxes=False`` leaves the SV text and leader lines
    unframed for the Figure 4 v6 presentation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    style.apply_style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    if not np.isfinite(marker_scale) or marker_scale <= 0:
        raise ValueError("marker_scale must be a finite positive number")
    if not np.isfinite(font_scale) or font_scale <= 0:
        raise ValueError("font_scale must be a finite positive number")
    sizes = {
        "discovery hit": 40 * marker_scale,
        "SV-linked hit": 90 * marker_scale,
        "meta-significant": 100 * marker_scale,
    }
    compact_legend = font_scale > 1.0
    alphas = {"discovery hit": 0.50, "SV-linked hit": 0.95, "meta-significant": 0.95}
    zorders = {"discovery hit": 2, "SV-linked hit": 4, "meta-significant": 5}

    for category in CATEGORY_ORDER:
        subset = plot_data.loc[plot_data["category"].eq(category)]
        if subset.empty:
            continue
        ax.scatter(
            subset["maf"],
            subset["abs_beta"],
            s=sizes[category],
            alpha=alphas[category],
            color=CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=0.45 if category == "discovery hit" else 0.8,
            rasterized=category == "discovery hit",
            zorder=zorders[category],
            label=(
                f"{category.replace(' hit', '')} ({len(subset):,})"
                if compact_legend
                else f"{category} ({len(subset):,})"
            ),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Minor allele frequency", fontsize=24 * font_scale)
    ax.set_ylabel(r"$|\beta|$ (log-odds)", fontsize=24 * font_scale)
    ax.tick_params(axis="both", labelsize=20 * font_scale)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.43 if compact_legend else 0.5, 1.01),
        ncol=3,
        fontsize=18 * font_scale,
        frameon=legend_frame,
        framealpha=0.95,
        columnspacing=0.7 if compact_legend else 1.5,
        handletextpad=0.35 if compact_legend else 0.6,
        handlelength=1.0 if compact_legend else 2.0,
        borderpad=0.2 if compact_legend else 0.4,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.84)

    sv_rows = (
        plot_data.loc[plot_data["category"].eq("SV-linked hit")]
        .sort_values(["maf", "abs_beta"], ascending=[True, False])
        .reset_index(drop=True)
    )
    annotation_texts = []
    # Candidate slots run from top-left across the sparse upper plot. Each box
    # is measured on the rendered canvas and assigned to the first low-cost
    # slot that avoids every hit and every already placed annotation.
    candidate_slots = tuple(
        (x, y)
        for y in np.linspace(0.90, 0.36, 10)
        for x in (0.18, 0.28, 0.38, 0.48, 0.58, 0.68)
    )
    targets: list[tuple[float, float]] = []
    for row in sv_rows.itertuples(index=False):
        annotation_texts.append(
            ax.text(
                row.maf,
                row.abs_beta,
                f"{row.sv_annotation}\n{row.phecode_string}",
                fontsize=13 * font_scale,
                fontweight="bold",
                color="#163B3D",
                ha="center",
                va="center",
                bbox=(
                    {
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": CATEGORY_COLORS["SV-linked hit"],
                        "linewidth": 1.0,
                        "alpha": 0.94,
                    }
                    if annotation_boxes
                    else {
                        # Invisible patch retained only so leader lines stop at
                        # the text boundary instead of crossing the glyphs.
                        "boxstyle": "square,pad=0.05",
                        "facecolor": "none",
                        "edgecolor": "none",
                        "linewidth": 0,
                        "alpha": 0,
                    }
                ),
                zorder=7,
            )
        )
        targets.append((float(row.maf), float(row.abs_beta)))
    if annotation_texts:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer)
        point_pixels = ax.transData.transform(
            plot_data[["maf", "abs_beta"]].to_numpy()
        )
        placed_boxes = []

        for text in annotation_texts:
            best: tuple[float, tuple[float, float], object] | None = None
            for rank, slot in enumerate(candidate_slots):
                data_position = tuple(
                    ax.transData.inverted().transform(ax.transAxes.transform(slot))
                )
                text.set_position(data_position)
                box = text.get_window_extent(renderer).expanded(1.06, 1.18)
                outside = int(
                    box.x0 < axes_box.x0
                    or box.x1 > axes_box.x1
                    or box.y0 < axes_box.y0
                    or box.y1 > axes_box.y1
                )
                point_overlap = int(
                    np.count_nonzero(
                        (point_pixels[:, 0] >= box.x0)
                        & (point_pixels[:, 0] <= box.x1)
                        & (point_pixels[:, 1] >= box.y0)
                        & (point_pixels[:, 1] <= box.y1)
                    )
                )
                label_overlap = sum(
                    int(box.overlaps(placed_box)) for placed_box in placed_boxes
                )
                score = (
                    outside * 1_000_000_000
                    + label_overlap * 10_000_000
                    + point_overlap * 1_000_000
                    + rank
                )
                if best is None or score < best[0]:
                    best = (score, data_position, box)
            assert best is not None
            text.set_position(best[1])
            placed_boxes.append(text.get_window_extent(renderer).expanded(1.06, 1.18))

        for text, target in zip(annotation_texts, targets):
            ax.add_patch(
                FancyArrowPatch(
                    posA=text.get_position(),
                    posB=target,
                    patchA=text.get_bbox_patch(),
                    arrowstyle="-",
                    color="#163B3D",
                    linewidth=2.0,
                    shrinkA=2,
                    shrinkB=4,
                    transform=ax.transData,
                    zorder=6,
                )
            )
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _valid_cached_source_data(path: Path, manifest: dict[str, object]) -> bool:
    """Validate the cached S9 row/schema contract, not only its file hash."""
    try:
        frame = pd.read_csv(path, sep="\t", dtype={"phecode": str}, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if not set(SOURCE_DATA_COLUMNS).issubset(frame.columns):
        return False
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return False
    expected_rows = counts.get("plot_rows")
    if expected_rows is None or len(frame) != expected_rows:
        return False
    if frame.duplicated(["variant_tag", "phecode"]).any():
        return False

    phecode_names = frame["phecode_string"].fillna("").astype(str).str.strip()
    if phecode_names.eq("").any():
        return False

    maf = pd.to_numeric(frame["maf"], errors="coerce")
    abs_beta = pd.to_numeric(frame["abs_beta"], errors="coerce")
    if (
        (~np.isfinite(maf) | maf.le(0) | maf.gt(0.5)).any()
        or (~np.isfinite(abs_beta) | abs_beta.lt(0)).any()
    ):
        return False
    if not set(frame["category"].astype(str)).issubset(CATEGORY_ORDER):
        return False
    sv_rows = _truthy(frame["is_sv_linked"])
    annotations = frame.loc[sv_rows, "sv_annotation"].fillna("").astype(str)
    return bool(
        annotations.str.contains(r"\([0-9][0-9,]* bp [^)]+\)$", regex=True).all()
    )


def _valid_cached_run(
    manifest_path: Path,
    inputs: dict[str, str],
    parameters: dict[str, object],
    *,
    source_data_path: Path,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("input_sha256") != inputs or manifest.get("parameters") != parameters:
        return False
    for output, expected_hash in manifest.get("output_sha256", {}).items():
        path = Path(output)
        if not path.exists() or _sha256(path) != expected_hash:
            return False
    return bool(manifest.get("output_sha256")) and _valid_cached_source_data(
        source_data_path, manifest
    )


def run(
    discovery_path: Path,
    meta_path: Path,
    sentinels_path: Path,
    *,
    sv_links_path: Path | None = None,
    cohort_dir: Path | None = None,
    analysis_dir: Path | None = None,
    figure_dir: Path | None = None,
    discovery_p_threshold: float = DISCOVERY_P_THRESHOLD,
    mac_min: int = MAC_MIN,
    expected_hit_count: int = EXPECTED_HIT_COUNT,
    expected_sv_region_count: int = EXPECTED_SV_REGION_COUNT,
    force: bool = False,
) -> dict[str, Path]:
    """Validate inputs, write audit/source data, and render paired PNG/PDF."""
    discovery_path = discovery_path.resolve()
    meta_path = meta_path.resolve()
    sentinels_path = sentinels_path.resolve()
    has_dedicated_sv_links = sv_links_path is not None
    sv_links_path = sv_links_path.resolve() if sv_links_path else sentinels_path
    for path in (discovery_path, meta_path, sentinels_path, sv_links_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty S9 input: {path}")

    analysis_dir = analysis_dir or config.ANALYSIS_DIR / "supplementary" / FIGURE_ID
    figure_dir = figure_dir or config.figure_dir("figure4") / "discovery"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = analysis_dir / "run_manifest.json"
    audit_path = analysis_dir / "figure_s9_audit.json"
    source_data_path = analysis_dir / "figure_s9_source_data.tsv"

    inputs = {
        str(path): _sha256(path)
        for path in (discovery_path, meta_path, sentinels_path, sv_links_path)
    }
    discovery = pd.read_csv(
        discovery_path, sep="\t", dtype={"phecode": str}, low_memory=False
    )
    discovery_variant = _first_column(
        discovery, ("variant_tag", "snp", "SNP", "variant_id", "tag")
    )
    mac_column = _first_column(discovery, ("mac_disc", "mac", "MAC"))
    needs_cohort_mac = mac_column is None or pd.to_numeric(
        discovery[mac_column], errors="coerce"
    ).isna().any()
    maf_column = _first_column(
        discovery,
        ("maf", "MAF", "minor_allele_frequency", "af", "AF", "EAF", "eaf"),
    )
    if maf_column is None:
        has_invalid_maf = True
    else:
        raw_maf = pd.to_numeric(discovery[maf_column], errors="coerce")
        folded_maf = np.minimum(raw_maf, 1.0 - raw_maf)
        has_invalid_maf = bool(
            (~np.isfinite(folded_maf) | folded_maf.le(0) | folded_maf.gt(0.5)).any()
        )
    use_cohort_data = cohort_dir is not None and (needs_cohort_mac or has_invalid_maf)
    parameters: dict[str, object] = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "discovery_p_threshold": discovery_p_threshold,
        "mac_min": mac_min,
        "expected_hit_count": expected_hit_count,
        "expected_sv_region_count": (
            expected_sv_region_count if has_dedicated_sv_links else None
        ),
        "cohort_dir": str(cohort_dir.resolve()) if use_cohort_data else None,
    }
    if use_cohort_data:
        if discovery_variant is None:
            raise S9ContractError("Discovery table needs variant_tag to locate cohort TSVs")
        cohort_manifest = [
            cohort_dir.resolve() / f"{_safe_tag(tag)}_cohort.tsv"
            for tag in sorted(discovery[discovery_variant].astype(str).unique())
        ]
        missing_cohorts = [path for path in cohort_manifest if not path.is_file()]
        if missing_cohorts:
            raise S9ContractError(
                f"Missing {len(missing_cohorts)} relevant cohort TSVs; first: {missing_cohorts[0]}"
            )
        parameters["cohort_manifest"] = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in cohort_manifest
        ]

    if not force and _valid_cached_run(
        manifest_path,
        inputs,
        parameters,
        source_data_path=source_data_path,
    ):
        print(f"[S9] cache hit: {manifest_path}")
        return {
            "png": figure_dir / f"{FIGURE_STEM}.png",
            "pdf": figure_dir / f"{FIGURE_STEM}.pdf",
            "source_data": source_data_path,
            "audit": analysis_dir / "figure_s9_audit.json",
            "manifest": manifest_path,
        }

    print(f"[S9] discovery input: {discovery_path}")
    print(f"[S9] meta input: {meta_path}")
    print(f"[S9] genic-sentinel input: {sentinels_path}")
    print(f"[S9] SV-link input: {sv_links_path}")
    print(
        f"[S9] parameters: p<={discovery_p_threshold:g}, MAC>={mac_min}, "
        f"expected rows={expected_hit_count}, "
        "expected SV-linked regions="
        f"{expected_sv_region_count if has_dedicated_sv_links else 'not enforced'}"
    )
    meta = pd.read_csv(meta_path, sep="\t", dtype={"phecode": str}, low_memory=False)
    sentinels = pd.read_csv(sentinels_path, sep="\t", low_memory=False)
    sv_links = (
        sentinels
        if sv_links_path == sentinels_path
        else pd.read_csv(sv_links_path, sep="\t", low_memory=False)
    )

    if needs_cohort_mac and cohort_dir is None:
        raise S9ContractError(
            "Discovery table has missing MAC values; provide --cohort-dir to reconstruct MAC"
        )
    if use_cohort_data:
        if cohort_dir is None or discovery_variant is None:
            raise S9ContractError(
                "Discovery table needs cohort data and a variant_tag column"
            )
        mac = compute_variant_mac(
            cohort_dir.resolve(), discovery[discovery_variant].astype(str)
        ).rename(columns={"mac_disc": "mac_disc_cohort"})
        discovery = discovery.merge(
            mac,
            left_on=discovery_variant,
            right_on="variant_tag",
            how="left",
            suffixes=("", "_cohort"),
        )

    try:
        plot_data, audit = prepare_s9_data(
            discovery,
            meta,
            sentinels,
            sv_links=sv_links,
            discovery_p_threshold=discovery_p_threshold,
            mac_min=mac_min,
            expected_hit_count=expected_hit_count,
            expected_sv_region_count=(
                expected_sv_region_count if has_dedicated_sv_links else None
            ),
        )
    except S9ContractError as error:
        if error.audit:
            failure_audit = {
                "status": "failed_contract",
                "error": str(error),
                **error.audit,
            }
            audit_path.write_text(
                json.dumps(failure_audit, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[S9] failure audit: {audit_path}")
        raise
    print(f"[S9] counts: {json.dumps(audit['counts'], sort_keys=True)}")
    print(f"[S9] SV linkage: {json.dumps(audit['sv_linkage'], sort_keys=True)}")
    print(f"[S9] categories: {json.dumps(audit['category_counts'], sort_keys=True)}")

    plot_data.to_csv(source_data_path, sep="\t", index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    figure = plot_s9(plot_data)
    png_path = figure_dir / f"{FIGURE_STEM}.png"
    pdf_path = figure_dir / f"{FIGURE_STEM}.pdf"
    import matplotlib as mpl

    # Shared style defaults to tight cropping, but S9 is a letter-landscape
    # manuscript panel and must retain an exact 11 x 8.5 inch page.
    with mpl.rc_context({"savefig.bbox": None}):
        figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches=None)
        figure.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches=None)
    import matplotlib.pyplot as plt

    plt.close(figure)
    if png_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise S9ContractError(f"Invalid PNG signature: {png_path}")
    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-2048:]:
        raise S9ContractError(f"Invalid PDF structure: {pdf_path}")

    outputs = (png_path, pdf_path, source_data_path, audit_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "row_unit": audit["row_unit"],
        "input_sha256": inputs,
        "parameters": parameters,
        "counts": audit["counts"],
        "sv_linkage": audit["sv_linkage"],
        "category_counts": audit["category_counts"],
        "output_sha256": {str(path.resolve()): _sha256(path) for path in outputs},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[S9] wrote: {png_path}")
    print(f"[S9] wrote: {pdf_path}")
    print(f"[S9] manifest: {manifest_path}")
    return {
        "png": png_path,
        "pdf": pdf_path,
        "source_data": source_data_path,
        "audit": audit_path,
        "manifest": manifest_path,
    }


def _required_path(value: str | None, environment_name: str) -> Path:
    if value:
        return Path(value)
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return Path(environment_value)
    raise S9ContractError(f"Provide the path or set {environment_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", help="Discovery PheWAS TSV (AOU_S9_DISCOVERY_TSV)")
    parser.add_argument("--meta", help="Meta-analysis TSV (AOU_S9_META_TSV)")
    parser.add_argument("--sentinels", help="Genic sentinel TSV (AOU_S9_SENTINELS_TSV)")
    parser.add_argument("--sv-links", help="Linked-sentinel TSV (AOU_S9_SV_LINKS_TSV)")
    parser.add_argument("--cohort-dir", help="Per-variant discovery cohort TSV directory")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--discovery-p-threshold", type=float, default=DISCOVERY_P_THRESHOLD)
    parser.add_argument("--mac-min", type=int, default=MAC_MIN)
    parser.add_argument("--expected-hit-count", type=int, default=EXPECTED_HIT_COUNT)
    parser.add_argument(
        "--expected-sv-region-count", type=int, default=EXPECTED_SV_REGION_COUNT
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    cohort_value = args.cohort_dir or os.environ.get("AOU_S9_COHORT_DIR")
    return run(
        _required_path(args.discovery, "AOU_S9_DISCOVERY_TSV"),
        _required_path(args.meta, "AOU_S9_META_TSV"),
        _required_path(args.sentinels, "AOU_S9_SENTINELS_TSV"),
        sv_links_path=_required_path(args.sv_links, "AOU_S9_SV_LINKS_TSV"),
        cohort_dir=Path(cohort_value) if cohort_value else None,
        analysis_dir=args.analysis_dir,
        figure_dir=args.figure_dir,
        discovery_p_threshold=args.discovery_p_threshold,
        mac_min=args.mac_min,
        expected_hit_count=args.expected_hit_count,
        expected_sv_region_count=args.expected_sv_region_count,
        force=args.force,
    )


if __name__ == "__main__":
    main()
