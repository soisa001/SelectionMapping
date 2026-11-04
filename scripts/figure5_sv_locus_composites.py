"""Build selection-led composites from the 17 S3 SV-hit rows.

The page unit is one row of the committed S3 selection reference: 17 selection
regions representing 14 distinct SV alleles. Selection-only rows create one
file; associated rows create one file per applicable phecode. The two groups
form independently numbered output series.
EHH, iHS, nucleotide diversity, and one shared gene track are always shown as
one Figure 3B-style panel. Every page also includes the exact IGV screenshot
for its SV allele from the committed source-extracted S12 asset set.
PheWAS and Figure 4-style
SuSiE panels are included only when an audited Supplementary Figure S9
``is_sv_linked`` association resolves to that exact selection row.  They are
omitted, with an explicit manifest reason, for rows without such an
association.  Association panels retain the actual association
``variant_tag``; linked SVs are never substituted for the EHH core or the
fine-mapping lead.

The workflow is restartable:

``tasks``
    Validate all 17 S3 rows, resolve the S9 rows, then write the exact SuSiE
    task table and 17-row page manifest.
``selection``
    Render the combined EHH/iHS/nucleotide-diversity/gene source panel.
``finemap``
    Stage or compute the exact 10k-discovery SuSiE results, validate their
    Figure 4 contract, and render the fine-map source panels with the Figure 4
    locus renderer.
``compose``
    Stage the exact PheWAS table, render fine-map panels from existing verified
    SuSiE results, validate the IGV asset manifest, and create titled/no-title
    letter-size PNG/PDF pages.
``all``
    Run missing SuSiE tasks, render every component panel, and compose the
    complete selection-only and per-phecode association series.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import re
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from PIL import Image

from scripts.common import config
from scripts import figure5_locus_composites as composites

ANALYSIS_CONTRACT = "supp_s3_sv_selection_pages_optional_phewas_finemap_igv_v18"
SELECTION_MAPPING_CONTRACT = "s3_selection_page_with_optional_s9_association_v2"
FINEMAP_PANEL_CONTRACT = "figure3_verified_10k_disc_locus_panel_v10"
IGV_ASSET_CONTRACT = "embedded_igv_screenshots_from_s12_pdf_v1"
FINEMAP_COHORT = "10k_disc"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_S9_SOURCE_DATA = Path(
    os.environ.get(
        "AOU_S9_SOURCE_DATA_TSV",
        REPO_ROOT / "analysis" / "supplementary" / "s9_phewas" / "figure_s9_source_data.tsv",
    )
)
DEFAULT_SELECTION_HITS = config.IHS_SV_HITS_TSV
DEFAULT_IHS_REGIONS = config.IHS_REGIONS_TSV
DEFAULT_IGV_DIR = REPO_ROOT / "assets" / "figure2" / "igv_sv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "figure5" / "loci"
DEFAULT_GENE_ANNOTATION_OBJECT = "hg38.refGene.gtf"
DEFAULT_WORK_DIR = Path("~/supp_sv_linked_locus_composites")
DEFAULT_EXPECTED_HITS = 4
DEFAULT_EXPECTED_SELECTION_REGIONS = 17
DEFAULT_EXPECTED_SV_ALLELES = 14
DEFAULT_SEED = 20260822
SV_HIGHLIGHT_COLOR = "#007C83"
SV_OUTPUT_PREFIX = "Supplementary_SV"
SV_PHEWAS_OUTPUT_PREFIX = "Supplementary_SV_Phewas"
SV_SELECTION_ONLY_HEIGHT_RATIOS = (0.62, 0.38)
SV_PHEWAS_HEIGHT_RATIOS = composites.COMPOSITE_HEIGHT_RATIOS
SV_ANNOTATION_RE = re.compile(
    r"^chr(?P<chrom>[0-9]+):(?P<position>[0-9,]+)\s+"
    r"\((?P<length>[0-9,]+)\s+bp\s+(?P<sv_type>[A-Za-z]+)\)$"
)


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES", "Y"})


def parse_sv_annotation(value: str) -> tuple[str, int, int, str]:
    """Parse S9's required source-backed ``chr:pos (length bp TYPE)`` label."""
    match = SV_ANNOTATION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid S9 SV annotation: {value!r}")
    chrom = f"chr{int(match.group('chrom'))}"
    position = int(match.group("position").replace(",", ""))
    length = int(match.group("length").replace(",", ""))
    sv_type = match.group("sv_type").upper()
    if position <= 0 or length < 50 or sv_type not in {"INS", "DEL"}:
        raise ValueError(f"Unsupported S9 SV annotation: {value!r}")
    return chrom, position, length, sv_type


def _read_tsv(path: Path, label: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, compression="infer", low_memory=False)


def _sv_allele_key(
    chrom: str,
    position: int | str,
    length_bp: int | str,
    sv_type: str,
) -> tuple[str, int, int, str]:
    normalized_chrom = f"chr{int(str(chrom).strip().lower().removeprefix('chr'))}"
    return (
        normalized_chrom,
        int(position),
        int(length_bp),
        str(sv_type).strip().upper(),
    )


def load_igv_screenshots(
    igv_dir: Path,
    *,
    expected_sv_alleles: int = DEFAULT_EXPECTED_SV_ALLELES,
) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], dict[str, Any]]:
    """Validate and index the source-extracted IGV screenshot assets."""
    igv_dir = igv_dir.expanduser().resolve()
    manifest_path = igv_dir / "manifest.tsv"
    manifest = _read_tsv(manifest_path, "IGV screenshot manifest")
    required = {
        "chrom", "sv_position", "sv_type", "sv_length_bp", "source_figure",
        "source_page", "source_image", "filename", "width_px", "height_px",
        "sha256", "source_pdf", "source_pdf_sha256",
    }
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(
            f"IGV screenshot manifest is missing required columns: {missing_columns}"
        )
    if len(manifest) != expected_sv_alleles:
        raise ValueError(
            f"Expected {expected_sv_alleles} IGV screenshots, found {len(manifest)}"
        )

    assets: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    filenames: set[str] = set()
    for _, row in manifest.iterrows():
        key = _sv_allele_key(
            row["chrom"], row["sv_position"], row["sv_length_bp"], row["sv_type"]
        )
        if key in assets:
            raise ValueError(f"Duplicate IGV screenshot allele in manifest: {key}")
        filename = str(row["filename"]).strip()
        if filename in filenames:
            raise ValueError(f"Duplicate IGV screenshot filename: {filename}")
        filenames.add(filename)
        path = (igv_dir / filename).resolve()
        if path.parent != igv_dir:
            raise ValueError(f"IGV screenshot must be directly beneath {igv_dir}: {filename}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"IGV screenshot is missing or empty: {path}")
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid IGV screenshot image: {path}") from exc
        expected_dimensions = (int(row["width_px"]), int(row["height_px"]))
        if (width, height) != expected_dimensions:
            raise ValueError(
                f"IGV screenshot dimensions differ for {path}: "
                f"{(width, height)} != {expected_dimensions}"
            )
        observed_sha256 = composites.sha256_file(path)
        expected_sha256 = str(row["sha256"]).strip().lower()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"IGV screenshot SHA256 differs for {path}: "
                f"{observed_sha256} != {expected_sha256}"
            )
        try:
            display_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display_path = str(path)
        assets[key] = {
            "path": path,
            "display_path": display_path,
            "chrom": key[0],
            "sv_position": key[1],
            "sv_length_bp": key[2],
            "sv_type": key[3],
            "sha256": observed_sha256,
            "width_px": width,
            "height_px": height,
            "source_figure": str(row["source_figure"]),
            "source_page": int(row["source_page"]),
            "source_image": str(row["source_image"]),
            "source_pdf": str(row["source_pdf"]),
            "source_pdf_sha256": str(row["source_pdf_sha256"]).strip().lower(),
        }

    source_pdf_hashes = sorted({asset["source_pdf_sha256"] for asset in assets.values()})
    if len(source_pdf_hashes) != 1:
        raise ValueError(
            f"IGV screenshot manifest must identify one source PDF hash: {source_pdf_hashes}"
        )
    provenance = {
        "contract": IGV_ASSET_CONTRACT,
        "manifest": str(manifest_path),
        "manifest_sha256": composites.sha256_file(manifest_path),
        "asset_count": len(assets),
        "source_pdf": sorted({asset["source_pdf"] for asset in assets.values()}),
        "source_pdf_sha256": source_pdf_hashes[0],
    }
    return assets, provenance


def _selection_distance(row: pd.Series, position: int) -> int:
    start = int(row["region_start"])
    stop = int(row["region_stop"])
    if start <= position <= stop:
        return 0
    return min(abs(position - start), abs(position - stop))


def _resolve_selection_row(candidates: pd.DataFrame, association_position: int) -> tuple[pd.Series, str]:
    if len(candidates) == 1:
        return candidates.iloc[0], "unique_sv_allele"

    direct = candidates[
        candidates["selection_peak_position"].eq(candidates["sv_position"])
        & candidates["sv_position"].eq(association_position)
    ]
    if len(direct) == 1:
        return direct.iloc[0], "association_at_direct_sv_peak"

    containing = candidates[
        candidates["region_start"].le(association_position)
        & candidates["region_stop"].ge(association_position)
    ].copy()
    pool = containing if not containing.empty else candidates.copy()
    method = "association_inside_selection_region" if not containing.empty else "nearest_selection_region"
    pool["_region_distance"] = pool.apply(
        _selection_distance, axis=1, position=association_position,
    )
    pool["_peak_distance"] = (
        pool["selection_peak_position"] - association_position
    ).abs()
    pool = pool.sort_values(
        ["_region_distance", "_peak_distance", "selection_peak_position"],
        kind="mergesort",
    )
    best = pool.iloc[0]
    tied = pool[
        pool["_region_distance"].eq(best["_region_distance"])
        & pool["_peak_distance"].eq(best["_peak_distance"])
    ]
    if len(tied) != 1:
        raise ValueError(
            "Ambiguous selection-region match for association position "
            f"{association_position}: {tied['selection_peak_position'].tolist()}"
        )
    return best, method


def _assign_sv_series(pages: pd.DataFrame) -> pd.DataFrame:
    """Assign independent, stable indices to selection and PheWAS SV series."""
    pages = pages.copy()
    has_association = pages["association_count"].astype(int).gt(0)
    pages["sv_series"] = np.where(has_association, "phewas", "selection")
    pages["sv_series_index"] = 0
    for series in ("selection", "phewas"):
        series_mask = pages["sv_series"].eq(series)
        series_codes, _ = pd.factorize(
            pages.loc[series_mask, "sv_index"], sort=False
        )
        pages.loc[series_mask, "sv_series_index"] = series_codes + 1
    pages["sv_series_index"] = pages["sv_series_index"].astype(int)
    return pages


def load_sv_linked_hits(
    source_data_path: Path,
    selection_hits_path: Path = DEFAULT_SELECTION_HITS,
    ihs_regions_path: Path = DEFAULT_IHS_REGIONS,
    *,
    expected_hits: int = DEFAULT_EXPECTED_HITS,
    expected_selection_regions: int = DEFAULT_EXPECTED_SELECTION_REGIONS,
    expected_sv_alleles: int = DEFAULT_EXPECTED_SV_ALLELES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return audited S9 associations and all selection-led page loci."""
    source = _read_tsv(source_data_path, "S9 source-data TSV")
    required_source = {
        "variant_tag", "phecode", "phecode_string", "p", "beta",
        "is_sv_linked", "sv_annotation", "is_meta_significant",
    }
    missing = sorted(required_source - set(source.columns))
    if missing:
        raise ValueError(f"S9 source-data TSV is missing required columns: {missing}")
    source = source[_truthy(source["is_sv_linked"])].copy()
    source["variant_tag"] = source["variant_tag"].astype(str).str.strip()
    source["phecode"] = source["phecode"].astype(str).str.strip()
    source = source.drop_duplicates(["variant_tag", "phecode"], keep="first").reset_index(drop=True)
    if expected_hits and len(source) != expected_hits:
        raise ValueError(f"Expected {expected_hits} SV-linked/lead S9 rows, found {len(source)}")
    if source.empty:
        raise ValueError("S9 source data contains no SV-linked/lead association rows")
    source["discovery_p"] = pd.to_numeric(source["p"], errors="coerce")
    source["discovery_beta"] = pd.to_numeric(source["beta"], errors="coerce")
    valid = (
        source["discovery_p"].gt(0)
        & source["discovery_p"].le(1)
        & np.isfinite(source["discovery_beta"])
    )
    if not bool(valid.all()):
        raise ValueError(f"S9 SV rows contain {int((~valid).sum())} invalid p/beta pairs")

    parsed_variants = source["variant_tag"].map(composites.parse_variant_tag)
    source[["chrom", "pos", "ref", "alt"]] = pd.DataFrame(
        parsed_variants.tolist(), index=source.index,
    )
    parsed_svs = source["sv_annotation"].map(parse_sv_annotation)
    source[["sv_chrom", "sv_position", "sv_length_bp", "sv_type"]] = pd.DataFrame(
        parsed_svs.tolist(), index=source.index,
    )

    selection = _read_tsv(selection_hits_path, "S3 SV-selection reference")
    required_selection = {
        "selection_chrom", "region_start", "region_stop", "selection_peak_position",
        "selection_peak_ihs", "selection_window_gene", "sv_position", "sv_type",
        "sv_length_bp", "sv_relation", "sv_gene_label",
    }
    missing = sorted(required_selection - set(selection.columns))
    if missing:
        raise ValueError(f"S3 SV-selection reference is missing columns: {missing}")
    for column in (
        "region_start", "region_stop", "selection_peak_position", "selection_peak_ihs",
        "sv_position", "sv_length_bp",
    ):
        selection[column] = pd.to_numeric(selection[column], errors="raise")
    for column in ("region_start", "region_stop", "selection_peak_position", "sv_position", "sv_length_bp"):
        selection[column] = selection[column].astype(int)
    selection["selection_chrom"] = selection["selection_chrom"].astype(str).str.strip()
    selection["sv_type"] = selection["sv_type"].astype(str).str.strip().str.upper()
    if expected_selection_regions and len(selection) != expected_selection_regions:
        raise ValueError(
            f"Expected {expected_selection_regions} S3 selection regions, found {len(selection)}"
        )
    allele_count = selection[
        ["selection_chrom", "sv_position", "sv_length_bp", "sv_type"]
    ].drop_duplicates().shape[0]
    if expected_sv_alleles and allele_count != expected_sv_alleles:
        raise ValueError(f"Expected {expected_sv_alleles} distinct S3 SV alleles, found {allele_count}")

    ihs = _read_tsv(ihs_regions_path, "linked iHS-region reference")
    required_ihs = {"chrom", "max_IHS_position", "max_emp_ihs", "IHS_REF", "IHS_ALT", "IHS_rs"}
    missing = sorted(required_ihs - set(ihs.columns))
    if missing:
        raise ValueError(f"Linked iHS-region reference is missing columns: {missing}")
    ihs["max_IHS_position"] = pd.to_numeric(ihs["max_IHS_position"], errors="coerce")
    ihs["max_emp_ihs"] = pd.to_numeric(ihs["max_emp_ihs"], errors="coerce")
    ihs = ihs.dropna(subset=["max_IHS_position", "max_emp_ihs"]).copy()
    ihs["max_IHS_position"] = ihs["max_IHS_position"].astype(int)
    ihs["chrom"] = ihs["chrom"].astype(str).str.strip()

    page_rows: list[dict[str, Any]] = []
    sv_index_by_allele: dict[tuple[str, int, int, str], int] = {}
    for selection_index, selected in selection.iterrows():
        ihs_match = ihs[
            ihs["chrom"].eq(selected["selection_chrom"])
            & ihs["max_IHS_position"].eq(int(selected["selection_peak_position"]))
        ]
        if len(ihs_match) != 1:
            raise ValueError(
                "Expected one linked iHS row for selection peak "
                f"{selected['selection_chrom']}:{int(selected['selection_peak_position'])}; "
                f"found {len(ihs_match)}"
            )
        ihs_row = ihs_match.iloc[0]
        relation = str(selected["sv_relation"]).strip().lower()
        if relation not in {"direct", "linked"}:
            raise ValueError(f"Unsupported S3 sv_relation {selected['sv_relation']!r}")
        gene = str(selected["selection_window_gene"]).split(";")[0].strip()
        if not gene or gene.casefold() == "nan":
            gene = str(selected["sv_gene_label"]).split("/")[0].strip()
        chrom = str(selected["selection_chrom"])
        core_pos = int(selected["selection_peak_position"])
        sv_position = int(selected["sv_position"])
        sv_length = int(selected["sv_length_bp"])
        sv_type = str(selected["sv_type"])
        allele_key = (chrom, sv_position, sv_length, sv_type)
        if allele_key not in sv_index_by_allele:
            sv_index_by_allele[allele_key] = len(sv_index_by_allele) + 1
        page_key = (
            f"{chrom.removeprefix('chr')}_{core_pos}_"
            f"sv_{sv_position}_{sv_length}bp_{sv_type}"
        )
        page_rows.append(
            {
                "page_index": int(selection_index) + 1,
                "sv_index": sv_index_by_allele[allele_key],
                "selection_page_key": page_key,
                "selection_chrom": chrom,
                "selection_region_start": int(selected["region_start"]),
                "selection_region_stop": int(selected["region_stop"]),
                "selection_core_pos": core_pos,
                "selection_ihs": float(selected["selection_peak_ihs"]),
                "selection_core_ref": str(ihs_row["IHS_REF"]),
                "selection_core_alt": str(ihs_row["IHS_ALT"]),
                "selection_core_rs": (
                    "" if pd.isna(ihs_row["IHS_rs"]) else str(ihs_row["IHS_rs"])
                ),
                "gene": gene,
                "selection_window_gene": str(selected["selection_window_gene"]).strip(),
                "sv_position": sv_position,
                "sv_length_bp": sv_length,
                "sv_type": sv_type,
                "sv_annotation": f"{chrom}:{sv_position} ({sv_length} bp {sv_type})",
                "sv_gene_label": str(selected["sv_gene_label"]).strip(),
                "selection_relation": relation,
                "selection_relation_label": "SV lead" if relation == "direct" else "SV-linked",
            }
        )
    pages = pd.DataFrame(page_rows)
    if pages["selection_page_key"].duplicated().any():
        duplicates = sorted(
            pages.loc[pages["selection_page_key"].duplicated(False), "selection_page_key"].unique()
        )
        raise ValueError(f"S3 selection rows do not have unique page keys: {duplicates}")
    selection = selection.copy()
    selection["selection_page_key"] = pages["selection_page_key"].to_numpy()

    resolved_rows: list[dict[str, Any]] = []
    mapping_counts: dict[str, int] = {}
    for _, association in source.iterrows():
        candidates = selection[
            selection["selection_chrom"].eq(association["sv_chrom"])
            & selection["sv_position"].eq(int(association["sv_position"]))
            & selection["sv_length_bp"].eq(int(association["sv_length_bp"]))
            & selection["sv_type"].eq(association["sv_type"])
        ]
        if candidates.empty:
            raise ValueError(
                f"No S3 selection reference matches {association['sv_annotation']} "
                f"for {association['variant_tag']} x {association['phecode']}"
            )
        selected, match_method = _resolve_selection_row(candidates, int(association["pos"]))
        mapping_counts[match_method] = mapping_counts.get(match_method, 0) + 1
        page = pages[
            pages["selection_page_key"].eq(str(selected["selection_page_key"]))
        ].iloc[0]
        resolved_rows.append(
            {
                **association.to_dict(),
                "gene": str(page["gene"]),
                "phecode_name": str(association["phecode_string"]).strip(),
                "p_meta": np.nan,
                "beta_meta": np.nan,
                "is_meta_significant": bool(_truthy(pd.Series([association["is_meta_significant"]])).iloc[0]),
                "selection_page_index": int(page["page_index"]),
                "sv_index": int(page["sv_index"]),
                "selection_page_key": str(page["selection_page_key"]),
                "selection_chrom": str(page["selection_chrom"]),
                "selection_region_start": int(page["selection_region_start"]),
                "selection_region_stop": int(page["selection_region_stop"]),
                "selection_core_pos": int(page["selection_core_pos"]),
                "selection_ihs": float(page["selection_ihs"]),
                "selection_core_ref": str(page["selection_core_ref"]),
                "selection_core_alt": str(page["selection_core_alt"]),
                "selection_core_rs": str(page["selection_core_rs"]),
                "selection_relation": str(page["selection_relation"]),
                "selection_relation_label": str(page["selection_relation_label"]),
                "selection_match_method": match_method,
                "sv_gene_label": str(page["sv_gene_label"]),
            }
        )

    hits = pd.DataFrame(resolved_rows)
    hits.insert(0, "figure_index", np.arange(1, len(hits) + 1, dtype=int))
    hit_columns = [
        "figure_index", "sv_index", "variant_tag", "chrom", "pos", "ref", "alt",
        "phecode", "phecode_name", "gene", "p_meta", "beta_meta", "discovery_p",
        "discovery_beta",
        "is_meta_significant", "sv_annotation", "sv_chrom", "sv_position", "sv_length_bp",
        "sv_type", "sv_gene_label", "selection_relation", "selection_relation_label",
        "selection_match_method", "selection_page_index", "selection_page_key",
        "selection_chrom", "selection_region_start",
        "selection_region_stop", "selection_core_pos", "selection_core_ref",
        "selection_core_alt", "selection_core_rs", "selection_ihs",
    ]
    hits = hits[hit_columns].copy()
    conflicting = hits.groupby("selection_page_key")["variant_tag"].nunique().gt(1)
    if conflicting.any():
        raise ValueError(
            "One selection page resolves to multiple association leads: "
            f"{sorted(conflicting[conflicting].index.tolist())}"
        )

    association_groups = hits.groupby("selection_page_key", sort=False)
    pages["association_count"] = pages["selection_page_key"].map(
        association_groups.size()
    ).fillna(0).astype(int)
    pages["association_variant_tag"] = pages["selection_page_key"].map(
        association_groups["variant_tag"].first()
    ).fillna("")
    pages["association_phecodes"] = pages["selection_page_key"].map(
        association_groups["phecode"].agg(lambda values: ";".join(map(str, values)))
    ).fillna("")
    has_association = pages["association_count"].gt(0)
    pages["phewas_panel_status"] = np.where(
        has_association, "included", "skipped_no_exact_s9_sv_linked_association"
    )
    pages["finemap_panel_status"] = np.where(
        has_association, "included", "skipped_no_exact_s9_sv_linked_association"
    )
    pages["selection_panel_status"] = "included"
    pages["igv_panel_status"] = "included"
    pages = _assign_sv_series(pages)
    page_series = pages.set_index("selection_page_key")
    hits["sv_series"] = hits["selection_page_key"].map(page_series["sv_series"])
    hits["sv_series_index"] = hits["selection_page_key"].map(
        page_series["sv_series_index"]
    ).astype(int)

    loci = pages[
        [
            "page_index", "sv_index", "sv_series", "sv_series_index",
            "selection_page_key", "selection_chrom",
            "selection_core_pos",
            "selection_core_ref", "selection_core_alt", "gene", "selection_ihs",
            "selection_core_rs", "sv_position", "sv_length_bp", "sv_type",
            "sv_annotation", "sv_gene_label", "selection_relation",
            "selection_relation_label", "selection_region_start", "selection_region_stop",
            "selection_window_gene", "association_count", "association_variant_tag",
            "association_phecodes", "phewas_panel_status", "finemap_panel_status",
            "selection_panel_status", "igv_panel_status",
        ]
    ].rename(
        columns={
            "selection_page_key": "variant_tag",
            "selection_chrom": "chrom",
            "selection_core_pos": "core_pos",
            "selection_core_ref": "ref",
            "selection_core_alt": "alt",
            "selection_ihs": "ihs_val",
            "selection_core_rs": "ihs_rs",
            "sv_position": "sv_pos",
        }
    )
    loci["min_p_meta"] = np.nan
    provenance = {
        "mapping_contract": SELECTION_MAPPING_CONTRACT,
        "s9_source_data": {
            "path": str(source_data_path.expanduser().resolve()),
            "sha256": composites.sha256_file(source_data_path.expanduser().resolve()),
        },
        "selection_reference": {
            "path": str(selection_hits_path.expanduser().resolve()),
            "sha256": composites.sha256_file(selection_hits_path.expanduser().resolve()),
            "regions": len(selection),
            "distinct_sv_alleles": allele_count,
        },
        "ihs_reference": {
            "path": str(ihs_regions_path.expanduser().resolve()),
            "sha256": composites.sha256_file(ihs_regions_path.expanduser().resolve()),
        },
        "association_rows": len(hits),
        "association_loci": int(hits["variant_tag"].nunique()),
        "selection_pages": len(loci),
        "selection_pages_with_association": int(has_association.sum()),
        "selection_pages_without_association": int((~has_association).sum()),
        "selection_series_sv_count": int(
            pages.loc[~has_association, "sv_series_index"].nunique()
        ),
        "phewas_series_sv_count": int(
            pages.loc[has_association, "sv_series_index"].nunique()
        ),
        "selection_match_methods": mapping_counts,
    }
    return hits, loci.reset_index(drop=True), provenance


def resolve_runtime_phewas_applicability(
    hits: pd.DataFrame,
    pages: pd.DataFrame,
    phewas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep PheWAS pages only when their exact target tests exist at runtime."""
    required_hit_columns = {"selection_page_key", "variant_tag", "phecode"}
    required_page_columns = {"variant_tag", "sv_index", "association_count"}
    required_phewas_columns = {"variant_tag", "phecode"}
    if missing := sorted(required_hit_columns - set(hits.columns)):
        raise ValueError(f"SV hit table lacks runtime-applicability columns: {missing}")
    if missing := sorted(required_page_columns - set(pages.columns)):
        raise ValueError(f"SV page table lacks runtime-applicability columns: {missing}")
    if missing := sorted(required_phewas_columns - set(phewas.columns)):
        raise ValueError(f"PheWAS table lacks runtime-applicability columns: {missing}")

    runtime_keys = set(
        zip(
            phewas["variant_tag"].astype(str),
            phewas["phecode"].astype(str),
        )
    )
    applicable_mask = np.fromiter(
        (
            (str(variant), str(phecode)) in runtime_keys
            for variant, phecode in zip(hits["variant_tag"], hits["phecode"])
        ),
        dtype=bool,
        count=len(hits),
    )
    applicable_hits = hits.loc[applicable_mask].copy().reset_index(drop=True)
    resolved_pages = pages.copy()
    source_has_association = resolved_pages["association_count"].astype(int).gt(0)

    if applicable_hits.empty:
        runtime_counts: dict[str, int] = {}
        runtime_variants: dict[str, str] = {}
        runtime_phecodes: dict[str, str] = {}
    else:
        groups = applicable_hits.groupby("selection_page_key", sort=False)
        runtime_counts = groups.size().astype(int).to_dict()
        runtime_variants = groups["variant_tag"].first().astype(str).to_dict()
        runtime_phecodes = groups["phecode"].agg(
            lambda values: ";".join(map(str, values))
        ).to_dict()

    page_keys = resolved_pages["variant_tag"].astype(str)
    resolved_pages["association_count"] = page_keys.map(runtime_counts).fillna(0).astype(int)
    resolved_pages["association_variant_tag"] = page_keys.map(runtime_variants).fillna("")
    resolved_pages["association_phecodes"] = page_keys.map(runtime_phecodes).fillna("")
    runtime_has_association = resolved_pages["association_count"].gt(0)
    missing_runtime_target = source_has_association & ~runtime_has_association
    skipped_status = np.where(
        missing_runtime_target,
        "skipped_no_runtime_phewas_target",
        "skipped_no_exact_s9_sv_linked_association",
    )
    resolved_pages["phewas_panel_status"] = np.where(
        runtime_has_association, "included", skipped_status
    )
    resolved_pages["finemap_panel_status"] = np.where(
        runtime_has_association, "included", skipped_status
    )
    resolved_pages["selection_panel_status"] = "included"
    resolved_pages["igv_panel_status"] = "included"
    resolved_pages["runtime_phewas_applicable"] = runtime_has_association
    resolved_pages = _assign_sv_series(resolved_pages)

    page_series = resolved_pages.set_index("variant_tag")
    applicable_hits["sv_series"] = applicable_hits["selection_page_key"].map(
        page_series["sv_series"]
    )
    applicable_hits["sv_series_index"] = applicable_hits["selection_page_key"].map(
        page_series["sv_series_index"]
    ).astype(int)
    provenance = {
        "audited_association_rows": len(hits),
        "runtime_association_rows": len(applicable_hits),
        "runtime_skipped_association_rows": int((~applicable_mask).sum()),
        "runtime_pages_with_phewas": int(runtime_has_association.sum()),
        "runtime_pages_without_phewas": int((~runtime_has_association).sum()),
        "runtime_pages_reclassified_to_selection": int(missing_runtime_target.sum()),
    }
    return applicable_hits, resolved_pages, provenance


def write_susie_tasks(hits: pd.DataFrame, path: Path) -> Path:
    """Write the exact association leads consumed by the fine-mapping notebook."""
    tasks = hits[
        ["variant_tag", "phecode", "phecode_name", "gene", "discovery_p", "discovery_beta"]
    ].rename(
        columns={
            "phecode_name": "phecode_string",
            "gene": "gene_in_window",
            "discovery_p": "p_d",
            "discovery_beta": "beta_d",
        }
    )
    # The notebook uses this field only as an explicit task-selection gate.  It
    # does not mean these S9 rows are meta-significant; the true status remains
    # in hit_manifest.tsv and no p_meta value is fabricated.
    tasks.insert(4, "meta_significant", "TRUE")
    tasks["p_meta"] = np.nan
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(path, sep="\t", index=False)
    return path


def expected_finemap_result_filename(hit: pd.Series) -> str:
    """Exact discovery-result name written by the current SuSiE notebook."""
    return f"susie_{hit['variant_tag']}__{hit['phecode']}__{FINEMAP_COHORT}.tsv"


def _figure4_locus(hit: pd.Series) -> dict[str, Any]:
    """Translate one audited S9 association row to Figure 4's locus schema."""
    from scripts import figure4_finemap

    phecode = str(hit["phecode"])
    sex_scope = figure4_finemap._sex_scope_lookup().get(phecode)
    if sex_scope not in {"Both", "Female", "Male"}:
        raise ValueError(f"No valid Figure 4 sex scope for phecode {phecode}")
    chrom = str(hit["chrom"]).removeprefix("chr")
    return {
        "chrom": f"chr{int(chrom)}",
        "lead_pos": int(hit["pos"]),
        "lead_ref": str(hit["ref"]),
        "lead_alt": str(hit["alt"]),
        "sv_pos": int(hit["sv_position"]),
        "sv_type": str(hit["sv_type"]).upper(),
        "phecode": phecode,
        "phecode_name": str(hit["phecode_name"]),
        "lead_gene": str(hit["gene"]),
        # The selection-window gene is not a fine-mapping conclusion. Keep the
        # Figure 4 gene track neutral for these association-driven panels.
        "finemap_gene": None,
        "bold_gene": None,
        "sex_scope": sex_scope,
        "meta_p": float("nan"),
        "variant_tag": str(hit["variant_tag"]),
        "source_kind": "audited S9 SV-linked/lead association",
    }


def _load_verified_finemap_result(path: Path, locus: dict[str, Any]) -> pd.DataFrame:
    """Load one result only after enforcing Figure 4's scientific contract."""
    from scripts import figure4_finemap

    figure4_finemap.validate_discovery_contract(
        path,
        expected_scope=locus["sex_scope"],
        expected_locus=locus,
    )
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "variant_id", "pos", "p_score", "pip", "cs", "is_sv", "sv_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fine-map result {path} is missing plotting columns: {missing}")
    if "r2_to_lead" not in frame.columns:
        if "r2_to_minp" not in frame.columns:
            raise ValueError(
                f"Fine-map result {path} lacks r2_to_minp/r2_to_lead"
            )
        frame["r2_to_lead"] = frame["r2_to_minp"]
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise")
    frame["p_score"] = pd.to_numeric(frame["p_score"], errors="coerce")
    frame["pip"] = pd.to_numeric(frame["pip"], errors="coerce")
    frame["cs"] = frame["cs"].fillna("").astype(str)
    frame["is_sv"] = _truthy(frame["is_sv"])
    frame["sv_type"] = frame["sv_type"].fillna("").astype(str).str.strip().str.upper()
    frame["r2_to_lead"] = pd.to_numeric(frame["r2_to_lead"], errors="coerce")
    if frame.empty or not bool(frame["p_score"].gt(0).any()):
        raise ValueError(f"Fine-map result {path} has no finite positive score-test p-values")
    if not bool(frame["pip"].notna().any()):
        raise ValueError(f"Fine-map result {path} has no finite SuSiE PIP values")
    if "neg_log_p" not in frame.columns:
        frame["neg_log_p"] = -np.log10(frame["p_score"].clip(lower=1e-300))
    else:
        frame["neg_log_p"] = pd.to_numeric(frame["neg_log_p"], errors="coerce")
    return frame


def _variant_log_record(row: pd.Series) -> dict[str, Any]:
    """Return JSON-safe fine-map fields for one source variant."""
    pip_value = pd.to_numeric(pd.Series([row["pip"]]), errors="coerce").iloc[0]
    p_value = pd.to_numeric(pd.Series([row["p_score"]]), errors="coerce").iloc[0]
    position = pd.to_numeric(pd.Series([row["pos"]]), errors="coerce").iloc[0]
    return {
        "variant_id": str(row["variant_id"]),
        "position": int(position) if pd.notna(position) else None,
        "pip": float(pip_value) if pd.notna(pip_value) else None,
        "p_value": float(p_value) if pd.notna(p_value) else None,
        "is_sv": bool(row["is_sv"]),
        "sv_type": str(row["sv_type"]) if bool(row["is_sv"]) else "",
    }


def _causal_set_names(value: Any) -> tuple[str, ...]:
    """Normalize the notebook's comma-separated ``cs`` membership column."""
    names = []
    for name in str(value).split(","):
        normalized = name.strip()
        if normalized and normalized.casefold() != "nan" and normalized not in names:
            names.append(normalized)
    return tuple(names)


def _sv_length_from_variant_id(variant_id: Any, sv_type: str) -> int | None:
    """Read the current notebook's ``chr:pos:TYPE:length`` SV identifier."""
    parts = str(variant_id).split(":")
    if len(parts) < 4 or parts[-2].strip().upper() != sv_type:
        return None
    try:
        return abs(int(parts[-1]))
    except ValueError:
        return None


def summarize_finemap_result(
    frame: pd.DataFrame,
    hit: pd.Series,
) -> dict[str, Any]:
    """Summarize every causal set plus top, minimum-p, and exact-SV rows."""
    ranked = frame.sort_values(
        ["pip", "p_score", "pos", "variant_id"],
        ascending=[False, True, True, True],
        na_position="last",
        kind="stable",
    )
    top_pip = _variant_log_record(ranked.iloc[0])
    min_p = _variant_log_record(
        frame.sort_values(
            ["p_score", "pip", "pos", "variant_id"],
            ascending=[True, False, True, True],
            na_position="last",
            kind="stable",
        ).iloc[0]
    )

    set_names = sorted(
        {name for value in frame["cs"] for name in _causal_set_names(value)}
    )
    causal_sets: list[dict[str, Any]] = []
    for set_name in set_names:
        member_mask = frame["cs"].map(
            lambda value, name=set_name: name in _causal_set_names(value)
        )
        members = frame.loc[member_mask].sort_values(
            ["pip", "p_score", "pos", "variant_id"],
            ascending=[False, True, True, True],
            na_position="last",
            kind="stable",
        )
        causal_sets.append(
            {
                "causal_set": set_name,
                "members": [
                    _variant_log_record(row) for _, row in members.iterrows()
                ],
            }
        )

    expected_sv_type = str(hit["sv_type"]).strip().upper()
    expected_sv_length = int(hit["sv_length_bp"])
    sv_mask = (
        frame["is_sv"]
        & frame["pos"].eq(int(hit["sv_position"]))
        & frame["sv_type"].eq(expected_sv_type)
        & frame["variant_id"].map(
            lambda value: _sv_length_from_variant_id(value, expected_sv_type)
            == expected_sv_length
        )
    )
    sv_rows = frame.loc[sv_mask].sort_values(
        ["pip", "p_score", "variant_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    )
    association_p = pd.to_numeric(
        pd.Series([hit.get("discovery_p", np.nan)]), errors="coerce"
    ).iloc[0]
    association_beta = pd.to_numeric(
        pd.Series([hit.get("discovery_beta", np.nan)]), errors="coerce"
    ).iloc[0]
    return {
        "variant_tag": str(hit["variant_tag"]),
        "phecode": str(hit["phecode"]),
        "phecode_name": str(hit["phecode_name"]),
        "phewas_association": {
            "p_value": float(association_p) if pd.notna(association_p) else None,
            "beta": (
                float(association_beta) if pd.notna(association_beta) else None
            ),
            "odds_ratio": (
                float(np.exp(association_beta))
                if pd.notna(association_beta)
                else None
            ),
        },
        "variant_count": len(frame),
        "causal_sets": causal_sets,
        "top_pip_variant": top_pip,
        "minimum_p_variant": min_p,
        "expected_sv": {
            "position": int(hit["sv_position"]),
            "sv_type": expected_sv_type,
            "length_bp": expected_sv_length,
        },
        "sv_matches": [
            _variant_log_record(row) for _, row in sv_rows.iterrows()
        ],
    }


def log_finemap_result(summary: dict[str, Any]) -> None:
    """Write complete per-PheWAS fine-map evidence to the workflow log."""
    logger = logging.getLogger("sv_linked_locus_composites")
    context = {
        key: summary[key]
        for key in ("variant_tag", "phecode", "phecode_name")
    }

    def emit(record_type: str, values: dict[str, Any]) -> None:
        payload = {"record_type": record_type, **context, **values}
        logger.info(
            "PHEWAS_FINEMAP | %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    emit(
        "summary",
        {
            "variant_count": summary["variant_count"],
            "causal_set_count": len(summary["causal_sets"]),
            "sv_match_count": len(summary["sv_matches"]),
            "phewas_association": summary["phewas_association"],
        },
    )
    emit("top_pip_variant", summary["top_pip_variant"])
    emit("minimum_p_variant", summary["minimum_p_variant"])
    if summary["sv_matches"]:
        for match_index, record in enumerate(summary["sv_matches"], start=1):
            emit(
                "exact_sv",
                {
                    "match_index": match_index,
                    "match_count": len(summary["sv_matches"]),
                    **record,
                },
            )
    else:
        emit("exact_sv_absent", summary["expected_sv"])
    for causal_set in summary["causal_sets"]:
        members = causal_set["members"]
        for member_index, record in enumerate(members, start=1):
            emit(
                "causal_set_member",
                {
                    "causal_set": causal_set["causal_set"],
                    "member_index": member_index,
                    "member_count": len(members),
                    **record,
                },
            )


def _resolve_finemap_result(
    hit: pd.Series,
    destination: Path,
    remote_dirs: Sequence[str],
    local_dirs: Sequence[Path],
    *,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the exact verified-TSV candidate from local or bucket sources."""
    filename = expected_finemap_result_filename(hit)
    for directory in local_dirs:
        candidate = directory.expanduser().resolve() / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, {
                "mode": "local result",
                "path": str(candidate),
                "size": candidate.stat().st_size,
                "sha256": composites.sha256_file(candidate),
            }

    destination.mkdir(parents=True, exist_ok=True)
    local = destination / filename
    failures: list[str] = []
    for remote_dir in remote_dirs:
        remote = f"{remote_dir.rstrip('/')}/{filename}"
        try:
            metadata_values = composites._stage_gcs_file(remote, local, force=force)
            if not local.is_file() or local.stat().st_size == 0:
                raise FileNotFoundError(f"Staged fine-map result is missing or empty: {local}")
            return local, metadata_values
        except OSError as exc:
            failures.append(str(exc))
    raise FileNotFoundError(
        f"Fine-map result {filename} was not found in any configured directory:\n"
        + "\n".join(failures)
    )


def render_figure4_finemap_panels(
    hits: pd.DataFrame,
    destination: Path,
    remote_dirs: Sequence[str],
    *,
    local_dirs: Sequence[Path] = (),
    dpi: int = composites.DEFAULT_DPI,
    force: bool = False,
) -> tuple[dict[int, Path], dict[str, Any]]:
    """Render titleless Figure 4-style panels from verified SuSiE result TSVs."""
    import matplotlib.pyplot as plt

    from scripts import figure3_ehh, figure4_finemap

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output: dict[int, Path] = {}
    provenance: dict[str, Any] = {}
    renderer_sha = composites.sha256_file(Path(figure4_finemap.__file__).resolve())
    gene_annotation = figure4_finemap._gene_annotation_path()
    if gene_annotation is None:
        raise FileNotFoundError(
            "Figure 4-style SV fine maps require the staged hg38.refGene.gtf"
        )
    gene_annotation_sha = (
        composites.sha256_file(gene_annotation) if gene_annotation is not None else None
    )

    for _, hit in hits.iterrows():
        index = int(hit["figure_index"])
        locus = _figure4_locus(hit)
        result_path, stage = _resolve_finemap_result(
            hit,
            destination,
            remote_dirs,
            local_dirs,
            force=force,
        )
        frame = _load_verified_finemap_result(result_path, locus)
        finemap_result_summary = summarize_finemap_result(frame, hit)
        # This runs before the panel-cache check so every applicable PheWAS
        # result is present in each workflow log, including reused panels.
        log_finemap_result(finemap_result_summary)
        sv_position_mask = pd.to_numeric(frame["pos"], errors="coerce").eq(
            locus["sv_pos"]
        )
        sv_result_matches = int(sv_position_mask.sum())
        sv_pvalue_matches = int(
            (
                sv_position_mask.to_numpy()
                & np.isfinite(
                    pd.to_numeric(
                        frame["neg_log_p"], errors="coerce"
                    ).to_numpy(dtype=float)
                )
            ).sum()
        )
        window_center = int(hit["selection_core_pos"])
        window_start = max(1, window_center - figure4_finemap.HALF_WINDOW)
        window_end = window_center + figure4_finemap.HALF_WINDOW
        genes = figure3_ehh.parse_gtf_region(
            str(gene_annotation), locus["chrom"], window_start, window_end
        )
        if not genes:
            raise ValueError(
                "Staged refGene annotation returned no genes for SV fine-map window "
                f"{locus['chrom']}:{window_start}-{window_end}"
            )
        gene_names = sorted(str(name) for name in genes)
        png_path = destination / composites.expected_finemap_filename(hit)
        pdf_path = png_path.with_suffix(".pdf")
        manifest_path = png_path.with_suffix(".json")
        source_contract = {
            "analysis_contract": FINEMAP_PANEL_CONTRACT,
            "axis_labels": [composites.style.P_VALUE_AXIS_LABEL, composites.style.LD_COLORBAR_LABEL],
            "figure4_analysis_contract": figure4_finemap.ANALYSIS_CONTRACT,
            "variant_tag": str(hit["variant_tag"]),
            "chrom": locus["chrom"],
            "phecode": str(hit["phecode"]),
            "sex_scope": locus["sex_scope"],
            "sv_pos": locus["sv_pos"],
            "sv_type": locus["sv_type"],
            "sv_marker": {
                "shape": figure4_finemap.sv_marker_shape(locus["sv_type"]),
                "edge_color": figure4_finemap.SV_OUTLINE_COLOR,
                "size": figure4_finemap.SV_MARKER_SIZE,
                "edge_linewidth": figure4_finemap.SV_MARKER_LINEWIDTH,
                "vertical_guide": False,
                "position_match_count": sv_result_matches,
                "pvalue_match_count": sv_pvalue_matches,
                "y_mode": (
                    "result_p_value"
                    if sv_pvalue_matches
                    else "position_only_top_of_axis"
                ),
            },
            "window_start": window_start,
            "window_end": window_end,
            "window_center": window_center,
            "window_center_source": "selection_core_pos",
            "result_path": str(result_path),
            "result_sha256": composites.sha256_file(result_path),
            "renderer_sha256": renderer_sha,
            "gene_annotation_path": str(gene_annotation) if gene_annotation else None,
            "gene_annotation_sha256": gene_annotation_sha,
            "gene_count": len(gene_names),
            "gene_names": gene_names,
            "panel_structure": "score_test_plus_shared_figure3_gene_track",
            "dpi": dpi,
            "show_title": False,
            "show_r2_legend": True,
            "expanded_gene_track": True,
            "gene_track_rows": figure4_finemap.SUPPLEMENTARY_GENE_TRACK_ROWS,
            "gene_track_renderer": "figure3_ehh.draw_gene_track",
            "source_figsize_inches": list(
                figure4_finemap.SUPPLEMENTARY_FIGSIZE_LOCUS
            ),
            "source_panel_aspect": figure4_finemap.SUPPLEMENTARY_PANEL_ASPECT,
            "best_p_marker_size": figure4_finemap.FINEMAP_BEST_P_MARKER_SIZE,
            "finemap_result_summary": finemap_result_summary,
        }
        contract_hash = composites._json_hash(source_contract)
        reused = False
        if not force and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous.get("contract_sha256") == contract_hash:
                    composites._validate_png(png_path)
                    composites._validate_pdf(pdf_path)
                    reused = True
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not reused:
            # ``plot_finemap`` applies Figure 4's large-text style globally.
            # Keep that styling inside this source panel so it cannot alter the
            # fixed 8.5 x 11 inch composite save contract that follows.
            with plt.rc_context():
                fig = figure4_finemap.plot_finemap(
                    frame,
                    locus,
                    show_title=False,
                    show_r2_legend=True,
                    highlight_gene=False,
                    sv_pos=locus["sv_pos"],
                    sv_type=locus["sv_type"],
                    expanded_gene_track=True,
                    window_center_pos=window_center,
                    shared_gene_track=genes,
                    editable_axis_labels=True,
                )
                fig.savefig(
                    png_path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches=fig.bbox_inches,
                    pad_inches=0,
                )
                fig.savefig(
                    pdf_path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches=fig.bbox_inches,
                    pad_inches=0,
                    metadata={
                        "Title": f"Fine map {hit['variant_tag']} {hit['phecode']}",
                        "Subject": FINEMAP_PANEL_CONTRACT,
                    },
                )
                plt.close(fig)
            outputs = {
                "png": composites._validate_png(png_path),
                "pdf": composites._validate_pdf(pdf_path),
            }
            composites._write_json_atomic(
                manifest_path,
                {
                    "analysis_contract": FINEMAP_PANEL_CONTRACT,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "contract_sha256": contract_hash,
                    "source_contract": source_contract,
                    "outputs": outputs,
                },
            )
        output[index] = png_path
        provenance[str(index)] = {
            "mode": "Figure 4 renderer from verified SuSiE TSV",
            "result_stage": stage,
            "result_path": str(result_path),
            "panel_png": str(png_path),
            "panel_pdf": str(pdf_path),
            "panel_manifest": str(manifest_path),
            "cache": "hit" if reused else "rendered",
            "finemap_result_summary": finemap_result_summary,
        }
    return output, provenance


def stage_selection_inputs(
    loci: pd.DataFrame,
    bucket: str,
    destination: Path,
    *,
    threads: int = composites.DEFAULT_THREADS,
    force: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Stage only the Figure 3 chromosomes needed by the SV locus series."""
    return composites.stage_selection_inputs(
        loci, bucket, destination, threads=threads, force=force,
    )


def resolve_local_finemaps(
    hits: pd.DataFrame,
    directories: Sequence[Path],
) -> tuple[dict[int, Path], dict[str, Any]]:
    """Resolve exact fine-map PNGs from directories staged with ``gcloud rsync``."""
    resolved_dirs = [directory.expanduser().resolve() for directory in directories]
    missing_dirs = [str(directory) for directory in resolved_dirs if not directory.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Local fine-map directories are missing: {missing_dirs}")
    output: dict[int, Path] = {}
    provenance: dict[str, Any] = {}
    for _, hit in hits.iterrows():
        filename = composites.expected_finemap_filename(hit)
        matches = [directory / filename for directory in resolved_dirs if (directory / filename).is_file()]
        if not matches:
            raise FileNotFoundError(
                f"Fine-map image {filename} is absent from local directories {resolved_dirs}"
            )
        path = matches[0]
        validation = composites._validate_png(path)
        index = int(hit["figure_index"])
        output[index] = path
        provenance[str(index)] = {
            "mode": "local gcloud-rsync stage",
            "path": str(path),
            **validation,
        }
    return output, provenance


def _expected_sv_relative_paths(expected_stems: set[str]) -> set[str]:
    return {
        relative
        for stem in expected_stems
        for relative in (
            f"title/{stem}.png",
            f"title/{stem}.pdf",
            f"png/{stem}.png",
            f"pdf/{stem}.pdf",
            f"source_zipped/{stem}.zip",
            f"{stem}.json",
        )
    }


def prune_stale_sv_outputs(output_dir: Path, expected_stems: set[str]) -> list[Path]:
    """Remove obsolete generated SV composites while preserving other series."""
    if not expected_stems:
        raise ValueError("Refusing to prune SV outputs without an expected output set")
    output_dir = output_dir.expanduser().resolve()
    expected_relative = _expected_sv_relative_paths(expected_stems)
    candidates = [
        *(output_dir / "title").glob(f"{SV_OUTPUT_PREFIX}*.png"),
        *(output_dir / "title").glob(f"{SV_OUTPUT_PREFIX}*.pdf"),
        *(output_dir / "png").glob(f"{SV_OUTPUT_PREFIX}*.png"),
        *(output_dir / "pdf").glob(f"{SV_OUTPUT_PREFIX}*.pdf"),
        *(output_dir / "source_zipped").glob(f"{SV_OUTPUT_PREFIX}*.zip"),
        *output_dir.glob(f"{SV_OUTPUT_PREFIX}*.json"),
    ]
    removed: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        relative = resolved.relative_to(output_dir).as_posix()
        if relative in expected_relative:
            continue
        resolved.unlink()
        removed.append(resolved)
    source_root = output_dir / "source_panels"
    for panel_dir in source_root.glob(f"{SV_OUTPUT_PREFIX}*"):
        if not panel_dir.is_dir() or panel_dir.name in expected_stems:
            continue
        shutil.rmtree(panel_dir)
        removed.append(panel_dir.resolve())
    return sorted(removed)


def _sv_page_output_stem_name(
    page: pd.Series,
    page_hits: pd.DataFrame,
    *,
    association_index: int | None = None,
) -> str:
    """Name independently numbered selection-only and PheWAS SV series."""
    if page_hits.empty:
        if association_index is not None:
            raise ValueError("Selection-only SV pages cannot have an association index")
        prefix = SV_OUTPUT_PREFIX
        expected_series = "selection"
        association_component = ""
        suffix = f"selection_{int(page['core_pos'])}"
    else:
        if len(page_hits) != 1:
            raise ValueError("Each SV PheWAS file must contain exactly one association")
        if association_index is None or association_index < 1:
            raise ValueError("SV PheWAS files require a positive association index")
        prefix = SV_PHEWAS_OUTPUT_PREFIX
        expected_series = "phewas"
        association_component = f"_{association_index}"
        hit = page_hits.iloc[0]
        suffix = composites._phecode_string_slug(
            hit["phecode_name"], fallback=str(hit["phecode"])
        )
    observed_series = str(page["sv_series"])
    if observed_series != expected_series:
        raise ValueError(
            f"SV page series mismatch: expected {expected_series}, found {observed_series}"
        )
    chrom = str(page["chrom"]).removeprefix("chr")
    return (
        f"{prefix}_{int(page['sv_series_index']):02d}{association_component}_{chrom}_"
        f"{int(page['sv_pos'])}_{int(page['sv_length_bp'])}bp_"
        f"{str(page['sv_type']).upper()}__{suffix}"
    )


def compose_selection_page(
    page: pd.Series,
    page_hits: pd.DataFrame,
    phewas: pd.DataFrame | None,
    selection_paths: dict[str, Path],
    finemap_paths: dict[int, Path],
    igv_asset: dict[str, Any],
    output_dir: Path,
    category_order: Sequence[str],
    parameters: composites.PlotParameters,
    *,
    sample_manifest_sha256: str,
    association_index: int | None = None,
    phewas_highlight_hits: pd.DataFrame | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Compose one S3 selection page, omitting inapplicable association panels."""
    import matplotlib.pyplot as plt

    page_index = int(page["page_index"])
    page_key = str(page["variant_tag"])
    expected_associations = int(page["association_count"])
    has_association = expected_associations > 0
    if has_association and len(page_hits) != 1:
        raise ValueError(
            f"Selection page {page_key} must receive exactly one of its "
            f"{expected_associations} associations per output; received {len(page_hits)}"
        )
    if has_association and (
        association_index is None
        or not 1 <= association_index <= expected_associations
    ):
        raise ValueError(
            f"Selection page {page_key} requires association_index between 1 and "
            f"{expected_associations}"
        )
    if not has_association and len(page_hits):
        raise ValueError(f"Selection-only page {page_key} received associations")
    if not has_association and association_index is not None:
        raise ValueError(f"Selection-only page {page_key} received an association index")
    if has_association and page_hits["variant_tag"].nunique() != 1:
        raise ValueError(f"Selection page {page_key} has multiple association leads")
    if has_association and (phewas is None or phewas.empty):
        raise ValueError(f"Selection page {page_key} requires a PheWAS table")
    if not has_association and finemap_paths:
        raise ValueError(f"Selection-only page {page_key} received fine-map panels")

    expected_igv_key = _sv_allele_key(
        page["chrom"], page["sv_pos"], page["sv_length_bp"], page["sv_type"]
    )
    observed_igv_key = _sv_allele_key(
        igv_asset["chrom"],
        igv_asset["sv_position"],
        igv_asset["sv_length_bp"],
        igv_asset["sv_type"],
    )
    if observed_igv_key != expected_igv_key:
        raise ValueError(
            f"IGV screenshot allele mismatch: {observed_igv_key} != {expected_igv_key}"
        )
    igv_path = Path(igv_asset["path"]).expanduser().resolve()
    if not igv_path.is_file() or igv_path.stat().st_size == 0:
        raise FileNotFoundError(f"IGV screenshot is missing or empty: {igv_path}")
    observed_igv_sha256 = composites.sha256_file(igv_path)
    if observed_igv_sha256 != str(igv_asset["sha256"]):
        raise ValueError(
            f"IGV screenshot changed after validation for {expected_igv_key}: {igv_path}"
        )

    locus_frame = pd.DataFrame()
    finemap_items: list[tuple[pd.Series, Path]] = []
    highlight_hits = page_hits if phewas_highlight_hits is None else phewas_highlight_hits
    if has_association:
        association_variant = str(page_hits.iloc[0]["variant_tag"])
        assert phewas is not None
        locus_frame = phewas[phewas["variant_tag"].eq(association_variant)].copy()
        if locus_frame.empty:
            raise ValueError(f"PheWAS table lacks association lead {association_variant}")
        target_keys = set(page_hits["phecode"].astype(str))
        observed_keys = set(locus_frame["phecode"].astype(str))
        missing = sorted(target_keys - observed_keys)
        if missing:
            raise ValueError(
                f"PheWAS table lacks targets for selection page {page_key}: {missing}"
            )
        highlight_variants = set(highlight_hits["variant_tag"].astype(str))
        if highlight_variants != {association_variant}:
            raise ValueError(
                "PheWAS highlight rows must use the page association lead; found "
                f"{sorted(highlight_variants)} instead of {association_variant}"
            )
        highlighted_keys = set(highlight_hits["phecode"].astype(str))
        missing_highlights = sorted(highlighted_keys - observed_keys)
        if missing_highlights:
            raise ValueError(
                f"PheWAS table lacks highlighted page hits: {missing_highlights}"
            )
        for _, hit in page_hits.sort_values("figure_index", kind="mergesort").iterrows():
            hit_index = int(hit["figure_index"])
            if hit_index not in finemap_paths:
                raise FileNotFoundError(
                    f"Fine-map panel is missing for {hit['variant_tag']} x {hit['phecode']}"
                )
            finemap_items.append((hit, finemap_paths[hit_index]))

    stem_name = _sv_page_output_stem_name(
        page,
        page_hits,
        association_index=association_index,
    )
    paths = composites._composite_output_paths(output_dir, stem_name)
    panel_names = ("A", "B", "C", "D") if has_association else ("A", "B")
    source_panel_paths = composites._source_panel_output_paths(
        output_dir, stem_name, page_index, panel_names
    )
    wide_selection_path = selection_paths.get("combined_wide_png")
    has_wide_selection = (
        wide_selection_path is not None and wide_selection_path.is_file()
    )
    active_selection_source = "combined_png"
    if not has_association:
        active_selection_source = (
            "combined_wide_png" if has_wide_selection else "combined_png_fallback"
        )
    page_contract_columns = (
        "page_index", "sv_index", "sv_series", "sv_series_index", "variant_tag",
        "chrom", "core_pos", "ref", "alt", "gene", "ihs_val", "sv_pos",
        "sv_length_bp", "sv_type",
        "sv_annotation",
        "sv_gene_label", "selection_relation", "selection_relation_label",
        "selection_region_start", "selection_region_stop", "association_count",
        "association_variant_tag", "association_phecodes", "phewas_panel_status",
        "finemap_panel_status", "selection_panel_status", "runtime_phewas_applicable",
        "igv_panel_status",
    )
    source_contract: dict[str, Any] = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "vector_composite_contract": composites.VECTOR_COMPOSITE_CONTRACT,
        "pdf_text_rc": composites.EDITABLE_PDF_RC,
        "phewas_axis_label": composites.style.P_VALUE_AXIS_LABEL,
        "page": {
            key: composites._json_scalar(page[key])
            for key in page_contract_columns
            if key in page.index
        },
        "associations": [
            {
                key: composites._json_scalar(hit[key])
                for key in (
                    "figure_index", "variant_tag", "phecode", "phecode_name",
                    "discovery_p", "discovery_beta", "sv_annotation",
                    "selection_match_method",
                )
            }
            for _, hit in page_hits.iterrows()
        ],
        "phewas_highlights": [
            {
                key: composites._json_scalar(hit[key])
                for key in (
                    "figure_index", "variant_tag", "phecode", "phecode_name",
                    "discovery_p", "discovery_beta", "selection_match_method",
                )
                if key in hit.index
            }
            for _, hit in highlight_hits.iterrows()
        ] if has_association else [],
        "panel_decisions": {
            "selection": str(page["selection_panel_status"]),
            "phewas": str(page["phewas_panel_status"]),
            "finemap": str(page["finemap_panel_status"]),
            "igv": str(page["igv_panel_status"]),
        },
        "sample_manifest_sha256": sample_manifest_sha256,
        "plot_parameters": composites.asdict(parameters),
        "selection": {
            "combined_sha256": composites.sha256_file(selection_paths["combined_png"]),
            "combined_pdf_sha256": composites.sha256_file(
                selection_paths["combined_pdf"]
            ),
            "wide_sha256": (
                composites.sha256_file(wide_selection_path)
                if has_wide_selection
                else None
            ),
            "wide_pdf_sha256": (
                composites.sha256_file(selection_paths["combined_wide_pdf"])
                if has_wide_selection
                else None
            ),
            "active_source": active_selection_source,
        },
        "igv": {
            key: composites._json_scalar(igv_asset[key])
            for key in (
                "display_path", "chrom", "sv_position", "sv_length_bp", "sv_type",
                "sha256", "width_px", "height_px",
                "source_figure", "source_page", "source_image", "source_pdf",
                "source_pdf_sha256",
            )
        },
        "visual_title_variants": {
            "titled": "overall title and panel subtitles",
            "no_title": "panel letters only",
        },
        "panel_layout": (
            "A selection; B IGV"
            if not has_association
            else "A full width; B lower left; C lower right; D IGV third row"
        ),
        "panel_height_ratios": list(
            SV_PHEWAS_HEIGHT_RATIOS
            if has_association
            else SV_SELECTION_ONLY_HEIGHT_RATIOS
        ),
        "output_layout": {
            "title_free_png_directory": str(output_dir / "png"),
            "title_free_pdf_directory": str(output_dir / "pdf"),
            "titled_directory": str(output_dir / "title"),
            "source_panel_directory": str(
                output_dir / "source_panels" / stem_name
            ),
            "source_zip": str(paths["source_zip"]),
            "shared_filename": paths["no_title_png"].name,
        },
    }
    if has_association:
        source_contract["association_index"] = int(association_index)
        source_contract["association_total"] = expected_associations
        source_contract["phewas_subset_sha256"] = composites._sha256_bytes(
            locus_frame.sort_values("phecode").to_csv(
                index=False, float_format="%.17g"
            ).encode("utf-8")
        )
        source_contract["finemap"] = [
            {
                "figure_index": int(hit["figure_index"]),
                "phecode": str(hit["phecode"]),
                "sha256": composites.sha256_file(path),
                "pdf_sha256": composites.sha256_file(path.with_suffix(".pdf")),
            }
            for hit, path in finemap_items
        ]
    contract_hash = composites._json_hash(source_contract)
    if not force and paths["manifest"].is_file():
        try:
            previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            if previous.get("contract_sha256") == contract_hash:
                for key in ("png", "no_title_png"):
                    composites._validate_png(paths[key])
                for key in ("pdf", "no_title_pdf"):
                    composites._validate_pdf(paths[key])
                composites._validate_source_panel_outputs(source_panel_paths)
                if not paths["source_zip"].is_file():
                    raise FileNotFoundError(paths["source_zip"])
                with composites.zipfile.ZipFile(paths["source_zip"]) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"Corrupt source archive: {paths['source_zip']}")
                return paths
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    source_panel_outputs: dict[str, dict[str, Any]] = {}

    def render_version(
        *,
        include_title: bool,
        png_path: Path,
        pdf_path: Path,
        save_sources: bool = False,
    ) -> None:
        fig = plt.figure(figsize=(8.5, 11), constrained_layout=False)
        if not has_association:
            grid = fig.add_gridspec(
                2,
                16,
                height_ratios=SV_SELECTION_ONLY_HEIGHT_RATIOS,
                left=0.065,
                right=0.985,
                top=0.895 if include_title else 0.965,
                bottom=0.06,
                hspace=0.30,
            )
            _group_axis, ax_selection = composites._show_grouped_selection_panel(
                fig,
                grid[0, 1:15],
                selection_paths,
                include_title=include_title,
                panel_label="A",
                wide=True,
            )
            ax_igv = fig.add_subplot(grid[1, 2:14])
            composites._show_image_panel(
                ax_igv,
                igv_path,
                "B",
                f"IGV validation: {page['sv_annotation']}" if include_title else None,
                anchor="N",
                fill_cell=False,
            )
        else:
            if len(finemap_items) != 1:
                raise ValueError(
                    f"SV PheWAS page {page_key} must contain exactly one fine-map panel"
                )
            grid = fig.add_gridspec(
                3,
                24,
                height_ratios=SV_PHEWAS_HEIGHT_RATIOS,
                left=0.055,
                right=0.985,
                top=0.895 if include_title else 0.965,
                bottom=0.055,
                hspace=0.34,
                wspace=0.70,
            )
            ax_phewas = fig.add_subplot(grid[0, 2:22])
            composites.draw_phewas_panel(
                ax_phewas,
                locus_frame,
                highlight_hits,
                page_hits.iloc[0],
                category_order,
                parameters,
                highlight_color=SV_HIGHLIGHT_COLOR,
                highlight_label="SV-linked/lead",
            )
            ax_phewas.text(
                -0.055, 1.02, "A", transform=ax_phewas.transAxes,
                fontsize=14, fontweight="bold", ha="right", va="bottom",
            )
            _group_axis, ax_selection = composites._show_grouped_selection_panel(
                fig,
                grid[1, 0:12],
                selection_paths,
                include_title=include_title,
                panel_label="B",
            )
            hit, finemap_path = finemap_items[0]
            ax_fine = fig.add_subplot(grid[1, 12:24])
            label = (
                f"{hit['phecode']}: {hit['phecode_name']}"
                if include_title else None
            )
            composites._show_image_panel(
                ax_fine,
                finemap_path,
                "C",
                label,
                strip_embedded_title=True,
                anchor="N",
                fill_cell=False,
            )
            ax_igv = fig.add_subplot(grid[2, 4:20])
            composites._show_image_panel(
                ax_igv,
                igv_path,
                "D",
                f"IGV validation: {page['sv_annotation']}" if include_title else None,
                anchor="N",
                fill_cell=False,
            )

        if include_title:
            association_suffix = ""
            if has_association:
                phecodes = ", ".join(page_hits["phecode"].astype(str))
                association_suffix = f"\nAudited S9 association: {phecodes}"
            fig.suptitle(
                f"{page['chrom']}:{int(page['core_pos']):,} · {page['gene']}\n"
                f"{page['selection_relation_label']}: {page['sv_annotation']}"
                f"{association_suffix}",
                fontsize=14.5,
                fontweight="bold",
                y=0.985,
            )
        metadata_values = {
            "Title": f"Supplementary SV selection composite {page_index}: {page_key}",
            "Subject": ANALYSIS_CONTRACT,
            "Keywords": (
                "selection SV EHH iHS nucleotide diversity PheWAS fine mapping IGV"
            ),
        }
        fig.savefig(
            png_path,
            dpi=parameters.dpi,
            facecolor="white",
            bbox_inches=fig.bbox_inches,
            pad_inches=0,
        )
        if not has_association:
            selection_png_key = (
                "combined_wide_png" if has_wide_selection else "combined_png"
            )
            selection_pdf_key = (
                "combined_wide_pdf" if has_wide_selection else "combined_pdf"
            )
            vector_overlays = ((ax_selection, selection_paths[selection_pdf_key]),)
        else:
            selection_png_key = "combined_png"
            selection_pdf_key = "combined_pdf"
            vector_overlays = (
                (ax_selection, selection_paths[selection_pdf_key]),
                (ax_fine, finemap_path.with_suffix(".pdf")),
            )
        composites._save_vector_composite_pdf(
            fig, pdf_path, vector_overlays, metadata=metadata_values
        )
        if save_sources:
            raster_panel_axes = (
                {"B": (ax_igv,)}
                if not has_association
                else {"A": (ax_phewas,), "D": (ax_igv,)}
            )
            source_panel_outputs.update(
                composites._save_source_panels(
                    fig,
                    raster_panel_axes,
                    source_panel_paths,
                    dpi=parameters.dpi,
                    metadata=metadata_values,
                )
            )
            selection_panel = "A" if not has_association else "B"
            source_panel_outputs[selection_panel] = composites._copy_native_source_panel(
                source_panel_paths[selection_panel],
                png_source=selection_paths[selection_png_key],
                pdf_source=selection_paths[selection_pdf_key],
            )
            if has_association:
                source_panel_outputs["C"] = composites._copy_native_source_panel(
                    source_panel_paths["C"],
                    png_source=finemap_path,
                    pdf_source=finemap_path.with_suffix(".pdf"),
                )
        plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("png", "pdf", "no_title_png", "no_title_pdf"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    render_version(include_title=True, png_path=paths["png"], pdf_path=paths["pdf"])
    render_version(
        include_title=False,
        png_path=paths["no_title_png"],
        pdf_path=paths["no_title_pdf"],
        save_sources=True,
    )
    source_zip_output = composites._write_source_zip(
        paths["source_zip"],
        source_panel_paths,
        figure_index=page_index,
        descriptors=(
            None
            if has_association
            else {"A": "ehh_ihs_pi_genes", "B": "igv_validation"}
        ),
    )
    outputs = {
        "png": composites._validate_png(paths["png"]),
        "pdf": composites._validate_pdf(paths["pdf"]),
        "no_title_png": composites._validate_png(paths["no_title_png"]),
        "no_title_pdf": composites._validate_pdf(paths["no_title_pdf"]),
        "source_panels": source_panel_outputs,
        "source_zip": source_zip_output,
    }
    composites._write_json_atomic(
        paths["manifest"],
        {
            "analysis_contract": ANALYSIS_CONTRACT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": contract_hash,
            "source_contract": source_contract,
            "outputs": outputs,
        },
    )
    return paths


def _configure_logging(work_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"sv_linked_locus_composites_{stamp}.log"
    logger = logging.getLogger("sv_linked_locus_composites")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, log_path


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "matplotlib", "adjustText", "Pillow", "scikit-allel")
    output: dict[str, str | None] = {}
    for package in packages:
        try:
            output[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            output[package] = None
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("tasks", "selection", "finemap", "compose", "all"),
        default="all",
    )
    parser.add_argument("--bucket", default=composites.DEFAULT_BUCKET)
    parser.add_argument("--s9-source-data", type=Path, default=DEFAULT_S9_SOURCE_DATA)
    parser.add_argument("--selection-hits", type=Path, default=DEFAULT_SELECTION_HITS)
    parser.add_argument("--ihs-regions", type=Path, default=DEFAULT_IHS_REGIONS)
    parser.add_argument(
        "--igv-dir",
        type=Path,
        default=DEFAULT_IGV_DIR,
        help="Directory containing the committed IGV screenshots and manifest.tsv.",
    )
    parser.add_argument("--phewas-object", default=composites.DEFAULT_PHEWAS_OBJECT)
    parser.add_argument(
        "--gene-annotation-object",
        default=DEFAULT_GENE_ANNOTATION_OBJECT,
        help="refGene GTF object beneath the workspace-bucket root.",
    )
    parser.add_argument(
        "--phewas-path", type=Path, default=None,
        help="Local PheWAS TSV staged with gcloud storage rsync; skips per-file GCS staging.",
    )
    parser.add_argument(
        "--finemap-gs-dir", action="append", default=None,
        help="Fine-map GCS directory; repeat for fallbacks.",
    )
    parser.add_argument(
        "--finemap-dir", action="append", type=Path, default=None,
        help=(
            "Local SuSiE result directory staged with gcloud storage rsync; "
            "repeat for fallbacks. Verified TSVs are redrawn with Figure 4 styling."
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=composites.DEFAULT_THREADS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-hits", type=int, default=DEFAULT_EXPECTED_HITS)
    parser.add_argument(
        "--expected-selection-pages",
        type=int,
        default=DEFAULT_EXPECTED_SELECTION_REGIONS,
    )
    parser.add_argument(
        "--expected-sv-alleles",
        type=int,
        default=DEFAULT_EXPECTED_SV_ALLELES,
    )
    parser.add_argument("--odds-ratio-cap", type=float, default=composites.DEFAULT_OR_CAP)
    parser.add_argument("--phewas-y-max", type=float, default=None)
    parser.add_argument("--marker-area-min", type=float, default=composites.DEFAULT_MARKER_MIN)
    parser.add_argument("--marker-area-max", type=float, default=composites.DEFAULT_MARKER_MAX)
    parser.add_argument("--phewas-threshold", type=float, default=composites.DEFAULT_PHEWAS_THRESHOLD)
    parser.add_argument(
        "--selection-half-window-bp", type=int,
        default=composites.DEFAULT_SELECTION_HALF_WINDOW_BP,
    )
    parser.add_argument("--dpi", type=int, default=composites.DEFAULT_DPI)
    parser.add_argument("--chunk-rows", type=int, default=composites.DEFAULT_CHUNK_ROWS)
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.selection_half_window_bp < 1:
        raise ValueError("--selection-half-window-bp must be positive")
    if args.selection_half_window_bp != composites.FINEMAP_HALF_WINDOW_BP:
        raise ValueError(
            "--selection-half-window-bp must equal the fine-map half-window "
            f"({composites.FINEMAP_HALF_WINDOW_BP} bp) so panels B and C share endpoints"
        )
    if not 0 < args.phewas_threshold < 1:
        raise ValueError("--phewas-threshold must be between zero and one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    bucket = composites._normalize_bucket(args.bucket)
    work_dir = args.work_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = _configure_logging(work_dir)
    logger.info("analysis_contract=%s phase=%s", ANALYSIS_CONTRACT, args.phase)
    logger.info(
        "bucket=%s work_dir=%s output_dir=%s threads=%s seed=%s",
        bucket, work_dir, output_dir, args.threads, args.seed,
    )

    hits, loci, mapping_provenance = load_sv_linked_hits(
        args.s9_source_data,
        args.selection_hits,
        args.ihs_regions,
        expected_hits=args.expected_hits,
        expected_selection_regions=args.expected_selection_pages,
        expected_sv_alleles=args.expected_sv_alleles,
    )
    hit_manifest_path = output_dir / "sv_linked_lead_hit_manifest.tsv"
    hits.to_csv(hit_manifest_path, sep="\t", index=False)
    page_manifest_path = output_dir / "sv_selection_page_manifest.tsv"
    page_manifest = loci.rename(
        columns={"variant_tag": "selection_page_key", "sv_pos": "sv_position"}
    )
    page_manifest.to_csv(page_manifest_path, sep="\t", index=False)
    susie_tasks_path = write_susie_tasks(
        hits, output_dir / "sv_linked_lead_susie_tasks.tsv"
    )
    logger.info(
        "SV page contract: %s selection pages; %s association rows across %s leads",
        len(loci), len(hits), hits["variant_tag"].nunique(),
    )

    phewas_remote = composites._gs_join(bucket, args.phewas_object)
    phewas_local = (
        args.phewas_path.expanduser().resolve()
        if args.phewas_path is not None
        else work_dir / "cache" / "inputs" / Path(args.phewas_object).name
    )
    table_stage: dict[str, Any] = {}
    finemap_paths: dict[int, Path] = {}
    finemap_stage: dict[str, Any] = {}
    selection_input_stage: dict[str, Any] = {}
    igv_provenance: dict[str, Any] = {}
    igv_assets: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    phewas = pd.DataFrame()
    phewas_metadata: dict[str, Any] = {}
    runtime_applicability: dict[str, Any] = {}
    remote_dirs = args.finemap_gs_dir or [
        composites._gs_join(bucket, directory) for directory in composites.DEFAULT_FINEMAP_DIRS
    ]
    remote_dirs = [composites._gs_join(bucket, directory) for directory in remote_dirs]
    finemap_cache = work_dir / "cache" / "finemap"
    # Keep controlled inputs rooted at the selected workspace bucket while all
    # transient SuSiE and panel artifacts remain beneath --work-dir.  The
    # recomputation launcher and Figure 4 renderer share this mutable config
    # module in the current Python process.
    os.environ["AOU_WORKSPACE_BUCKET_GS"] = bucket
    os.environ["AOU_SOURCE_BUCKET"] = bucket
    os.environ["AOU_SUSIE_DIR"] = str(finemap_cache)
    os.environ["AOU_SUSIE_GS_DIR"] = remote_dirs[0]
    config.WORKSPACE_BUCKET_GS = bucket
    config.SUSIE_FINEMAP_DIR = str(finemap_cache)
    config.SUSIE_FINEMAP_GS_DIR = remote_dirs[0]

    gene_annotation_local = (
        finemap_cache / "workspace_inputs" / "hg38.refGene.gtf"
    )
    if args.phase in {"selection", "finemap", "compose", "all"}:
        gene_annotation_remote = composites._gs_join(
            bucket, args.gene_annotation_object
        )
        table_stage["finemap_gene_annotation"] = composites._stage_gcs_file(
            gene_annotation_remote,
            gene_annotation_local,
            force=args.force,
        )
        logger.info(
            "staged Figure 4 gene annotation: %s (%s)",
            gene_annotation_local,
            table_stage["finemap_gene_annotation"]["cache"],
        )

    if args.phase in {"compose", "all"}:
        if args.phewas_path is not None:
            if not phewas_local.is_file() or phewas_local.stat().st_size == 0:
                raise FileNotFoundError(f"Locally staged PheWAS table is missing or empty: {phewas_local}")
            table_stage["phewas"] = {
                "mode": "local gcloud-rsync stage",
                "path": str(phewas_local),
                "size": phewas_local.stat().st_size,
                "sha256": composites.sha256_file(phewas_local),
            }
        else:
            table_stage["phewas"] = composites._stage_gcs_file(
                phewas_remote, phewas_local, force=args.force,
            )
        phewas, phewas_metadata = composites.load_phewas_for_variants(
            phewas_local,
            set(hits["variant_tag"]),
            chunk_rows=args.chunk_rows,
        )
        hits, loci, runtime_applicability = resolve_runtime_phewas_applicability(
            hits,
            loci,
            phewas,
        )
        mapping_provenance["runtime_phewas_applicability"] = runtime_applicability
        hits.to_csv(hit_manifest_path, sep="\t", index=False)
        loci.rename(
            columns={"variant_tag": "selection_page_key", "sv_pos": "sv_position"}
        ).to_csv(page_manifest_path, sep="\t", index=False)
        write_susie_tasks(hits, susie_tasks_path)
        logger.info(
            "runtime PheWAS applicability: %s target rows; %s PheWAS pages; "
            "%s pages reclassified to selection-only",
            runtime_applicability["runtime_association_rows"],
            runtime_applicability["runtime_pages_with_phewas"],
            runtime_applicability["runtime_pages_reclassified_to_selection"],
        )
        igv_assets, igv_provenance = load_igv_screenshots(
            args.igv_dir,
            expected_sv_alleles=args.expected_sv_alleles,
        )
        expected_igv_keys = {
            _sv_allele_key(
                row["chrom"], row["sv_pos"], row["sv_length_bp"], row["sv_type"]
            )
            for _, row in loci.iterrows()
        }
        observed_igv_keys = set(igv_assets)
        missing_igv = sorted(expected_igv_keys - observed_igv_keys)
        unused_igv = sorted(observed_igv_keys - expected_igv_keys)
        if missing_igv or unused_igv:
            raise ValueError(
                "IGV screenshot assets do not match the runtime SV allele set: "
                f"missing={missing_igv}, unused={unused_igv}"
            )
        igv_provenance["selection_allele_count"] = len(expected_igv_keys)
        logger.info(
            "validated %s exact IGV screenshots from %s",
            len(igv_assets),
            igv_provenance["manifest"],
        )

    if args.phase in {"finemap", "compose", "all"}:
        local_finemap_dirs = [
            finemap_cache,
            *(path.expanduser().resolve() for path in (args.finemap_dir or [])),
        ]
        try:
            finemap_paths, finemap_stage = render_figure4_finemap_panels(
                hits,
                finemap_cache,
                remote_dirs,
                local_dirs=local_finemap_dirs,
                dpi=args.dpi,
                force=args.force,
            )
        except (FileNotFoundError, ValueError) as exc:
            if args.phase not in {"finemap", "all"}:
                raise FileNotFoundError(
                    f"Verified SuSiE result/panel is unavailable: {exc}. "
                    "Run --phase finemap or --phase all on Workbench first."
                ) from exc
            logger.info(
                "verified SuSiE results are missing or invalid; running the exact task table: %s",
                susie_tasks_path,
            )
            from scripts import recompute_figure4_susie

            recompute_figure4_susie.ensure_figure4_outputs(task_table=susie_tasks_path)
            finemap_paths, finemap_stage = render_figure4_finemap_panels(
                hits,
                finemap_cache,
                remote_dirs,
                local_dirs=local_finemap_dirs,
                dpi=args.dpi,
                force=args.force,
            )
        logger.info(
            "rendered/validated %s Figure 4-style fine-map panels from verified TSVs",
            len(finemap_paths),
        )

    selection_dir = work_dir / "selection"
    selection_paths = {
        variant_tag: composites._selection_paths(selection_dir, variant_tag)
        for variant_tag in dict.fromkeys(loci["variant_tag"])
    }
    selection_provenance: dict[str, Any] = {}
    if args.phase in {"selection", "all"}:
        from scripts import figure3_ehh

        selection_sources, selection_input_stage = stage_selection_inputs(
            loci,
            bucket,
            work_dir / "cache" / "selection_inputs",
            threads=args.threads,
            force=args.force,
        )
        figure3_ehh.VCF_PATTERN = selection_sources["vcf_pattern"]
        figure3_ehh.IHS_PATTERN = selection_sources["ihs_pattern"]
        figure3_ehh.GTF_PATH = str(gene_annotation_local)
        figure3_ehh.HARDMASK_PATH = selection_sources["hardmask_path"]
        logger.info(
            "staged Figure 3 selection inputs for chromosomes: %s",
            ", ".join(selection_input_stage["chromosomes"]),
        )
        selection_paths, selection_provenance = composites.render_selection_loci(
            loci,
            loci[["variant_tag"]],
            selection_dir,
            work_dir / "cache" / "selection",
            source_identity=mapping_provenance,
            threads=args.threads,
            selection_half_window_bp=args.selection_half_window_bp,
            force=args.force,
        )
        logger.info("rendered/validated selection panels for %s S3 rows", len(selection_paths))

    outputs: dict[str, dict[str, Path]] = {}
    shared_scales: dict[str, float] = {}
    expected_stems: set[str] = set()
    pruned_local: list[Path] = []
    if args.phase in {"compose", "all"}:
        for variant_tag, paths in selection_paths.items():
            try:
                composites._validate_png(paths["combined_png"])
            except (FileNotFoundError, ValueError) as exc:
                raise FileNotFoundError(
                    f"Selection panel missing for {variant_tag}; run --phase selection first: {exc}"
                ) from exc
        category_order = list(dict.fromkeys(phewas["phecode_category"].astype(str)))
        shared_scales = composites.resolve_shared_phewas_scales(
            phewas,
            odds_ratio_cap=args.odds_ratio_cap,
            phewas_y_max=args.phewas_y_max,
        )
        parameters = composites.PlotParameters(
            odds_ratio_cap=shared_scales["odds_ratio_cap"],
            marker_area_min=args.marker_area_min,
            marker_area_max=args.marker_area_max,
            phewas_threshold=args.phewas_threshold,
            phewas_y_max=shared_scales["phewas_y_max"],
            dpi=args.dpi,
        )
        _, sample_hash = composites._sample_manifest()
        for _, page in loci.sort_values("page_index", kind="mergesort").iterrows():
            index = int(page["page_index"])
            igv_key = _sv_allele_key(
                page["chrom"], page["sv_pos"], page["sv_length_bp"], page["sv_type"]
            )
            igv_asset = igv_assets[igv_key]
            page_hits = hits[
                hits["selection_page_key"].eq(str(page["variant_tag"]))
            ].sort_values("figure_index", kind="mergesort")
            if page_hits.empty:
                output_key = f"selection:{index}"
                outputs[output_key] = compose_selection_page(
                    page,
                    page_hits,
                    None,
                    selection_paths[str(page["variant_tag"])],
                    {},
                    igv_asset,
                    output_dir,
                    category_order,
                    parameters,
                    sample_manifest_sha256=sample_hash,
                    force=args.force,
                )
                expected_stems.add(outputs[output_key]["manifest"].stem)
                continue

            for association_index, (_, hit) in enumerate(
                page_hits.iterrows(), start=1
            ):
                hit_frame = hit.to_frame().T
                hit_index = int(hit["figure_index"])
                output_key = f"phewas:{index}:{association_index}"
                outputs[output_key] = compose_selection_page(
                    page,
                    hit_frame,
                    phewas,
                    selection_paths[str(page["variant_tag"])],
                    {hit_index: finemap_paths[hit_index]},
                    igv_asset,
                    output_dir,
                    category_order,
                    parameters,
                    sample_manifest_sha256=sample_hash,
                    association_index=association_index,
                    phewas_highlight_hits=page_hits,
                    force=args.force,
                )
                expected_stems.add(outputs[output_key]["manifest"].stem)
        pruned_local = prune_stale_sv_outputs(output_dir, expected_stems)
        logger.info(
            "rendered/validated %s titled/no-title PNG/PDF SV files; "
            "pruned %s obsolete local SV files",
            len(outputs),
            len(pruned_local),
        )

    run_manifest = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "git_sha": composites._git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "parameters": {
            "bucket": bucket,
            "output_dir": str(output_dir),
            "igv_dir": str(args.igv_dir.expanduser().resolve()),
            "phewas_object": phewas_remote,
            "phewas_path": str(phewas_local),
            "gene_annotation_object": composites._gs_join(
                bucket, args.gene_annotation_object
            ),
            "finemap_gs_dirs": remote_dirs,
            "finemap_dirs": [str(path.expanduser().resolve()) for path in (args.finemap_dir or [])],
            "threads": args.threads,
            "seed": args.seed,
            "expected_hits": args.expected_hits,
            "expected_selection_pages": args.expected_selection_pages,
            "expected_sv_alleles": args.expected_sv_alleles,
            "odds_ratio_cap": args.odds_ratio_cap,
            "phewas_y_max": args.phewas_y_max,
            "phewas_threshold": args.phewas_threshold,
            "selection_half_window_bp": args.selection_half_window_bp,
            "dpi": args.dpi,
            "chunk_rows": args.chunk_rows,
            "force": args.force,
        },
        "mapping": mapping_provenance,
        "runtime_phewas_applicability": runtime_applicability,
        "hit_manifest": {
            "path": str(hit_manifest_path),
            "sha256": composites.sha256_file(hit_manifest_path),
            "association_rows": len(hits),
            "association_loci": int(hits["variant_tag"].nunique()),
        },
        "page_manifest": {
            "path": str(page_manifest_path),
            "sha256": composites.sha256_file(page_manifest_path),
            "selection_pages": len(loci),
            "pages_with_association": int(loci["association_count"].gt(0).sum()),
            "pages_without_association": int(loci["association_count"].eq(0).sum()),
        },
        "susie_tasks": {
            "path": str(susie_tasks_path),
            "sha256": composites.sha256_file(susie_tasks_path),
            "rows": len(hits),
        },
        "table_stage": table_stage,
        "finemap_stage": finemap_stage,
        "selection_input_stage": selection_input_stage,
        "igv_assets": igv_provenance,
        "selection": selection_provenance,
        "phewas": phewas_metadata,
        "shared_phewas_scales": shared_scales,
        "outputs": {
            str(index): {key: str(path) for key, path in paths.items()}
            for index, paths in outputs.items()
        },
        "pruned_stale_local_sv_outputs": [str(path) for path in pruned_local],
        "log": str(log_path),
    }
    run_manifest_path = output_dir / "sv_selection_run_manifest.json"
    composites._write_json_atomic(run_manifest_path, run_manifest)
    logger.info("run manifest: %s", run_manifest_path)

    return {
        "work_dir": work_dir,
        "output_dir": output_dir,
        "hit_manifest": hit_manifest_path,
        "page_manifest": page_manifest_path,
        "susie_tasks": susie_tasks_path,
        "run_manifest": run_manifest_path,
        "outputs": outputs,
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in result.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
