"""Build the 17 PheWAS/selection/10k-fine-map supplementary composites.

The workflow is Workbench-native and split into restartable phases:

``stage``
    Copy the two aggregate TSVs and verified fine-map result TSVs from the
    study workspace bucket, then render titleless native source panels.
``selection``
    Stage only the required chromosome VCFs and indexes from the workspace
    bucket, then reuse Figure 3B's renderer to draw EHH, iHS, nucleotide
    diversity, and one shared gene track per unique meta-significant locus.
``compose``
    Draw an odds-ratio-aware PheWAS Manhattan panel and assemble titled and
    no-title letter-size PNG/PDF composites per meta-significant pair.

The expected production contract is 17 hits at 13 unique loci.  Every phase is
idempotent.  Cache decisions depend on the hit manifest, the 1,025-sample
manifest, plotting parameters, and the relevant input/output fingerprints, not
on the repository commit alone.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# Workbench notebook kernels set the inline backend.  This command is a
# headless batch renderer, and the isolated uv environment may not contain the
# matplotlib-inline package, so sanitize the inherited value before Matplotlib
# validates it during import.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream

from scripts import figure3_locus_combined
from scripts.common import config, gcs, samples, style

ANALYSIS_CONTRACT = "supp_phewas_combined_selection_10k_finemap_v23"
# Composite-only revisions must not invalidate valid controlled-data selection
# panels. Bump this independently only when EHH/iHS inputs or rendering change.
SELECTION_CACHE_CONTRACT = "supp_phewas_combined_selection_10k_finemap_v14"
FINEMAP_PANEL_CONTRACT = "supp_phewas_native_shared_gene_finemap_v1"
DEFAULT_BUCKET = "gs://rw-migration-aou-rw-a3e31f4d"
DEFAULT_META_OBJECT = "phewas_meta/meta_analysis_results_genic_meff.tsv"
DEFAULT_PHEWAS_OBJECT = "phewas_sv_v5/phewas_all_filtered.mac20.converged.tsv"
DEFAULT_FINEMAP_DIRS = ("susie_finemap_meta_dual", "susie_finemap_meta")
DEFAULT_GENE_ANNOTATION_OBJECT = "hg38.refGene.gtf"
DEFAULT_HARDMASK_OBJECT = "hardmask.hg38.v4.over99.bed"
DEFAULT_SELECTION_VCF_OBJECT = "SV_biallelic/{chrom}.biallelic.vcf.gz"
DEFAULT_SELECTION_IHS_OBJECT = "ihs_output_v4/{chrom}.ihs.out.100bins.norm.ann"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "figure5" / "loci"
DEFAULT_THREADS = 4
DEFAULT_SEED = 20260715
DEFAULT_EXPECTED_HITS = 17
DEFAULT_EXPECTED_LOCI = 13
DEFAULT_OR_CAP = 4.0
DEFAULT_OR_LEGEND_FOLDS = (1.0, 2.0, 3.0, 4.0)
DEFAULT_MARKER_MIN = 36.0
DEFAULT_MARKER_MAX = 220.0
DEFAULT_PHEWAS_THRESHOLD = 1e-3
DEFAULT_SELECTION_HALF_WINDOW_BP = 500_000
FINEMAP_HALF_WINDOW_BP = 500_000
COMPOSITE_HEIGHT_RATIOS = (0.58, 0.71, 0.71)
# The selection plot is drawn natively at the aspect ratio of its destination
# cell.  A 12-inch source width retains the established Figure 3 typography
# while avoiding any non-uniform scaling when it is reduced onto the page.
SELECTION_COMPACT_PANEL_ASPECT = 1.3405649529424606
SELECTION_WIDE_PANEL_ASPECT = 1.2729933256530441
SELECTION_SOURCE_WIDTH_IN = 12.0
SELECTION_COMPACT_FIGSIZE = (
    SELECTION_SOURCE_WIDTH_IN,
    SELECTION_SOURCE_WIDTH_IN / SELECTION_COMPACT_PANEL_ASPECT,
)
SELECTION_WIDE_FIGSIZE = (
    SELECTION_SOURCE_WIDTH_IN,
    SELECTION_SOURCE_WIDTH_IN / SELECTION_WIDE_PANEL_ASPECT,
)
DEFAULT_DPI = 200
DEFAULT_CHUNK_ROWS = 250_000
# Retain only the immediate y-tick/colorbar-label band above the first
# chromatic plot row. A larger fraction can retain long ``chr4:...`` title
# fragments and make their former width shrink panel C after embedding.
FINEMAP_TITLE_HEADROOM_FRACTION = 0.060
FINEMAP_TOP_PADDING_FRACTION = 0.012
FINEMAP_SIDE_LABEL_HEIGHT_PX = 64
FINEMAP_POSTCROP_TOP_PADDING_PX = 16
PHEWAS_CATEGORY_BACKGROUND_COLORS = ("#6baed6", "#bdbdbd")
PHEWAS_CATEGORY_LABEL_ROTATION = 40.0
# Matplotlib's default Type 3 fonts contain glyph drawings. Embed TrueType
# fonts with Unicode maps instead, including mathtext and the pi axis label.
# Apply at output time so caller styles cannot silently restore Type 3.
EDITABLE_PDF_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False}
VECTOR_COMPOSITE_CONTRACT = "native_panels_with_transformed_crop_boxes_v2"


@dataclass(frozen=True)
class PlotParameters:
    odds_ratio_cap: float = DEFAULT_OR_CAP
    marker_area_min: float = DEFAULT_MARKER_MIN
    marker_area_max: float = DEFAULT_MARKER_MAX
    phewas_threshold: float = DEFAULT_PHEWAS_THRESHOLD
    phewas_y_max: float | None = None
    dpi: int = DEFAULT_DPI


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_bucket(value: str) -> str:
    value = str(value).strip().rstrip("/")
    if not value:
        raise ValueError("Workspace bucket cannot be empty")
    return value if value.startswith("gs://") else f"gs://{value.lstrip('/')}"


def _gs_join(bucket: str, value: str) -> str:
    value = str(value).strip()
    return value.rstrip("/") if value.startswith("gs://") else (
        f"{_normalize_bucket(bucket)}/{value.lstrip('/').rstrip('/')}"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_bytes(payload.encode("utf-8"))


def _json_scalar(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remote_cache_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".remote.json")


def _stage_gcs_file(remote: str, local: Path, *, force: bool = False) -> dict[str, Any]:
    """Stage one object atomically and retain the remote integrity metadata."""
    local = local.expanduser().resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    remote_metadata = gcs.object_metadata(remote)
    sidecar = _remote_cache_sidecar(local)

    if not force and local.is_file() and local.stat().st_size > 0:
        if remote_metadata is not None and sidecar.is_file():
            try:
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if (
                cached.get("remote") == remote
                and cached.get("remote_metadata") == remote_metadata
                and int(remote_metadata.get("size", -1)) == local.stat().st_size
            ):
                return {
                    "remote": remote,
                    "local": str(local),
                    "size": local.stat().st_size,
                    "remote_metadata": remote_metadata,
                    "cache": "metadata_match",
                }
        if gcs.gcs_same_file(local, remote):
            return {
                "remote": remote,
                "local": str(local),
                "size": local.stat().st_size,
                "remote_metadata": remote_metadata,
                "cache": "md5_match",
            }

    temporary = local.with_suffix(local.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    result = gcs.gcs_cp(remote, temporary)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        if temporary.exists():
            temporary.unlink()
        detail = (result.stderr or result.stdout or "copy failed").strip()
        raise FileNotFoundError(f"Could not stage {remote}: {detail}")
    if remote_metadata is not None and temporary.stat().st_size != int(remote_metadata["size"]):
        observed = temporary.stat().st_size
        temporary.unlink()
        raise OSError(
            f"Size mismatch while staging {remote}: expected {remote_metadata['size']}, "
            f"observed {observed}"
        )
    temporary.replace(local)
    _write_json_atomic(
        sidecar,
        {
            "remote": remote,
            "remote_metadata": remote_metadata,
            "local_size": local.stat().st_size,
            "staged_utc": _utc_now(),
        },
    )
    return {
        "remote": remote,
        "local": str(local),
        "size": local.stat().st_size,
        "remote_metadata": remote_metadata,
        "cache": "downloaded",
    }


def _find_header_row(path: Path, required: Sequence[str]) -> int:
    required_set = {column.strip() for column in required}
    with path.open("rt", encoding="utf-8-sig", errors="replace") as stream:
        for index, line in enumerate(stream):
            columns = {column.strip() for column in line.rstrip("\n").split("\t")}
            if required_set <= columns:
                return index
            if index >= 100:
                break
    raise ValueError(f"No TSV header containing {sorted(required_set)} in {path}")


def _first_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    column_list = list(columns)
    present = set(column_list)
    exact = next((column for column in candidates if column in present), None)
    if exact is not None:
        return exact
    casefolded = {str(column).casefold(): column for column in column_list}
    return next(
        (casefolded[column.casefold()] for column in candidates if column.casefold() in casefolded),
        None,
    )


def parse_variant_tag(value: str) -> tuple[int, int, str, str]:
    parts = str(value).strip().replace("chr", "", 1).split("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid variant_tag {value!r}; expected CHR_POS_REF_ALT")
    chrom, pos = int(parts[0]), int(parts[1])
    if not 1 <= chrom <= 22 or pos <= 0 or not parts[2] or not parts[3]:
        raise ValueError(f"Invalid autosomal variant_tag {value!r}")
    return chrom, pos, parts[2], parts[3]


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES", "Y"})


def load_meta_hits(
    path: Path,
    *,
    expected_hits: int = DEFAULT_EXPECTED_HITS,
    expected_loci: int = DEFAULT_EXPECTED_LOCI,
) -> pd.DataFrame:
    """Load and validate the ordered meta-significant variant-phecode manifest."""
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Meta-analysis table is missing or empty: {path}")
    header_row = _find_header_row(path, ("variant_tag", "phecode"))
    frame = pd.read_csv(path, sep="\t", skiprows=header_row, dtype=str, low_memory=False)
    frame.columns = frame.columns.str.strip()
    if "meta_significant" not in frame.columns:
        raise ValueError("Meta-analysis table must include meta_significant")
    frame = frame.dropna(subset=["variant_tag", "phecode"]).copy()
    frame = frame[_truthy(frame["meta_significant"])].copy()
    frame["variant_tag"] = frame["variant_tag"].astype(str).str.strip()
    frame["phecode"] = frame["phecode"].astype(str).str.strip()
    frame = frame.drop_duplicates(["variant_tag", "phecode"], keep="first").reset_index(drop=True)

    parsed = frame["variant_tag"].map(parse_variant_tag)
    frame[["chrom", "pos", "ref", "alt"]] = pd.DataFrame(parsed.tolist(), index=frame.index)
    name_column = _first_column(frame.columns, ("phecode_string", "phecode_name", "phenotype"))
    gene_column = _first_column(frame.columns, ("gene_in_window", "gene", "nearest_gene"))
    meta_p_column = _first_column(frame.columns, ("p_meta", "meta_p", "pvalue_meta"))
    meta_beta_column = _first_column(frame.columns, ("beta_meta", "meta_beta"))
    frame["phecode_name"] = (
        frame[name_column].fillna(frame["phecode"]).astype(str).str.strip()
        if name_column
        else frame["phecode"]
    )
    frame["gene"] = frame[gene_column].fillna("").astype(str).str.strip() if gene_column else ""
    frame["p_meta"] = (
        pd.to_numeric(frame[meta_p_column], errors="coerce") if meta_p_column else np.nan
    )
    frame["beta_meta"] = (
        pd.to_numeric(frame[meta_beta_column], errors="coerce")
        if meta_beta_column
        else np.nan
    )
    variant_codes, _ = pd.factorize(frame["variant_tag"], sort=False)
    frame.insert(0, "figure_index", np.arange(1, len(frame) + 1, dtype=int))
    frame.insert(1, "variant_index", variant_codes + 1)
    keep = [
        "figure_index", "variant_index", "variant_tag", "chrom", "pos", "ref",
        "alt", "phecode", "phecode_name", "gene", "p_meta", "beta_meta",
    ]
    frame = frame[keep].copy()
    n_loci = int(frame["variant_tag"].nunique())
    if expected_hits and len(frame) != expected_hits:
        raise ValueError(f"Expected {expected_hits} meta-significant hits, found {len(frame)}")
    if expected_loci and n_loci != expected_loci:
        raise ValueError(f"Expected {expected_loci} unique loci, found {n_loci}")
    return frame


def load_phewas_for_variants(
    path: Path,
    variant_tags: set[str],
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream the large aggregate table and retain only the 13 target loci."""
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"PheWAS table is missing or empty: {path}")
    header_row = _find_header_row(path, ("variant_tag", "phecode"))
    header = pd.read_csv(path, sep="\t", skiprows=header_row, nrows=0)
    columns = [str(column).strip() for column in header.columns]
    p_column = _first_column(
        columns, ("p_value", "p", "pval", "pvalue", "p_ind", "discovery_p", "p_d")
    )
    beta_column = _first_column(
        columns, ("beta", "beta_ind", "b", "discovery_beta", "beta_d", "log_odds")
    )
    or_column = _first_column(columns, ("odds_ratio", "or", "OR"))
    name_column = _first_column(columns, ("phecode_string", "phecode_name", "phenotype"))
    category_column = _first_column(columns, ("phecode_category", "category"))
    if p_column is None or (beta_column is None and or_column is None):
        raise ValueError(
            "PheWAS table needs p/p_value and either beta/log-odds or odds_ratio; "
            f"found columns {columns}"
        )
    selected = ["variant_tag", "phecode", p_column]
    selected += [column for column in (beta_column, or_column, name_column, category_column) if column]
    selected = list(dict.fromkeys(selected))

    retained: list[pd.DataFrame] = []
    rows_scanned = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        skiprows=header_row,
        usecols=selected,
        dtype={"variant_tag": str, "phecode": str},
        chunksize=chunk_rows,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        chunk["variant_tag"] = chunk["variant_tag"].astype(str).str.strip()
        keep = chunk["variant_tag"].isin(variant_tags)
        if keep.any():
            retained.append(chunk.loc[keep].copy())
    if not retained:
        raise ValueError("None of the meta-significant variant tags occur in the PheWAS table")
    frame = pd.concat(retained, ignore_index=True)
    frame["phecode"] = frame["phecode"].astype(str).str.strip()
    frame["p_value"] = pd.to_numeric(frame[p_column], errors="coerce")
    if beta_column:
        frame["beta"] = pd.to_numeric(frame[beta_column], errors="coerce")
    else:
        odds = pd.to_numeric(frame[or_column], errors="coerce")
        frame["beta"] = np.log(odds.where(odds > 0))
    if or_column:
        frame["odds_ratio"] = pd.to_numeric(frame[or_column], errors="coerce")
    else:
        frame["odds_ratio"] = np.exp(frame["beta"].clip(-700, 700))
    frame["phecode_name"] = (
        frame[name_column].fillna(frame["phecode"]).astype(str).str.strip()
        if name_column
        else frame["phecode"]
    )
    frame["phecode_category"] = (
        frame[category_column].fillna("Other").astype(str).str.strip()
        if category_column
        else frame["phecode"].str.split("_", n=1).str[0]
    )
    finite = (
        np.isfinite(frame["p_value"])
        & np.isfinite(frame["beta"])
        & np.isfinite(frame["odds_ratio"])
        & (frame["p_value"] > 0)
        & (frame["p_value"] <= 1)
        & (frame["odds_ratio"] > 0)
    )
    if not bool(finite.all()):
        raise ValueError(f"Target-locus PheWAS rows include {int((~finite).sum())} invalid tests")

    duplicates = frame.duplicated(["variant_tag", "phecode"], keep=False)
    if duplicates.any():
        duplicate_rows = frame.loc[duplicates, ["variant_tag", "phecode", "p_value", "beta"]]
        conflicts = duplicate_rows.groupby(["variant_tag", "phecode"]).agg(
            p_values=("p_value", "nunique"), betas=("beta", "nunique")
        )
        if bool(((conflicts["p_values"] > 1) | (conflicts["betas"] > 1)).any()):
            raise ValueError("Conflicting duplicate variant-phecode tests in PheWAS table")
        frame = frame.drop_duplicates(["variant_tag", "phecode"], keep="first")

    missing_variants = sorted(variant_tags - set(frame["variant_tag"]))
    if missing_variants:
        raise ValueError(f"PheWAS table is missing target variants: {missing_variants}")
    relevant = frame[
        [
            "variant_tag", "phecode", "phecode_name", "phecode_category",
            "p_value", "beta", "odds_ratio",
        ]
    ].sort_values(["variant_tag", "phecode"], kind="mergesort")
    subset_hash = _sha256_bytes(relevant.to_csv(index=False, float_format="%.17g").encode("utf-8"))
    return relevant.reset_index(drop=True), {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "rows_scanned": rows_scanned,
        "rows_retained": len(relevant),
        "variants_retained": int(relevant["variant_tag"].nunique()),
        "relevant_subset_sha256": subset_hash,
    }


def odds_ratio_folds(odds_ratio: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return reciprocal-symmetric OR magnitude (for example, 2 == 0.5)."""
    odds = np.asarray(odds_ratio, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 0):
        raise ValueError("Odds ratios must be positive and finite")
    return np.maximum(odds, 1.0 / odds)


def odds_ratio_marker_areas(
    odds_ratio: Sequence[float] | np.ndarray,
    *,
    cap: float = DEFAULT_OR_CAP,
    minimum: float = DEFAULT_MARKER_MIN,
    maximum: float = DEFAULT_MARKER_MAX,
) -> np.ndarray:
    """Symmetric marker areas on a log2 odds-ratio scale.

    OR=2 and OR=0.5 receive the same area.  Values beyond ``cap`` or ``1/cap``
    saturate, while OR values near 1 remain visible at ``minimum``.
    """
    if cap <= 1 or minimum <= 0 or maximum < minimum:
        raise ValueError("Require cap > 1 and 0 < minimum <= maximum")
    fold = odds_ratio_folds(odds_ratio)
    scaled = np.clip(np.log2(fold) / math.log2(cap), 0.0, 1.0)
    return minimum + (maximum - minimum) * scaled


def resolve_shared_phewas_scales(
    frame: pd.DataFrame,
    *,
    odds_ratio_cap: float | None = None,
    phewas_y_max: float | None = None,
) -> dict[str, float]:
    """Resolve one OR-size ceiling and one y-axis ceiling for every locus."""
    if frame.empty:
        raise ValueError("Cannot resolve shared PheWAS scales from an empty table")
    observed_or_fold_max = float(odds_ratio_folds(frame["odds_ratio"]).max())
    resolved_or_cap = observed_or_fold_max if odds_ratio_cap is None else float(odds_ratio_cap)
    # A completely null fixture still needs a nonzero log-scale denominator.
    resolved_or_cap = max(resolved_or_cap, math.nextafter(1.0, math.inf))
    if not math.isfinite(resolved_or_cap) or resolved_or_cap <= 1:
        raise ValueError("The shared OR-fold ceiling must be finite and greater than one")

    p_values = np.asarray(frame["p_value"], dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any(p_values <= 0) or np.any(p_values > 1):
        raise ValueError("PheWAS p-values must be finite and in (0, 1]")
    observed_neglogp_max = float((-np.log10(np.clip(p_values, 1e-300, 1.0))).max())
    if phewas_y_max is None:
        label_headroom = max(0.75, min(2.0, observed_neglogp_max * 0.15))
        resolved_y_max = math.ceil((observed_neglogp_max + label_headroom) * 2.0) / 2.0
    else:
        resolved_y_max = float(phewas_y_max)
    if not math.isfinite(resolved_y_max) or resolved_y_max <= observed_neglogp_max:
        raise ValueError(
            "The shared PheWAS y-axis maximum must exceed the observed maximum "
            f"-log10(p) ({observed_neglogp_max:.4g})"
        )
    return {
        "observed_or_fold_max": observed_or_fold_max,
        "odds_ratio_cap": resolved_or_cap,
        "observed_neglogp_max": observed_neglogp_max,
        "phewas_y_max": resolved_y_max,
    }


def _category_layout(frame: pd.DataFrame, category_order: Sequence[str], *,
                     category_widths: dict[str, float] | None = None) -> pd.DataFrame:
    frame = frame.copy()
    order_rank = {value: index for index, value in enumerate(category_order)}
    frame["_category_rank"] = frame["phecode_category"].map(order_rank).fillna(len(order_rank))
    frame["_or_fold"] = odds_ratio_folds(frame["odds_ratio"])
    is_meta = frame.get("_meta", pd.Series(False, index=frame.index)).astype(bool)
    # Meta hits lead each category by p-value. Background tests then use
    # ascending p-values, so their plotted -log10(p) heights descend from left
    # to right within each category.
    frame["_meta_rank"] = np.where(is_meta, 0, 1)
    frame["_within_primary"] = frame["p_value"]
    frame["_within_secondary"] = -frame["_or_fold"]
    frame = frame.sort_values(
        ["_category_rank", "_meta_rank", "_within_primary", "_within_secondary", "phecode"],
        kind="mergesort",
    ).reset_index(drop=True)
    positions = np.zeros(len(frame), dtype=float)
    centers: dict[str, float] = {}
    cursor = 0.0
    for category, indices in frame.groupby("phecode_category", sort=False).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        if category_widths is None:
            local = np.arange(len(index_array), dtype=float) + cursor
            center, next_cursor = float(local.mean()), float(local[-1] + 3.0)
        else:
            width = float(category_widths[str(category)])
            if not math.isfinite(width) or width <= 0:
                raise ValueError(f"Invalid PheWAS category width for {category}: {width}")
            center, next_cursor = cursor + width / 2, cursor + width
            local = (np.linspace(cursor + 0.04 * width, cursor + 0.96 * width, len(index_array))
                     if len(index_array) > 1 else np.asarray([center]))
        positions[index_array] = local
        centers[str(category)] = center
        cursor = next_cursor
    frame["_x"] = positions
    frame.attrs["category_centers"] = centers
    if category_widths is not None:
        frame.attrs["category_xlim"] = (0.0, cursor)
    return frame


def _short_category(value: str) -> str:
    replacements = {
        "Endocrine/Metab": "Endocrine/\nmetabolic",
        "Congenital anomalies": "Congenital\nanomalies",
        "Gastrointestinal": "Gastro-\nintestinal",
        "Genitourinary": "Genito-\nurinary",
        "Musculoskeletal": "Musculo-\nskeletal",
        "Muscloskeletal": "Musculo-\nskeletal",
        "Pregnancy complications": "Pregnancy\ncomplications",
        "Hematopoietic": "Hemato-\npoietic",
        "Dermatological": "Dermato-\nlogical",
        "Neurological": "Neuro",
        "Sense organs": "Sense\norgans",
        "Symptoms": "Signs/\nsymptoms",
    }
    return replacements.get(str(value), re.sub(r"\s+", "\n", str(value), count=1))


def draw_phewas_panel(
    ax: plt.Axes,
    locus_frame: pd.DataFrame,
    meta_hits: pd.DataFrame,
    target_hit: pd.Series,
    category_order: Sequence[str],
    parameters: PlotParameters,
    *,
    highlight_color: str = "#d95f02",
    highlight_label: str | None = "Meta-significant",
    nominal_label: str | None = "Nominal significance",
    threshold_inclusive: bool = True,
    font_scale: float = 1.0,
    category_widths: dict[str, float] | None = None,
    category_labels: dict[str, str] | None = None,
) -> None:
    """Draw one per-locus PheWAS panel with explicit OR size/direction semantics."""
    frame = locus_frame.copy()
    meta_phecodes = set(meta_hits["phecode"].astype(str))
    target_phecode = str(target_hit["phecode"])
    frame["_meta"] = frame["phecode"].isin(meta_phecodes)
    frame["_target"] = frame["phecode"] == target_phecode
    frame["_nominal"] = (
        frame["p_value"] <= parameters.phewas_threshold
        if threshold_inclusive
        else frame["p_value"] < parameters.phewas_threshold
    )
    frame = _category_layout(frame, category_order, category_widths=category_widths)
    frame["_neglogp"] = -np.log10(frame["p_value"].clip(lower=1e-300))
    frame["_area"] = odds_ratio_marker_areas(
        frame["odds_ratio"],
        cap=parameters.odds_ratio_cap,
        minimum=parameters.marker_area_min,
        maximum=parameters.marker_area_max,
    )
    categories = list(category_order)
    colors = {
        category: PHEWAS_CATEGORY_BACKGROUND_COLORS[index % 2]
        for index, category in enumerate(categories)
    }
    base_colors = frame["phecode_category"].map(colors).fillna("#bdbdbd").to_numpy()

    for positive, marker in ((True, "^"), (False, "v")):
        direction = (frame["beta"] >= 0) if positive else (frame["beta"] < 0)
        background = direction & ~frame["_meta"] & ~frame["_nominal"]
        if background.any():
            ax.scatter(
                frame.loc[background, "_x"], frame.loc[background, "_neglogp"],
                s=frame.loc[background, "_area"], marker=marker,
                c=base_colors[background.to_numpy()], alpha=0.52,
                edgecolors="none", zorder=2,
            )
        nominal = direction & frame["_nominal"] & ~frame["_meta"]
        if nominal.any():
            ax.scatter(
                frame.loc[nominal, "_x"], frame.loc[nominal, "_neglogp"],
                s=frame.loc[nominal, "_area"], marker=marker,
                c="#f2c14e", alpha=1.0, edgecolors="black", linewidths=1.0, zorder=4,
            )
        meta = direction & frame["_meta"]
        if meta.any():
            ax.scatter(
                frame.loc[meta, "_x"], frame.loc[meta, "_neglogp"],
                s=frame.loc[meta, "_area"], marker=marker,
                c=highlight_color, alpha=0.95, edgecolors="black", linewidths=1.0, zorder=6,
            )

    threshold_y = -math.log10(parameters.phewas_threshold)
    ax.axhline(threshold_y, color="#c0392b", lw=1.2, ls="--", alpha=0.85, zorder=1)
    centers = frame.attrs["category_centers"]
    tick_categories = [category for category in category_order if category in centers]
    ax.set_xticks([centers[category] for category in tick_categories])
    ax.set_xticklabels(
        [category_labels[category] if category_labels is not None else _short_category(category)
         for category in tick_categories],
        rotation=PHEWAS_CATEGORY_LABEL_ROTATION,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax.set_ylabel(style.P_VALUE_AXIS_LABEL, fontsize=10.5 * font_scale, parse_math=False)
    ax.tick_params(axis="x", labelsize=7.5 * font_scale, pad=1.5)
    ax.tick_params(axis="y", labelsize=7.5 * font_scale)
    ax.grid(axis="y", alpha=0.18, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    y_max = parameters.phewas_y_max
    if y_max is None:
        y_max = resolve_shared_phewas_scales(
            frame, odds_ratio_cap=parameters.odds_ratio_cap,
        )["phewas_y_max"]
    ax.set_ylim(-0.15, y_max)
    if category_widths is not None:
        ax.set_xlim(frame.attrs["category_xlim"])

    hit_rows = frame[frame["_meta"] | frame["_nominal"]].sort_values(
        ["p_value", "phecode"], kind="mergesort",
    )
    label_texts = []
    target_x: list[float] = []
    target_y: list[float] = []
    x_step = (float(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025 if category_widths is not None
              else max(1.25, float(frame["_x"].max() - frame["_x"].min()) * 0.025))
    y_step = max(0.25, float(y_max) * 0.045)
    for label_index, (_, row) in enumerate(hit_rows.iterrows()):
        horizontal_direction = -1.0 if label_index % 2 == 0 else 1.0
        initial_x = float(row["_x"]) + horizontal_direction * x_step * (
            1.0 + 0.25 * (label_index // 2)
        )
        initial_y = float(row["_neglogp"]) + y_step * (1.0 + 1.25 * (label_index % 2))
        label_texts.append(ax.text(
            initial_x, initial_y,
            str(row["phecode_name"]),
            ha="center", va="bottom", fontsize=8.2 * font_scale,
            fontweight="bold",
            color="black",
            zorder=8,
        ))
        target_x.append(float(row["_x"]))
        target_y.append(float(row["_neglogp"]))
    if label_texts:
        adjust_text(
            label_texts,
            x=hit_rows["_x"].to_numpy(dtype=float),
            y=hit_rows["_neglogp"].to_numpy(dtype=float),
            target_x=target_x,
            target_y=target_y,
            ax=ax,
            expand=(1.08, 1.12),
            force_text=(1.15, 0.32),
            force_static=(0.28, 0.08),
            force_pull=(0.015, 0.008),
            force_explode=(0.80, 0.25),
            max_move=(40, 16),
            ensure_inside_axes=True,
            # Experimental crossing swaps can move unequal-width labels back
            # outside the axes after bounding them in the compact slot layout.
            prevent_crossings=category_widths is None,
            min_arrow_len=2,
            iter_lim=250,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.55},
        )

    shape_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#6baed6",
               markeredgecolor="black", markersize=7, label="OR > 1"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#6baed6",
               markeredgecolor="black", markersize=7, label="OR < 1"),
    ]
    if highlight_label is not None:
        shape_handles.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=highlight_color,
            markeredgecolor="black", markersize=6, label=highlight_label,
        ))
    if nominal_label is not None:
        shape_handles.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#f2c14e",
            markeredgecolor="black", markersize=6, label=nominal_label,
        ))
    legend_shapes = ax.legend(
        handles=shape_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
        ncol=2, fontsize=7.2 * font_scale, frameon=False, columnspacing=1.0,
        handletextpad=0.35, borderaxespad=0.0,
    )
    ax.add_artist(legend_shapes)
    size_values = (
        np.asarray(DEFAULT_OR_LEGEND_FOLDS, dtype=float)
        if np.isclose(parameters.odds_ratio_cap, DEFAULT_OR_CAP)
        else np.geomspace(1.0, parameters.odds_ratio_cap, 4)
    )
    size_handles = [
        Line2D(
            [0], [0], marker="^", color="none", markerfacecolor="#777777",
            markeredgecolor="black", markeredgewidth=0.5,
            markersize=math.sqrt(float(odds_ratio_marker_areas(
                [value], cap=parameters.odds_ratio_cap,
                minimum=parameters.marker_area_min, maximum=parameters.marker_area_max,
            )[0])),
            label=f"{value:.2g}×",
        )
        for value in size_values
    ]
    ax.legend(
        handles=size_handles, title="OR fold (log scale; cap 4×)", loc="lower right",
        bbox_to_anchor=(1.0, 1.005), borderaxespad=0.0,
        ncol=4, fontsize=7.0 * font_scale, title_fontsize=7.2 * font_scale, frameon=False,
        columnspacing=0.8, handletextpad=0.25,
    )

def _validate_png(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"PNG missing or empty: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width < 200 or height < 150:
        raise ValueError(f"PNG is unexpectedly small ({width}x{height}): {path}")
    return {"path": str(path), "size": path.stat().st_size, "width": width, "height": height,
            "sha256": sha256_file(path)}


def _validate_pdf(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1000:
        raise FileNotFoundError(f"PDF missing, empty, or truncated: {path}")
    if path.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"Invalid PDF header: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def _crop_white_array(array: np.ndarray) -> np.ndarray:
    """Crop white margins from an RGB array without changing its pixel aspect."""
    content = np.any(array < 248, axis=2)
    if not content.any():
        return array
    ys, xs = np.where(content)
    margin = 8
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, array.shape[1])
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, array.shape[0])
    return array[y0:y1, x0:x1]


def _crop_white(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        array = np.asarray(image)
    return _crop_white_array(array)


def _strip_embedded_top_title(array: np.ndarray) -> np.ndarray:
    """Remove a rasterized title block separated from a plot by white space."""
    if array.ndim != 3 or array.shape[0] < 20:
        return array
    height, width = array.shape[:2]
    # The fine-map title is grayscale, while the locus points and full-height
    # viridis colorbar are chromatic. Prefer that stable boundary over exact
    # blank rows, which can be interrupted by antialiasing or a colorbar edge.
    color_spread = np.ptp(array, axis=2)
    chromatic_rows = (
        (color_spread >= 12) & np.any(array < 245, axis=2)
    ).sum(axis=1) >= max(3, int(width * 0.002))
    run_length = max(6, int(height * 0.012))
    if len(chromatic_rows) >= run_length:
        sustained = np.convolve(
            chromatic_rows.astype(np.int16),
            np.ones(run_length, dtype=np.int16),
            mode="valid",
        ) >= math.ceil(run_length * 0.60)
        starts = np.flatnonzero(sustained)
        if len(starts):
            sustained_start = int(starts[0])
            first_chromatic = np.flatnonzero(
                chromatic_rows[sustained_start:sustained_start + run_length]
            )
            plot_start = sustained_start + int(first_chromatic[0])
            if plot_start <= int(height * 0.03):
                return array
            if plot_start <= int(height * 0.55):
                # The top y tick and colorbar maximum extend above the first
                # chromatic plot row. Preserve the lower part of that side-label
                # band, while clearing the full-width upper title line and the
                # centered lower title line. Long locus titles can otherwise
                # leave clipped fragments at either edge of the composite.
                headroom = max(
                    FINEMAP_SIDE_LABEL_HEIGHT_PX + 8,
                    int(height * FINEMAP_TITLE_HEADROOM_FRACTION),
                )
                crop_start = max(0, plot_start - headroom)
                cropped = array[crop_start:, :, :].copy()
                title_band_height = plot_start - crop_start
                # Long Figure 4 titles can extend all the way to both image
                # edges. Clear the upper part across the full width, then keep
                # only the lower outer side-label bands containing the top
                # y-axis tick and colorbar maximum.
                title_left = int(width * 0.07)
                title_right = int(width * 0.86)
                # Keep a complete font-height outer band for the top y tick and
                # colorbar maximum.  A proportional 32% band retained only the
                # lower half of ``1.0`` in production fine-map rasters.
                side_label_height = min(
                    title_band_height - 1,
                    FINEMAP_SIDE_LABEL_HEIGHT_PX,
                )
                full_width_stop = max(1, title_band_height - side_label_height)
                cropped[:full_width_stop, :, :] = 255
                cropped[
                    full_width_stop:title_band_height,
                    title_left:title_right,
                    :,
                ] = 255
                top_padding = max(8, int(height * FINEMAP_TOP_PADDING_FRACTION))
                return np.pad(
                    cropped,
                    ((top_padding, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=255,
                )

    row_ink = np.any(array < 248, axis=2).sum(axis=1)
    search_end = max(1, int(height * 0.40))
    # A colorbar edge or antialiased speck can cross an otherwise blank title
    # gap, so treat rows with only a few inked pixels as blank.
    blank_ink_limit = max(2, int(width * 0.0025))
    blank = row_ink[:search_end] <= blank_ink_limit
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_blank in enumerate(blank):
        if is_blank and run_start is None:
            run_start = index
        elif not is_blank and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(blank)))

    minimum_gap = max(3, int(height * 0.004))
    candidates = [
        (start, end)
        for start, end in runs
        if end - start >= minimum_gap
        and (row_ink[:start] > blank_ink_limit).any()
        and (row_ink[end:] > blank_ink_limit).any()
    ]
    if not candidates:
        return array
    _, crop_start = max(candidates, key=lambda run: (run[1] - run[0], run[1]))
    return array[max(0, crop_start - 2):, :, :]


def _show_image_panel(
    ax: plt.Axes,
    path: Path,
    panel: str,
    label: str | None,
    *,
    strip_embedded_title: bool = False,
    anchor: str = "C",
    fill_cell: bool = False,
    preserve_canvas: bool = False,
) -> None:
    image = (
        np.asarray(Image.open(path).convert("RGB"))
        if preserve_canvas
        else _crop_white(path)
    )
    if strip_embedded_title:
        image = _strip_embedded_top_title(image)
        # The initial crop necessarily included the long title. Once that title
        # is gone, crop again so its former width cannot shrink the real plot
        # inside a same-size B/C grid cell.
        image = _crop_white_array(image)
        image = np.pad(
            image,
            ((FINEMAP_POSTCROP_TOP_PADDING_PX, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )
    ax.imshow(image, aspect="auto" if fill_cell else "equal")
    ax.set_anchor(anchor)
    ax.axis("off")
    if panel:
        ax.text(-0.012, 1.02, panel, transform=ax.transAxes, fontsize=14,
                fontweight="bold", ha="right", va="bottom")
    if label:
        ax.text(0.012, 1.02, label, transform=ax.transAxes, fontsize=9.5,
                ha="left", va="bottom")


def _show_grouped_selection_panel(
    fig: plt.Figure,
    grid_spec: Any,
    selection_paths: dict[str, Path],
    *,
    include_title: bool,
    panel_label: str = "B",
    wide: bool = False,
) -> tuple[plt.Axes, plt.Axes]:
    """Embed the Figure 3B-style EHH/iHS/pi/gene source as one panel."""
    group_axis = fig.add_subplot(grid_spec)
    group_axis.axis("off")
    group_axis.patch.set_alpha(0.0)
    group_axis.set_zorder(3)
    group_axis.text(
        -0.012,
        1.02,
        panel_label,
        transform=group_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    if include_title:
        group_axis.text(
            0.012,
            1.02,
            "EHH, iHS, nucleotide diversity, and genes (±500 kb)",
            transform=group_axis.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
        )

    selection_axis = fig.add_subplot(grid_spec)
    source_key = "combined_wide_png" if wide else "combined_png"
    source_path = selection_paths.get(source_key)
    if source_path is None or not source_path.is_file():
        source_path = selection_paths["combined_png"]
    _show_image_panel(
        selection_axis,
        source_path,
        "",
        None,
        anchor="N",
        fill_cell=False,
        preserve_canvas=True,
    )
    return group_axis, selection_axis


def _safe_name(value: str, maximum: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    if len(value) <= maximum:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[:maximum - 13]}_{digest}"


def _phecode_string_slug(
    value: str,
    *,
    fallback: str | None = None,
    maximum: int = 80,
) -> str:
    """Return the short phenotype label used in reader-facing filenames."""
    # Slash-delimited phrases can identify distinct phecodes (for example,
    # ``Irregular menstrual cycle`` and ``Irregular menstrual cycle/bleeding``).
    # Keep the complete label so those hit-level outputs cannot resolve to the
    # same path and silently overwrite one another.
    label = str(value)
    label = re.sub(r"\[[^\]]*\]", "", label)
    label = label.replace("[", "").replace("]", "")
    label = " ".join(label.split()).strip().lower()
    if not label and fallback is not None:
        label = str(fallback).strip().lower()
    slug = _safe_name(label, maximum=maximum).lower()
    if not slug:
        raise ValueError("Phecode string cannot be converted to a filename label")
    return slug


def _phewas_output_stem_name(
    hit: pd.Series,
    *,
    output_prefix: str = "Supplementary_Phewas",
) -> str:
    index_field = "variant_index" if "variant_index" in hit.index else "sv_index"
    if index_field not in hit.index:
        raise ValueError("Hit lacks variant_index/sv_index for grouped output naming")
    phenotype = _phecode_string_slug(
        hit["phecode_name"], fallback=str(hit["phecode"])
    )
    return (
        f"{output_prefix}_{int(hit[index_field]):02d}_"
        f"{_safe_name(hit['variant_tag'])}__{phenotype}"
    )


def _composite_output_paths(output_dir: Path, stem_name: str) -> dict[str, Path]:
    """Separate title-free PNG/PDF files and keep titled copies in title/."""
    root_stem = output_dir / stem_name
    title_stem = output_dir / "title" / stem_name
    return {
        "png": Path(f"{title_stem}.png"),
        "pdf": Path(f"{title_stem}.pdf"),
        "no_title_png": output_dir / "png" / f"{stem_name}.png",
        "no_title_pdf": output_dir / "pdf" / f"{stem_name}.pdf",
        "source_zip": output_dir / "source_zipped" / f"{stem_name}.zip",
        "manifest": Path(f"{root_stem}.json"),
    }


@plt.rc_context(EDITABLE_PDF_RC)
def _save_vector_composite_pdf(
    fig: plt.Figure,
    path: Path,
    overlays: Sequence[tuple[plt.Axes, Path]],
    *,
    metadata: dict[str, str],
) -> None:
    """Save a page and place native panel PDFs into their Matplotlib cells.

    Matplotlib cannot import a PDF as an artist without rasterizing it.  The
    preview PNG therefore continues to use the matching PNG panels, while the
    publication PDF is assembled from the native vector panel PDFs.
    """
    missing = [str(source) for _, source in overlays if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Native vector panel PDF(s) required for composite output: "
            + ", ".join(missing)
        )
    for _, source in overlays:
        _validate_vector_pdf(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_path = path.with_suffix(".vector-base.partial.pdf")
    output_path = path.with_suffix(".partial.pdf")
    image_visibility: list[tuple[Any, bool]] = []
    try:
        for axis, _ in overlays:
            for artist in axis.images:
                image_visibility.append((artist, artist.get_visible()))
                artist.set_visible(False)
        fig.savefig(
            base_path,
            facecolor="white",
            bbox_inches=fig.bbox_inches,
            pad_inches=0,
            metadata=metadata,
        )
        writer = PdfWriter(clone_from=str(base_path))
        if len(writer.pages) != 1:
            raise ValueError(f"Composite base PDF must have one page: {base_path}")
        page = writer.pages[0]
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        for axis, source in overlays:
            source_writer = PdfWriter(clone_from=str(source))
            if len(source_writer.pages) != 1:
                raise ValueError(f"Panel PDF must have one page: {source}")
            source_page = source_writer.pages[0]
            source_width = float(source_page.mediabox.width)
            source_height = float(source_page.mediabox.height)
            if source_width <= 0 or source_height <= 0:
                raise ValueError(f"Panel PDF has an invalid media box: {source}")
            box = axis.get_position()
            cell_x = box.x0 * page_width
            cell_y = box.y0 * page_height
            cell_width = box.width * page_width
            cell_height = box.height * page_height
            scale = min(cell_width / source_width, cell_height / source_height)
            rendered_width = source_width * scale
            rendered_height = source_height * scale
            translate_x = cell_x + (cell_width - rendered_width) / 2.0
            translate_y = cell_y + cell_height - rendered_height
            # Transform the panel and its clipping rectangle together. Moving
            # only its content leaves the original crop box behind, cutting
            # off EHH curves/legends when the panel sits high on the page.
            page.merge_transformed_page(
                source_page,
                Transformation().scale(scale).translate(translate_x, translate_y),
                over=True,
            )
        writer.add_metadata({f"/{key}": str(value) for key, value in metadata.items()})
        with output_path.open("wb") as stream:
            writer.write(stream)
        output_path.replace(path)
    finally:
        for artist, visible in image_visibility:
            artist.set_visible(visible)
        base_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _copy_native_source_panel(
    panel_paths: dict[str, Path],
    *,
    png_source: Path,
    pdf_source: Path,
) -> dict[str, Any]:
    """Copy a panel's preview PNG and native vector PDF into source_panels."""
    _validate_png(png_source)
    _validate_vector_pdf(pdf_source)
    for destination in panel_paths.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_source, panel_paths["png"])
    shutil.copy2(pdf_source, panel_paths["pdf"])
    return {
        "png": _validate_png(panel_paths["png"]),
        "pdf": _validate_vector_pdf(panel_paths["pdf"]),
    }


def _validate_vector_pdf(path: Path, *, allow_images: bool = False) -> dict[str, Any]:
    """Validate a readable PDF and reject raster image objects by default."""
    validation = _validate_pdf(path)
    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError(f"PDF contains no pages: {path}")
    image_objects = sum(len(page.images) for page in reader.pages)
    if image_objects and not allow_images:
        raise ValueError(
            f"Vector PDF unexpectedly contains {image_objects} raster image object(s): {path}"
        )
    return {
        **validation,
        "pages": len(reader.pages),
        "raster_image_objects": image_objects,
        "vector_only": image_objects == 0,
    }


def _validate_editable_panel_pdf(path: Path, *, allow_images: bool = False) -> dict[str, Any]:
    """Require real, embedded TrueType text in a native vector source panel."""
    validation = _validate_vector_pdf(path, allow_images=allow_images)
    reader = PdfReader(path)
    fonts: list[str] = []
    visited: set[int] = set()
    visited_streams: dict[int, Any] = {}
    has_text = False

    def inspect_text(stream: Any) -> None:
        nonlocal has_text
        if stream is None or id(stream) in visited_streams:
            return
        visited_streams[id(stream)] = stream
        content = stream if isinstance(stream, ContentStream) else ContentStream(stream, reader)
        for operands, operator in content.operations:
            if operator not in {b"Tj", b"TJ", b"'", b'"'} or not operands:
                continue
            values = operands[-1] if operator == b"TJ" else operands[-1:]
            if any(isinstance(value, (str, bytes)) and value.strip() for value in values):
                has_text = True
                return

    def inspect_resources(resources: Any) -> None:
        resources = resources.get_object()
        if id(resources) in visited:
            return
        visited.add(id(resources))
        font_resources = resources["/Font"].get_object() if "/Font" in resources else {}
        for ref in font_resources.values():
            font = ref.get_object()
            if font.get("/Subtype") != "/Type0" or not font.get("/ToUnicode"):
                raise ValueError(f"PDF text is not embedded Unicode TrueType: {path}")
            descendants = font.get("/DescendantFonts", [])
            if not descendants:
                raise ValueError(f"PDF text lacks an embedded TrueType font: {path}")
            for descendant_ref in descendants:
                descendant = descendant_ref.get_object()
                descriptor = descendant.get("/FontDescriptor")
                if (
                    descendant.get("/Subtype") != "/CIDFontType2"
                    or descriptor is None
                    or not descriptor.get_object().get("/FontFile2")
                ):
                    raise ValueError(f"PDF text lacks an embedded TrueType font: {path}")
            fonts.append(str(font.get("/BaseFont", "")))
        if "/XObject" in resources:
            for ref in resources["/XObject"].get_object().values():
                obj = ref.get_object()
                if obj.get("/Subtype") == "/Form":
                    inspect_text(obj)
                    if "/Resources" in obj:
                        inspect_resources(obj["/Resources"])

    # A dense selection plot can paint the same marker Form thousands of
    # times. Validate text operators and fonts once per stream/resource tree
    # instead of expanding every marker through extract_text's safety limit.
    for page in reader.pages:
        inspect_text(page.get_contents())
        inspect_resources(page["/Resources"])
    if not fonts or not has_text:
        raise ValueError(f"PDF panel contains no editable text: {path}")
    return {**validation, "editable_text": True, "embedded_truetype_fonts": sorted(set(fonts))}


def _write_source_zip(
    zip_path: Path,
    panel_paths: dict[str, dict[str, Path]],
    *,
    figure_index: int,
    descriptors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write one deterministic PDF-only archive of source panels for one page."""
    panel_descriptors = {
        "A": "phewas",
        "B": "ehh_ihs_pi_genes",
        "C": "phecode_fine_map",
        "D": "igv_validation",
    }
    panel_descriptors.update(descriptors or {})
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_suffix(".partial.zip")
    members: list[str] = []
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for panel, formats in panel_paths.items():
            source = formats["pdf"]
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Source panel is missing or empty: {source}")
            member = (
                f"fig_{figure_index:02d}_{panel}_"
                f"{panel_descriptors.get(panel, 'panel')}.pdf"
            )
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
            members.append(member)
    partial.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or sorted(archive.namelist()) != sorted(members):
            raise ValueError(f"Source archive failed integrity validation: {zip_path}")
    return {
        "path": str(zip_path),
        "size": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "members": members,
    }


def _source_panel_output_paths(
    output_dir: Path,
    stem_name: str,
    figure_index: int,
    panels: Sequence[str],
) -> dict[str, dict[str, Path]]:
    """Return collision-free PNG/PDF paths for the rendered page sources."""
    panel_dir = output_dir / "source_panels" / stem_name
    return {
        panel: {
            "png": panel_dir / f"fig_{figure_index:02d}_{panel}.png",
            "pdf": panel_dir / f"fig_{figure_index:02d}_{panel}.pdf",
        }
        for panel in panels
    }


@plt.rc_context(EDITABLE_PDF_RC)
def _save_source_panels(
    fig: plt.Figure,
    panel_axes: dict[str, Sequence[plt.Axes]],
    panel_paths: dict[str, dict[str, Path]],
    *,
    dpi: int,
    metadata: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Crop each panel from the rendered page without rescaling its geometry."""
    all_axes = tuple(fig.axes)
    outputs: dict[str, dict[str, Any]] = {}
    for panel, axes in panel_axes.items():
        included_axes = set(axes)
        prior_visibility = {axis: axis.get_visible() for axis in all_axes}
        try:
            for axis in all_axes:
                axis.set_visible(axis in included_axes)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            figure_box = fig.bbox
            display_boxes = [axis.get_tightbbox(renderer) for axis in axes]
            display_box = Bbox.union(display_boxes)
            display_box = Bbox.from_extents(
                max(figure_box.x0, display_box.x0 - 2.0),
                max(figure_box.y0, display_box.y0 - 2.0),
                min(figure_box.x1, display_box.x1 + 2.0),
                min(figure_box.y1, display_box.y1 + 2.0),
            )
            inches_box = display_box.transformed(fig.dpi_scale_trans.inverted())
            paths = panel_paths[panel]
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                paths["png"],
                dpi=dpi,
                facecolor="white",
                bbox_inches=inches_box,
                pad_inches=0,
            )
            fig.savefig(
                paths["pdf"],
                dpi=dpi,
                facecolor="white",
                bbox_inches=inches_box,
                pad_inches=0,
                metadata=metadata,
            )
        finally:
            for axis, visible in prior_visibility.items():
                axis.set_visible(visible)
        outputs[panel] = {
            "png": _validate_png(paths["png"]),
            # IGV is an intentional screenshot; its panel letter still uses
            # editable text. PheWAS panels must remain entirely vector.
            "pdf": _validate_editable_panel_pdf(
                paths["pdf"], allow_images=any(axis.images for axis in axes),
            ),
        }
    return outputs


def _validate_source_panel_outputs(
    panel_paths: dict[str, dict[str, Path]],
) -> dict[str, dict[str, Any]]:
    return {
        panel: {
            "png": _validate_png(paths["png"]),
            "pdf": _validate_pdf(paths["pdf"]),
        }
        for panel, paths in panel_paths.items()
    }


def expected_finemap_filename(hit: pd.Series) -> str:
    return f"pval_disc_{hit['variant_tag']}__{hit['phecode']}.png"


def expected_finemap_result_filename(hit: pd.Series) -> str:
    """Return the verified 10k-discovery result used to redraw panel C."""
    from scripts import figure4_finemap

    return (
        f"susie_{hit['variant_tag']}__{hit['phecode']}__"
        f"{figure4_finemap.FINEMAP_COHORT}.tsv"
    )


def _standard_finemap_variant_log_record(row: pd.Series) -> dict[str, Any]:
    """Return the PIP and association evidence for one fine-map variant."""
    pip_value = pd.to_numeric(pd.Series([row.get("pip")]), errors="coerce").iloc[0]
    p_value = pd.to_numeric(
        pd.Series([row.get("p_score")]), errors="coerce"
    ).iloc[0]
    position = pd.to_numeric(pd.Series([row.get("pos")]), errors="coerce").iloc[0]
    is_sv = bool(row.get("is_sv", False))
    return {
        "variant_id": str(row.get("variant_id", "")),
        "position": int(position) if pd.notna(position) else None,
        "pip": float(pip_value) if pd.notna(pip_value) else None,
        "p_value": float(p_value) if pd.notna(p_value) else None,
        "is_sv": is_sv,
        "sv_type": str(row.get("sv_type", "")) if is_sv else "",
    }


def _standard_finemap_causal_set_names(value: Any) -> tuple[str, ...]:
    names: list[str] = []
    for name in str(value).split(","):
        normalized = name.strip()
        if (
            normalized
            and normalized.casefold() != "nan"
            and normalized not in names
        ):
            names.append(normalized)
    return tuple(names)


def summarize_standard_finemap_result(
    frame: pd.DataFrame,
    hit: pd.Series,
) -> dict[str, Any]:
    """Summarize causal sets and top-PIP/minimum-p evidence for one hit."""
    required = {"variant_id", "pos", "p_score", "pip", "cs"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fine-map evidence log is missing columns: {missing}")
    ranked = frame.sort_values(
        ["pip", "p_score", "pos", "variant_id"],
        ascending=[False, True, True, True],
        na_position="last",
        kind="stable",
    )
    minimum_p = frame.sort_values(
        ["p_score", "pip", "pos", "variant_id"],
        ascending=[True, False, True, True],
        na_position="last",
        kind="stable",
    )
    set_names = sorted(
        {
            name
            for value in frame["cs"]
            for name in _standard_finemap_causal_set_names(value)
        }
    )
    causal_sets: list[dict[str, Any]] = []
    for set_name in set_names:
        members = frame.loc[
            frame["cs"].map(
                lambda value, name=set_name: (
                    name in _standard_finemap_causal_set_names(value)
                )
            )
        ].sort_values(
            ["pip", "p_score", "pos", "variant_id"],
            ascending=[False, True, True, True],
            na_position="last",
            kind="stable",
        )
        causal_sets.append(
            {
                "causal_set": set_name,
                "members": [
                    _standard_finemap_variant_log_record(row)
                    for _, row in members.iterrows()
                ],
            }
        )
    association_p = pd.to_numeric(
        pd.Series([hit.get("p_meta", np.nan)]), errors="coerce"
    ).iloc[0]
    association_beta = pd.to_numeric(
        pd.Series([hit.get("beta_meta", np.nan)]), errors="coerce"
    ).iloc[0]
    return {
        "variant_tag": str(hit["variant_tag"]),
        "phecode": str(hit["phecode"]),
        "phecode_name": str(hit["phecode_name"]),
        "phewas_association": {
            "p_value": float(association_p) if pd.notna(association_p) else None,
            "beta": (
                float(association_beta)
                if pd.notna(association_beta)
                else None
            ),
            "odds_ratio": (
                float(np.exp(association_beta))
                if pd.notna(association_beta)
                else None
            ),
        },
        "variant_count": len(frame),
        "causal_sets": causal_sets,
        "top_pip_variant": _standard_finemap_variant_log_record(ranked.iloc[0]),
        "minimum_p_variant": _standard_finemap_variant_log_record(
            minimum_p.iloc[0]
        ),
    }


def log_standard_finemap_result(summary: dict[str, Any]) -> None:
    """Write complete per-PheWAS SNV fine-map evidence to the workflow log."""
    logger = logging.getLogger("supp_locus_composites")
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
            "phewas_association": summary["phewas_association"],
        },
    )
    emit("top_pip_variant", summary["top_pip_variant"])
    emit("minimum_p_variant", summary["minimum_p_variant"])
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


def _selection_paths(selection_dir: Path, variant_tag: str) -> dict[str, Path]:
    safe = _safe_name(variant_tag)
    return {
        "combined_png": selection_dir / f"selection_combined_{safe}.png",
        "combined_pdf": selection_dir / f"selection_combined_{safe}.pdf",
        "combined_wide_png": selection_dir / f"selection_combined_wide_{safe}.png",
        "combined_wide_pdf": selection_dir / f"selection_combined_wide_{safe}.pdf",
        "manifest": selection_dir / f"selection_{safe}.json",
    }


def _sample_manifest() -> tuple[set[str], str]:
    ids = samples.load_aou_sample_list(download=True)
    if ids is None:
        raise FileNotFoundError(
            f"Could not load canonical sample manifest {samples.AOU_SAMPLE_LIST_GS}"
        )
    if len(ids) != samples.N_AOU:
        raise ValueError(f"Expected {samples.N_AOU} unrelated AoU IDs, found {len(ids)}")
    if set(samples.load_excluded_sample_ids()) & ids:
        raise ValueError("Canonical sample manifest still contains excluded related IDs")
    digest = _sha256_bytes(("\n".join(sorted(ids)) + "\n").encode("utf-8"))
    return ids, digest


def _file_stat(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def stage_selection_inputs(
    loci: pd.DataFrame,
    bucket: str,
    destination: Path,
    *,
    threads: int = DEFAULT_THREADS,
    force: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Stage only the Figure 3 chromosomes needed by a locus series."""
    if threads < 1:
        raise ValueError("threads must be positive")
    if "chrom" not in loci.columns:
        raise ValueError("Selection loci are missing the chrom column")
    chromosome_set = set(loci["chrom"].astype(str).str.strip())
    invalid = sorted(
        chrom for chrom in chromosome_set
        if re.fullmatch(r"chr(?:[1-9]|1[0-9]|2[0-2])", chrom) is None
    )
    if invalid:
        raise ValueError(f"Invalid selection chromosomes: {invalid}")
    chromosomes = sorted(
        chromosome_set,
        key=lambda value: int(value.removeprefix("chr")),
    )
    if not chromosomes:
        raise ValueError("Selection loci contain no chromosomes")

    bucket = _normalize_bucket(bucket)
    destination = destination.expanduser().resolve()
    vcf_dir = destination / "SV_biallelic"
    ihs_dir = destination / "ihs_output_v4"
    hardmask_path = destination / DEFAULT_HARDMASK_OBJECT
    hardmask_stage = _stage_gcs_file(
        _gs_join(bucket, DEFAULT_HARDMASK_OBJECT),
        hardmask_path,
        force=force,
    )

    def stage_chromosome(chrom: str) -> tuple[str, dict[str, Any]]:
        vcf_object = DEFAULT_SELECTION_VCF_OBJECT.format(chrom=chrom)
        ihs_object = DEFAULT_SELECTION_IHS_OBJECT.format(chrom=chrom)
        vcf_path = vcf_dir / f"{chrom}.biallelic.vcf.gz"
        ihs_path = ihs_dir / f"{chrom}.ihs.out.100bins.norm.ann"
        vcf_stage = _stage_gcs_file(
            _gs_join(bucket, vcf_object), vcf_path, force=force,
        )
        index_failures: list[str] = []
        index_path: Path | None = None
        index_stage: dict[str, Any] | None = None
        for suffix in (".tbi", ".csi"):
            candidate = Path(f"{vcf_path}{suffix}")
            try:
                index_stage = _stage_gcs_file(
                    _gs_join(bucket, f"{vcf_object}{suffix}"),
                    candidate,
                    force=force,
                )
                index_path = candidate
                break
            except OSError as exc:
                index_failures.append(str(exc))
                candidate.unlink(missing_ok=True)
        if index_path is None or index_stage is None:
            raise FileNotFoundError(
                f"No tabix/CSI index could be staged for {vcf_object}:\n"
                + "\n".join(index_failures)
            )
        ihs_stage = _stage_gcs_file(
            _gs_join(bucket, ihs_object), ihs_path, force=force,
        )
        return chrom, {
            "vcf": vcf_stage,
            "vcf_index": index_stage,
            "vcf_index_path": str(index_path),
            "ihs": ihs_stage,
        }

    chrom_stages: dict[str, Any] = {}
    worker_count = min(max(1, threads), len(chromosomes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(stage_chromosome, chrom) for chrom in chromosomes]
        for future in concurrent.futures.as_completed(futures):
            chrom, stage = future.result()
            chrom_stages[chrom] = stage

    paths = {
        "vcf_pattern": str(vcf_dir / "{chrom}.biallelic.vcf.gz"),
        "ihs_pattern": str(ihs_dir / "{chrom}.ihs.out.100bins.norm.ann"),
        "hardmask_path": str(hardmask_path),
    }
    return paths, {
        "mode": "workspace-bucket root stage",
        "bucket": bucket,
        "destination": str(destination),
        "chromosomes": chromosomes,
        "hardmask": hardmask_stage,
        "by_chromosome": {
            chrom: chrom_stages[chrom] for chrom in chromosomes
        },
        "paths": paths,
    }


@plt.rc_context(EDITABLE_PDF_RC)
def _selection_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-safe Figure 3B-style selection renderer for one unique locus."""
    from scripts import figure3_ehh, figure3_locus_combined

    figure3_ehh.CACHE_DIR = Path(payload["cache_dir"])
    figure3_ehh.GTF_PATH = payload["gtf_path"]
    figure3_ehh.HARDMASK_PATH = payload["hardmask_path"]
    figure3_ehh.IHS_PATTERN = payload["ihs_pattern"]
    figure3_ehh.EHH_HALF_WINDOW = int(payload["selection_half_window_bp"])
    figure3_ehh.IHSPI_HALF_WINDOW = int(payload["selection_half_window_bp"])
    locus = payload["locus"]
    paths = {key: Path(value) for key, value in payload["paths"].items() if key != "manifest"}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    chrom = str(locus["chrom"])
    core_pos = int(locus["core_pos"])
    vcf_path = payload["vcf_path"]

    with plt.rc_context(figure3_ehh._RC):
        selection_savefig = {
            key: value
            for key, value in figure3_ehh._SAVEFIG_KW.items()
            if key not in {"bbox_inches", "pad_inches"}
        }
        fig = figure3_locus_combined.plot_combined_locus(
            locus["gene"], chrom, core_pos, locus["ref"], locus["alt"],
            float(locus["ihs_val"]), locus["ihs_rs"], vcf_path,
            payload["gtf_path"],
            sv_pos=locus.get("sv_pos"),
            sv_type=locus.get("sv_type"),
            figsize=tuple(payload.get("compact_figsize", SELECTION_COMPACT_FIGSIZE)),
        )
        fig.savefig(
            paths["combined_png"], bbox_inches=fig.bbox_inches,
            pad_inches=0, **selection_savefig,
        )
        fig.savefig(
            paths["combined_pdf"], bbox_inches=fig.bbox_inches,
            pad_inches=0, **selection_savefig,
        )
        fig.set_size_inches(
            *tuple(payload.get("wide_figsize", SELECTION_WIDE_FIGSIZE)),
            forward=True,
        )
        fig.savefig(
            paths["combined_wide_png"], bbox_inches=fig.bbox_inches,
            pad_inches=0, **selection_savefig,
        )
        fig.savefig(
            paths["combined_wide_pdf"], bbox_inches=fig.bbox_inches,
            pad_inches=0, **selection_savefig,
        )
        plt.close(fig)
    return {"variant_tag": payload["variant_tag"], "paths": {k: str(v) for k, v in paths.items()}}


def render_selection_loci(
    loci: pd.DataFrame,
    hits: pd.DataFrame,
    selection_dir: Path,
    cache_root: Path,
    *,
    source_identity: dict[str, Any],
    threads: int = DEFAULT_THREADS,
    selection_half_window_bp: int = DEFAULT_SELECTION_HALF_WINDOW_BP,
    force: bool = False,
) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Render supplied loci as Figure 3B-style combined 1,025-sample panels."""
    from scripts import figure3_ehh

    if selection_half_window_bp < 1:
        raise ValueError("selection_half_window_bp must be positive")
    _, sample_hash = _sample_manifest()
    required_columns = {
        "variant_tag", "chrom", "core_pos", "ref", "alt", "gene",
        "ihs_val", "ihs_rs", "min_p_meta",
    }
    missing_columns = sorted(required_columns - set(loci.columns))
    if missing_columns:
        raise ValueError(f"Selection loci are missing required columns: {missing_columns}")
    loci = loci.copy()
    duplicated = loci["variant_tag"].astype(str).duplicated(keep=False)
    if duplicated.any():
        values = sorted(loci.loc[duplicated, "variant_tag"].astype(str).unique())
        raise ValueError(f"Selection loci contain duplicate panel keys: {values}")
    expected_tags = list(dict.fromkeys(hits["variant_tag"].tolist()))
    locus_by_tag = {str(row["variant_tag"]): row for _, row in loci.iterrows()}
    missing = sorted(set(expected_tags) - set(locus_by_tag))
    if missing:
        raise ValueError(f"EHH/iHS locus selection is missing target variants: {missing}")

    required_static = [figure3_ehh.GTF_PATH, figure3_ehh.HARDMASK_PATH]
    for path in required_static:
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"Required selection input is missing: {path}")

    output_map: dict[str, dict[str, Path]] = {}
    pending: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    base_contract = {
        "analysis_contract": SELECTION_CACHE_CONTRACT,
        "sample_manifest_sha256": sample_hash,
        "sample_count": samples.N_AOU,
        **source_identity,
        "ehh_half_window": selection_half_window_bp,
        "ihspi_half_window": selection_half_window_bp,
        "pi_bin_size": figure3_ehh.PI_BIN_SIZE,
        "minimum_maf": figure3_ehh.MIN_MAF,
        "compact_render_figsize_inches": list(SELECTION_COMPACT_FIGSIZE),
        "wide_render_figsize_inches": list(SELECTION_WIDE_FIGSIZE),
    }
    cache_dir = cache_root / _json_hash(base_contract)[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)

    for variant_tag in expected_tags:
        row = locus_by_tag[variant_tag]
        chrom = str(row["chrom"])
        vcf_path = figure3_ehh.VCF_PATTERN.format(chrom=chrom)
        ihs_path = figure3_ehh.IHS_PATTERN.format(chrom=chrom)
        index_candidates = [vcf_path + ".tbi", vcf_path + ".csi"]
        if not Path(vcf_path).is_file() or not any(Path(path).is_file() for path in index_candidates):
            raise FileNotFoundError(f"Indexed phased VCF is unavailable for {variant_tag}: {vcf_path}")
        if not Path(ihs_path).is_file():
            raise FileNotFoundError(f"Normalized iHS input is unavailable for {variant_tag}: {ihs_path}")
        paths = _selection_paths(selection_dir, variant_tag)
        output_map[variant_tag] = paths
        source_contract = {
            **base_contract,
            # Rendering-only settings belong outside base_contract: changing
            # fonts must redraw PDFs while retaining the EHH/pi NPZ cache.
            "pdf_text_rc": EDITABLE_PDF_RC,
            "variant_tag": variant_tag,
            "vcf": _file_stat(vcf_path),
            "vcf_index": _file_stat(next(path for path in index_candidates if Path(path).is_file())),
            "ihs": _file_stat(ihs_path),
            "gtf": _file_stat(figure3_ehh.GTF_PATH),
            "hardmask": _file_stat(figure3_ehh.HARDMASK_PATH),
        }
        if "sv_pos" in row.index and pd.notna(row["sv_pos"]):
            sv_type = (
                str(row["sv_type"]).strip().upper()
                if "sv_type" in row.index and pd.notna(row["sv_type"])
                else "DEL"
            )
            source_contract["sv_marker"] = {
                "position": int(row["sv_pos"]),
                "type": sv_type,
                "shape": "^" if sv_type == "INS" else "v",
                "size": figure3_ehh.SV_GENE_TRACK_MARKER_SIZE,
                "placement": "top_of_gene_axis_immediately_below_pi",
            }
        source_stats[variant_tag] = source_contract
        contract_hash = _json_hash(source_contract)
        cached = False
        if not force and paths["manifest"].is_file():
            try:
                previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                cached = previous.get("contract_sha256") == contract_hash
                if cached:
                    _validate_png(paths["combined_png"])
                    _validate_editable_panel_pdf(paths["combined_pdf"])
                    _validate_png(paths["combined_wide_png"])
                    _validate_editable_panel_pdf(paths["combined_wide_pdf"])
            except (OSError, ValueError, json.JSONDecodeError):
                cached = False
        if cached:
            continue
        locus_keys = (
            "chrom", "core_pos", "ref", "alt", "gene", "ihs_val", "ihs_rs",
            "min_p_meta", "sv_type",
        )
        locus = {
            key: (float(row[key]) if key in {"ihs_val", "min_p_meta"} else row[key])
            for key in locus_keys
            if key in row.index
        }
        if "sv_pos" in row.index and pd.notna(row["sv_pos"]):
            locus["sv_pos"] = int(row["sv_pos"])
        pending.append(
            {
                "variant_tag": variant_tag,
                "locus": locus,
                "vcf_path": vcf_path,
                "gtf_path": figure3_ehh.GTF_PATH,
                "hardmask_path": figure3_ehh.HARDMASK_PATH,
                "ihs_pattern": figure3_ehh.IHS_PATTERN,
                "selection_half_window_bp": selection_half_window_bp,
                "cache_dir": str(cache_dir),
                "paths": {key: str(value) for key, value in paths.items()},
                "compact_figsize": list(SELECTION_COMPACT_FIGSIZE),
                "wide_figsize": list(SELECTION_WIDE_FIGSIZE),
                "contract_sha256": contract_hash,
            }
        )

    if pending:
        worker_count = min(max(1, threads), len(pending))
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_selection_worker, payload): payload for payload in pending}
            for future in concurrent.futures.as_completed(future_map):
                payload = future_map[future]
                result = future.result()
                variant_tag = result["variant_tag"]
                paths = output_map[variant_tag]
                outputs = {
                    "combined_png": _validate_png(paths["combined_png"]),
                    "combined_pdf": _validate_editable_panel_pdf(paths["combined_pdf"]),
                    "combined_wide_png": _validate_png(paths["combined_wide_png"]),
                    "combined_wide_pdf": _validate_editable_panel_pdf(paths["combined_wide_pdf"]),
                }
                _write_json_atomic(
                    paths["manifest"],
                    {
                        "analysis_contract": SELECTION_CACHE_CONTRACT,
                        "created_utc": _utc_now(),
                        "variant_tag": variant_tag,
                        "contract_sha256": payload["contract_sha256"],
                        "source_contract": source_stats[variant_tag],
                        "outputs": outputs,
                    },
                )
    return output_map, {
        "sample_manifest_sha256": sample_hash,
        "sample_count": samples.N_AOU,
        "selection_cache_dir": str(cache_dir),
        "selection_loci": len(output_map),
        "selection_half_window_bp": selection_half_window_bp,
    }


def render_selection_panels(
    meta_path: Path,
    hits: pd.DataFrame,
    selection_dir: Path,
    cache_root: Path,
    *,
    bucket: str | None = None,
    selection_input_dir: Path | None = None,
    threads: int = DEFAULT_THREADS,
    selection_half_window_bp: int = DEFAULT_SELECTION_HALF_WINDOW_BP,
    force: bool = False,
) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Resolve loci, optionally stage bucket inputs, and render selection panels."""
    from scripts import figure3_ehh

    figure3_ehh.META_TSV = str(meta_path)
    loci = figure3_ehh.select_loci()
    loci["variant_tag"] = loci.apply(
        lambda row: (
            f"{str(row['chrom']).replace('chr', '')}_{int(row['core_pos'])}_"
            f"{row['ref']}_{row['alt']}"
        ),
        axis=1,
    )
    source_identity: dict[str, Any] = {"meta_sha256": sha256_file(meta_path)}
    input_stage: dict[str, Any] = {}
    if (bucket is None) != (selection_input_dir is None):
        raise ValueError("bucket and selection_input_dir must be supplied together")
    if bucket is not None and selection_input_dir is not None:
        normalized_bucket = _normalize_bucket(bucket)
        sources, input_stage = stage_selection_inputs(
            loci,
            normalized_bucket,
            selection_input_dir,
            threads=threads,
            force=force,
        )
        gene_annotation = selection_input_dir / DEFAULT_GENE_ANNOTATION_OBJECT
        input_stage["gene_annotation"] = _stage_gcs_file(
            _gs_join(normalized_bucket, DEFAULT_GENE_ANNOTATION_OBJECT),
            gene_annotation,
            force=force,
        )
        figure3_ehh.VCF_PATTERN = sources["vcf_pattern"]
        figure3_ehh.IHS_PATTERN = sources["ihs_pattern"]
        figure3_ehh.GTF_PATH = str(gene_annotation)
        figure3_ehh.HARDMASK_PATH = sources["hardmask_path"]
        source_identity["workspace_bucket"] = normalized_bucket

    paths, provenance = render_selection_loci(
        loci,
        hits,
        selection_dir,
        cache_root,
        source_identity=source_identity,
        threads=threads,
        selection_half_window_bp=selection_half_window_bp,
        force=force,
    )
    if input_stage:
        provenance["input_stage"] = input_stage
    return paths, provenance


def stage_finemaps(
    hits: pd.DataFrame,
    destination: Path,
    remote_dirs: Sequence[str],
    *,
    threads: int = DEFAULT_THREADS,
    force: bool = False,
) -> tuple[dict[int, Path], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)

    def stage_hit(row: pd.Series) -> tuple[int, Path, dict[str, Any]]:
        filename = expected_finemap_filename(row)
        local = destination / filename
        failures: list[str] = []
        for remote_dir in remote_dirs:
            remote = f"{remote_dir.rstrip('/')}/{filename}"
            try:
                metadata_values = _stage_gcs_file(remote, local, force=force)
                _validate_png(local)
                return int(row["figure_index"]), local, metadata_values
            except (OSError, ValueError) as exc:
                failures.append(str(exc))
        raise FileNotFoundError(
            f"Fine-map image {filename} was not found in any configured directory:\n"
            + "\n".join(failures)
        )

    output: dict[int, Path] = {}
    metadata_map: dict[str, Any] = {}
    worker_count = min(max(1, threads), len(hits))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(stage_hit, row) for _, row in hits.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            index, path, metadata_values = future.result()
            output[index] = path
            metadata_map[str(index)] = metadata_values
    return output, metadata_map


def _resolve_standard_finemap_result(
    hit: pd.Series,
    destination: Path,
    remote_dirs: Sequence[str],
    local_dirs: Sequence[Path],
    *,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    filename = expected_finemap_result_filename(hit)
    destination.mkdir(parents=True, exist_ok=True)
    checked: set[Path] = set()
    for directory in (destination, *local_dirs):
        candidate = directory.expanduser().resolve() / filename
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, {
                "mode": "local",
                "path": str(candidate),
                "size": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }

    local = destination / filename
    failures: list[str] = []
    for remote_dir in remote_dirs:
        remote = f"{remote_dir.rstrip('/')}/{filename}"
        try:
            staged = _stage_gcs_file(remote, local, force=force)
            return local, staged
        except (OSError, ValueError, FileNotFoundError) as exc:
            failures.append(str(exc))
    raise FileNotFoundError(
        f"Fine-map result {filename} was not found in configured local/remote "
        "directories:\n" + "\n".join(failures)
    )


def render_standard_finemap_panels(
    hits: pd.DataFrame,
    destination: Path,
    remote_dirs: Sequence[str],
    *,
    local_dirs: Sequence[Path] = (),
    gene_annotation: Path,
    dpi: int = DEFAULT_DPI,
    force: bool = False,
) -> tuple[dict[int, Path], dict[str, Any]]:
    """Redraw standard SNV fine maps on the same native gene canvas as B."""
    from scripts import figure3_ehh, figure4_finemap

    destination = destination.expanduser().resolve()
    gene_annotation = gene_annotation.expanduser().resolve()
    if not gene_annotation.is_file() or gene_annotation.stat().st_size == 0:
        raise FileNotFoundError(
            f"Fine-map gene annotation is missing or empty: {gene_annotation}"
        )
    gene_annotation_sha = sha256_file(gene_annotation)
    renderer_sha = {
        "figure3_ehh": sha256_file(Path(figure3_ehh.__file__).resolve()),
        "figure4_finemap": sha256_file(Path(figure4_finemap.__file__).resolve()),
    }
    sex_lookup = figure4_finemap._sex_scope_lookup()
    output: dict[int, Path] = {}
    provenance: dict[str, Any] = {}

    for _, hit in hits.iterrows():
        index = int(hit["figure_index"])
        chrom_number, lead_pos, lead_ref, lead_alt = parse_variant_tag(
            str(hit["variant_tag"])
        )
        phecode = str(hit["phecode"])
        sex_scope = sex_lookup.get(phecode)
        if sex_scope not in {"Both", "Female", "Male"}:
            raise ValueError(f"No valid sex scope for fine-map phecode {phecode}")
        locus = {
            "variant_tag": str(hit["variant_tag"]),
            "chrom": f"chr{chrom_number}",
            "lead_pos": lead_pos,
            "lead_ref": lead_ref,
            "lead_alt": lead_alt,
            "lead_gene": str(hit.get("gene", "")),
            "finemap_gene": str(hit.get("gene", "")),
            "bold_gene": None,
            "phecode": phecode,
            "phecode_name": str(hit.get("phecode_name", phecode)),
            "sex_scope": sex_scope,
            "meta_p": float(hit.get("p_meta", np.nan)),
            "meta_beta": float(hit.get("beta_meta", np.nan)),
        }
        result_path, stage = _resolve_standard_finemap_result(
            hit,
            destination,
            remote_dirs,
            local_dirs,
            force=force,
        )
        figure4_finemap.validate_discovery_contract(
            result_path,
            expected_scope=sex_scope,
            expected_locus=locus,
        )
        frame = pd.read_csv(result_path, sep="\t", low_memory=False)
        if "r2_to_lead" not in frame.columns and "r2_to_minp" in frame.columns:
            frame["r2_to_lead"] = frame["r2_to_minp"]
        if "neg_log_p" not in frame.columns and "p_score" in frame.columns:
            p_values = pd.to_numeric(frame["p_score"], errors="coerce").clip(
                lower=1e-300
            )
            frame["neg_log_p"] = -np.log10(p_values)
        required = {"pos", "neg_log_p", "r2_to_lead"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Verified fine-map result {result_path} is missing {missing}"
            )
        frame["pos"] = pd.to_numeric(frame["pos"], errors="raise")
        frame["neg_log_p"] = pd.to_numeric(
            frame["neg_log_p"], errors="raise"
        )
        frame["r2_to_lead"] = pd.to_numeric(
            frame["r2_to_lead"], errors="coerce"
        ).fillna(0.0)
        finemap_result_summary = summarize_standard_finemap_result(frame, hit)
        # Log on every run, including panel-cache hits, so the workflow log is
        # a complete record of every PheWAS/fine-map result in this series.
        log_standard_finemap_result(finemap_result_summary)

        window_start = max(1, lead_pos - figure4_finemap.HALF_WINDOW)
        window_end = lead_pos + figure4_finemap.HALF_WINDOW
        genes = figure3_ehh.parse_gtf_region(
            str(gene_annotation), locus["chrom"], window_start, window_end
        )
        if not genes:
            raise ValueError(
                "Staged refGene annotation returned no genes for fine-map window "
                f"{locus['chrom']}:{window_start}-{window_end}"
            )

        png_path = destination / expected_finemap_filename(hit)
        pdf_path = png_path.with_suffix(".pdf")
        manifest_path = png_path.with_suffix(".json")
        source_contract = {
            "analysis_contract": FINEMAP_PANEL_CONTRACT,
            "axis_labels": [style.P_VALUE_AXIS_LABEL, style.LD_COLORBAR_LABEL],
            "figure4_analysis_contract": figure4_finemap.ANALYSIS_CONTRACT,
            "variant_tag": str(hit["variant_tag"]),
            "phecode": phecode,
            "window_start": window_start,
            "window_end": window_end,
            "window_center": lead_pos,
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "result_stage": stage,
            "renderer_sha256": renderer_sha,
            "gene_annotation_path": str(gene_annotation),
            "gene_annotation_sha256": gene_annotation_sha,
            "gene_names": sorted(str(name) for name in genes),
            "gene_track_renderer": "figure3_ehh.draw_gene_track",
            "source_figsize_inches": list(
                figure4_finemap.SUPPLEMENTARY_FIGSIZE_LOCUS
            ),
            "source_panel_aspect": figure4_finemap.SUPPLEMENTARY_PANEL_ASPECT,
            "dpi": dpi,
            "show_title": False,
            "finemap_result_summary": finemap_result_summary,
        }
        contract_hash = _json_hash(source_contract)
        reused = False
        if not force and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous.get("contract_sha256") == contract_hash:
                    _validate_png(png_path)
                    _validate_pdf(pdf_path)
                    reused = True
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not reused:
            with plt.rc_context():
                figure = figure4_finemap.plot_finemap(
                    frame,
                    locus,
                    show_title=False,
                    show_r2_legend=True,
                    highlight_gene=False,
                    expanded_gene_track=True,
                    window_center_pos=lead_pos,
                    shared_gene_track=genes,
                    editable_axis_labels=True,
                )
                figure.savefig(
                    png_path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches=figure.bbox_inches,
                    pad_inches=0,
                )
                figure.savefig(
                    pdf_path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches=figure.bbox_inches,
                    pad_inches=0,
                    metadata={
                        "Title": f"Fine map {hit['variant_tag']} {phecode}",
                        "Subject": FINEMAP_PANEL_CONTRACT,
                    },
                )
                plt.close(figure)
            outputs = {
                "png": _validate_png(png_path),
                "pdf": _validate_pdf(pdf_path),
            }
            _write_json_atomic(
                manifest_path,
                {
                    "analysis_contract": FINEMAP_PANEL_CONTRACT,
                    "created_utc": _utc_now(),
                    "contract_sha256": contract_hash,
                    "source_contract": source_contract,
                    "outputs": outputs,
                },
            )
        output[index] = png_path
        provenance[str(index)] = {
            "cache": "hit" if reused else "rendered",
            "panel": str(png_path),
            "panel_manifest": str(manifest_path),
            "source_contract": source_contract,
        }
    return output, provenance


def write_standard_susie_tasks(hits: pd.DataFrame, path: Path) -> Path:
    """Write the exact standard-locus task table consumed by the notebook."""
    required = {"variant_tag", "phecode", "phecode_name", "gene", "p_meta"}
    missing = sorted(required - set(hits.columns))
    if missing:
        raise ValueError(f"Standard SuSiE hits are missing task columns: {missing}")
    tasks = hits[
        ["variant_tag", "phecode", "phecode_name", "gene", "p_meta"]
    ].rename(
        columns={
            "phecode_name": "phecode_string",
            "gene": "gene_in_window",
        }
    )
    tasks.insert(3, "meta_significant", "TRUE")
    tasks = tasks.drop_duplicates(["variant_tag", "phecode"], keep="first")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(path, sep="\t", index=False)
    return path


def render_standard_finemap_panels_with_recompute(
    hits: pd.DataFrame,
    destination: Path,
    remote_dirs: Sequence[str],
    *,
    local_dirs: Sequence[Path],
    gene_annotation: Path,
    task_path: Path,
    phase: str,
    dpi: int,
    force: bool,
) -> tuple[dict[int, Path], dict[str, Any]]:
    """Render strict current panels, recomputing invalid results in ``all``."""
    try:
        return render_standard_finemap_panels(
            hits,
            destination,
            remote_dirs,
            local_dirs=local_dirs,
            gene_annotation=gene_annotation,
            dpi=dpi,
            force=force,
        )
    except (FileNotFoundError, ValueError) as exc:
        if phase != "all":
            raise FileNotFoundError(
                f"Verified standard SuSiE result/panel is unavailable: {exc}. "
                "Run --phase all on Workbench to regenerate the current "
                "carrier-filtered result."
            ) from exc
        task_path = write_standard_susie_tasks(hits, task_path)
        logging.getLogger("supp_locus_composites").info(
            "verified standard SuSiE results are missing or invalid; "
            "running the exact task table: %s",
            task_path,
        )
        from scripts import recompute_figure4_susie

        recompute_figure4_susie.ensure_figure4_outputs(task_table=task_path)
        return render_standard_finemap_panels(
            hits,
            destination,
            remote_dirs,
            local_dirs=local_dirs,
            gene_annotation=gene_annotation,
            dpi=dpi,
            force=force,
        )


def compose_hit(
    hit: pd.Series,
    hits: pd.DataFrame,
    phewas: pd.DataFrame,
    selection_paths: dict[str, Path],
    finemap_png: Path,
    output_dir: Path,
    category_order: Sequence[str],
    parameters: PlotParameters,
    *,
    sample_manifest_sha256: str,
    force: bool = False,
    analysis_contract: str = ANALYSIS_CONTRACT,
    output_prefix: str = "Supplementary_Phewas",
    highlight_color: str = "#d95f02",
    highlight_label: str = "Meta-significant",
    association_summary: str | None = None,
    extra_hit_contract_columns: Sequence[str] = (),
) -> dict[str, Path]:
    index = int(hit["figure_index"])
    stem_name = _phewas_output_stem_name(hit, output_prefix=output_prefix)
    paths = _composite_output_paths(output_dir, stem_name)
    source_panel_paths = _source_panel_output_paths(
        output_dir, stem_name, index, ("A", "B", "C")
    )
    locus_frame = phewas[phewas["variant_tag"] == hit["variant_tag"]].copy()
    meta_hits = hits[hits["variant_tag"] == hit["variant_tag"]].copy()
    target_rows = locus_frame[locus_frame["phecode"] == str(hit["phecode"])]
    if target_rows.empty:
        raise ValueError(f"PheWAS table lacks target {hit['variant_tag']} x {hit['phecode']}")
    target = target_rows.iloc[0]
    source_contract = {
        "analysis_contract": analysis_contract,
        "hit": {
            key: _json_scalar(hit[key])
            for key in (
                "figure_index", "variant_index", "sv_index", "variant_tag",
                "phecode", "phecode_name", "gene", "p_meta", "beta_meta",
                *extra_hit_contract_columns,
            )
            if key in hit.index
        },
        "sample_manifest_sha256": sample_manifest_sha256,
        "plot_parameters": asdict(parameters),
        "rendering_contract": {
            "vector_composite_contract": VECTOR_COMPOSITE_CONTRACT,
            "pdf_text_rc": EDITABLE_PDF_RC,
            "phewas_axis_label": style.P_VALUE_AXIS_LABEL,
            "phewas_category_background_colors": list(
                PHEWAS_CATEGORY_BACKGROUND_COLORS
            ),
            "phewas_category_label_rotation_degrees": (
                PHEWAS_CATEGORY_LABEL_ROTATION
            ),
            "selection_height_ratios": list(
                figure3_locus_combined.COMBINED_HEIGHT_RATIOS
            ),
            "finemap_complete_side_label_height_px": FINEMAP_SIDE_LABEL_HEIGHT_PX,
            "finemap_postcrop_top_padding_px": FINEMAP_POSTCROP_TOP_PADDING_PX,
        },
        "visual_title_variants": {
            "titled": "overall title; B left and C right on one row; C title removed",
            "no_title": "panel letters only; B left and C right; C title removed",
        },
        "panel_layout": "A full width; B lower left; C lower right",
        "panel_height_ratios": list(COMPOSITE_HEIGHT_RATIOS),
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
        "phewas_subset_sha256": _sha256_bytes(
            locus_frame.sort_values("phecode").to_csv(index=False, float_format="%.17g").encode("utf-8")
        ),
        "selection": {
            "combined_sha256": sha256_file(selection_paths["combined_png"]),
            "combined_pdf_sha256": sha256_file(selection_paths["combined_pdf"]),
        },
        "finemap_sha256": sha256_file(finemap_png),
        "finemap_pdf_sha256": sha256_file(finemap_png.with_suffix(".pdf")),
    }
    if (
        analysis_contract != ANALYSIS_CONTRACT
        or output_prefix != "Supplementary_Phewas"
        or highlight_color != "#d95f02"
        or highlight_label != "Meta-significant"
        or association_summary is not None
        or extra_hit_contract_columns
    ):
        source_contract["series_rendering"] = {
            "output_prefix": output_prefix,
            "highlight_color": highlight_color,
            "highlight_label": highlight_label,
            "association_summary": association_summary,
        }
    contract_hash = _json_hash(source_contract)
    if not force and paths["manifest"].is_file():
        try:
            previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            if previous.get("contract_sha256") == contract_hash:
                _validate_png(paths["png"])
                _validate_pdf(paths["pdf"])
                _validate_png(paths["no_title_png"])
                _validate_pdf(paths["no_title_pdf"])
                _validate_source_panel_outputs(source_panel_paths)
                if not paths["source_zip"].is_file():
                    raise FileNotFoundError(paths["source_zip"])
                with zipfile.ZipFile(paths["source_zip"]) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"Corrupt source archive: {paths['source_zip']}")
                return paths
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    meta_text = (
        association_summary
        if association_summary is not None
        else (
            f"meta p={hit['p_meta']:.2g}"
            if pd.notna(hit["p_meta"])
            else "meta-significant"
        )
    )
    gene_text = f" · {hit['gene']}" if str(hit["gene"]).strip() else ""

    source_panel_outputs: dict[str, dict[str, Any]] = {}

    def render_version(
        *,
        include_title: bool,
        png_path: Path,
        pdf_path: Path,
        save_sources: bool = False,
    ) -> None:
        fig = plt.figure(figsize=(8.5, 11), constrained_layout=False)
        grid = fig.add_gridspec(
            3,
            24,
            height_ratios=COMPOSITE_HEIGHT_RATIOS,
            left=0.055,
            right=0.985,
            top=0.895 if include_title else 0.965,
            bottom=0.055,
            hspace=0.34,
            wspace=0.70,
        )
        ax_phewas = fig.add_subplot(grid[0, 2:22])
        draw_phewas_panel(
            ax_phewas,
            locus_frame,
            meta_hits,
            hit,
            category_order,
            parameters,
            highlight_color=highlight_color,
            highlight_label=highlight_label,
        )
        ax_phewas.text(
            -0.055,
            1.02,
            "A",
            transform=ax_phewas.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        _group_axis, ax_selection = _show_grouped_selection_panel(
            fig,
            grid[1, 0:12],
            selection_paths,
            include_title=include_title,
        )
        ax_fine = fig.add_subplot(grid[1, 12:24])
        _show_image_panel(
            ax_fine,
            finemap_png,
            "C",
            None,
            strip_embedded_title=False,
            anchor="N",
            fill_cell=False,
            preserve_canvas=True,
        )

        if include_title:
            fig.suptitle(
                f"{hit['variant_tag']}{gene_text} → {hit['phecode']} {hit['phecode_name']}\n"
                f"OR={target['odds_ratio']:.3g} · discovery p={target['p_value']:.2g} · "
                f"{meta_text}",
                fontsize=14.5,
                fontweight="bold",
                y=0.989,
            )
        metadata_values = {
            "Title": (
                f"Supplementary locus composite {index}: "
                f"{hit['variant_tag']} {hit['phecode']}"
            ),
            "Subject": analysis_contract,
            "Keywords": "PheWAS EHH iHS fine mapping All of Us",
        }
        fig.savefig(
            png_path,
            dpi=parameters.dpi,
            facecolor="white",
            bbox_inches=fig.bbox_inches,
            pad_inches=0,
        )
        selection_pdf_key = "combined_pdf"
        selection_pdf = selection_paths[selection_pdf_key]
        finemap_pdf = finemap_png.with_suffix(".pdf")
        _save_vector_composite_pdf(
            fig,
            pdf_path,
            ((ax_selection, selection_pdf), (ax_fine, finemap_pdf)),
            metadata=metadata_values,
        )
        if save_sources:
            source_panel_outputs.update(
                _save_source_panels(
                    fig,
                    {
                        "A": (ax_phewas,),
                    },
                    source_panel_paths,
                    dpi=parameters.dpi,
                    metadata=metadata_values,
                )
            )
            source_panel_outputs["B"] = _copy_native_source_panel(
                source_panel_paths["B"],
                png_source=selection_paths["combined_png"],
                pdf_source=selection_pdf,
            )
            source_panel_outputs["C"] = _copy_native_source_panel(
                source_panel_paths["C"],
                png_source=finemap_png,
                pdf_source=finemap_pdf,
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
    source_zip_output = _write_source_zip(
        paths["source_zip"], source_panel_paths, figure_index=index
    )
    outputs = {
        "png": _validate_png(paths["png"]),
        "pdf": _validate_vector_pdf(paths["pdf"]),
        "no_title_png": _validate_png(paths["no_title_png"]),
        "no_title_pdf": _validate_vector_pdf(paths["no_title_pdf"]),
        "source_panels": source_panel_outputs,
        "source_zip": source_zip_output,
    }
    _write_json_atomic(
        paths["manifest"],
        {
            "analysis_contract": analysis_contract,
            "created_utc": _utc_now(),
            "contract_sha256": contract_hash,
            "source_contract": source_contract,
            "outputs": outputs,
        },
    )
    return paths


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "matplotlib", "adjustText", "Pillow", "scikit-allel")
    values: dict[str, str | None] = {}
    for package in packages:
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = None
    return values


def _configure_logging(work_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"supp_locus_composites_{stamp}.log"
    logger = logging.getLogger("supp_locus_composites")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger, log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage", "selection", "compose", "all"), default="all")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--meta-object", default=DEFAULT_META_OBJECT)
    parser.add_argument("--phewas-object", default=DEFAULT_PHEWAS_OBJECT)
    parser.add_argument(
        "--finemap-gs-dir", action="append", default=None,
        help="Fine-map GCS directory; repeat for fallbacks. Defaults to the dual and legacy dirs.",
    )
    parser.add_argument(
        "--finemap-dir",
        action="append",
        type=Path,
        default=None,
        help="Local verified SuSiE result directory; repeat for fallbacks.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("~/supp_phewas_locus_composites"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Single directory for all figures and provenance outputs.",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-hits", type=int, default=DEFAULT_EXPECTED_HITS)
    parser.add_argument("--expected-loci", type=int, default=DEFAULT_EXPECTED_LOCI)
    parser.add_argument(
        "--odds-ratio-cap", type=float, default=DEFAULT_OR_CAP,
        help="Shared OR-fold ceiling (default: 4, matching the 1x-4x triangle legend).",
    )
    parser.add_argument(
        "--phewas-y-max", type=float, default=None,
        help="Optional shared -log10(p) ceiling; defaults to the padded observed maximum.",
    )
    parser.add_argument("--marker-area-min", type=float, default=DEFAULT_MARKER_MIN)
    parser.add_argument("--marker-area-max", type=float, default=DEFAULT_MARKER_MAX)
    parser.add_argument("--phewas-threshold", type=float, default=DEFAULT_PHEWAS_THRESHOLD)
    parser.add_argument(
        "--selection-half-window-bp", type=int, default=DEFAULT_SELECTION_HALF_WINDOW_BP,
        help="Half-window for the combined selection panel (default: 500000 bp).",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.selection_half_window_bp < 1:
        raise ValueError("--selection-half-window-bp must be positive")
    if args.selection_half_window_bp != FINEMAP_HALF_WINDOW_BP:
        raise ValueError(
            "--selection-half-window-bp must equal the fine-map half-window "
            f"({FINEMAP_HALF_WINDOW_BP} bp) so panels B and C share endpoints"
        )
    if not 0 < args.phewas_threshold < 1:
        raise ValueError("--phewas-threshold must be between zero and one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    bucket = _normalize_bucket(args.bucket)
    work_dir = args.work_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_dir = work_dir / "cache" / "inputs"
    finemap_dir = work_dir / "cache" / "finemap"
    selection_dir = work_dir / "selection"
    logger, log_path = _configure_logging(work_dir)
    logger.info("analysis_contract=%s phase=%s", ANALYSIS_CONTRACT, args.phase)
    logger.info(
        "bucket=%s work_dir=%s output_dir=%s threads=%s seed=%s",
        bucket, work_dir, output_dir, args.threads, args.seed,
    )

    meta_remote = _gs_join(bucket, args.meta_object)
    phewas_remote = _gs_join(bucket, args.phewas_object)
    meta_local = input_dir / Path(args.meta_object).name
    phewas_local = input_dir / Path(args.phewas_object).name
    table_stage: dict[str, Any] = {}
    for label, remote, local in (
        ("meta", meta_remote, meta_local), ("phewas", phewas_remote, phewas_local)
    ):
        if args.phase == "selection" and label == "phewas":
            continue
        table_stage[label] = _stage_gcs_file(remote, local, force=args.force)
        logger.info("staged %s: %s (%s)", label, local, table_stage[label]["cache"])

    hits = load_meta_hits(
        meta_local, expected_hits=args.expected_hits, expected_loci=args.expected_loci,
    )
    logger.info("hit contract: %s pairs across %s loci", len(hits), hits["variant_tag"].nunique())
    hit_manifest_path = output_dir / "hit_manifest.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    hits.to_csv(hit_manifest_path, sep="\t", index=False)

    remote_dirs = args.finemap_gs_dir or [
        _gs_join(bucket, directory) for directory in DEFAULT_FINEMAP_DIRS
    ]
    remote_dirs = [_gs_join(bucket, directory) for directory in remote_dirs]
    # Keep notebook recomputation and panel staging rooted in this run's
    # selected workspace bucket/cache rather than ambient process defaults.
    os.environ["AOU_WORKSPACE_BUCKET_GS"] = bucket
    os.environ["AOU_SOURCE_BUCKET"] = bucket
    os.environ["AOU_SUSIE_DIR"] = str(finemap_dir)
    os.environ["AOU_SUSIE_GS_DIR"] = remote_dirs[0]
    config.WORKSPACE_BUCKET_GS = bucket
    config.SUSIE_FINEMAP_DIR = str(finemap_dir)
    config.SUSIE_FINEMAP_GS_DIR = remote_dirs[0]
    finemap_paths: dict[int, Path] = {}
    finemap_stage: dict[str, Any] = {}
    susie_task_path = output_dir / "standard_locus_susie_tasks.tsv"
    if args.phase in {"stage", "compose", "all"}:
        gene_annotation = input_dir.parent / "selection_inputs" / DEFAULT_GENE_ANNOTATION_OBJECT
        table_stage["finemap_gene_annotation"] = _stage_gcs_file(
            _gs_join(bucket, DEFAULT_GENE_ANNOTATION_OBJECT),
            gene_annotation,
            force=args.force,
        )
        local_finemap_dirs = [
            finemap_dir,
            *(path.expanduser().resolve() for path in (args.finemap_dir or [])),
        ]
        finemap_paths, finemap_stage = render_standard_finemap_panels_with_recompute(
            hits,
            finemap_dir,
            remote_dirs,
            local_dirs=local_finemap_dirs,
            gene_annotation=gene_annotation,
            task_path=susie_task_path,
            phase=args.phase,
            dpi=args.dpi,
            force=args.force,
        )
        logger.info(
            "rendered/validated %s native Figure 4-style fine-map panels from TSVs",
            len(finemap_paths),
        )

    selection_paths: dict[str, dict[str, Path]] = {
        variant_tag: _selection_paths(selection_dir, variant_tag)
        for variant_tag in dict.fromkeys(hits["variant_tag"])
    }
    selection_provenance: dict[str, Any] = {}
    if args.phase in {"selection", "all"}:
        selection_paths, selection_provenance = render_selection_panels(
            meta_local, hits, selection_dir, work_dir / "cache" / "selection",
            bucket=bucket,
            selection_input_dir=work_dir / "cache" / "selection_inputs",
            threads=args.threads,
            selection_half_window_bp=args.selection_half_window_bp,
            force=args.force,
        )
        if "input_stage" in selection_provenance:
            logger.info(
                "staged Figure 3 inputs from bucket for chromosomes: %s",
                ", ".join(selection_provenance["input_stage"]["chromosomes"]),
            )
        logger.info("rendered/validated selection panels for %s loci", len(selection_paths))

    outputs: dict[int, dict[str, Path]] = {}
    phewas_metadata: dict[str, Any] = {}
    shared_scales: dict[str, float] = {}
    if args.phase in {"compose", "all"}:
        for variant_tag, paths in selection_paths.items():
            try:
                _validate_png(paths["combined_png"])
            except (FileNotFoundError, ValueError) as exc:
                raise FileNotFoundError(
                    f"Selection panel missing for {variant_tag}; run --phase selection first: {exc}"
                ) from exc
        phewas, phewas_metadata = load_phewas_for_variants(
            phewas_local, set(hits["variant_tag"]), chunk_rows=args.chunk_rows,
        )
        target_keys = set(zip(hits["variant_tag"], hits["phecode"]))
        observed_keys = set(zip(phewas["variant_tag"], phewas["phecode"]))
        missing_targets = sorted(target_keys - observed_keys)
        if missing_targets:
            raise ValueError(f"PheWAS table is missing meta-significant target tests: {missing_targets}")
        category_order = list(dict.fromkeys(phewas["phecode_category"].astype(str)))
        shared_scales = resolve_shared_phewas_scales(
            phewas,
            odds_ratio_cap=args.odds_ratio_cap,
            phewas_y_max=args.phewas_y_max,
        )
        logger.info(
            "shared PheWAS scales: observed OR-fold max=%.6g cap=%.6g; "
            "observed -log10(p) max=%.6g y_max=%.6g",
            shared_scales["observed_or_fold_max"], shared_scales["odds_ratio_cap"],
            shared_scales["observed_neglogp_max"], shared_scales["phewas_y_max"],
        )
        parameters = PlotParameters(
            odds_ratio_cap=shared_scales["odds_ratio_cap"],
            marker_area_min=args.marker_area_min,
            marker_area_max=args.marker_area_max,
            phewas_threshold=args.phewas_threshold,
            phewas_y_max=shared_scales["phewas_y_max"],
            dpi=args.dpi,
        )
        _, sample_hash = _sample_manifest()
        for _, hit in hits.iterrows():
            index = int(hit["figure_index"])
            outputs[index] = compose_hit(
                hit, hits, phewas, selection_paths[str(hit["variant_tag"])],
                finemap_paths[index], output_dir, category_order, parameters,
                sample_manifest_sha256=sample_hash, force=args.force,
            )
        logger.info(
            "rendered/validated %s titled/no-title letter-size PNG/PDF composites",
            len(outputs),
        )

    run_manifest = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "created_utc": _utc_now(),
        "phase": args.phase,
        "git_sha": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "parameters": {
            "bucket": bucket,
            "output_dir": str(output_dir),
            "meta_object": meta_remote,
            "phewas_object": phewas_remote,
            "finemap_gs_dirs": remote_dirs,
            "finemap_dirs": [
                str(path.expanduser().resolve()) for path in (args.finemap_dir or [])
            ],
            "threads": args.threads,
            "seed": args.seed,
            "expected_hits": args.expected_hits,
            "expected_loci": args.expected_loci,
            "requested_odds_ratio_cap": args.odds_ratio_cap,
            "requested_phewas_y_max": args.phewas_y_max,
            "marker_area_min": args.marker_area_min,
            "marker_area_max": args.marker_area_max,
            "phewas_threshold": args.phewas_threshold,
            "selection_half_window_bp": args.selection_half_window_bp,
            "dpi": args.dpi,
            "chunk_rows": args.chunk_rows,
            "force": args.force,
        },
        "hit_manifest": {
            "path": str(hit_manifest_path),
            "sha256": sha256_file(hit_manifest_path),
            "hits": len(hits),
            "loci": int(hits["variant_tag"].nunique()),
        },
        "table_stage": table_stage,
        "finemap_stage": finemap_stage,
        "susie_tasks": (
            {
                "path": str(susie_task_path),
                "sha256": sha256_file(susie_task_path),
                "rows": len(pd.read_csv(susie_task_path, sep="\t")),
            }
            if susie_task_path.is_file()
            else None
        ),
        "selection": selection_provenance,
        "phewas": phewas_metadata,
        "shared_phewas_scales": shared_scales,
        "outputs": {
            str(index): {key: str(path) for key, path in paths.items()}
            for index, paths in outputs.items()
        },
        "log": str(log_path),
    }
    run_manifest_path = output_dir / "run_manifest.json"
    _write_json_atomic(run_manifest_path, run_manifest)
    logger.info("run manifest: %s", run_manifest_path)

    return {
        "work_dir": work_dir,
        "output_dir": output_dir,
        "hit_manifest": hit_manifest_path,
        "run_manifest": run_manifest_path,
        "outputs": outputs,
        "log": log_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception:
        logging.getLogger("supp_locus_composites").exception("workflow failed")
        return 1
    print(json.dumps({key: str(value) for key, value in result.items() if key != "outputs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
