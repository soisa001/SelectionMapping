"""AoU sample bookkeeping using controlled cohort and exclusion manifests.

Participant identifiers are loaded only at runtime from the Workbench.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

from . import config

# --- Cohort accounting (hardcoded from the VCF; do not edit casually) ---------
N_AOU = 1025                 # AoU samples present in the VCF == the analysis set
N_1000G_IN_VCF = 827         # 1000G (HG/NA) reference IDs in the joint VCF
VCF_TOTAL_SAMPLES = 1852     # 1025 AoU + 827 1000G


# Canonical 1,025 good-sample list (post-VWB migration), one sample ID per line.
# This is the authoritative analysis set; use it directly to restrict any query.
AOU_SAMPLE_LIST_GS = "gs://rw-migration-aou-rw-a3e31f4d/AOU_sample_list.txt"
AOU_SAMPLE_LIST_LOCAL = config.RAW_DATA_DIR / "AOU_sample_list.txt"


def load_excluded_sample_ids() -> frozenset[str]:
    """Read the controlled exclusion list, one ID per line, without logging IDs."""
    path = Path(os.environ.get(
        "AOU_EXCLUDED_SAMPLE_IDS_FILE", config.RAW_DATA_DIR / "excluded_related_sample_ids.txt"
    )).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "Controlled exclusion manifest is required. Set AOU_EXCLUDED_SAMPLE_IDS_FILE "
            "to the existing Workbench list (one ID per line)."
        )
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)) or any(
        any(character.isspace() or character == "," for character in value)
        or value.lower() in {"sample_id", "person_id", "research_id"}
        for value in values
    ):
        raise ValueError("Exclusion manifest must contain unique IDs, one per line, without a header.")
    return frozenset(values)


def _download_sample_list() -> bool:
    """Best-effort copy of the GCS sample list to the local cache.

    Uses ``gcloud storage cp`` with a billing project (requester-pays aware).
    """
    from . import gcs
    if not gcs.available():
        return False
    AOU_SAMPLE_LIST_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    gcs.gcs_cp(AOU_SAMPLE_LIST_GS, AOU_SAMPLE_LIST_LOCAL)
    return AOU_SAMPLE_LIST_LOCAL.exists()


def load_aou_sample_list(download: bool = True) -> set[str] | None:
    """Return the 1,025 good-sample IDs, or None if unavailable.

    Reads the local cache (``data/raw/AOU_sample_list.txt``); if absent and
    ``download`` is True, fetches it from ``AOU_SAMPLE_LIST_GS`` via
    ``gcloud storage cp``. Controlled cohort exclusions are applied defensively.
    """
    if not AOU_SAMPLE_LIST_LOCAL.exists() and download:
        _download_sample_list()
    if not AOU_SAMPLE_LIST_LOCAL.exists():
        return None
    ids = {ln.strip() for ln in AOU_SAMPLE_LIST_LOCAL.read_text().splitlines() if ln.strip()}
    ids -= load_excluded_sample_ids()
    return ids or None


def is_excluded(sample_id) -> bool:
    """True if a sample ID is one of the always-excluded related individuals."""
    return str(sample_id) in load_excluded_sample_ids()


def exclude_related(df: pd.DataFrame, *, sample_col: str = "sample_id") -> pd.DataFrame:
    """Drop the always-excluded related individuals from a per-sample frame.

    No-op (with a warning) if ``sample_col`` is absent, since we then cannot
    identify the samples to remove.
    """
    if sample_col not in df.columns:
        warnings.warn(
            f"[samples] no '{sample_col}' column; cannot apply cohort exclusions.",
            stacklevel=2,
        )
        return df
    return df[~df[sample_col].astype(str).isin(load_excluded_sample_ids())].copy()


# Vendored 1000 Genomes reference mapping (public sample IDs -> AFR/EUR),
# restricted to the 827 joint-set 1kg samples (409 AFR + 418 EUR) in
# 1kg_sample_list.txt. Used to label/select 1kg samples in the ADMIXTURE plot.
REF_1KG_CSV = config.ASSETS_DIR / "1kg_ref_populations.csv"
REF_1KG_LIST = config.ASSETS_DIR / "1kg_sample_list.txt"


def load_1kg_reference() -> dict[str, str]:
    """Return {1000G sample_id -> superpop ('AFR'|'EUR')} from the vendored CSV."""
    df = pd.read_csv(REF_1KG_CSV, dtype=str)
    return dict(zip(df["sample_id"].str.strip(), df["superpop"].str.strip()))


def load_unrelated_ids() -> set[str] | None:
    """Return the unrelated (1,025) sample IDs from the local cache, or None.

    Back-compat wrapper over ``load_aou_sample_list``; does not trigger a
    download (use ``load_aou_sample_list(download=True)`` for that).
    """
    return load_aou_sample_list(download=False)


def check_sample_count(
    df: pd.DataFrame,
    expected: int = N_AOU,
    *,
    sample_col: str = "sample_id",
    label: str = "",
    strict: bool = False,
) -> pd.DataFrame:
    """Verify a per-sample frame has ``expected`` unique samples (default 1025).

    On mismatch, warns (or raises when ``strict``). Returns ``df`` for chaining.
    """
    n = df[sample_col].nunique() if sample_col in df.columns else len(df)
    if n != expected:
        msg = f"[samples] {label or 'frame'}: expected {expected} samples, found {n}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return df


def filter_to_unrelated(
    df: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    label: str = "",
    strict: bool = False,
) -> pd.DataFrame:
    """Restrict an AoU frame to the 1,025-sample analysis set.

    Always drops the controlled exclusion list; if an unrelated-sample list is present
    it is additionally intersected. The result is checked against ``N_AOU``.
    """
    out = exclude_related(df, sample_col=sample_col)
    ids = load_unrelated_ids()
    if ids is not None and sample_col in out.columns:
        out = out[out[sample_col].astype(str).isin(ids)].copy()
    return check_sample_count(
        out, N_AOU, sample_col=sample_col,
        label=label or "AoU analysis set", strict=strict,
    )
