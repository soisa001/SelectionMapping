"""Publication CSV tables for the corrected Figure 2 admixture workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import (
    figure4_meta,
    admixture_finemap,
    admixture_inputs,
    admixture_lai,
    admixture_phewas,
    admixture_plrs,
    admixture_summary,
)
from .common import config

OUTPUT_FILENAMES = {
    "outliers": "Figure_2_1025_all_outlier_regions.csv",
    "plrs": "Figure_2_1025_plrs_regions.csv",
    "phewas": "Figure_2_1025_phewas_results.csv",
}

PHEWAS_COLUMNS = [
    "chrom",
    "start",
    "end",
    "direction",
    "peak_z",
    "genes_in_region",
    "phecode",
    "phecode_string",
    "beta_disc",
    "se_disc",
    "p_disc",
    "n_disc",
    "beta_rep",
    "se_rep",
    "p_rep",
    "n_rep",
    "beta_meta",
    "se_meta",
    "p_meta",
    "same_direction",
    "rep_nominal",
    "meta_significant",
]

META_ANNOTATION_COLUMNS = ["genes_in_region", "phecode_string"]
META_GENE_SOURCE_CONTRACT = admixture_finemap.GENE_TABLE_CONTRACT
META_PHECODE_SOURCE_CONTRACT = "snv_phewas_tables_then_discovery_phetk_v1"

OUTLIER_LEADING_COLUMNS = [
    "chrom",
    "start",
    "end",
    "direction",
    "peak_z",
    "genes_in_region",
    "region_id",
]

PLRS_METRIC_COLUMNS = [
    "n_snvs",
    "n_components",
    "n_train",
    "n_test",
    "n_source_samples",
    "r2_holdout",
    "rmse_holdout",
    "r2_null",
    "r2_threshold",
    "retained_for_phewas",
    "model_contract",
]


def output_dir() -> Path:
    """Use the requested main Figure 2 folder without replacing plot files."""
    path = config.figure_dir("figure2_sources")
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_paths() -> dict[str, Path]:
    return {key: output_dir() / filename for key, filename in OUTPUT_FILENAMES.items()}


def _read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required Figure 2_1025 table source is missing: {path}"
        )
    try:
        return pd.read_csv(path, sep="\t", **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Required Figure 2_1025 table source is empty: {path}"
        ) from exc


def _ordered_columns(frame: pd.DataFrame, leading: list[str]) -> list[str]:
    return [*leading, *[column for column in frame.columns if column not in leading]]


def _with_region_genes(regions: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    result = regions.copy()
    result["chrom"] = result["chrom"].map(admixture_inputs.normalize_chrom)
    gene_lookup = admixture_summary._genes_in_regions(result, genes)
    result["genes_in_region"] = (
        result["region_id"].astype(str).map(gene_lookup).fillna("")
    )
    return result


def _phecode_strings(
    meta: pd.DataFrame, discovery: pd.DataFrame
) -> dict[str, str]:
    """Use canonical SNV-PheWAS labels, then current PheTK labels as fallback."""
    lookup = admixture_summary._phecode_string_lookup(meta)
    if "phecode_string" in discovery:
        descriptions = discovery[["phecode", "phecode_string"]].dropna()
        lookup.update(_description_lookup(descriptions))
    lookup.update(_snv_phewas_phecode_strings())
    return lookup


def _description_lookup(frame: pd.DataFrame) -> dict[str, str]:
    if "phecode" not in frame:
        return {}
    description = next(
        (
            column
            for column in (
                "phecode_string",
                "phecode_description",
                "description",
                "meaning",
            )
            if column in frame
        ),
        None,
    )
    if description is None:
        return {}
    values = frame[["phecode", description]].copy()
    values["phecode"] = values["phecode"].fillna("").astype(str).str.strip()
    values[description] = values[description].fillna("").astype(str).str.strip()
    values = values[
        values["phecode"].ne("")
        & values[description].ne("")
        & values[description].str.lower().ne("nan")
    ].drop_duplicates("phecode")
    return dict(zip(values["phecode"], values[description]))


def _snv_phewas_phecode_strings() -> dict[str, str]:
    """Load PhecodeX labels from the same tables used by the SNV PheWAS."""
    lookup: dict[str, str] = {}
    for path in snv_phewas_source_paths():
        try:
            frame = pd.read_csv(path, sep="\t", dtype={"phecode": str})
        except (OSError, pd.errors.EmptyDataError):
            continue
        lookup.update(_description_lookup(frame))
    return lookup


def snv_phewas_source_paths() -> list[Path]:
    """Stage and return the SNV-PheWAS label sources."""
    sources = (
        (config.PHEWAS_META_RESULTS_TSV, config.PHEWAS_META_RESULTS_GS),
        (config.PHEWAS_DISCOVERY_TSV, config.PHEWAS_DISCOVERY_GS),
    )
    paths = []
    for local, remote in sources:
        path = figure4_meta._stage_gcs_file(local, remote, force=False)
        if path.is_file() and path.stat().st_size > 0:
            paths.append(path)
    return paths


def annotate_meta_analysis(
    meta: pd.DataFrame,
    regions: pd.DataFrame,
    genes: pd.DataFrame,
    phecode_strings: dict[str, str],
) -> pd.DataFrame:
    """Add old-table phenotype and outlier-region gene annotations."""
    if "region_id" not in meta or "phecode" not in meta:
        raise ValueError("Meta-analysis table requires region_id and phecode columns")
    result = meta.copy()
    region_ids = result["region_id"].astype(str)
    phecodes = result["phecode"].astype(str)
    gene_lookup = admixture_summary._genes_in_regions(regions, genes)
    annotated_genes = region_ids.map(gene_lookup).fillna("")
    if "genes_in_region" in result:
        existing = result["genes_in_region"].fillna("").astype(str)
        result["genes_in_region"] = existing.where(
            existing.str.strip().ne(""), annotated_genes
        )
    else:
        result["genes_in_region"] = annotated_genes
    annotated_descriptions = phecodes.map(phecode_strings).fillna("")
    if "phecode_string" in result:
        existing = result["phecode_string"].fillna("").astype(str)
        result["phecode_string"] = existing.where(
            existing.str.strip().ne(""), annotated_descriptions
        )
    else:
        result["phecode_string"] = annotated_descriptions

    ordered = [
        column
        for column in result.columns
        if column not in META_ANNOTATION_COLUMNS
    ]
    phecode_index = ordered.index("phecode")
    ordered[phecode_index:phecode_index] = ["genes_in_region"]
    phecode_index = ordered.index("phecode")
    ordered[phecode_index + 1:phecode_index + 1] = ["phecode_string"]
    return result[ordered]


def annotate_meta_analysis_file() -> Path:
    """Persist annotations from the SNV-PheWAS Phecode and gene-track sources."""
    meta_path = admixture_phewas.analysis_dir() / "meta_analysis_results.tsv"
    regions = _read_tsv(
        admixture_lai.analysis_dir() / "outlier_regions.tsv",
        dtype={"region_id": str},
    )
    discovery = _read_tsv(
        admixture_phewas.analysis_dir() / "phewas.discovery.tsv.gz",
        dtype={"region_id": str, "phecode": str},
    )
    meta = _read_tsv(meta_path, dtype={"region_id": str, "phecode": str})
    genes = admixture_finemap._gene_table(force=False)
    annotated = annotate_meta_analysis(
        meta, regions, genes, _phecode_strings(meta, discovery)
    )
    missing_descriptions = annotated["phecode_string"].fillna("").astype(str).str.strip().eq("")
    if missing_descriptions.any():
        missing = sorted(
            annotated.loc[missing_descriptions, "phecode"].astype(str).unique()
        )
        raise ValueError(
            "Meta-analysis PhecodeX descriptions are missing for: "
            + ", ".join(missing)
        )
    temporary = meta_path.with_suffix(meta_path.suffix + ".tmp")
    annotated.to_csv(temporary, sep="\t", index=False)
    temporary.replace(meta_path)
    return meta_path


def build_tables() -> dict[str, pd.DataFrame]:
    """Build all-outlier, fitted-PLRS, and combined PheWAS result tables."""
    regions = _read_tsv(
        admixture_lai.analysis_dir() / "outlier_regions.tsv",
        dtype={"region_id": str},
    )
    models = _read_tsv(
        admixture_plrs.analysis_dir() / "plrs_models_summary.tsv",
        dtype={"region_id": str},
    )
    discovery = _read_tsv(
        admixture_phewas.analysis_dir() / "phewas.discovery.tsv.gz",
        dtype={"region_id": str, "phecode": str},
    )
    meta = _read_tsv(
        admixture_phewas.analysis_dir() / "meta_analysis_results.tsv",
        dtype={"region_id": str, "phecode": str},
    )
    genes = admixture_finemap._gene_table(force=False)

    outliers = _with_region_genes(regions, genes)
    outliers = outliers[_ordered_columns(outliers, OUTLIER_LEADING_COLUMNS)]
    outliers = outliers.sort_values(
        ["chrom", "start"],
        key=lambda values: (
            values.str.replace("chr", "", regex=False).astype(int)
            if values.name == "chrom"
            else values
        ),
    ).reset_index(drop=True)

    model_columns = [
        "region_id",
        *[column for column in PLRS_METRIC_COLUMNS if column in models.columns],
    ]
    plrs = outliers.merge(
        models[model_columns], on="region_id", how="inner", validate="one_to_one"
    )
    plrs_leading = [*OUTLIER_LEADING_COLUMNS, *PLRS_METRIC_COLUMNS]
    plrs = plrs[
        _ordered_columns(plrs, [column for column in plrs_leading if column in plrs])
    ]

    phecode_strings = _phecode_strings(meta, discovery)
    meta = annotate_meta_analysis(meta, regions, genes, phecode_strings)
    phewas = admixture_summary._meta_summary_table(meta, regions, genes, phecode_strings)
    phewas = phewas[PHEWAS_COLUMNS].copy()
    for column in ("same_direction", "rep_nominal", "meta_significant"):
        phewas[column] = phewas[column].map({True: "TRUE", False: "FALSE"})
    for column in ("start", "end", "n_disc", "n_rep"):
        numeric = pd.to_numeric(phewas[column], errors="coerce")
        phewas[column] = numeric.map(
            lambda value: str(int(value)) if pd.notna(value) else "Not replicable"
        )
    for column in (
        "beta_rep",
        "se_rep",
        "p_rep",
        "beta_meta",
        "se_meta",
        "p_meta",
    ):
        phewas[column] = phewas[column].where(
            pd.to_numeric(phewas[column], errors="coerce").notna(),
            "Not replicable",
        )
    return {"outliers": outliers, "plrs": plrs, "phewas": phewas}


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, na_rep: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, na_rep=na_rep, float_format="%.10g")
    temporary.replace(path)


def write_tables() -> dict[str, Path]:
    tables = build_tables()
    paths = output_paths()
    _write_csv_atomic(tables["outliers"], paths["outliers"])
    _write_csv_atomic(tables["plrs"], paths["plrs"])
    _write_csv_atomic(tables["phewas"], paths["phewas"], na_rep="Not replicable")
    print(
        "[figure2_1025][tables] "
        f"outliers={len(tables['outliers']):,}; PLRS={len(tables['plrs']):,}; "
        f"PheWAS={len(tables['phewas']):,}; directory={output_dir()}",
        flush=True,
    )
    return paths
