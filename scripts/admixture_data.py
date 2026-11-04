"""Load admixture reference LAI products from the Workbench analysis cache."""
from __future__ import annotations

import json

import pandas as pd

from . import admixture_finemap, admixture_lai, admixture_phewas, admixture_plrs
from .common.genome import CHROM_SIZES

FIGURE_ID = "admixture_reference"


def _true_flag(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _real_finemap(region: dict) -> tuple[pd.DataFrame, dict]:
    meta_path = admixture_phewas.analysis_dir() / "meta_analysis_results.tsv"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return pd.DataFrame(), {}
    try:
        meta = pd.read_csv(meta_path, sep="\t", dtype={"phecode": str})
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(), {}
    required = {"region_id", "phecode", "meta_significant", "p_meta"}
    if meta.empty or not required.issubset(meta.columns):
        return pd.DataFrame(), {}
    hits = meta[
        meta["region_id"].astype(str).eq(str(region["region_id"]))
        & _true_flag(meta["meta_significant"])
    ].copy()
    if hits.empty:
        return pd.DataFrame(), {}
    hits["p_meta"] = pd.to_numeric(hits["p_meta"], errors="coerce")
    hits = hits.sort_values("p_meta", na_position="last")
    for _, hit in hits.iterrows():
        table_path = (
            admixture_finemap.analysis_dir()
            / f"susie_{admixture_finemap.hit_tag(hit)}.tsv.gz"
        )
        frame = admixture_finemap.load_current_cache(table_path, hit)
        if frame is not None:
            return frame, hit.to_dict()
    return pd.DataFrame(), hits.iloc[0].to_dict()


def _real_plrs(region: dict) -> tuple[pd.DataFrame, dict]:
    holdout_path = admixture_plrs.analysis_dir() / "plrs_holdout_predictions.tsv.gz"
    summary_path = admixture_plrs.analysis_dir() / "plrs_models_summary.tsv"
    holdout = pd.DataFrame()
    metrics = {"region_id": region["region_id"]}
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path, sep="\t", dtype={"region_id": str})
        holdout = holdout[
            holdout["region_id"].eq(str(region["region_id"]))
        ].copy()
    if holdout.empty and admixture_plrs.model_path(str(region["region_id"])).exists():
        try:
            holdout = admixture_plrs.reconstruct_holdout_predictions(
                str(region["region_id"])
            )
            print(
                f"[admixture_reference] reconstructed held-out PLRS predictions for "
                f"{region['region_id']} from its cached model",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[admixture_reference] held-out PLRS panel unavailable for "
                f"{region['region_id']}: {exc}",
                flush=True,
            )
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep="\t", dtype={"region_id": str})
        match = summary[summary["region_id"].eq(str(region["region_id"]))]
        if not match.empty:
            metrics.update(match.iloc[0].to_dict())
    return holdout, metrics


def _real() -> dict:
    root = admixture_lai.analysis_dir()
    region = json.loads(
        (root / "display_region.json").read_text(encoding="utf-8")
    )
    finemap, finemap_hit = _real_finemap(region)
    gene_path = admixture_finemap.analysis_dir() / "refgene_transcripts.tsv.gz"
    finemap_genes = (
        pd.read_csv(gene_path, sep="\t") if gene_path.exists() else pd.DataFrame(
            columns=["chrom", "start", "end", "gene", "strand"]
        )
    )
    plrs_holdout, plrs_metrics = _real_plrs(region)
    return {
        "tracts": pd.read_csv(root / "karyogram_segments.tsv.gz", sep="\t"),
        "masks": pd.read_csv(root / "karyogram_masks.tsv.gz", sep="\t"),
        "genome_eur": pd.read_csv(root / "genome_scan_plot.tsv.gz", sep="\t"),
        "region_zoom": pd.read_csv(root / "region_zoom.tsv.gz", sep="\t"),
        "regions": pd.read_csv(root / "outlier_regions.tsv", sep="\t"),
        "stats": json.loads((root / "scan_stats.json").read_text(encoding="utf-8")),
        "display_region": region,
        "finemap": finemap,
        "finemap_hit": finemap_hit,
        "finemap_genes": finemap_genes,
        "plrs_holdout": plrs_holdout,
        "plrs_metrics": plrs_metrics,
        "chrom_sizes": CHROM_SIZES,
    }


def build(clean_run_data: bool = False) -> dict:
    del clean_run_data  # recomputation is orchestrated by recompute_admixture_reference
    return _real()
