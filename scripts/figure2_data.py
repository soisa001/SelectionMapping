"""Figure 2 data facade for the local-ancestry admixture-mapping workflow.

The controlled-data products retain their established ``analysis/admixture_reference``
cache and bucket contract.  Renumbering the manuscript figure must not force an
expensive Workbench recomputation or duplicate controlled-data products.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import admixture_data, admixture_finemap, admixture_lai, admixture_plrs, figure2_sv

LOCUS_FLANK_BP = 100_000


def locus_window(region: dict) -> tuple[int, int]:
    """Return the common admixture/fine-map display window."""
    return (
        max(1, int(region["start"]) - LOCUS_FLANK_BP),
        int(region["end"]) + LOCUS_FLANK_BP,
    )


def _ensure_region_scan(chrom: int) -> tuple[np.ndarray, np.ndarray, int]:
    scan_path = admixture_lai._scan_cache(chrom)
    if not scan_path.exists():
        lai_vcf = admixture_lai.stage_lai_chromosome(chrom, force=False)
        mapping = admixture_lai.parse_ancestry_header(lai_vcf)
        eur_index, _, _, _ = admixture_lai.ancestry_indices(mapping)
        pos, fraction, n_samples = admixture_lai.scan_chromosome(lai_vcf, eur_index)
        temporary = scan_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            pos=pos,
            eur_frac=fraction,
            n_samples=np.asarray([n_samples], dtype=np.int64),
        )
        temporary.replace(scan_path)
        print(
            f"[figure2_sources] built chr{chrom} marker scan for the 100 kb locus view",
            flush=True,
        )
    return admixture_lai._load_chromosome_scan(scan_path)


def _expanded_region_zoom(region: dict) -> pd.DataFrame:
    window_start, window_end = locus_window(region)
    pos, fraction, _ = _ensure_region_scan(int(region["chrom"]))
    keep = (pos >= window_start) & (pos <= window_end)
    stats = json.loads(
        (admixture_lai.analysis_dir() / "scan_stats.json").read_text(encoding="utf-8")
    )
    zoom = pd.DataFrame({
        "chrom": int(region["chrom"]),
        "pos": pos[keep],
        "eur_frac": fraction[keep],
    })
    zoom["eur_z"] = (
        zoom["eur_frac"] - float(stats["mean_eur"])
    ) / float(stats["sd_eur"])
    zoom["in_region"] = zoom["pos"].between(
        int(region["start"]), int(region["end"]), inclusive="both"
    )
    return zoom


def prepare_locus_outputs(*, force: bool = False) -> dict:
    """Prepare only the expanded Figure 2 locus products on Workbench."""
    if not admixture_lai.real_outputs_available():
        raise FileNotFoundError(f"Required LAI products are missing: {admixture_lai.analysis_dir()}")
    region = json.loads(
        (admixture_lai.analysis_dir() / "display_region.json").read_text(
            encoding="utf-8"
        )
    )
    _expanded_region_zoom(region)
    finemap, _ = admixture_data._real_finemap(region)
    if force or finemap.empty:
        print(
            "[figure2_sources] the displayed fine-map cache is not on the 100 kb "
            "contract; regenerating only this region",
            flush=True,
        )
        finemap, _ = admixture_finemap.ensure_region_cache(region, force=force)
    if finemap.empty:
        raise FileNotFoundError(
            "No current 100 kb fine-map cache is available for the displayed "
            f"admixture region {region['region_id']}"
        )
    admixture_finemap._gene_table(force=force)
    window_start, window_end = locus_window(region)
    regional_svs = figure2_sv.ensure_cache(
        region, window_start, window_end, force=force
    )
    association_audit = figure2_sv.match_finemap_svs(regional_svs, finemap)
    audit_path = (
        admixture_finemap.analysis_dir() / "display_region_sv_association.tsv"
    )
    association_audit.to_csv(audit_path, sep="\t", index=False)
    if not association_audit.empty:
        columns = [
            "variant_id", "sv_type", "pos", "n_carrier_haplotypes",
            "alt_allele_count", "called_allele_count", "alt_allele_frequency",
            "n_evaluable_haplotypes", "n_eur_haplotypes",
            "n_afr_haplotypes", "n_other_haplotypes",
            "overall_eur_frac", "nearest_lai_distance_bp",
            "exact_finemap_match", "beta_score",
            "p_score", "neg_log10_p", "r2_to_min_p",
        ]
        print(
            "[figure2_sources][sv-finemap] exact SV/type association audit:\n"
            + association_audit[columns].to_string(index=False),
            flush=True,
        )
    return {
        "status": "ready",
        "region_id": region["region_id"],
        "window_start": window_start,
        "window_end": window_end,
        "n_svs": len(regional_svs),
    }


def _plrs_models() -> pd.DataFrame:
    path = admixture_plrs.analysis_dir() / "plrs_models_summary.tsv"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required Figure 2 PLRS model summary is missing: {path}")
    models = pd.read_csv(path, sep="\t", dtype={"region_id": str})
    required = {"region_id", "n_snvs", "r2_holdout"}
    missing = required.difference(models.columns)
    if missing:
        raise ValueError(
            f"Figure 2 PLRS model summary lacks columns {sorted(missing)}: {path}"
        )
    return models


def build(clean_run_data: bool = False) -> dict:
    """Load the established local-ancestry products for Figure 2."""
    data = admixture_data.build(clean_run_data=clean_run_data)
    region = data["display_region"]
    data["locus_window"] = locus_window(region)
    prepare_locus_outputs(force=clean_run_data)
    data["region_zoom"] = _expanded_region_zoom(region)
    data["finemap"], data["finemap_hit"] = admixture_data._real_finemap(region)
    data["finemap_genes"] = admixture_finemap._gene_table(force=False)
    data["regional_svs"] = figure2_sv.match_finemap_svs(
        figure2_sv.load_cache(), data["finemap"]
    )
    data["plrs_models"] = _plrs_models()
    return data
