"""Central configuration: repo paths and figure/panel metadata.

Everything downstream (data scripts, plotting scripts, notebooks) resolves its
paths through here so the layout can move without touching call sites.

Directory contract (see README):
    data/      raw, immutable source data (read-only; never written to)
    analysis/  regenerable intermediates written by *_data.py scripts
    figures/   individual panels + composite figures written by *_plots.py
    scripts/   importable python modules the notebooks call
    notebooks/ Figure_<N>.ipynb orchestrators
"""
from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a repo marker is found.

    Works whether called from a notebook (cwd == notebooks/) or a script.
    """
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "requirements.txt").exists():
            return candidate
    # Fallback: two levels up from this file (scripts/common/config.py -> repo).
    return Path(__file__).resolve().parents[2]


REPO_ROOT: Path = find_repo_root()

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis"
FIGURES_DIR: Path = REPO_ROOT / "figures"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
ASSETS_DIR: Path = REPO_ROOT / "assets"

# Vendored static reference assets (tracked in git).
US_STATES_GEOJSON: Path = ASSETS_DIR / "us_states.geojson"
WORLD_COUNTRIES_GEOJSON: Path = ASSETS_DIR / "world_countries.geojson"

# Figure 3A iHS selection scan: per-20kb-window max |iHS|, empirical rank, simulation
# p-value and gene annotations across the 22 autosomes (gzipped TSV, tracked).
IHS_WINDOWS_TSV: Path = ASSETS_DIR / "figure2" / "ihs_windows.acaf_resolved.annotated.sim_pval.tsv.gz"
# Merged significant iHS regions (sig p, SV/SNV-linked): precise per-region peak
# position + iHS + gene annotations, used to place the Figure 3A gene labels.
IHS_REGIONS_TSV: Path = ASSETS_DIR / "figure2" / "ihs_significant_regions.linked.tsv.gz"
# Figure 3A's manuscript-table S3 subset: 17 selection-hit rows that are either
# themselves SVs or linked to an SV. Some rows share one SV (14 unique alleles),
# so the plot circles all 17 hits but labels each SV locus only once.
IHS_SV_HITS_TSV: Path = ASSETS_DIR / "figure2" / "selection_sv_hits.s3.tsv"

# Figure 3 selection-locus EHH + iHS/pi showcase loci (Workbench-gated). The phased
# VCFs, pre-computed IHS output, meta-analysis TSV and reference files live under
# the migration workspace mount. This is anchored at $HOME (``~/workspace/...``)
# and expanded to an absolute path so it resolves regardless of the notebook's
# working directory. Override via env var.
MIGRATION_WORKSPACE: str = os.path.expanduser(
    os.environ.get("AOU_MIGRATION_WORKSPACE", "~/workspace/rw-migration-aou-rw-a3e31f4d")
)


def migration_path(relative: str) -> str:
    """Path to a resource under the migration workspace mount (Figure 3 selection-locus inputs)."""
    return os.path.join(MIGRATION_WORKSPACE, relative)


def home_path(relative: str) -> str:
    """Absolute path under $HOME for a Workbench working-dir resource."""
    return os.path.expanduser(os.path.join("~", relative))


# Figure 4 inputs are persisted in the Researcher Workbench workspace bucket. The
# This study's migration bucket is the default source. Use the repo-specific
# ``AOU_WORKSPACE_BUCKET_GS`` override only when deliberately pointing at a
# different copy; a generic Workbench ``WORKSPACE_BUCKET`` environment value
# must not silently redirect these Figure 4 inputs.
def _normalize_gs_bucket(value: str | None) -> str:
    value = (value or "").strip().rstrip("/")
    if not value:
        return ""
    return value if value.startswith("gs://") else f"gs://{value.lstrip('/')}"


DEFAULT_WORKSPACE_BUCKET_GS: str = "gs://rw-migration-aou-rw-a3e31f4d"
WORKSPACE_BUCKET_GS: str = _normalize_gs_bucket(
    os.environ.get("AOU_WORKSPACE_BUCKET_GS", DEFAULT_WORKSPACE_BUCKET_GS)
)


def workspace_bucket_path(relative: str) -> str:
    """Return ``relative`` beneath the configured study ``gs://`` bucket."""
    if not WORKSPACE_BUCKET_GS:
        return ""
    return f"{WORKSPACE_BUCKET_GS}/{relative.lstrip('/')}"


# Figure 3 enrichment: permutation enrichment of the iHS-selected windows
# for LD-linked GWAS / eQTL / sQTL overlap. The cached products
# (enrichment_results/permutation_pool_linkage.tsv, permutation_results.tsv) live
# in the study workspace bucket alongside the other inputs, so this defaults to
# WORKSPACE_BUCKET_GS. Once fetched and verified, the linkage cache is tracked
# under assets/ for panel rendering. The source pipeline (LD from
# the SV_biallelic panel VCFs + ld_cache, GWAS Catalog + MAGE QTL scan) only runs
# to regenerate that cache. Override the source bucket with AOU_FIG2_ENRICH_BUCKET.
ENRICHMENT_BUCKET_GS: str = _normalize_gs_bucket(
    os.environ.get("AOU_FIG2_ENRICH_BUCKET", WORKSPACE_BUCKET_GS)
)


def enrichment_bucket_path(relative: str) -> str:
    """Return ``relative`` beneath the Figure 3 enrichment source bucket."""
    if not ENRICHMENT_BUCKET_GS:
        return ""
    return f"{ENRICHMENT_BUCKET_GS}/{relative.lstrip('/')}"


# Tracked cache (assets/): the per-window LD-linkage booleans (the permutation
# "hitlist" linkage) and the summary results table. With IHS_WINDOWS_TSV and the
# fixed permutation seed these reproduce the observed rates and the null
# distribution exactly, so the panel needs no bucket access to replot.
IHS_ENRICHMENT_LINKAGE_TSV: Path = (
    ASSETS_DIR / "figure2" / "enrichment_permutation_pool_linkage.tsv.gz"
)
IHS_ENRICHMENT_RESULTS_TSV: Path = (
    ASSETS_DIR / "figure2" / "enrichment_permutation_results.tsv"
)


def phewas_sv_path(filename: str) -> str:
    """Workspace-bucket path under ``phewas_sv_v5/``.

    Example: ``phewas_sv_path('10_100434627_C_T_cohort.tsv')`` resolves to
    ``gs://rw-migration-aou-rw-a3e31f4d/phewas_sv_v5/10_100434627_C_T_cohort.tsv``
    with the default bucket.
    """
    return workspace_bucket_path(f"phewas_sv_v5/{filename}")


# Local staging paths remain $HOME-anchored and env-overridable. Their paired GCS
# constants are the authoritative Workbench sources and are copied locally by
# the Figure 4 loaders before pandas/matplotlib consume them.
PHEWAS_META_RESULTS_TSV: str = os.environ.get(
    "AOU_META_RESULTS_TSV", home_path("phewas_meta/meta_analysis_results_genic_meff.tsv"))
PHEWAS_META_RESULTS_GS: str = os.environ.get(
    "AOU_META_RESULTS_GS",
    workspace_bucket_path("phewas_meta/meta_analysis_results_genic_meff.tsv"),
)
PHEWAS_DISCOVERY_TSV: str = os.environ.get(
    "AOU_DISCOVERY_TSV", home_path("phewas_sv_v5/phewas_all_results_svonly_all_p0.01.tsv"))
PHEWAS_DISCOVERY_GS: str = os.environ.get(
    "AOU_DISCOVERY_GS",
    phewas_sv_path("phewas_all_results_svonly_all_p0.01.tsv"),
)

# Full MAC- and convergence-filtered PheWAS frame (every tested variant x
# phecode cell, not just the p<=0.01 tail). This is the source for the Figure 4A
# volcano of nominally significant hits (p<=1e-3); the discovery p0.01 table
# above only carries the suggestive tail and cannot draw the full cloud. Written
# by the PheWAS notebook as ``phewas_all_filtered.mac20.converged.tsv``.
PHEWAS_FILTERED_TSV: str = os.environ.get(
    "AOU_PHEWAS_FILTERED_TSV",
    home_path("phewas_sv_v5/phewas_all_filtered.mac20.converged.tsv"))
PHEWAS_FILTERED_GS: str = os.environ.get(
    "AOU_PHEWAS_FILTERED_GS",
    phewas_sv_path("phewas_all_filtered.mac20.converged.tsv"),
)

# The 963 genic merged iHS sentinel loci define the Figure 4 PheWAS universe.
# Keep this paired with S9's ``AOU_S9_SENTINELS_TSV`` input so the volcano and
# MAF/effect panels cannot silently use different variant sets.
PHEWAS_GENIC_SENTINELS_TSV: str = os.environ.get(
    "AOU_PHEWAS_GENIC_SENTINELS_TSV",
    os.environ.get(
        "AOU_S9_SENTINELS_TSV",
        home_path(
            "ihs_output_v4/"
            "all_windows.acaf_resolved.annotated.with_sim_pval.merged_regions.tsv"
        ),
    ),
)
PHEWAS_GENIC_SENTINELS_GS: str = os.environ.get(
    "AOU_PHEWAS_GENIC_SENTINELS_GS",
    workspace_bucket_path(
        "ihs_output_v4/"
        "all_windows.acaf_resolved.annotated.with_sim_pval.merged_regions.tsv"
    ),
)

# The attached SuSiE notebook writes discovery-cohort TSVs under this
# legacy-named directory. Figure 4 redraws every locus from the verified TSVs.
SUSIE_FINEMAP_DIR: str = os.environ.get(
    "AOU_SUSIE_DIR", home_path("susie_finemap_meta_dual"))
SUSIE_FINEMAP_GS_DIR: str = os.environ.get(
    "AOU_SUSIE_GS_DIR", workspace_bucket_path("susie_finemap_meta_dual"))

# admixture reference starts from the marker-level FLARE outputs in ``lai_masked`` and
# stages every other discovery/replication input beneath the same normalized
# migration-bucket root.  Local paths are deliberately $HOME-anchored so the
# Workbench notebook can be launched from any directory.
ADMIXTURE_WORK_DIR: str = os.environ.get(
    "AOU_FIG4_WORK_DIR", home_path("figure4_local_ancestry"))
LAI_MASKED_DIR: str = os.environ.get(
    "AOU_LAI_DIR", home_path("lai_masked"))
LAI_MASKED_GS_DIR: str = os.environ.get(
    "AOU_LAI_GS_DIR", workspace_bucket_path("lai_masked"))

ADMIXTURE_DISCOVERY_VCF_GS: str = os.environ.get(
    "AOU_FIG4_DISCOVERY_VCF_GS",
    workspace_bucket_path(
        "SV_10k/AoU12k.vs.AoU-HPRC-v1.1b.AF_gt_pt005."
        "25kb_windows.0d0e0eb4.kage.glimpse.collisionless.merged.vcf.gz"
    ),
)
# Figure 2 SV positions and allele-frequency provenance use the phased
# 1,025-person SNV/SV panel matching the LAI cohort. This is deliberately
# distinct from the AoU12k merged VCF used for association fine-mapping above.
FIGURE2_SV_VCF_PATTERN: str = os.environ.get(
    "AOU_FIG5_SV_VCF_PATTERN",
    migration_path("SV_biallelic/{chrom}.biallelic.vcf.gz"),
)
FIGURE2_SV_VCF_GS_PATTERN: str = os.environ.get(
    "AOU_FIG5_SV_VCF_GS_PATTERN",
    workspace_bucket_path("SV_biallelic/{chrom}.biallelic.vcf.gz"),
)
ADMIXTURE_DISCOVERY_COV_GS: str = os.environ.get(
    "AOU_FIG4_DISCOVERY_COV_GS", phewas_sv_path("covariates_base.tsv"))
ADMIXTURE_DISCOVERY_PCA_GS: str = os.environ.get(
    "AOU_FIG4_DISCOVERY_PCA_GS", phewas_sv_path("pca_scores_afr.tsv.bgz"))
ADMIXTURE_DISCOVERY_PHE_GS: str = os.environ.get(
    "AOU_FIG4_DISCOVERY_PHE_GS", phewas_sv_path("aou_phecodeX_counts.tsv"))

ADMIXTURE_REPLICATION_COV_GS: str = os.environ.get(
    "AOU_FIG4_REPLICATION_COV_GS",
    workspace_bucket_path("phewas_acaf_replication/covariates_base.tsv"),
)
ADMIXTURE_REPLICATION_PCA_GS: str = os.environ.get(
    "AOU_FIG4_REPLICATION_PCA_GS",
    workspace_bucket_path("phewas_acaf_replication/pca_scores_afr.tsv.bgz"),
)
ADMIXTURE_REPLICATION_PHE_GS: str = os.environ.get(
    "AOU_FIG4_REPLICATION_PHE_GS",
    workspace_bucket_path("phewas_acaf_replication/aou_phecodeX_counts.tsv"),
)
ADMIXTURE_REPLICATION_SHARD_INDEX_GS: str = os.environ.get(
    "AOU_FIG4_REPLICATION_SHARD_INDEX_GS",
    workspace_bucket_path("phewas_acaf_replication/acaf_shard_index.tsv"),
)
ADMIXTURE_ACAF_GS_DIR: str = os.environ.get(
    "AOU_FIG4_ACAF_GS_DIR",
    f"{os.environ.get('AOU_CONTROLLED_BUCKET', 'gs://vwb-aou-datasets-controlled')}"
    "/v8/wgs/short_read/snpindel/"
    "acaf_threshold/vcf",
)
ADMIXTURE_HARDMASK_GS: str = os.environ.get(
    "AOU_FIG4_HARDMASK_GS", workspace_bucket_path("hardmask.hg38.v4.over99.bed"))
ADMIXTURE_GTF_GS: str = os.environ.get(
    "AOU_FIG4_GTF_GS", workspace_bucket_path("hg38.refGene.gtf"))
ADMIXTURE_MAGE_EQTL_GS: str = os.environ.get(
    "AOU_FIG4_MAGE_EQTL_GS",
    workspace_bucket_path(
        "ihs_output_v4/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt.gz"
    ),
)
ADMIXTURE_MAGE_SQTL_GS: str = os.environ.get(
    "AOU_FIG4_MAGE_SQTL_GS",
    workspace_bucket_path(
        "ihs_output_v4/sQTL_finemapping.geneMerged.leadVariant.MAGE.v1.0.txt.gz"
    ),
)
ADMIXTURE_GWAS_GS: str = os.environ.get(
    "AOU_FIG4_GWAS_GS",
    workspace_bucket_path("GWAS_catalog/gwas_catalog_associations.tsv"),
)

# Fig 1E demographic inference (dadi 3D-SFS fit). The observed spectrum and
# demes best-fit model are tracked; panel E is drawn directly during Figure 1
# rendering rather than read from a generated asset.
DEMOGRAPHY_DIR: Path = ASSETS_DIR / "demography"
DEMOGRAPHY_SFS: Path = DEMOGRAPHY_DIR / "3pop_AFR_NFE_AMR_combined.fs"
DEMOGRAPHY_DEMES_YAML: Path = DEMOGRAPHY_DIR / "3pop_2mig.yaml"
# 100 bootstrap best-fit parameter sets (one dadi fit per line: loglik, 8 model
# params, misid, theta). Drives the Fig 1F/1G bootstrap scatters + correlogram.
DEMOGRAPHY_BOOTSTRAP: Path = DEMOGRAPHY_DIR / "runs_3d_demo_fits_pulse_admix.txt.gz"
# Exactly 100 bootstrap joint SFSs used to recompute the C1-vs-C2 Godambe LRT
# adjustment. The tarball contains fs/0_AFR_EUR_AOU.fs through fs/99_....fs.
DEMOGRAPHY_BOOTSTRAP_SFS_ARCHIVE: Path = (
    DEMOGRAPHY_DIR / "s1_demography_bootstrap_sfs.tar.gz"
)
# AoU-marginal observed vs. best-fit-model SFS + residuals (precomputed with dadi
# from the .fs data + best-fit params; tracked so Fig 1F needs no dadi at plot
# time). Columns: allele_count, data, model, poisson_resid.
DEMOGRAPHY_AOU_FIT: Path = DEMOGRAPHY_DIR / "aou_sfs_fit.csv"
# Supplementary Figure S1 fixed-result model comparison.  These aggregate
# tables are regenerated from ``DEMOGRAPHY_SFS`` and the saved parameter sets;
# no optimization or participant-level data are involved.
DEMOGRAPHY_S1_MODEL_SUMMARY: Path = (
    DEMOGRAPHY_DIR / "s1_demography_model_summary.tsv"
)
DEMOGRAPHY_S1_PARAMETERS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_parameters.tsv"
)
DEMOGRAPHY_S1_WIDE_PARAMETERS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_nested_parameters_wide.tsv"
)
DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_nested_parameters_physical.tsv"
)
DEMOGRAPHY_S1_HARMONIZED_PARAMETERS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_harmonized_parameters.tsv"
)
DEMOGRAPHY_S1_COMPARISONS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_model_comparisons.tsv"
)
DEMOGRAPHY_S1_CANDIDATE_CHECKS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_candidate_checks.tsv"
)
DEMOGRAPHY_S1_MARGINAL_SFS: Path = (
    DEMOGRAPHY_DIR / "s1_demography_sfs_marginals.tsv.gz"
)
DEMOGRAPHY_S1_PROVENANCE: Path = (
    DEMOGRAPHY_DIR / "s1_demography_provenance.json"
)


# --------------------------------------------------------------------------
# All of Us controlled-tier storage (post-VWB migration)
# --------------------------------------------------------------------------
# The VWB (Verily Workbench) migration renamed the controlled dataset bucket:
#   OLD: gs://fc-aou-datasets-controlled/...
#   NEW: gs://vwb-aou-datasets-controlled/...
# Override with the AOU_CONTROLLED_BUCKET env var if the name changes again.
AOU_CONTROLLED_BUCKET: str = os.environ.get(
    "AOU_CONTROLLED_BUCKET", "gs://vwb-aou-datasets-controlled"
)


def _ancestry_aux(filename: str, version: str) -> str:
    """gs:// path to a file under the controlled short-read ancestry aux dir."""
    return (
        f"{AOU_CONTROLLED_BUCKET}/{version}/wgs/short_read/snpindel/aux/"
        f"ancestry/{filename}"
    )


def ancestry_preds_path(version: str = "v7") -> str:
    """gs:// path to ancestry_preds.tsv (Fig 1A PCA input; lives under v7)."""
    return _ancestry_aux("ancestry_preds.tsv", version)


def training_pca_path(version: str = "v8") -> str:
    """gs:// path to training_pca.tsv (HGDP+1KG training scores; under v8)."""
    return _ancestry_aux("training_pca.tsv", version)


def eigenvalues_path(version: str = "v8") -> str:
    """gs:// path to eigenvalues.txt (matches training_pca; under v8)."""
    return _ancestry_aux("eigenvalues.txt", version)


# Resolved defaults for the ancestry aux files used by Fig 1B/1D.
ANCESTRY_PREDS_GS: str = ancestry_preds_path()   # v7 (per-sample PC scores)
TRAINING_PCA_GS: str = training_pca_path()        # v8 (HGDP+1KG training scores)
EIGENVALUES_GS: str = eigenvalues_path()          # v8 (matches training)

# gnomAD HGDP+1KG geographic metadata (public bucket) for Fig 1D IDW anchoring.
HGDP_1KG_META_TSV: str = (
    "gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/"
    "hgdp_1kg_v2/metadata_and_qc/gnomad_meta_updated.tsv"
)

# Fig 1D IDW parameters.
IDW_N_PCS: int = 16      # PCs used for PC-space distance
IDW_K: int = 20          # k nearest reference neighbours
# Phase 1 Figure 1D is intentionally limited to 1KG AFR/EUR source pops.
AFR_EUR_EXTENT: tuple[float, float, float, float] = (-25.0, 65.0, -37.0, 72.0)
# Retained for other world-map utilities; Figure 1D itself uses lon/lat over the
# AFR/EUR extent above.
WORLD_ROBINSON_CRS: str = "ESRI:54030"


# Publication names are separate from persisted working-repository caches.
# In particular, publication figure2 must never read the selection cache.
ANALYSIS_SOURCE_IDS = {
    "figure1": "figure1",
    "figure2": "figure5_v2",
    "figure2_sources": "figure5",
    "figure3": "figure2",
    "figure4": "figure3",
    "admixture_reference": "figure4",
}


def analysis_dir(figure_id: str) -> Path:
    """Per-figure intermediate directory, created on demand."""
    # Reuse established analysis locations after publication renumbering.
    d = ANALYSIS_DIR / ANALYSIS_SOURCE_IDS.get(figure_id, figure_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure_dir(figure_id: str) -> Path:
    """Per-figure output directory (holds panels/ and the composite)."""
    d = FIGURES_DIR / figure_id
    (d / "panels").mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Figure / panel specification
# --------------------------------------------------------------------------
# Individual panel descriptions and standalone rendering options.

FIGURES = {'figure1': {'number': 1,
             'title': 'Demographic history of AoU individuals',
             'standalone_sizes': {'D': (7.0, 8.4), 'E': (10.5, 5.2)},
             'standalone_title_pad': {'A': 34, 'B': 48, 'C': 32},
             'panels': {'A': 'Sampling locations of AoU individuals',
                        'B': 'PCA of AoU individuals vs. v8 cohort by genetic ancestry',
                        'C': 'Unsupervised ADMIXTURE (k=2) of AoU individuals',
                        'D': 'IDW origins (Phase 1 vs 1KG AFR/EUR source populations)',
                        'E': 'Best-fit joint-SFS demographic model (AoU/AFR/EUR)',
                        'F': 'AoU-marginal SFS: data vs. model + residuals',
                        'G': 'Bootstrap uncertainty: initial vs. final EUR size',
                        'H': 'Bootstrap admixture fraction'},
             'axis_limits': {'D': {'ylim': (-4.0, 66.0)}}},
 'figure3': {'number': 3,
             'title': 'Genome-wide selection scan and candidate loci',
             'standalone_sizes': {'A': (8.4, 4.4),
                                  'B': (6.0, 4.0),
                                  'C': (6.0, 4.0),
                                  'D': (6.0, 4.0),
                                  'E': (6.0, 4.0),
                                  'F': (6.0, 4.0),
                                  'G': (6.0, 4.0)},
             'standalone_title_skip': ('B', 'C', 'D', 'E', 'F', 'G'),
             'standalone_title_pad': {'A': 46},
             'panels': {'A': 'Genome-wide selection scan',
                        'B': 'LCT',
                        'C': 'SLC24A5',
                        'D': 'ANKRD31',
                        'E': 'Enrichment'}},
 'figure4': {'number': 4,
             'title': 'PheWAS of selected loci',
             'standalone_title_skip': ('A', 'B', 'C', 'D', 'E'),
             'panels': {'A': 'PheWAS discovery',
                        'B': 'MAF and effect size',
                        'C': 'Discovery and meta-analysis',
                        'D': 'Meta-significant associations'}},
 'admixture_reference': {'number': 'admixture_reference',
                         'title': 'Local ancestry across the AoU cohort',
                         'standalone_sizes': {'A': (6.0, 5.5),
                                              'B': (6.0, 3.2),
                                              'C': (3.2, 3.0),
                                              'D': (3.2, 3.0),
                                              'E': (3.2, 3.2)},
                         'panels': {'A': 'Admixture tracts for a representative AoU individual',
                                    'B': 'EUR ancestry fraction across the genome',
                                    'C': 'Highlighted outlier region with 50 kb flanks',
                                    'D': 'Regional score-test p-values',
                                    'E': 'Held-out PLRS calibration for the highlighted outlier '
                                         'region'}},
 'figure2_sources': {'number': '2_sources',
                     'title': 'Admixture mapping across the AoU cohort',
                     'standalone_sizes': {'A': (6.0, 3.2),
                                          'B': (3.2, 3.2),
                                          'C': (3.2, 3.2),
                                          'D': (3.2, 3.2),
                                          'E': (4.8, 3.4),
                                          'F': (4.8, 4.2)},
                     'panels': {'A': 'EUR ancestry fraction across the genome',
                                'B': 'Held-out PLRS calibration for the highlighted outlier region',
                                'C': 'Held-out PLRS performance across regional models',
                                'D': 'Regional PLRS model size and held-out performance',
                                'E': 'Highlighted outlier region with SV positions at '
                                     'nearest-marker LAI',
                                'F': 'Regional score-test p-values, SVs, and exon/intron gene '
                                     'track'}},
 'figure2': {'number': 2,
             'title': 'Admixture mapping across the AoU cohort',
             'standalone_sizes': {'A': (6.0, 3.2),
                                  'B': (3.2, 3.2),
                                  'C': (3.2, 3.2),
                                  'D': (3.2, 3.2),
                                  'E': (4.8, 3.4),
                                  'F': (4.8, 4.2)},
             'panels': {'A': 'EUR ancestry fraction across the genome',
                        'B': 'Held-out PLRS calibration for the highlighted outlier region',
                        'C': 'Held-out PLRS performance across regional models',
                        'D': 'Regional PLRS model size and held-out performance',
                        'E': 'Highlighted outlier region with SV positions at nearest-marker LAI',
                        'F': 'Regional score-test p-values, SVs, and exon/intron gene track'},
             'standalone_title_skip': ('A', 'B', 'C', 'D', 'E', 'F'),
             'panel_kwargs': {'F': {'colorbar_side': 'left'}}}}

# Vector and raster formats emitted for each panel.
FIGURE_FORMATS: tuple[str, ...] = ("pdf", "png")


def figure_ids() -> list[str]:
    return list(FIGURES)
