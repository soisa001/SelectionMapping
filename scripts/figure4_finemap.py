"""Figure 3 SuSiE fine-mapping locus plots.

For every meta-significant SNV lead x phecode, SuSiE fine-maps a +/-500 kb
window. The canonical locus panel shows score-test p-values coloured by r^2 to
the minimum-p variant over a prominent Figure 2-style gene track.

    Fig 3B : the LOC100507053 lead locus, which fine-maps to **ADH1B**.
    Fig 3C : the STEAP2 lead locus, which fine-maps to **CFAP69**.
    Fig 3D : the chr1:35,126,025 hypertension locus near **ZMYM1**.
    Fig 3E : the chr2:85,632,712 breast lump/mass locus in **USP39**.

Data source: the attached discovery-cohort SuSiE notebook's
``susie_<tag>__10k_disc.tsv`` products. They are staged from
``config.SUSIE_FINEMAP_GS_DIR`` (the normalized ``gs://`` workspace bucket)
into ``config.SUSIE_FINEMAP_DIR`` and redrawn through one shared renderer so all
loci use current typography and gene-track styling. Gene tracks use the refGene
GTF when present (``figure2_ehh.GTF_PATH``), else a small built-in gene set for
the four manuscript loci.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs, samples, style

FIGURE_ID = "figure4"

# Figure 4 currently fine-maps only the 10k imputed discovery cohort. Generated
# panels are accepted only when their TSV records the 20-PC, sex-aware contract.
FINEMAP_COHORT = "10k_disc"
FINEMAP_N_PCS = 20
ANALYSIS_CONTRACT = "10k_disc_20pc_sexaware_1mb_carriers20_v4"
FINEMAP_MIN_CARRIERS = 20
PC_COUNT_COLUMN = "_n_pcs"
CONTRACT_COLUMN = "_analysis_contract"
MIN_CARRIERS_COLUMN = "_min_carriers"
WINDOW_START_COLUMN = "_window_start"
WINDOW_END_COLUMN = "_window_end"

# The four manuscript loci. ``lead_pos`` is the exact iHS/meta lead SNV (the variant
# overlapping the lead gene): it centres the +/-HALF_WINDOW window, and the real
# susie_<tag>.tsv is matched by that exact tag (susie_<chr>_<pos>_*) or by region
# overlap. ``finemap_gene`` is the credible-set target the locus fine-maps to.
HALF_WINDOW = 500_000
FINEMAP_LOCI = {
    "B": {"lead_gene": "LOC100507053", "finemap_gene": "ADH1B",
          "bold_gene": None,
          "chrom": "chr4", "lead_pos": 99_103_864, "phecode": "MB_280.1",
          "phecode_name": "Alcohol use disorders", "sex_scope": "Both",
          "lead_ref": "C", "lead_alt": "T"},
    "C": {"lead_gene": "STEAP2", "finemap_gene": "CFAP69",
          "bold_gene": "CFAP69",
          "chrom": "chr7", "lead_pos": 90_206_732, "phecode": "GU_626.1",
          "phecode_name": "Irregular menstrual cycle/bleeding", "sex_scope": "Female",
          "lead_ref": "G", "lead_alt": "A"},
    "D": {"lead_gene": "ZMYM1", "finemap_gene": "ZMYM1",
          "bold_gene": "ZMYM1",
          "chrom": "chr1", "lead_pos": 35_126_025, "phecode": "CV_401",
          "phecode_name": "Hypertension", "sex_scope": "Both",
          "lead_ref": "A", "lead_alt": "G"},
    "E": {"lead_gene": "USP39", "finemap_gene": "USP39",
          "bold_gene": "USP39",
          "chrom": "chr2", "lead_pos": 85_632_712, "phecode": "CA_142.1",
          "phecode_name": "Lump or mass in breast", "sex_scope": "Both",
          "lead_ref": "T", "lead_alt": "A"},
}

# Figure 4 v5 replaces the bottom-row fine maps with the two requested
# discovery associations linked to these exact S9 structural variants.  The
# association lead (``variant_tag``) is resolved from S9 source data at render
# time; it is deliberately not guessed or hard-coded here.
LINKED_SV_PANEL_SPECS = {
    "F": {
        "sv_chrom": "chr22",
        "sv_pos": 20_401_764,
        "sv_length": 90,
        "sv_type": "INS",
        "phecode": "GU_592",
        "phecode_name": "Cystitis and urethritis",
    },
    "G": {
        "sv_chrom": "chr2",
        "sv_pos": 131_129_733,
        "sv_length": 111,
        "sv_type": "INS",
        "phecode": "EM_247.5",
        "phecode_name": "Disorders of calcium metabolism",
    },
}

FONTSIZE = 34
TITLE_FONTSIZE = 28
TICK_FONTSIZE = 28
GENE_FONTSIZE = 26
FINEMAP_MARKER_SIZE = 120
FINEMAP_BEST_P_MARKER_SIZE = 360
FINEMAP_SV_MARKER_SIZE = 200
SV_MARKER_SIZE = 1600
SV_OUTLINE_COLOR = "#d95f02"
SV_MARKER_LINEWIDTH = 8.0
# The refGene windows for panels B and C contain 16 and 11 displayable genes,
# respectively.  Keep both tracks complete while retaining a guard against
# pathologically dense windows among the additional hit-level plots.
MAX_TRACK_GENES = 16
GENE_TRACK_ROWS = 6
SUPPLEMENTARY_GENE_TRACK_ROWS = 10
SUPPLEMENTARY_GENE_LABEL_UNIT_FRACTION = 0.0
SUPPLEMENTARY_GENE_LABEL_PAD_FRACTION = 0.24
SUPPLEMENTARY_GENE_LABEL_EDGE_FRACTION = 0.05
SUPPLEMENTARY_PANEL_ASPECT = 1.3405649529424606
SUPPLEMENTARY_FIGSIZE_LOCUS = (12, 12 / SUPPLEMENTARY_PANEL_ASPECT)
SUPPLEMENTARY_FINEMAP_HEIGHT_RATIOS = (7.5, 4.9)
_FIGSIZE_LOCUS = (12, 11.0)
_FIGSIZE_COMPACT = (12, 9.5)
_SMALL_RNA_RE = re.compile(
    r"^(MIR|MIRLET|SNOR|SNORD|SNORA|SCARNA|RNU|RN7|RNA5|VTRNA|RNY)",
    re.IGNORECASE,
)

def on_workbench() -> bool:
    return os.path.isdir(config.SUSIE_FINEMAP_DIR) or bool(
        config.SUSIE_FINEMAP_GS_DIR and gcs.available()
    )


def _parse_variant_tag(tag) -> tuple[str, int, str, str] | None:
    parts = str(tag).strip().split("_", 3)
    if len(parts) != 4:
        return None
    try:
        chrom = parts[0].removeprefix("chr")
        return f"chr{int(chrom)}", int(parts[1]), parts[2], parts[3]
    except (TypeError, ValueError):
        return None


def _first_gene(value, fallback: str) -> str:
    if value is None or pd.isna(value):
        return fallback
    gene = str(value).split(";")[0].strip()
    return gene if gene and gene.lower() != "nan" else fallback


def meta_significant_loci(*, clean_run_data: bool = False) -> list[dict]:
    """Return every meta-significant variant x phecode as a plotting locus."""
    from . import figure4_meta

    meta = figure4_meta.load_meta_results(clean_run_data=clean_run_data).copy()
    if "meta_significant" in meta.columns:
        flag = meta["meta_significant"]
        meta = meta[
            flag.astype(str).str.strip().str.upper().isin({"TRUE", "1"})
            | flag.eq(True)  # noqa: E712
        ].copy()
    variant_col = figure4_meta._col(
        meta, ["variant_tag", "snp", "SNP", "variant_id", "tag"]
    )
    phecode_col = figure4_meta._col(meta, ["phecode", "Phecode", "phenotype"])
    name_col = figure4_meta._col(
        meta, ["phecode_string", "phecode_description", "description", "meaning"]
    )
    gene_col = figure4_meta._col(
        meta, ["gene_in_window", "gene", "gene_name", "nearest_gene"]
    )
    p_col = figure4_meta._col(meta, ["p_meta", "P_meta", "pval_meta"])
    if variant_col is None or phecode_col is None:
        raise ValueError("meta-significant locus plotting needs variant_tag and phecode")

    sex_frame = pd.read_csv(config.ASSETS_DIR / "phecodeX_sex.csv", dtype=str)
    sex_lookup = dict(zip(
        sex_frame["phecode"].str.strip(),
        sex_frame["sex"].str.strip(),
    ))

    loci = []
    seen = set()
    showcase = {
        (loc["chrom"], loc["lead_pos"], loc["phecode"]): loc
        for loc in FINEMAP_LOCI.values()
    }
    for _, row in meta.iterrows():
        parsed = _parse_variant_tag(row[variant_col])
        if parsed is None:
            continue
        chrom, pos, ref, alt = parsed
        phecode = str(row[phecode_col]).strip()
        key = (chrom, pos, phecode)
        if key in seen:
            continue
        seen.add(key)
        matched = showcase.get(key, {})
        fallback_gene = str(row[variant_col])
        lead_gene = matched.get(
            "lead_gene", _first_gene(row.get(gene_col) if gene_col else None, fallback_gene)
        )
        sex_scope = matched.get("sex_scope", sex_lookup.get(phecode))
        if sex_scope not in {"Both", "Female", "Male"}:
            raise ValueError(f"no valid sex scope for meta-significant phecode {phecode}")
        loci.append({
            "chrom": chrom,
            "lead_pos": pos,
            "lead_ref": ref,
            "lead_alt": alt,
            "phecode": phecode,
            "phecode_name": (
                str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else phecode
            ),
            "lead_gene": lead_gene,
            "finemap_gene": matched.get("finemap_gene", lead_gene),
            "bold_gene": matched.get(
                "bold_gene", matched.get("finemap_gene", lead_gene)
            ),
            "sex_scope": sex_scope,
            "meta_p": (
                float(pd.to_numeric(row[p_col], errors="coerce")) if p_col else np.nan
            ),
            "variant_tag": str(row[variant_col]),
        })
    if p_col:
        loci.sort(key=lambda locus: np.nan_to_num(locus["meta_p"], nan=np.inf))
    return loci


def _v5_sv_annotation(spec: dict) -> str:
    return (
        f"{spec['sv_chrom']}:{spec['sv_pos']} "
        f"({spec['sv_length']} bp {spec['sv_type']})"
    )


def _sex_scope_lookup() -> dict[str, str]:
    frame = pd.read_csv(config.ASSETS_DIR / "phecodeX_sex.csv", dtype=str)
    return dict(zip(frame["phecode"].str.strip(), frame["sex"].str.strip()))


def linked_sv_loci(
    source_data: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Resolve the v5 F/G fine-map leads from audited S9 association rows.

    The S9 annotation identifies the linked SV, while ``variant_tag`` supplies
    the actual discovery association lead used to centre the +/-500 kb SuSiE
    window.  Controlled runs require exactly one source row for each requested
    SV x phecode pair.
    """
    from . import figure4_s9

    if source_data is None:
        source_data, _ = figure4_s9.load_source_data()
    else:
        source_data = source_data.copy()

    sex_lookup = _sex_scope_lookup()
    loci: dict[str, dict] = {}
    for letter, spec in LINKED_SV_PANEL_SPECS.items():
        annotation = _v5_sv_annotation(spec)
        matches = source_data.loc[
            source_data["phecode"].astype(str).str.strip().eq(spec["phecode"])
            & source_data["sv_annotation"].fillna("").astype(str).str.strip().eq(annotation)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Figure 4 panel {letter} requires exactly one S9 row for "
                f"{annotation} x {spec['phecode']}; found {len(matches)}"
            )
        row = matches.iloc[0]
        parsed = _parse_variant_tag(row["variant_tag"])
        if parsed is None:
            raise ValueError(f"Invalid S9 variant_tag: {row['variant_tag']}")
        source_kind = "audited S9 source data"

        chrom, pos, ref, alt = parsed
        if chrom != spec["sv_chrom"] or abs(pos - spec["sv_pos"]) > HALF_WINDOW:
            raise ValueError(
                f"Figure 4 v5 panel {letter} lead {chrom}:{pos} does not place "
                f"the linked SV {annotation} inside the +/-{HALF_WINDOW:,} bp window"
            )
        sex_scope = sex_lookup.get(spec["phecode"])
        if sex_scope not in {"Both", "Female", "Male"}:
            raise ValueError(f"no valid sex scope for v5 phecode {spec['phecode']}")

        lead_gene = _first_gene(
            row.get("gene_in_window"),
            f"SV_{spec['sv_chrom']}_{spec['sv_pos']}",
        )
        p_value = pd.to_numeric(row.get("p", row.get("p_value", np.nan)), errors="coerce")
        loci[letter] = {
            "chrom": chrom,
            "lead_pos": pos,
            "lead_ref": ref,
            "lead_alt": alt,
            "phecode": spec["phecode"],
            "phecode_name": str(row.get("phecode_string", spec["phecode_name"])).strip(),
            "lead_gene": lead_gene,
            "finemap_gene": lead_gene,
            "bold_gene": None,
            "sex_scope": sex_scope,
            "meta_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "variant_tag": str(row["variant_tag"]),
            "sv_annotation": annotation,
            "sv_chrom": spec["sv_chrom"],
            "sv_pos": spec["sv_pos"],
            "sv_length": spec["sv_length"],
            "sv_type": spec["sv_type"],
            "source_kind": source_kind,
        }
    return loci


def unrelated_panel_indices(
    vcf_sample_ids,
    *,
    allowed_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return VCF indices and IDs for the canonical 1,025-sample AoU panel.

    This helper is intended for the SuSiE analysis notebook.  It removes the
    controlled cohort exclusions, intersects with the authoritative bucket sample list when
    available, and raises unless the result contains exactly 1,025 unique IDs.
    The returned indices must be applied to every genotype matrix before model
    fitting; passing ``N_AOU`` while retaining all VCF genotype columns is not
    sufficient.
    """
    vcf_ids = [str(sample_id) for sample_id in vcf_sample_ids]
    if len(vcf_ids) != len(set(vcf_ids)):
        raise ValueError("fine-map VCF contains duplicate sample IDs")

    excluded = set(samples.load_excluded_sample_ids())
    if allowed_ids is None:
        allowed_ids = samples.load_aou_sample_list(download=True)
    if allowed_ids is None:
        # The panel-only VCF may already be the complete AoU cohort.  This
        # fallback is still strict: joint AoU+1KG VCFs cannot accidentally pass.
        allowed = set(vcf_ids) - excluded
    else:
        allowed = {str(sample_id) for sample_id in allowed_ids} - excluded

    keep = np.asarray(
        [i for i, sample_id in enumerate(vcf_ids)
         if sample_id in allowed and sample_id not in excluded],
        dtype=int,
    )
    kept_ids = [vcf_ids[i] for i in keep]
    if excluded.intersection(kept_ids):
        raise ValueError("related sample IDs remain in the fine-map panel")
    if len(kept_ids) != samples.N_AOU or len(set(kept_ids)) != samples.N_AOU:
        raise ValueError(
            "fine-map panel must contain exactly "
            f"{samples.N_AOU} unrelated AoU samples; found {len(set(kept_ids))}"
        )
    return keep, kept_ids


def _local_patterns(locus: dict, kind: str) -> list[str]:
    """Local glob patterns for outputs written by the attached SuSiE notebook."""
    cn = locus["chrom"].replace("chr", "")
    pos = locus["lead_pos"]
    phe = locus["phecode"]
    if kind == "plot":
        stems = ("pval_disc",)
        suffix = f"*__{phe}.png"
    elif kind == "result":
        stems = ("susie",)
        suffix = f"*__{phe}__*.tsv"
    else:
        raise ValueError(f"unknown SuSiE artifact kind: {kind}")
    directories = [Path(config.SUSIE_FINEMAP_DIR)]
    if kind == "plot":
        directories.append(config.figure_dir(FIGURE_ID) / "loci")
    return [
        pattern
        for directory in directories
        for stem in stems
        for pattern in (
            str(directory / f"{stem}_{cn}_{pos}_{suffix}"),
            str(directory / f"{stem}_chr{cn}_{pos}_{suffix}"),
        )
    ]


def _find_local_artifact(locus: dict, kind: str) -> str | None:
    hits = sorted({p for pat in _local_patterns(locus, kind) for p in glob.glob(pat)})
    return hits[0] if hits else None


def _find_discovery_result(locus: dict) -> str | None:
    """Return the 10k-discovery TSV used in a generated panel."""
    hits = sorted({p for pat in _local_patterns(locus, "result")
                   for p in glob.glob(pat)
                   if f"__{FINEMAP_COHORT}.tsv" in os.path.basename(p)})
    return hits[0] if hits else None


def validate_discovery_contract(
    path: str | os.PathLike,
    *,
    expected_scope: str | None = None,
    expected_locus: dict | None = None,
) -> None:
    """Raise unless a result records the current cohort and 1-Mb window contract."""
    try:
        head = pd.read_csv(path, sep="\t", nrows=10)
    except Exception as exc:
        raise ValueError(f"could not read fine-map provenance from {path}") from exc
    required = (
        PC_COUNT_COLUMN, CONTRACT_COLUMN, "_sex_scope",
        WINDOW_START_COLUMN, WINDOW_END_COLUMN, MIN_CARRIERS_COLUMN,
    )
    missing = [c for c in required
               if c not in head.columns]
    if missing:
        raise ValueError(
            f"refusing unverified fine-map artifact {path}: missing {missing}; "
            "rerun the 10k discovery analysis"
        )
    counts = pd.to_numeric(head[PC_COUNT_COLUMN], errors="coerce").dropna().unique()
    contracts = head[CONTRACT_COLUMN].dropna().astype(str).unique()
    carrier_thresholds = pd.to_numeric(
        head[MIN_CARRIERS_COLUMN], errors="coerce"
    ).dropna().unique()
    scopes = set(head["_sex_scope"].dropna().astype(str).unique())
    if len(counts) != 1 or int(counts[0]) != FINEMAP_N_PCS:
        raise ValueError(
            f"refusing fine-map artifact {path}: expected {PC_COUNT_COLUMN}="
            f"{FINEMAP_N_PCS}, found {counts.tolist()}"
        )
    if list(contracts) != [ANALYSIS_CONTRACT]:
        raise ValueError(f"refusing fine-map artifact {path}: wrong analysis contract")
    if (
        len(carrier_thresholds) != 1
        or int(carrier_thresholds[0]) != FINEMAP_MIN_CARRIERS
    ):
        raise ValueError(
            f"refusing fine-map artifact {path}: expected {MIN_CARRIERS_COLUMN}="
            f"{FINEMAP_MIN_CARRIERS}, found {carrier_thresholds.tolist()}"
        )
    if not scopes or not scopes.issubset({"Both", "Female", "Male"}):
        raise ValueError(f"refusing fine-map artifact {path}: invalid sex scope {scopes}")
    if expected_scope is not None and scopes != {expected_scope}:
        raise ValueError(
            f"refusing fine-map artifact {path}: expected sex scope "
            f"{expected_scope}, found {scopes}"
        )
    if expected_locus is not None:
        center = int(expected_locus["lead_pos"])
        expected_start = max(1, center - HALF_WINDOW)
        expected_end = center + HALF_WINDOW
        starts = pd.to_numeric(
            head[WINDOW_START_COLUMN], errors="coerce"
        ).dropna().unique()
        ends = pd.to_numeric(
            head[WINDOW_END_COLUMN], errors="coerce"
        ).dropna().unique()
        if list(starts) != [expected_start] or list(ends) != [expected_end]:
            raise ValueError(
                f"refusing fine-map artifact {path}: expected full window "
                f"{expected_locus['chrom']}:{expected_start}-{expected_end}, "
                f"found starts={starts.tolist()}, ends={ends.tolist()}"
            )


def _stage_remote_artifacts(locus: dict, kind: str, *, force: bool = False) -> None:
    """Stage matching SuSiE outputs from the workspace ``gs://`` bucket."""
    if _find_local_artifact(locus, kind) is not None and not force:
        return
    gs_dir = config.SUSIE_FINEMAP_GS_DIR.rstrip("/")
    if not gs_dir or not gcs.available():
        return
    local_dir = Path(config.SUSIE_FINEMAP_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    # Both gcloud and gsutil accept wildcards in a gs:// source. Try the precise
    # phecode patterns first, then a locus-wide pattern so slightly different
    # phenotype suffixes from earlier notebook runs are staged for inspection.
    cn = locus["chrom"].replace("chr", "")
    pos = locus["lead_pos"]
    stems, ext = (("pval_disc",), "png") if kind == "plot" else (("susie",), "tsv")
    remote_basenames = [os.path.basename(p) for p in _local_patterns(locus, kind)]
    remote_basenames += [name for stem in stems for name in (
        f"{stem}_{cn}_{pos}_*.{ext}",
        f"{stem}_chr{cn}_{pos}_*.{ext}",
    )]
    errors = []
    for basename in dict.fromkeys(remote_basenames):
        remote = f"{gs_dir}/{basename}"
        result = gcs.gcs_cp(remote, local_dir)
        if result.returncode == 0 and _find_local_artifact(locus, kind) is not None:
            return
        if result.stderr:
            errors.append(result.stderr.strip())
    import warnings
    detail = errors[-1][:300] if errors else "no matching object was copied"
    warnings.warn(
        f"[figure4] could not stage {kind} artifacts for {locus['lead_gene']} "
        f"from {gs_dir}: {detail}",
        stacklevel=2,
    )


# --- Data loading -------------------------------------------------------------
def _find_result_tsv(locus: dict) -> str | None:
    """Find the pipeline susie_<tag>.tsv whose region covers this locus."""
    discovery = _find_discovery_result(locus)
    if discovery is not None:
        return discovery
    d = config.SUSIE_FINEMAP_DIR
    if not os.path.isdir(d):
        return None
    cn = locus["chrom"].replace("chr", "")
    pos = locus["lead_pos"]
    # 1) Exact lead tag: susie_<chr>_<pos>_<ref>_<alt>.tsv (chr with or without 'chr').
    for pat in (f"susie_{cn}_{pos}_*.tsv", f"susie_chr{cn}_{pos}_*.tsv",
                f"susie_{cn}_{pos}.tsv"):
        hits = glob.glob(os.path.join(d, pat))
        if hits:
            return sorted(hits)[0]
    # 2) Fall back to region overlap: any susie file on this chrom covering lead_pos.
    for path in sorted(glob.glob(os.path.join(d, "susie_*.tsv"))):
        try:
            head = pd.read_csv(path, sep="\t", usecols=["chrom", "pos"], nrows=20000)
        except Exception:
            continue
        if head.empty:
            continue
        chrom = str(head["chrom"].iloc[0]).replace("chr", "")
        if chrom != cn:
            continue
        if head["pos"].min() <= pos <= head["pos"].max():
            return path
    return None


def load_finemap_result(
    locus: dict,
    *,
    clean_run_data: bool = False,
) -> pd.DataFrame:
    """Stage and load a verified SuSiE result for the requested locus."""
    _stage_remote_artifacts(locus, "result", force=clean_run_data)
    path = _find_result_tsv(locus)
    if path is not None:
        validate_discovery_contract(
            path,
            expected_scope=locus.get("sex_scope"),
            expected_locus=locus,
        )
        df = pd.read_csv(path, sep="\t")
        # The attached discovery-cohort notebook colours against the minimum-p
        # variant and writes ``r2_to_minp``; retain compatibility with the
        # original single-cohort column consumed by this fallback renderer.
        if "r2_to_lead" not in df.columns and "r2_to_minp" in df.columns:
            df["r2_to_lead"] = df["r2_to_minp"]
        if "neg_log_p" not in df.columns and "p_score" in df.columns:
            p = pd.to_numeric(df["p_score"], errors="coerce").clip(lower=1e-300)
            df["neg_log_p"] = -np.log10(p)
        df["pos"] = pd.to_numeric(df["pos"], errors="raise")
        if "_is_lead" not in df.columns:
            if "lead_variant" in df.columns:
                lead_flag = df["lead_variant"]
                if lead_flag.dtype == bool:
                    df["_is_lead"] = lead_flag
                else:
                    df["_is_lead"] = lead_flag.astype(str).str.upper().isin({"TRUE", "1"})
            else:
                lead = int((df["pos"] - locus["lead_pos"]).abs().idxmin())
                df["_is_lead"] = df.index == lead
        return df
    raise FileNotFoundError(
        f"Verified SuSiE result missing for {locus['variant_tag']} x {locus['phecode']} "
        f"under {config.SUSIE_FINEMAP_DIR}. Run scripts.recompute_figure4_susie first."
    )


def _gene_annotation_path() -> Path | None:
    """Return the current refGene source, if one has been staged locally."""
    try:
        from . import figure3_ehh
        gtf_candidates = [
            Path(figure3_ehh.GTF_PATH),
            Path(config.SUSIE_FINEMAP_DIR) / "workspace_inputs" / "hg38.refGene.gtf",
        ]
        return next((path for path in gtf_candidates if path.exists()), None)
    except Exception:
        return None


def _genes_for_track(locus: dict, start: int, end: int):
    """Load the locus gene track from the staged refGene annotation."""
    from . import figure3_ehh
    gtf_path = _gene_annotation_path()
    if gtf_path is None:
        raise FileNotFoundError("Stage hg38.refGene.gtf before rendering fine-map gene tracks.")
    genes = figure3_ehh.parse_gtf_region(str(gtf_path), locus["chrom"], start, end)
    return [
        {"name": name, "tx_start": tx["tx_start"], "tx_end": tx["tx_end"],
         "strand": tx["strand"], "exons": tx.get("exons", [])}
        for name, tx in genes.items()
    ]


# --- Plot ---------------------------------------------------------------------
def sv_marker_shape(sv_type: str) -> str:
    """Return the structural-variant marker requested for locus panels."""
    normalized = str(sv_type).strip().upper()
    if normalized == "INS":
        return "^"
    if normalized == "DEL":
        return "v"
    raise ValueError(f"Unsupported SV type {sv_type!r}; expected INS or DEL")


def finemap_variant_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Classify fine-map rows without treating the string ``False`` as true."""
    if "is_sv" in df.columns:
        normalized = (
            df["is_sv"].fillna(False).astype(str).str.strip().str.lower()
        )
    else:
        normalized = pd.Series("false", index=df.index)
    true_values = {"true", "1", "1.0", "t", "yes", "y"}
    false_values = {"false", "0", "0.0", "f", "no", "n", "", "none", "nan"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        values = sorted(normalized[invalid].unique())
        raise ValueError(f"Unsupported is_sv values in fine-map result: {values}")
    is_sv = normalized.isin(true_values).to_numpy(bool)
    sv_types = (
        df.get("sv_type", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )
    unsupported = is_sv & ~np.isin(sv_types, ("INS", "DEL"))
    if unsupported.any():
        values = sorted(set(sv_types[unsupported]))
        raise ValueError(
            f"Unsupported sv_type values in fine-map result: {values}; "
            "expected INS or DEL"
        )
    return {
        "snv": ~is_sv,
        "ins": is_sv & (sv_types == "INS"),
        "del": is_sv & (sv_types == "DEL"),
    }


def plot_finemap(
    df: pd.DataFrame,
    locus: dict,
    *,
    fs: float = FONTSIZE,
    show_title: bool = True,
    show_r2_legend: bool = True,
    reserve_r2_legend_space: bool = False,
    highlight_gene: bool = True,
    sv_pos: int | None = None,
    sv_type: str | None = None,
    expanded_gene_track: bool = False,
    window_center_pos: int | None = None,
    shared_gene_track: dict | None = None,
    editable_axis_labels: bool = False,
):
    """Score-test locus plot with r^2 colouring and a prominent gene track."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    center = (
        int(locus["lead_pos"])
        if window_center_pos is None
        else int(window_center_pos)
    )
    start, end = center - HALF_WINDOW, center + HALF_WINDOW
    use_bp_axis = shared_gene_track is not None
    plot_positions = df["pos"].to_numpy(float)
    if not use_bp_axis:
        plot_positions = plot_positions / 1e6
    nlp = df["neg_log_p"].to_numpy(float)
    r2 = np.nan_to_num(df["r2_to_lead"].to_numpy(float))
    fg = locus.get("finemap_gene")
    bold_gene = locus.get("bold_gene", fg) if highlight_gene else None

    style.apply_style()
    figure_size = (
        _FIGSIZE_COMPACT
        if reserve_r2_legend_space
        else SUPPLEMENTARY_FIGSIZE_LOCUS if expanded_gene_track else _FIGSIZE_LOCUS
    )
    fig = plt.figure(
        figsize=figure_size,
        constrained_layout=not reserve_r2_legend_space,
    )
    if not reserve_r2_legend_space:
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.04)
    # Dense loci can contain eight long gene symbols.  Give the six-row gene
    # track enough physical height that each transcript and its label remain a
    # distinct unit after the panel is embedded in the composite figure.
    if reserve_r2_legend_space:
        # Fixed normalized positions make the left and right composite panels
        # identical even though only the right panel draws the shared r2 scale.
        ax_pv = fig.add_axes([0.097, 0.631, 0.647, 0.353])
        ax_g = fig.add_axes([0.097, 0.095, 0.647, 0.494], sharex=ax_pv)
        colorbar_axis = fig.add_axes([0.841, 0.631, 0.036, 0.353])
    else:
        height_ratios = (
            SUPPLEMENTARY_FINEMAP_HEIGHT_RATIOS
            if expanded_gene_track
            else (3.0, 4.2)
        )
        gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.04)
        ax_pv = fig.add_subplot(gs[0])
        ax_g = fig.add_subplot(gs[1], sharex=ax_pv)
        colorbar_axis = None

    genes = None if use_bp_axis else _genes_for_track(locus, start, end)
    order = np.argsort(r2, kind="stable")
    marker_sizes = np.full(len(df), FINEMAP_MARKER_SIZE, dtype=float)
    finite_p = np.flatnonzero(np.isfinite(nlp))
    if len(finite_p):
        best_p_index = int(finite_p[np.nanargmax(nlp[finite_p])])
        marker_sizes[best_p_index] = FINEMAP_BEST_P_MARKER_SIZE
    variant_masks = finemap_variant_masks(df)
    sc = None
    for variant_kind, marker in (("snv", "o"), ("ins", "^"), ("del", "v")):
        group_order = order[variant_masks[variant_kind][order]]
        if not len(group_order):
            continue
        sizes = marker_sizes[group_order]
        if variant_kind != "snv":
            sizes = np.maximum(sizes, FINEMAP_SV_MARKER_SIZE)
        collection = ax_pv.scatter(
            plot_positions[group_order],
            nlp[group_order],
            s=sizes,
            c=r2[group_order],
            marker=marker,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.9,
            edgecolor="none",
        )
        collection.set_gid(f"finemap-{variant_kind}")
        if sc is None:
            sc = collection
    if sc is None:
        raise ValueError("Fine-map result contains no variants to plot")
    if (sv_pos is None) != (sv_type is None):
        raise ValueError("sv_pos and sv_type must be provided together")
    if sv_pos is not None and sv_type is not None:
        if not start <= int(sv_pos) <= end:
            raise ValueError(
                f"SV position {sv_pos:,} is outside fine-map window "
                f"{start:,}-{end:,}"
            )
        sv_x = int(sv_pos) if use_bp_axis else int(sv_pos) / 1e6
        positions = pd.to_numeric(df["pos"], errors="coerce").to_numpy(float)
        position_matches = np.flatnonzero(positions == int(sv_pos))
        matches = position_matches[np.isfinite(nlp[position_matches])]
        marker_kwargs = {
            "marker": sv_marker_shape(sv_type),
            "s": SV_MARKER_SIZE,
            "edgecolor": SV_OUTLINE_COLOR,
            "linewidth": SV_MARKER_LINEWIDTH,
            "zorder": 7,
            "clip_on": False,
        }
        if len(matches):
            match = int(matches[np.nanargmax(nlp[matches])])
            face_color = plt.get_cmap("viridis")(float(np.clip(r2[match], 0, 1)))
            ax_pv.scatter(
                [sv_x],
                [nlp[match]],
                facecolor=[face_color],
                **marker_kwargs,
            )
            label_y = nlp[match]
            label_transform = ax_pv.transData
        else:
            ax_pv.scatter(
                [sv_x],
                [0.92],
                facecolor="white",
                transform=ax_pv.get_xaxis_transform(),
                **marker_kwargs,
            )
            label_y = 0.92
            label_transform = ax_pv.get_xaxis_transform()
        ax_pv.annotate(
            f"{str(sv_type).strip().upper()} SV",
            xy=(sv_x, label_y),
            xycoords=label_transform,
            xytext=(28, 16),
            textcoords="offset points",
            color=SV_OUTLINE_COLOR,
            fontsize=max(12, fs * 0.45),
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=8,
        )
    # The minimum-p variant remains LD-coloured and larger than ordinary points,
    # while retaining its circle/INS-triangle/DEL-triangle variant-class shape.
    # The iHS lead receives no special outline, guide line, or legend entry.
    ax_pv.set_ylabel(
        style.P_VALUE_AXIS_LABEL if editable_axis_labels else r"$-\log_{10}(p)$",
        fontsize=fs,
        parse_math=not editable_axis_labels,
    )
    ax_pv.tick_params(labelbottom=False, labelsize=TICK_FONTSIZE)
    ax_pv.grid(axis="y", alpha=0.25, lw=0.8)
    for spine in ("top", "right"):
        ax_pv.spines[spine].set_visible(False)
    if show_r2_legend or reserve_r2_legend_space:
        cbar = (
            fig.colorbar(sc, cax=colorbar_axis)
            if colorbar_axis is not None
            else fig.colorbar(sc, ax=ax_pv, fraction=0.046, pad=0.02)
        )
        cbar.set_label(
            style.LD_COLORBAR_LABEL if editable_axis_labels else r"$r^2$ to min-p variant",
            fontsize=GENE_FONTSIZE,
            parse_math=not editable_axis_labels,
        )
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        cbar.ax.set_ylim(0.0, 1.0)
        # Matplotlib rasterizes dense colorbar meshes by default.  These
        # supplementary panels are publication-vector PDFs, so retain the
        # gradient as vector paths along with the points, labels, and genes.
        if cbar.solids is not None:
            cbar.solids.set_rasterized(False)
        if not show_r2_legend:
            # Keep the same reserved colorbar axis while making the shared
            # scale visually blank on the left panel.
            cbar.solids.set_alpha(0)
            cbar.outline.set_alpha(0)
            cbar.ax.tick_params(color=(0, 0, 0, 0), labelcolor=(0, 0, 0, 0))
            cbar.ax.yaxis.label.set_alpha(0)

    if use_bp_axis:
        # Supplementary B and C use the same annotation dictionary, window,
        # lane packing, and gene-label renderer. Keeping C in base-pair units
        # lets Figure 3's renderer be reused directly rather than approximated
        # by a visually similar but independently packed track.
        from scripts import figure3_ehh, figure3_locus_combined

        figure3_ehh.draw_gene_track(
            ax_g,
            shared_gene_track,
            start,
            end,
            highlight_gene=None,
            gene_fontsize=figure3_locus_combined.GENE_FONTSIZE_COMBINED,
            # Panel B draws the large SV triangle in this band.  Panel C keeps
            # the band empty so both shared gene tracks have exactly the same
            # total height and lane coordinates without duplicating the SV.
            reserve_sv_marker_band=sv_pos is not None,
        )
        ax_g.set_xticks([start, center, end])
        ax_g.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x / 1e6:.2f} Mb")
        )
    else:
        _draw_gene_track(
            ax_g,
            genes,
            start,
            end,
            highlight=None if expanded_gene_track else bold_gene,
            required=fg,
            fs=fs,
            minimum_rows=(
                SUPPLEMENTARY_GENE_TRACK_ROWS
                if expanded_gene_track
                else GENE_TRACK_ROWS
            ),
            label_unit_fraction=(
                SUPPLEMENTARY_GENE_LABEL_UNIT_FRACTION
                if expanded_gene_track
                else 0.014
            ),
            label_pad_fraction=(
                SUPPLEMENTARY_GENE_LABEL_PAD_FRACTION
                if expanded_gene_track
                else 0.020
            ),
            label_edge_fraction=(
                SUPPLEMENTARY_GENE_LABEL_EDGE_FRACTION
                if expanded_gene_track
                else 0.10
            ),
        )
        ax_g.set_xlabel("Position (bp)", fontsize=fs)
        ax_g.set_xticks([start / 1e6, center / 1e6, end / 1e6])
        ax_g.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f} Mb"))
    ax_g.tick_params(labelsize=TICK_FONTSIZE)

    scope = locus.get("sex_scope")
    if not scope and "_sex_scope" in df.columns:
        scopes = df["_sex_scope"].dropna().astype(str).unique()
        scope = scopes[0] if len(scopes) else None
    n_samples = int(df["_n_samples"].iloc[0]) if "_n_samples" in df.columns else None
    n_cases = int(df["_n_cases"].iloc[0]) if "_n_cases" in df.columns else None
    cohort_bits = ["10k discovery"]
    if scope:
        cohort_bits.append(str(scope))
    if n_samples is not None:
        cohort_bits.append(f"n={n_samples:,}")
    if n_cases is not None:
        cohort_bits.append(f"cases={n_cases:,}")
    meta_p = locus.get("meta_p", np.nan)
    meta_text = f"; meta p={meta_p:.2g}" if np.isfinite(meta_p) else ""
    ref_alt = ""
    if locus.get("lead_ref") and locus.get("lead_alt"):
        ref_alt = f" {locus['lead_ref']}>{locus['lead_alt']}"
    lead_gene = locus.get("lead_gene", locus.get("variant_tag", ""))
    if show_title:
        fig.suptitle(
            f"{locus['chrom']}:{center:,}{ref_alt} — {lead_gene} → "
            f"{locus['phecode']} {locus['phecode_name']}\n"
            f"{' · '.join(cohort_bits)}{meta_text}",
            fontsize=TITLE_FONTSIZE,
        )
    return fig


def _draw_gene_track(
    ax,
    genes,
    start,
    end,
    *,
    highlight=None,
    required=None,
    fs=FONTSIZE,
    minimum_rows=GENE_TRACK_ROWS,
    label_unit_fraction=0.014,
    label_pad_fraction=0.020,
    label_edge_fraction=0.10,
):
    import matplotlib.pyplot as plt
    ax.set_xlim(start / 1e6, end / 1e6)
    ax.set_ylabel("Genes", fontsize=fs)
    ax.set_yticks([])
    if not genes:
        ax.text(0.5, 0.5, "(no annotated genes)", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=TICK_FONTSIZE)
        return
    genes = [
        gene for gene in genes
        if not _SMALL_RNA_RE.match(str(gene["name"]))
        and gene["tx_end"] - gene["tx_start"] >= 1_000
    ]
    if not genes:
        ax.text(0.5, 0.5, "(no annotated genes)", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=TICK_FONTSIZE)
        return
    if len(genes) > MAX_TRACK_GENES:
        midpoint = (start + end) / 2
        ranked = sorted(
            genes,
            key=lambda gene: (
                gene["name"] != required,
                abs((gene["tx_start"] + gene["tx_end"]) / 2 - midpoint),
            ),
        )
        genes = ranked[:MAX_TRACK_GENES]
    genes = sorted(genes, key=lambda gene: gene["tx_start"])
    # Pack using an approximate label footprint, not only transcript bounds.
    # Long symbols such as LOC100507053 otherwise collide with a neighbouring
    # short gene even when their genomic intervals do not overlap.
    label_unit = (end - start) * label_unit_fraction
    label_pad = (end - start) * label_pad_fraction
    rows = [float("-inf")] * minimum_rows
    row_of: list[int] = []
    for gene in genes:
        # Pack around the displayed label position.  Labels for transcripts at
        # a window edge are clipped inward below; using the unclipped midpoint
        # here allowed edge labels such as ADH7 and MTTP to collide.
        label_center = np.clip(
            (gene["tx_start"] + gene["tx_end"]) / 2,
            start + (end - start) * label_edge_fraction,
            end - (end - start) * label_edge_fraction,
        )
        label_half_width = label_unit * max(len(str(gene["name"])), 4)
        collision_start = min(gene["tx_start"], label_center - label_half_width)
        collision_end = max(gene["tx_end"], label_center + label_half_width)
        chosen_row = None
        for ri, last in enumerate(rows):
            if collision_start > last + label_pad:
                chosen_row = ri
                break
        if chosen_row is None:
            # The full 3C refGene track needs a seventh row.  Grow instead of
            # forcing a colliding annotation back into one of the six standard
            # rows; unusually dense hit-level loci remain readable as well.
            chosen_row = len(rows)
            rows.append(float("-inf"))
        rows[chosen_row] = collision_end
        row_of.append(chosen_row)
    row_height = 0.6
    row_gap = 3.0
    # Do not reserve empty rows at loci with fewer annotations, but retain the
    # full six-row capacity at crowded loci such as STEAP2/CFAP69.
    used_rows = max(row_of) + 1
    ax.set_ylim(-0.7, (used_rows - 1) * row_gap + 2.8)
    min_exon_width_mb = (end - start) / 1e6 * 0.002
    for gene, row in zip(genes, row_of):
        y = row * row_gap + 0.5
        start_mb = gene["tx_start"] / 1e6
        end_mb = gene["tx_end"] / 1e6
        ax.plot([start_mb, end_mb], [y, y], color="black", linewidth=1.8,
                solid_capstyle="butt", zorder=2)
        for exon_start, exon_end in gene.get("exons", []):
            exon_start_mb = max(exon_start, start) / 1e6
            exon_end_mb = min(exon_end, end) / 1e6
            if exon_end_mb <= exon_start_mb:
                continue
            ax.add_patch(plt.Rectangle(
                (exon_start_mb, y - row_height / 2),
                max(exon_end_mb - exon_start_mb, min_exon_width_mb),
                row_height,
                facecolor="black", edgecolor="none", alpha=0.85, zorder=3,
            ))
        label_x = np.clip(
            (start_mb + end_mb) / 2,
            start / 1e6 + (end - start) / 1e6 * label_edge_fraction,
            end / 1e6 - (end - start) / 1e6 * label_edge_fraction,
        )
        ax.text(
            label_x, y + row_height / 2 + 0.35, gene["name"],
            ha="center", va="bottom", fontsize=GENE_FONTSIZE,
            fontstyle="italic", color="black",
            fontweight="bold" if gene["name"] == highlight else "normal",
            zorder=4,
        )
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


# --- Orchestration ------------------------------------------------------------
def panel_image_path(letter: str, ext: str = "png"):
    loc = FINEMAP_LOCI[letter]
    return config.figure_dir(FIGURE_ID) / f"Figure_4{letter}_finemap_{loc['lead_gene']}.{ext}"


def compact_panel_image_path(letter: str, ext: str = "png") -> Path:
    """Titleless B--E source used by the shared-r2 composite arrangement."""
    loc = FINEMAP_LOCI[letter]
    return (
        config.figure_dir(FIGURE_ID)
        / "arrangements"
        / f"Figure_4{letter}_finemap_{loc['lead_gene']}_compact.{ext}"
    )


def linked_sv_panel_image_path(letter: str, ext: str = "png") -> Path:
    """Titleless v5 fine-map panel for one requested S9-linked SV hit."""
    spec = LINKED_SV_PANEL_SPECS[letter]
    token = f"{spec['sv_chrom']}_{spec['sv_pos']}_{spec['phecode']}"
    return (
        config.figure_dir(FIGURE_ID)
        / "arrangements"
        / f"Figure_4{letter}_finemap_{token}_linked_sv.{ext}"
    )


def _safe_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("_") or "NA"


def locus_image_path(locus: dict, ext: str = "png") -> Path:
    """Canonical per-hit image beneath ``figures/figure4/loci``."""
    chrom = locus["chrom"].removeprefix("chr")
    stem = (
        f"finemap_chr{chrom}_{locus['lead_pos']}_"
        f"{_safe_token(locus['phecode'])}_{_safe_token(locus.get('lead_gene', 'NA'))}"
    )
    out_dir = config.figure_dir(FIGURE_ID) / "loci"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.{ext}"


def _save_locus_figure(fig, locus: dict) -> Path:
    for ext in config.FIGURE_FORMATS:
        fig.savefig(
            locus_image_path(locus, ext),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    return locus_image_path(locus)


def _render_dependency_paths() -> list[Path]:
    """Files whose changes require Figure 4 locus panels to be redrawn."""
    paths = [Path(__file__)]
    gene_annotation = _gene_annotation_path()
    if gene_annotation is not None:
        paths.append(gene_annotation)
    return paths


def _render_targets_current(locus: dict, targets: list[Path]) -> bool:
    """True when plots postdate the result, renderer, and gene annotation."""
    if not all(path.exists() for path in targets):
        return False
    result = _find_discovery_result(locus)
    if result is None:
        return False
    try:
        validate_discovery_contract(
            result,
            expected_scope=locus.get("sex_scope"),
            expected_locus=locus,
        )
    except ValueError:
        return False
    dependencies = [Path(result), *_render_dependency_paths()]
    newest_dependency = max(path.stat().st_mtime for path in dependencies)
    return all(path.stat().st_mtime >= newest_dependency for path in targets)


def render_all_locus_panels(
    clean_run_data: bool = False,
    clean_run_plots: bool = False,
    *,
    loci: list[dict] | None = None,
) -> list[Path]:
    """Render one consistently styled locus panel per meta-significant hit."""
    import matplotlib.pyplot as plt

    saved: list[Path] = []
    if loci is None:
        loci = meta_significant_loci(clean_run_data=clean_run_data)
    for locus in loci:
        targets = [locus_image_path(locus, ext) for ext in config.FIGURE_FORMATS]
        if not clean_run_plots and _render_targets_current(locus, targets):
            saved.append(locus_image_path(locus))
            continue
        df = load_finemap_result(locus, clean_run_data=clean_run_data)
        fig = plot_finemap(df, locus)
        saved.append(_save_locus_figure(fig, locus))
        plt.close(fig)
    return saved


def render_finemap_panels(clean_run_data: bool = False,
                          clean_run_plots: bool = False) -> list:
    """Render every hit, then save the four manuscript loci as Figure 4B--E."""
    import matplotlib.pyplot as plt

    all_loci = meta_significant_loci(clean_run_data=clean_run_data)
    render_all_locus_panels(
        clean_run_data=clean_run_data,
        clean_run_plots=clean_run_plots,
        loci=all_loci,
    )
    dynamic = {
        (locus["chrom"], locus["lead_pos"], locus["phecode"]): locus
        for locus in all_loci
    }
    saved: list[Path] = []
    for letter, locus in FINEMAP_LOCI.items():
        key = (locus["chrom"], locus["lead_pos"], locus["phecode"])
        source_locus = dynamic.get(key, locus)
        source = locus_image_path(source_locus)
        if not source.exists():
            df = load_finemap_result(source_locus, clean_run_data=clean_run_data)
            fig = plot_finemap(df, source_locus)
            _save_locus_figure(fig, source_locus)
            plt.close(fig)
        for ext in config.FIGURE_FORMATS:
            source_ext = locus_image_path(source_locus, ext)
            target = panel_image_path(letter, ext)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_ext, target)

        compact_targets = [
            compact_panel_image_path(letter, ext) for ext in config.FIGURE_FORMATS
        ]
        if clean_run_plots or not _render_targets_current(
                source_locus, compact_targets):
            df = load_finemap_result(source_locus, clean_run_data=clean_run_data)
            fig = plot_finemap(
                df,
                source_locus,
                show_title=False,
                show_r2_legend=letter in {"C", "E"},
                reserve_r2_legend_space=True,
                # Compact locus C is panel E in the v4/v5 composites.
                highlight_gene=letter != "C",
            )
            for target in compact_targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=150, facecolor="white")
            plt.close(fig)
        saved.append(panel_image_path(letter))
    return saved


def render_linked_sv_panels(
    clean_run_data: bool = False,
    clean_run_plots: bool = False,
) -> list[Path]:
    """Render v5 panels F/G from the chr22 and chr2 S9-linked hits."""
    import matplotlib.pyplot as plt

    loci = linked_sv_loci()
    saved: list[Path] = []
    for letter, locus in loci.items():
        print(
            f"[figure4-linked-sv] panel {letter}: {locus['sv_annotation']} x "
            f"{locus['phecode']} {locus['phecode_name']}; association lead="
            f"{locus['variant_tag']}; window=+/-{HALF_WINDOW:,} bp; "
            f"marker_size={FINEMAP_MARKER_SIZE}; source={locus['source_kind']}"
        )
        targets = [
            linked_sv_panel_image_path(letter, ext) for ext in config.FIGURE_FORMATS
        ]
        if not clean_run_plots and _render_targets_current(locus, targets):
            saved.append(linked_sv_panel_image_path(letter))
            continue
        df = load_finemap_result(
            locus,
            clean_run_data=clean_run_data,
        )
        fig = plot_finemap(
            df,
            locus,
            show_title=False,
            show_r2_legend=letter == "G",
            reserve_r2_legend_space=True,
        )
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=150, facecolor="white")
            print(f"[figure4-linked-sv] wrote: {target}")
        plt.close(fig)
        saved.append(linked_sv_panel_image_path(letter))
    return saved
