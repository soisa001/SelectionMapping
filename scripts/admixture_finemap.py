"""Discovery-cohort SuSiE fine-mapping centered on LAI outlier regions."""
from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd

from . import admixture_inputs, admixture_lai, admixture_mask, admixture_phewas, figure2_sv
from .common import config, style
from .susie_rscript import run_susie

WINDOW_PAD_BP = 100_000
FINEMAP_CONTRACT = "figure5_lai_region_susie_10k_20pc_100kb_v3"
GENE_TABLE_CONTRACT = "refgene_longest_transcript_with_exons_v2"
QC_CONTRACT = "observed_analysis_cohort_mac_gt_carriers_ge_missing_le_v1"
MIN_MAC = int(os.environ.get("AOU_FIG4_FINEMAP_MIN_MAC", "20"))
MIN_CARRIERS = int(os.environ.get("AOU_FIG4_FINEMAP_MIN_CARRIERS", "20"))
MAX_MISSING = float(os.environ.get("AOU_FIG4_FINEMAP_MAX_MISSING", "0.05"))
L_SUSIE = int(os.environ.get("AOU_FIG4_FINEMAP_L", "10"))
ANALYSIS_SUBDIR = ContextVar("admixture_finemap_analysis_subdir", default="finemap")


def filter_parameters() -> dict:
    """Include QC semantics as well as thresholds in every fine-map cache."""
    if MIN_MAC < 0 or MIN_CARRIERS < 1 or not 0 <= MAX_MISSING < 1:
        raise ValueError("Invalid fine-map MAC/carrier/missingness thresholds")
    return {
        "qc_contract": QC_CONTRACT, "min_mac": MIN_MAC, "mac_operator": ">",
        "min_carriers": MIN_CARRIERS, "max_missing": MAX_MISSING,
        "susie_l": L_SUSIE, "genotype_source": "GT_hard_calls",
        "additional_vcf_filter": "none", "additional_gq_dp_hwe_filter": "none",
        **admixture_mask.Hardmask().parameters,
    }


def analysis_dir() -> Path:
    path = admixture_lai.analysis_dir().parent / ANALYSIS_SUBDIR.get()
    path.mkdir(parents=True, exist_ok=True)
    return path


def gene_table_path() -> Path:
    """Annotation is shared by the full-variant and FLARE-panel analyses."""
    path = admixture_lai.analysis_dir().parent / "finemap" / "refgene_transcripts.tsv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def hit_tag(hit: pd.Series | dict) -> str:
    """Return the stable filename tag for one region-phecode fine map."""
    return f"{_safe(hit['region_id'])}__{_safe(hit['phecode'])}"


def expected_window(hit: pd.Series | dict) -> tuple[int, int]:
    """Return the exact outlier-boundary +/- 100 kb fine-mapping window."""
    return (
        max(1, int(hit["start"]) - WINDOW_PAD_BP),
        int(hit["end"]) + WINDOW_PAD_BP,
    )


def cache_matches_hit(frame: pd.DataFrame, hit: pd.Series | dict) -> bool:
    """Reject stale SuSiE tables built under a different region contract."""
    parameters = filter_parameters()
    required = {
        "analysis_contract", "region_id", "phecode",
        "window_start", "window_end", *parameters,
    }
    if frame.empty or not required.issubset(frame.columns):
        return False
    window_start, window_end = expected_window(hit)
    return (
        frame["analysis_contract"].astype(str).eq(FINEMAP_CONTRACT).all()
        and frame["region_id"].astype(str).eq(str(hit["region_id"])).all()
        and frame["phecode"].astype(str).eq(str(hit["phecode"])).all()
        and pd.to_numeric(frame["window_start"], errors="coerce").eq(window_start).all()
        and pd.to_numeric(frame["window_end"], errors="coerce").eq(window_end).all()
        and all(frame[key].eq(value).all() for key, value in parameters.items())
    )


def load_current_cache(path: Path, hit: pd.Series | dict) -> pd.DataFrame | None:
    """Load a SuSiE table only when it covers the requested 100 kb flanks."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, sep="\t", dtype={"phecode": str})
    except (OSError, EOFError, pd.errors.EmptyDataError):
        return None
    return frame if cache_matches_hit(frame, hit) else None


def ensure_region_cache(
    region: pd.Series | dict,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build/reuse only the best meta-significant fine map for one region."""
    meta_path = admixture_phewas.analysis_dir() / "meta_analysis_results.tsv"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required meta-analysis table is missing: {meta_path}")
    meta = pd.read_csv(meta_path, sep="\t", dtype={"phecode": str})
    required = {"region_id", "phecode", "meta_significant", "p_meta"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Meta-analysis table lacks {sorted(required - set(meta))}")
    significant = meta["meta_significant"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0"}
    )
    hits = meta[
        meta["region_id"].astype(str).eq(str(region["region_id"])) & significant
    ].copy()
    hits["p_meta"] = pd.to_numeric(hits["p_meta"], errors="coerce")
    hits = hits.sort_values("p_meta", na_position="last")
    if hits.empty:
        raise ValueError(
            f"No meta-significant phenotype is available for {region['region_id']}"
        )
    for _, hit in hits.iterrows():
        table_path = analysis_dir() / f"susie_{hit_tag(hit)}.tsv.gz"
        cached = None if force else load_current_cache(table_path, hit)
        if cached is not None:
            return cached, hit.to_dict()

    vcf_path = admixture_inputs.stage_discovery_vcf(force=False)
    covariates, counts = admixture_phewas.load_cohort("discovery")
    scopes = admixture_phewas.phecode_sex_lookup()
    failures = []
    for _, hit in hits.iterrows():
        table_path = analysis_dir() / f"susie_{hit_tag(hit)}.tsv.gz"
        window_start, window_end = expected_window(hit)
        print(
            f"[figure2_sources] fine-mapping {hit_tag(hit)} at "
            f"chr{int(hit['chrom'])}:{window_start}-{window_end}",
            flush=True,
        )
        try:
            frame = fine_map_hit(hit, vcf_path, covariates, counts, scopes)
        except Exception as exc:
            failures.append(f"{hit_tag(hit)}: {exc}")
            continue
        frame.to_csv(table_path, sep="\t", index=False, compression="gzip")
        return frame, hit.to_dict()
    raise RuntimeError(
        "No 100 kb fine map completed for the displayed region: "
        + "; ".join(failures)
    )


def _extract_genotypes(vcf_path: Path, chrom: int, start: int, end: int, *,
                       allowed_variants: frozenset[str] | None = None,
                       include_svs: bool = False) -> dict:
    try:
        from cyvcf2 import VCF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cyvcf2 is required for admixture reference fine-mapping") from exc
    hardmask = admixture_mask.Hardmask()
    vcf = VCF(str(vcf_path))
    samples = np.asarray(vcf.samples, dtype=str)
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from the discovery VCF")
    variants, positions, columns = [], [], []
    sv_types, sv_lengths = [], []
    n_biallelic, n_masked, n_not_in_panel = 0, 0, 0
    try:
        for record in vcf(f"{token}:{max(1, start)}-{end}"):
            if len(record.ALT or []) != 1:
                continue
            # Indexed queries can return a spanning deletion anchored before
            # the requested interval. All plots/filters use the POS anchor.
            if not start <= int(record.POS) <= end:
                continue
            n_biallelic += 1
            if hardmask.contains(chrom, int(record.POS)):
                n_masked += 1
                continue
            identifier = admixture_inputs.variant_id(record.CHROM, record.POS, record.REF, record.ALT[0])
            # QC is applied later, in the phenotype/covariate/sex-filtered
            # discovery cohort, never in all samples of the source VCF.
            try:
                info_type = record.INFO.get("SVTYPE")
                info_length = record.INFO.get("SVLEN")
            except (KeyError, TypeError):
                info_type = info_length = None
            sv = figure2_sv.classify_sv(
                record.REF,
                record.ALT[0],
                info_type=info_type,
                info_length=info_length,
            )
            if (allowed_variants is not None and identifier not in allowed_variants
                    and not (include_svs and sv is not None)):
                n_not_in_panel += 1
                continue
            dosage = admixture_inputs.genotype_dosage(record)
            columns.append(dosage)
            positions.append(int(record.POS))
            variants.append(identifier)
            sv_types.append(sv[0] if sv is not None else "")
            sv_lengths.append(int(sv[1]) if sv is not None else 0)
    finally:
        vcf.close()
    if not columns:
        raise ValueError(f"No eligible variants in chr{chrom}:{start}-{end}; "
                         f"biallelic={n_biallelic}, masked={n_masked}, not_in_panel={n_not_in_panel}")
    print(f"[figure2_sources][finemap-mask] chr{chrom}:{start}-{end}; "
          f"biallelic={n_biallelic}; excluded={n_masked}; not_in_panel={n_not_in_panel}; "
          f"mask={json.dumps(hardmask.provenance, sort_keys=True)}", flush=True)
    return {
        "samples": samples,
        "variant_id": variants,
        "pos": np.asarray(positions, dtype=np.int64),
        "sv_type": np.asarray(sv_types, dtype=object),
        "sv_length_bp": np.asarray(sv_lengths, dtype=np.int64),
        "X": np.column_stack(columns).astype(float),
        "mask_counts": {"n_variants_before_hardmask": n_biallelic,
                        "n_fail_hardmask": n_masked,
                        **({"n_variants_before_panel": n_biallelic - n_masked,
                            "n_fail_panel": n_not_in_panel,
                            "n_variants_after_panel": len(columns)} if allowed_variants is not None else {})},
    }


def _residualize(value: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coefficient, *_ = np.linalg.lstsq(design, value, rcond=None)
    return value - design @ coefficient


def _r2_to_reference(X: np.ndarray, reference_index: int) -> np.ndarray:
    centered = X - X.mean(axis=0)
    reference = centered[:, reference_index]
    numerator = centered.T @ reference
    denominator = np.sqrt(np.square(centered).sum(axis=0) *
                          np.square(reference).sum())
    return np.square(np.divide(numerator, denominator,
                               out=np.full_like(numerator, np.nan, dtype=float),
                               where=denominator > 0))


def _prepare_analysis(hit: pd.Series, vcf_path: Path,
                      covariates: pd.DataFrame, counts: pd.DataFrame,
                      scopes: dict[str, str], *, allowed_variants: frozenset[str] | None = None,
                      include_svs: bool = False) -> dict:
    chrom = int(hit["chrom"])
    window_start, window_end = expected_window(hit)
    options = {} if allowed_variants is None else {"allowed_variants": allowed_variants}
    if include_svs:
        options["include_svs"] = True
    genotype = _extract_genotypes(vcf_path, chrom, window_start, window_end, **options)
    sample_ids = genotype["samples"]
    aligned = covariates.reindex(sample_ids)
    has_covariates = aligned[admixture_phewas.COVARIATE_COLUMNS].notna().all(axis=1).to_numpy()
    phecode = str(hit["phecode"])
    group = counts[counts["phecode"].astype(str) == phecode]
    if group.empty:
        raise ValueError(f"No discovery phenotype counts for {phecode}")
    scope = scopes.get(phecode)
    if scope not in {"Both", "Female", "Male"}:
        raise ValueError(f"No valid sex scope for {phecode}")
    sex = aligned["sex_at_birth"].to_numpy(dtype=float)
    y_all = admixture_phewas.phenotype_vector(group, sample_ids, sex, scope)
    mask = has_covariates & np.isfinite(y_all)
    y = y_all[mask]
    if int(y.sum()) < admixture_phewas.MIN_CASES:
        raise ValueError(f"Only {int(y.sum())} cases for {phecode}")
    covariate_matrix = aligned.loc[
        mask, admixture_phewas.COVARIATE_COLUMNS
    ].to_numpy(dtype=float)
    X = genotype["X"][mask]
    filter_parameters()  # Validate configuration before accepting variants.
    called = np.isfinite(X)
    if not np.isin(X[called], [0, 1, 2]).all():
        raise ValueError("Fine mapping requires biallelic diploid hard genotype calls")
    n_called = called.sum(axis=0)
    allele_count = np.where(called, X, 0).sum(axis=0).astype(int)
    allele_number = 2 * n_called
    mac = np.minimum(allele_count, allele_number - allele_count)
    missing_rate = (~called).sum(axis=0) / len(X)
    carriers = called & (X > 0)
    n_carriers = carriers.sum(axis=0)
    # Counts include observed calls only. Mean imputation cannot rescue a
    # rare variant or turn a missing genotype into a counted carrier.
    feature_means = np.divide(allele_count, n_called,
                              out=np.full(X.shape[1], np.nan), where=n_called > 0)
    centered = np.where(called, X - feature_means, 0)
    variable = np.square(centered).sum(axis=0) > 0
    # Restored panel SVs use observed MAC > 20 without an ALT-carrier cutoff.
    # A minor allele carried by fewer than 20 homozygotes can therefore qualify.
    panel_sv = ((np.asarray(genotype["sv_type"]) != "")
                & (allowed_variants is not None) & include_svs)
    mac_pass = mac > np.where(panel_sv, 20, MIN_MAC)
    carriers_pass = panel_sv | (n_carriers >= MIN_CARRIERS)
    keep = (missing_rate <= MAX_MISSING) & mac_pass & carriers_pass & variable
    qc_counts = {
        **genotype.get("mask_counts", {}),
        "n_variants_before_qc": X.shape[1], "n_variants_after_qc": int(keep.sum()),
        # Exclusion counts overlap: a variant can fail more than one check.
        "n_fail_mac": int((~mac_pass).sum()),
        "n_fail_carriers": int((~carriers_pass).sum()),
        "n_panel_svs_before_qc": int(panel_sv.sum()),
        "n_panel_svs_after_qc": int((panel_sv & keep).sum()),
        "n_fail_missingness": int((missing_rate > MAX_MISSING).sum()),
        "n_fail_monomorphic": int((~variable).sum()),
    }
    print(f"[figure2_sources][finemap-qc] {hit_tag(hit)}; n={len(y)}; cases={int(y.sum())}; "
          f"panel_include_svs={include_svs and allowed_variants is not None}; "
          f"panel_sv_min_mac=20; panel_sv_min_carriers=0; "
          f"parameters={json.dumps(filter_parameters(), sort_keys=True)}; "
          f"counts={json.dumps(qc_counts, sort_keys=True)}", flush=True)
    if not keep.any():
        raise ValueError(f"No fine-map variants pass MAC > {MIN_MAC}, "
                         f"carriers >= {MIN_CARRIERS}, missingness <= {MAX_MISSING}")
    variant_qc = {
        "mac": mac, "alt_allele_count": allele_count, "allele_number": allele_number,
        "n_called": n_called, "missing_rate": missing_rate,
        "n_carriers": n_carriers,
        "n_case_carriers": carriers[y == 1].sum(axis=0),
        "n_control_carriers": carriers[y == 0].sum(axis=0),
        "n_called_cases": called[y == 1].sum(axis=0),
        "n_called_controls": called[y == 0].sum(axis=0),
    }
    X = np.where(called[:, keep], X[:, keep], feature_means[keep])
    return {
        "chrom": chrom, "window_start": window_start, "window_end": window_end,
        "scope": scope, "phecode": phecode,
        "X": X, "y": y, "covariates": covariate_matrix,
        "variant_id": np.asarray(genotype["variant_id"], dtype=object)[keep],
        "pos": genotype["pos"][keep],
        **{key: values[keep] for key, values in variant_qc.items()},
        "qc_counts": qc_counts,
        "sv_type": genotype["sv_type"][keep],
        "sv_length_bp": genotype["sv_length_bp"][keep],
    }


def fine_map_hit(hit: pd.Series, vcf_path: Path, covariates: pd.DataFrame,
                 counts: pd.DataFrame, scopes: dict[str, str], *,
                 allowed_variants: frozenset[str] | None = None,
                 include_svs: bool = False) -> pd.DataFrame:
    options = {} if allowed_variants is None else {"allowed_variants": allowed_variants}
    if include_svs:
        options["include_svs"] = True
    prepared = _prepare_analysis(hit, vcf_path, covariates, counts, scopes, **options)
    X, y, Z = prepared["X"], prepared["y"], prepared["covariates"]
    X_residual = _residualize(X, Z)
    y_residual = _residualize(y, Z)
    diagnostics = {}
    pip, credible_sets = run_susie(X_residual, y_residual, L=L_SUSIE,
                                   diagnostics=diagnostics)
    if not diagnostics["susie_converged"]:
        raise RuntimeError(f"SuSiE did not converge: {json.dumps(diagnostics)}")
    beta, se, p_value = admixture_phewas.score_test_batch(X, y, Z)
    finite = np.where(np.isfinite(p_value) & (p_value >= 0) & (p_value <= 1), p_value, np.inf)
    if not np.isfinite(finite).any():
        raise ValueError("No finite score-test p-values available to select a lead")
    reference_index = int(np.argmin(finite))
    r2 = _r2_to_reference(X, reference_index)
    frame = pd.DataFrame({
        "variant_id": prepared["variant_id"],
        "chrom": f"chr{prepared['chrom']}",
        "pos": prepared["pos"],
        "is_sv": prepared["sv_type"] != "",
        "sv_type": prepared["sv_type"],
        "sv_length_bp": prepared["sv_length_bp"],
        "pip": pip,
        "beta_score": beta,
        "se_score": se,
        "p_score": p_value,
        "neg_log10_p": -np.log10(np.clip(p_value, 1e-300, 1.0)),
        "r2_to_min_p": r2,
        "mac": prepared["mac"],
        "credible_set": "",
    })
    for name, indices in credible_sets.items():
        for index in indices:
            previous = str(frame.at[index, "credible_set"])
            frame.at[index, "credible_set"] = f"{previous},{name}" if previous else name
    frame["minimum_p_variant"] = False
    frame.at[reference_index, "minimum_p_variant"] = True
    frame["region_id"] = str(hit["region_id"])
    frame["phecode"] = prepared["phecode"]
    frame["sex_scope"] = prepared["scope"]
    frame["n_samples"] = len(y)
    frame["n_cases"] = int(y.sum())
    frame["n_pcs"] = admixture_phewas.N_PCS
    frame["analysis_contract"] = FINEMAP_CONTRACT
    frame["window_start"] = prepared["window_start"]
    frame["window_end"] = prepared["window_end"]
    for key in ("alt_allele_count", "allele_number", "n_called", "missing_rate",
                "n_carriers", "n_case_carriers", "n_control_carriers",
                "n_called_cases", "n_called_controls"):
        frame[key] = prepared[key]
    for key, value in {**filter_parameters(), **prepared["qc_counts"], **diagnostics}.items():
        frame[key] = value
    lead, method = select_lead(frame)
    frame["lead_variant"] = frame.index == lead.name
    frame["lead_selection"] = method
    frame["r2_to_lead"] = _r2_to_reference(X, int(lead.name))
    return frame


def select_lead(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use the highest-PIP credible-set member, else the minimum-p variant."""
    members = frame["credible_set"].fillna("").astype(str).str.strip().ne("")
    if members.any():
        return frame.loc[frame.loc[members, "pip"].idxmax()], "credible_set_max_pip"
    return frame.loc[frame["p_score"].idxmin()], "minimum_p_no_credible_set"


def summarize_fit(frame: pd.DataFrame) -> dict:
    """Keep the plotted lead distinct from the maximum PIP statistic."""
    parameters = filter_parameters()
    lead, method = select_lead(frame)
    max_pip = float(frame["pip"].max())
    top = frame.loc[frame["pip"].idxmax()] if max_pip > 0 else None
    sets = {name for value in frame["credible_set"].fillna("").astype(str)
            for name in value.split(",") if name}
    result = {
        "n_variants": len(frame), "n_samples": int(frame["n_samples"].iloc[0]),
        "n_cases": int(frame["n_cases"].iloc[0]), "n_credible_sets": len(sets),
        "n_positive_pip": int(frame["pip"].gt(0).sum()),
        "sum_pip": float(frame["pip"].sum()),
        "pip_status": "all_zero" if max_pip == 0 else "nonzero",
        "top_pip_variant": "" if top is None else top["variant_id"],
        "top_pip": max_pip, "lead_variant": lead["variant_id"],
        "lead_selection": method, "lead_pos": int(lead["pos"]),
        "lead_pip": float(lead["pip"]), "lead_p_score": float(lead["p_score"]),
    }
    for prefix, row in (("lead", lead), ("top_pip", top)):
        for key in ("mac", "n_carriers", "n_case_carriers", "n_control_carriers",
                    "missing_rate", "n_called_cases", "n_called_controls", "p_score", "pos"):
            result[f"{prefix}_{key}"] = row[key] if row is not None else np.nan
    for key in frame:
        if (key.startswith(("susie_", "n_fail_", "n_variants_", "panel_", "n_panel_"))
                or key in parameters or key in {"variant_set", "sv_min_mac", "sv_mac_operator", "sv_min_carriers"}):
            result[key] = frame[key].iloc[0]
    return result


def _gene_table(*, force: bool = False) -> pd.DataFrame:
    cache = gene_table_path()
    if cache.exists() and not force:
        cached = pd.read_csv(cache, sep="\t")
        required = {"contract", "transcript_id", "exons"}
        if required.issubset(cached.columns) and cached["contract"].astype(str).eq(
            GENE_TABLE_CONTRACT
        ).all():
            return cached
    gtf = admixture_inputs.stage_file(
        config.ADMIXTURE_GTF_GS, Path("reference") / "hg38.refGene.gtf",
        force=force, required=False,
    )
    if gtf is None:
        return pd.DataFrame(columns=["chrom", "start", "end", "gene", "strand"])
    transcripts: dict[tuple[str, str], dict] = {}
    with open(gtf, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in {"transcript", "exon"}:
                continue
            gene_match = re.search(r'gene_name "([^"]+)"', fields[8])
            if gene_match is None:
                gene_match = re.search(r'gene_id "([^"]+)"', fields[8])
            transcript_match = re.search(r'transcript_id "([^"]+)"', fields[8])
            if gene_match is None or transcript_match is None:
                continue
            key = (gene_match.group(1), transcript_match.group(1))
            transcript = transcripts.setdefault(key, {
                "chrom": admixture_inputs.normalize_chrom(fields[0]),
                "start": None,
                "end": None,
                "gene": key[0],
                "transcript_id": key[1],
                "strand": fields[6],
                "exons": [],
            })
            start, end = int(fields[3]), int(fields[4])
            if fields[2] == "transcript":
                transcript["start"], transcript["end"] = start, end
            else:
                transcript["exons"].append((start, end))
    rows = []
    for transcript in transcripts.values():
        exons = sorted(transcript["exons"])
        start = transcript["start"] or (exons[0][0] if exons else None)
        end = transcript["end"] or (exons[-1][1] if exons else None)
        if start is None or end is None:
            continue
        rows.append({
            **transcript,
            "start": int(start),
            "end": int(end),
            "exons": json.dumps(exons, separators=(",", ":")),
            "contract": GENE_TABLE_CONTRACT,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["length"] = frame["end"] - frame["start"]
        frame = frame.loc[
            frame.groupby("gene")["length"].idxmax()
        ].drop(columns="length")
    frame.to_csv(cache, sep="\t", index=False, compression="gzip")
    return frame


def plot_finemap(frame: pd.DataFrame, hit: pd.Series, genes: pd.DataFrame,
                 output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors
    style.apply_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.42], "hspace": 0.08},
    )
    x = frame["pos"].to_numpy() / 1e6
    r2 = frame["r2_to_min_p"].fillna(0).to_numpy()
    norm = colors.Normalize(0, 1)
    axes[0].scatter(x, frame["pip"], c=r2, cmap="viridis", norm=norm,
                    s=18, edgecolor="none", rasterized=True)
    axes[0].set_ylabel("PIP")
    axes[1].scatter(x, frame["neg_log10_p"], c=r2, cmap="viridis", norm=norm,
                    s=18, edgecolor="none", rasterized=True)
    axes[1].set_ylabel(r"$-\log_{10}p$")
    start, end = int(frame["window_start"].iloc[0]), int(frame["window_end"].iloc[0])
    chrom = str(frame["chrom"].iloc[0])
    local_genes = genes[(genes["chrom"] == chrom) &
                        (genes["start"] <= end) & (genes["end"] >= start)]
    for row_index, row in enumerate(local_genes.itertuples(index=False)):
        y = -(row_index % 3)
        axes[2].plot([row.start / 1e6, row.end / 1e6], [y, y], lw=2.2)
        axes[2].text((row.start + row.end) / 2e6, y + 0.18, row.gene,
                     ha="center", va="bottom", fontsize=8, fontstyle="italic")
    axes[2].set_ylim(-2.7, 0.8)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Genes")
    axes[2].set_xlabel(f"Position on {chrom} (Mb)")
    region_start, region_end = int(hit["start"]), int(hit["end"])
    for axis in axes[:2]:
        axis.axvspan(region_start / 1e6, region_end / 1e6,
                    color=style.PALETTE["yellow"], alpha=0.14)
    fig.suptitle(
        f"{hit['region_id']} - {hit['phecode']} (10k discovery)", fontsize=13
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes[:2], fraction=0.025, pad=0.02,
    )
    colorbar.set_label(r"$r^2$ to minimum-$p$ variant")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(*, force: bool = False) -> dict[str, Path]:
    meta_path = admixture_phewas.analysis_dir() / "meta_analysis_results.tsv"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return {}
    try:
        meta = pd.read_csv(meta_path, sep="\t", dtype={"phecode": str})
    except pd.errors.EmptyDataError:
        return {}
    if meta.empty or "meta_significant" not in meta.columns:
        return {}
    hits = meta[meta["meta_significant"].astype(bool)].copy()
    if hits.empty:
        print("[admixture_reference] no meta-significant LAI-dose hits to fine-map")
        return {}
    vcf_path = admixture_inputs.stage_discovery_vcf()
    covariates, counts = admixture_phewas.load_cohort("discovery")
    scopes = admixture_phewas.phecode_sex_lookup()
    genes = None
    summary_rows = []
    outputs = {}
    for hit_number, hit in enumerate(hits.itertuples(index=False), start=1):
        hit_series = pd.Series(hit._asdict())
        tag = hit_tag(hit_series)
        table_path = analysis_dir() / f"susie_{tag}.tsv.gz"
        plot_path = analysis_dir() / f"finemap_{tag}.png"
        frame = None if force else load_current_cache(table_path, hit_series)
        if frame is None:
            window_start, window_end = expected_window(hit_series)
            print(
                f"[admixture_reference] fine-map {hit_number}/{len(hits)} {tag}: "
                f"running chr{int(hit_series['chrom'])}:{window_start}-{window_end}",
                flush=True,
            )
            try:
                frame = fine_map_hit(hit_series, vcf_path, covariates, counts, scopes)
            except Exception as exc:
                print(f"[admixture_reference] fine-map skipped {tag}: {exc}")
                continue
            frame.to_csv(table_path, sep="\t", index=False, compression="gzip")
            print(
                f"[admixture_reference] fine-map {hit_number}/{len(hits)} {tag}: "
                f"completed {len(frame):,} variants",
                flush=True,
            )
        else:
            print(
                f"[admixture_reference] fine-map {hit_number}/{len(hits)} {tag}: "
                "reused matching 100 kb cache",
                flush=True,
            )
        if (force or not plot_path.exists()
                or plot_path.stat().st_mtime < table_path.stat().st_mtime):
            if genes is None:
                genes = _gene_table()
            plot_finemap(frame, hit_series, genes, plot_path)
        summary_rows.append({
            "region_id": hit_series["region_id"],
            "phecode": hit_series["phecode"],
            **summarize_fit(frame),
            "minimum_p_variant": frame.loc[
                frame["minimum_p_variant"].astype(bool), "variant_id"
            ].iloc[0],
            "n_variants": len(frame),
            "n_samples": int(frame["n_samples"].iloc[0]),
            "analysis_contract": FINEMAP_CONTRACT,
        })
        outputs[tag] = plot_path
    summary_path = analysis_dir() / "finemap_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    outputs["summary"] = summary_path
    return outputs
