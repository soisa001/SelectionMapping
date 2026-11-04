"""Resume 10k fine-mapping for every corrected discovery-nominal LAI hit.

The publication CSV selects associations; the corrected meta table supplies
their canonical region IDs. No replication or meta-significance filter applies.
Only missing/invalid per-association fits are recomputed, using the same SuSiE,
20-PC, sex-aware discovery analysis as Figure 2f.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from . import admixture_finemap as fm
from . import admixture_inputs, admixture_phewas, figure2_1025
from . import figure2_loci_plots as plots
from . import figure2_locus_figures as locus_figures
from . import figure2_locus_results as locus_results
from .common import config

KEYS = ["chrom", "start", "end", "phecode"]
_PANEL_ONLY = ContextVar("figure2_loci_panel_only", default=False)
_INCLUDE_SVS = ContextVar("figure2_loci_include_svs", default=False)


def _scope_suffix():
    return "_panel_sv" if _INCLUDE_SVS.get() else "_panel" if _PANEL_ONLY.get() else ""


def source_table() -> Path:
    return config.figure_dir("figure2_sources") / "Figure_2_1025_phewas_results.csv"


def output_dir() -> Path:
    return config.figure_dir("figure2_sources") / f"loci{_scope_suffix()}"


def _stage_name(kind, hit):
    prefix = f"nominal{_scope_suffix()}"
    return f"{prefix}_{kind}_{fm.hit_tag(hit)}"


@contextmanager
def _variant_scope(panel_only, include_svs=False):
    """Isolate panel fits and plot manifests while sharing upstream inputs."""
    mode = _PANEL_ONLY.set(panel_only)
    sv_mode = _INCLUDE_SVS.set(include_svs)
    directory = fm.ANALYSIS_SUBDIR.set(f"finemap{_scope_suffix()}")
    contract = fm.FINEMAP_CONTRACT
    if panel_only:
        fm.FINEMAP_CONTRACT += "_flare_panel_snvs_svs_v1" if include_svs else "_flare_panel_snvs_v1"
    try:
        yield
    finally:
        fm.FINEMAP_CONTRACT = contract
        fm.ANALYSIS_SUBDIR.reset(directory)
        _INCLUDE_SVS.reset(sv_mode)
        _PANEL_ONLY.reset(mode)


def plot_paths(hit: pd.Series | dict) -> list[Path]:
    chrom = str(hit["chrom"]).removeprefix("chr")
    stem = (
        f"chr{chrom}_{int(hit['start'])}_{int(hit['end'])}__{fm._safe(hit['phecode'])}"
    )
    # Append extensions: with_suffix would truncate dotted PhecodeX identifiers.
    return [output_dir() / f"{stem}.{extension}" for extension in ("png", "pdf")]


def select_hits(publication: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Validate the published selection against the corrected analysis cache."""
    publication, meta = publication.copy(), meta.copy()
    for frame, required in (
        (publication, {*KEYS, "p_disc"}),
        (meta, {*KEYS, "region_id", "p_discovery"}),
    ):
        if not required.issubset(frame):
            raise ValueError(
                f"Nominal fine-map source lacks {sorted(required - set(frame))}"
            )
        frame["chrom"] = frame["chrom"].astype(str).str.removeprefix("chr").astype(int)
        if not frame["chrom"].between(1, 22).all():
            raise ValueError("Nominal admixture loci must be autosomal")
        for column in ("start", "end"):
            values = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(values).all() or not values.eq(values.astype(int)).all():
                raise ValueError(f"Invalid {column} coordinates in nominal source")
            frame[column] = values.astype(int)
        if (frame["start"] < 1).any() or (frame["end"] < frame["start"]).any():
            raise ValueError("Invalid nominal locus interval")
        if frame["phecode"].isna().any() or frame.duplicated(KEYS).any():
            raise ValueError("Missing or duplicated region-phenotype association")
        frame["phecode"] = frame["phecode"].astype(str)
    if meta["region_id"].isna().any():
        raise ValueError("Missing corrected region IDs")
    for frame, column in ((publication, "p_disc"), (meta, "p_discovery")):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(0, 1).all():
            raise ValueError(f"Invalid {column} in nominal source")
    selected = publication[
        publication["p_disc"] < admixture_phewas.DISCOVERY_P_THRESHOLD
    ].copy()
    expected = meta[meta["p_discovery"] < admixture_phewas.DISCOVERY_P_THRESHOLD]
    if selected.empty:
        raise ValueError("No discovery associations pass p < 0.001")
    selected = selected.merge(
        expected[[*KEYS, "region_id", "p_discovery"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all() or not np.allclose(
        selected["p_disc"],
        selected["p_discovery"],
        rtol=1e-8,
        atol=0,
    ):
        raise ValueError(
            "Publication CSV and corrected discovery associations disagree; refresh the Figure_2_1025 tables"
        )
    return (
        selected.drop(columns=["_merge", "p_discovery"])
        .sort_values(
            ["chrom", "start", "end", "phecode"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _valid_frame(frame: pd.DataFrame | None, hit, *, panel=None) -> bool:
    required = {
        "pos",
        "variant_id",
        "pip",
        "p_score",
        "neg_log10_p",
        "r2_to_min_p",
        "sv_type",
        "minimum_p_variant",
        "n_samples",
        "n_cases",
        "n_pcs",
        "sex_scope",
        "credible_set", "lead_variant", "lead_selection", "r2_to_lead",
        "mac", "n_carriers", "missing_rate", "n_case_carriers", "n_control_carriers",
        "n_called_cases", "n_called_controls", "susie_converged", "susie_niter",
        "susie_prior_variances", "susie_n_active_effects", "susie_version",
    }
    if (
        frame is None
        or not required.issubset(frame)
        or not fm.cache_matches_hit(frame, hit)
    ):
        return False
    if panel is not None and not panel.valid_frame(frame, hit):
        return False
    start, end = fm.expected_window(hit)
    if frame["variant_id"].isna().any() or frame["variant_id"].duplicated().any():
        return False
    for column, lower, upper in (("pos", start, end), ("pip", 0, 1), ("p_score", 0, 1)):
        if (
            not pd.to_numeric(frame[column], errors="coerce")
            .between(lower, upper)
            .all()
        ):
            return False
    lead = frame["minimum_p_variant"].astype(str).str.lower().isin({"true", "1", "1.0"})
    selected, method = fm.select_lead(frame)
    expected_lead = frame.index == selected.name
    plotted_lead = frame["lead_variant"].astype(str).str.lower().isin({"true", "1", "1.0"})
    panel_sv = (frame["sv_type"].fillna("").ne("")
                & (panel is not None and panel.include_svs))
    return (
        lead.sum() == 1
        and (plotted_lead == expected_lead).all()
        and frame["lead_selection"].eq(method).all()
        and frame["susie_converged"].astype(str).str.lower().eq("true").all()
        and pd.to_numeric(frame["mac"], errors="coerce").gt(np.where(panel_sv, 20, fm.MIN_MAC)).all()
        and (panel_sv | pd.to_numeric(frame["n_carriers"], errors="coerce").ge(fm.MIN_CARRIERS)).all()
        and pd.to_numeric(frame["missing_rate"], errors="coerce").between(0, fm.MAX_MISSING).all()
        and frame["n_samples"].nunique() == 1
        and frame["n_cases"].nunique() == 1
        and pd.to_numeric(frame["n_pcs"], errors="coerce").eq(20).all()
        and np.isfinite(pd.to_numeric(frame["neg_log10_p"], errors="coerce")).all()
    )


def _legacy_verified(table: Path, hit, cohort: dict) -> bool:
    """Adopt the already-verified Figure 2f fit without changing its contract."""
    path = figure2_1025._stage_manifest_path("finemap")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        contract = saved["contract"]
        expected = {
            "analysis_contract": fm.FINEMAP_CONTRACT,
            "discovery_cohort": cohort,
            "region_id": str(hit["region_id"]),
            "phecode": str(hit["phecode"]),
            "window_pad_bp": fm.WINDOW_PAD_BP,
            **fm.filter_parameters(),
        }
        return (
            all(contract.get(key) == value for key, value in expected.items())
            and admixture_phewas.MIN_PHECODE_COUNT == 2
            and figure2_1025._stage_current("finemap", contract, [table])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_plots(paths: list[Path]) -> None:
    from PIL import Image

    png, pdf = paths
    with Image.open(png) as image:
        image.verify()
    content = pdf.read_bytes()
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-1024:]:
        raise ValueError(f"Incomplete PDF: {pdf}")
    if b"/Subtype /Image" in content or b"/Subtype /Type3" in content:
        raise ValueError(f"PDF must contain native vectors and editable text: {pdf}")


def run(
    *, force: bool = False, render: bool = True, ids: tuple[str, ...] | None = None,
    panel_only: bool = False, include_svs: bool = False,
    scan_linked_svs: bool | None = None,
) -> dict:
    """Resume loci; panel_only + include_svs selects the separate loci_panel_sv run."""
    if include_svs and not panel_only:
        raise ValueError("include_svs requires panel_only=True; ordinary loci already includes SVs")
    if scan_linked_svs is None:
        scan_linked_svs = include_svs
    table = source_table()
    root = figure2_1025.analysis_root()
    meta_path = root / "phewas" / "meta_analysis_results.tsv"
    audit_path = root / "cohorts" / "discovery_cohort_audit_1025.json"
    samples_path = root / "cohorts" / "discovery_analysis_samples_1025.tsv"
    for path in (table, meta_path, audit_path, samples_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required corrected Figure 2 source is missing: {path}. Run Figure_2_1025 first; legacy analysis/admixture_reference caches cannot substitute."
            )
    hits = select_hits(
        pd.read_csv(table, dtype={"phecode": str}),
        pd.read_csv(meta_path, sep="\t", dtype={"phecode": str}),
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sample_ids = pd.read_csv(samples_path, sep="\t", dtype=str)["person_id"]
    if (
        sample_ids.duplicated().any()
        or figure2_1025._sample_sha(sample_ids.tolist()) != audit["sample_sha256"]
        or len(sample_ids) != audit["retained_after_pca_and_complete_covariates"]
        or audit["n_pcs"] != 20
        or audit["internal_related_pairs"] != 0
    ):
        raise ValueError(
            "Corrected discovery sample manifest is inconsistent with its cohort audit"
        )
    cohort = figure2_1025._cohort_contract("discovery")
    ids = ids or figure2_1025.training_ids()
    with figure2_1025.analysis_context(ids):
        from . import figure2_mask_audit
        mask_audit = figure2_mask_audit.run()
        # Bind explicitly to discovery even if a replication audit is unavailable.
        fm.FINEMAP_CONTRACT = f"figure2_susie_100kb_20pc_qc_unrelated_{cohort['sample_sha256'][:12]}_v1"
        with _variant_scope(panel_only, include_svs):
            if panel_only:
                from .figure2_marker_panel import MarkerPanel
                panel = MarkerPanel.prepare(hits, ids, include_svs=include_svs)
            else:
                panel = None
            result = _run_hits(
                hits, table, meta_path, cohort, audit, force=force, render=render, panel=panel
            )
            result["mask_audit"] = mask_audit
            if panel is not None:
                result["panel_manifest"] = panel.manifest_path
            if render:
                result["two_panel_figures"] = locus_figures.save_hits(hits, ids, force=force, panel=panel)
            if scan_linked_svs:
                from .figure2_sv_linkage import run_locus_linkage
                linked_dir = run_locus_linkage(
                    result["summary"], output_dir() / "sv_linkage",
                    threads=max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))),
                )
                result["linked_sv_table"] = linked_dir / "linked_svs.csv"
                result["linked_sv_manifest"] = linked_dir / "manifest.json"
            return result


def _run_hits(hits, table, meta_path, cohort, audit, *, force, render, panel=None):
    from .run_all_figure_notebooks import DEFAULT_SEED

    started = time.perf_counter()
    runtime = {
        "threads": {
            name: os.environ.get(name, "4")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "output_directory": str(output_dir()),
        "work_directory": str(admixture_inputs.work_dir()),
        "hardmask_source": fm.admixture_mask.Hardmask().provenance,
    }
    parameters = {
        "analysis_contract": fm.FINEMAP_CONTRACT,
        "discovery_cohort": cohort,
        "window_pad_bp": fm.WINDOW_PAD_BP,
        **fm.filter_parameters(),
        "min_cases": admixture_phewas.MIN_CASES,
        "min_phecode_count": admixture_phewas.MIN_PHECODE_COUNT,
        "covariates": admixture_phewas.COVARIATE_COLUMNS,
        "sex_scopes_sha256": figure2_1025._sha256_file(
            config.ASSETS_DIR / "phecodeX_sex.csv"
        ),
        "discovery_vcf_source": config.ADMIXTURE_DISCOVERY_VCF_GS,
        "cohort_sources": audit.get("sources", {}),
        "seed": int(os.environ.get("AOU_FIGURE_SEED", str(DEFAULT_SEED))),
    }
    if panel is not None:
        parameters["variant_set"] = panel.variant_set
        if panel.include_svs:
            parameters.update(sv_min_mac=20, sv_mac_operator=">", sv_min_carriers=0)
        runtime["panel_manifest"] = figure2_1025._output_fingerprint(panel.manifest_path)
    print(
        f"[figure2_sources][loci] selection={table}; p_disc < {admixture_phewas.DISCOVERY_P_THRESHOLD}; "
        f"associations={len(hits)}; regions={hits['region_id'].nunique()}; output={output_dir()}; "
        f"parameters={json.dumps(parameters, sort_keys=True)}; "
        f"threads={os.environ.get('OMP_NUM_THREADS', '4')}",
        flush=True,
    )
    genes = fm._gene_table(force=False) if render else None
    gene_path = fm.gene_table_path()
    if render and (
        not gene_path.is_file()
        or not {"chrom", "start", "end", "gene", "exons"}.issubset(genes)
    ):
        raise FileNotFoundError(
            f"A complete exon/intron gene annotation cache is required: {gene_path}"
        )
    inputs = None
    rows, failures = [], []
    provenance = {
        "source_table": figure2_1025._output_fingerprint(table),
        "source_meta": figure2_1025._output_fingerprint(meta_path),
        "parameters": parameters,
        "runtime": runtime,
        "selection": "p_disc < 0.001; no replication/meta filter",
        "expected_associations": len(hits),
        "completed_associations": 0,
        "status": "running",
        "render": render,
        "failures": failures,
        "started_utc": figure2_1025._utc_now(),
    }
    run_path = output_dir() / "run_manifest.json"
    figure2_1025._atomic_json(run_path, provenance)
    for index, hit in hits.iterrows():
        tag = fm.hit_tag(hit)
        table_path = fm.analysis_dir() / f"susie_{tag}.tsv.gz"
        stage = _stage_name("finemap", hit)
        contract = {
            **parameters,
            "region_id": str(hit["region_id"]),
            "phecode": str(hit["phecode"]),
            "chrom": int(hit["chrom"]),
            "start": int(hit["start"]),
            "end": int(hit["end"]),
            **(panel.parameters(hit) if panel is not None else {}),
        }
        paths = plot_paths(hit)
        row = {key: hit[key] for key in [*KEYS, "region_id", "p_disc"]}
        row.update(
            {
                "fine_map_table": str(table_path),
                "png": str(paths[0]),
                "pdf": str(paths[1]),
            }
        )
        try:
            current = not force and figure2_1025._stage_current(
                stage, contract, [table_path]
            )
            # Once a per-hit manifest exists, a failed check must not fall back to legacy.
            legacy = (
                panel is None and not force
                and not figure2_1025._stage_manifest_path(stage).exists()
                and _legacy_verified(table_path, hit, cohort)
            )
            frame = (
                fm.load_current_cache(table_path, hit) if current or legacy else None
            )
            rebuilt = not _valid_frame(frame, hit, panel=panel)
            if rebuilt:
                if inputs is None:
                    covariates, counts = admixture_phewas.load_cohort("discovery")
                    if (
                        figure2_1025._sample_sha(covariates.index.astype(str).tolist())
                        != cohort["sample_sha256"]
                    ):
                        raise ValueError(
                            "Loaded discovery cohort differs from the fine-map sample manifest"
                        )
                    inputs = (
                        admixture_inputs.stage_discovery_vcf(force=False),
                        covariates,
                        counts,
                        admixture_phewas.phecode_sex_lookup(),
                    )
                print(
                    f"[figure2_sources][loci] {index + 1}/{len(hits)} fitting {tag}; window={fm.expected_window(hit)}",
                    flush=True,
                )
                options = {} if panel is None else {"allowed_variants": panel.variants(hit)}
                if panel is not None and panel.include_svs:
                    options["include_svs"] = True
                frame = fm.fine_map_hit(hit, *inputs, **options)
                if panel is not None:
                    for key, value in panel.parameters(hit).items():
                        frame[key] = value
                if not _valid_frame(frame, hit, panel=panel):
                    raise ValueError(
                        "Fine-map output failed variant/statistic validation"
                    )
                temporary = table_path.with_name(table_path.name + ".tmp")
                frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
                if not _valid_frame(
                    pd.read_csv(temporary, sep="\t", compression="gzip"), hit, panel=panel
                ):
                    raise ValueError("Fine-map table failed read-back validation")
                temporary.replace(table_path)
            if rebuilt or legacy:
                figure2_1025._write_stage_manifest(stage, contract, [table_path])
            if render:
                plot_contract = {
                    "plot_contract": plots.PLOT_CONTRACT,
                    "fine_map_sha256": figure2_1025._sha256_file(table_path),
                    "gene_sha256": figure2_1025._sha256_file(gene_path),
                    "p_disc": float(hit["p_disc"]),
                    "phecode_string": str(hit.get("phecode_string", "")),
                }
                plot_stage = _stage_name("plot", hit)
                if force or not figure2_1025._stage_current(
                    plot_stage, plot_contract, paths
                ):
                    plots.render(frame, hit, genes, paths)
                    _validate_plots(paths)
                    figure2_1025._write_stage_manifest(plot_stage, plot_contract, paths)
            row.update(
                status="ready",
                **fm.summarize_fit(frame),
                **locus_results.summary_fields(frame),
                sex_scope=frame["sex_scope"].iloc[0],
                error="",
            )
            print(
                f"[figure2_sources][loci] {index + 1}/{len(hits)} {tag}: "
                f"{'computed' if rebuilt else 'cache verified'}; variants={len(frame)}; "
                f"n={row['n_samples']}; CS={row['n_credible_sets']}; "
                f"PIP={row['pip_status']}; lead={row['lead_selection']}; PDF/PNG={render}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint each failed fit, then fail the run
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures.append(f"{tag}: {exc}")
            print(f"[figure2_sources][loci] FAILED {failures[-1]}", flush=True)
        rows.append(row)
        # Persist progress after every association, so interruptions remain reviewable.
        summary_path = output_dir() / "fine_mapping_summary.tsv"
        figure2_1025._atomic_tsv(pd.DataFrame(rows), summary_path)
        provenance.update(
            completed_associations=len(rows) - len(failures),
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        figure2_1025._atomic_json(run_path, provenance)
    provenance.update(
        status="failed" if failures else "ready", completed_utc=figure2_1025._utc_now()
    )
    figure2_1025._atomic_json(run_path, provenance)
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(hits)} nominal fine maps failed; rerun to resume. "
            + "; ".join(failures)
        )
    return {
        "status": "ready",
        "n_associations": len(hits),
        "directory": output_dir(),
        "summary": summary_path,
    }
