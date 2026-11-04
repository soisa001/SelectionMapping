"""Required input failures must propagate to the notebook runner."""
import subprocess
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import (
    admixture_data, admixture_lai, figure1_data, figure1_demography, figure1_idw,
    figure1_query, figure2_data, figure3_data, figure3_enrichment,
    figure4_finemap, figure4_meta, figure4_s9, figure4_volcano,
    recompute_admixture_reference, recompute_figure4_susie,
)
from scripts.common import config, gcs


@pytest.mark.parametrize("loader", [
    admixture_data.build, figure2_data.prepare_locus_outputs,
    figure3_data._build_scan, figure4_meta.load_meta_results,
    figure4_s9.load_source_data, figure4_volcano.load_phewas_results,
    figure1_data._build_bootstrap, figure1_demography.load_demes_model,
])
def test_missing_required_source_raises(loader, tmp_path, monkeypatch):
    missing = tmp_path / "missing.tsv"
    monkeypatch.setattr(admixture_lai, "analysis_dir", lambda: tmp_path)
    monkeypatch.setattr(admixture_lai, "real_outputs_available", lambda: False)
    monkeypatch.setattr(config, "IHS_WINDOWS_TSV", missing)
    monkeypatch.setattr(config, "DEMOGRAPHY_BOOTSTRAP", missing)
    monkeypatch.setattr(config, "DEMOGRAPHY_DEMES_YAML", missing)
    monkeypatch.setattr(figure4_meta, "_stage_gcs_file", lambda *a, **k: missing)
    monkeypatch.setattr(figure4_s9, "source_data_path", lambda: missing)
    monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    with pytest.raises(FileNotFoundError):
        loader()


def test_ancestry_staging_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(figure1_data, "ANCESTRY_TSV_INPUT", tmp_path / "ancestry.tsv")
    copy = Mock(return_value=subprocess.CompletedProcess([], 1, "", "access denied"))
    monkeypatch.setattr(gcs, "gcs_cp", copy)
    with pytest.raises(RuntimeError, match="access denied"):
        figure1_data._build_pca()


def test_missing_admixture_input_raises(monkeypatch):
    from scripts import figure1_admixture
    monkeypatch.setattr(figure1_admixture, "load_admixture_frame", lambda: None)
    with pytest.raises(FileNotFoundError, match="ADMIXTURE products"):
        figure1_data._annotated_admixture()


@pytest.mark.parametrize("source", ["idw", "residence", "sfs", "enrichment", "admixture", "susie"])
def test_source_computation_errors_propagate(source, tmp_path, monkeypatch):
    failure = Mock(side_effect=RuntimeError("source failed"))
    if source == "idw":
        monkeypatch.setattr(figure1_idw, "load_idw_frames", lambda: None)
        monkeypatch.setattr(figure1_idw, "build_idw", failure)
        loader = figure1_data._build_idw
    elif source == "residence":
        monkeypatch.setattr(figure1_data, "STATE_SAMPLES_INPUT", tmp_path / "missing.csv")
        monkeypatch.setattr(figure1_query, "fetch_state_samples", failure)
        loader = figure1_data._build_state_counts
    elif source == "sfs":
        monkeypatch.setattr(figure1_demography, "load_aou_fit", lambda: None)
        monkeypatch.setattr(figure1_demography, "compute_aou_fit", failure)
        loader = figure1_data._build_sfs_aou_fit
    elif source == "enrichment":
        monkeypatch.setattr(figure3_enrichment, "_load_windows", lambda: pd.DataFrame())
        monkeypatch.setattr(figure3_enrichment, "_resolve_linkage_cache", lambda: None)
        monkeypatch.setattr(figure3_enrichment, "regenerate_cache", failure)
        loader = figure3_enrichment.build
    elif source == "admixture":
        monkeypatch.setattr(recompute_admixture_reference, "run_all", failure)
        loader = recompute_admixture_reference.ensure_admixture_outputs
    else:
        monkeypatch.setattr(recompute_figure4_susie, "_verify_figure4_outputs", failure)
        monkeypatch.setattr(recompute_figure4_susie, "_stage_figure4_outputs", failure)
        loader = recompute_figure4_susie.ensure_figure4_outputs
    with pytest.raises(RuntimeError, match="source failed"):
        loader()


def test_valid_cached_inputs_are_reused(monkeypatch):
    frames = (pd.DataFrame({"sample_id": ["test"]}), pd.DataFrame({"pop_code": ["YRI"]}))
    rebuild = Mock(side_effect=AssertionError("must reuse validated inputs"))
    monkeypatch.setattr(figure1_idw, "load_idw_frames", lambda: frames)
    monkeypatch.setattr(figure1_idw, "build_idw", rebuild)
    assert figure1_data._build_idw() is frames
    cached_fit = pd.DataFrame({"allele_count": [1], "data": [2.0], "model": [2.1]})
    monkeypatch.setattr(figure1_demography, "load_aou_fit", lambda: cached_fit)
    monkeypatch.setattr(figure1_demography, "compute_aou_fit", rebuild)
    assert figure1_data._build_sfs_aou_fit() is cached_fit


def test_finemap_requires_result_and_gene_annotation(monkeypatch):
    monkeypatch.setattr(figure4_finemap, "_stage_remote_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(figure4_finemap, "_find_result_tsv", lambda *a: None)
    locus = {"variant_tag": "1_100_A_G", "phecode": "TEST", "chrom": "chr1"}
    with pytest.raises(FileNotFoundError, match="Verified SuSiE"):
        figure4_finemap.load_finemap_result(locus)
    monkeypatch.setattr(figure4_finemap, "_gene_annotation_path", lambda: None)
    with pytest.raises(FileNotFoundError, match="refGene"):
        figure4_finemap._genes_for_track(locus, 1, 200)


def test_gene_parser_failure_is_not_hidden(tmp_path, monkeypatch):
    from scripts import figure3_ehh
    monkeypatch.setattr(figure4_finemap, "_gene_annotation_path", lambda: tmp_path / "genes.gtf")
    monkeypatch.setattr(figure3_ehh, "parse_gtf_region", Mock(side_effect=ValueError("bad GTF")))
    with pytest.raises(ValueError, match="bad GTF"):
        figure4_finemap._genes_for_track({"chrom": "chr1"}, 1, 200)
