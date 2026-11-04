from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pymupdf
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import figure2_1025, figure2_1025_tables, figure2_plots
from scripts.common import panel_io, config

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")


def _runner():
    path = config.REPO_ROOT / "scripts" / "run_all_figure_notebooks.py"
    spec = importlib.util.spec_from_file_location("figure_runner_1025", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_alias_and_artifact_contract() -> None:
    runner = _runner()
    notebook = runner.resolve_notebooks(["2_1025"], )[0]
    assert notebook.name == "Figure_2.ipynb"
    artifacts = runner.expected_artifacts(notebook)
    assert artifacts[0].as_posix().endswith("figures/figure2/panels/Figure_2A.png")
    names = {path.name for path in artifacts}
    for letter in "ABCDEF":
        assert f"Figure_2{letter}.png" in names
        assert f"Figure_2{letter}.pdf" in names
    assert "Figure_2.pdf" not in names



def test_figure2_1025_panel_a_pdf_uses_vector_markers(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "chrom": np.repeat([1, 2], 20),
            "pos": np.tile(np.arange(1, 21) * 1_000, 2),
            "eur_frac": np.linspace(0.1, 0.9, 40),
            "eur_z": np.r_[np.zeros(38), -4.0, 4.0],
            "telomere_trimmed": False,
        }
    )
    data = {
        "genome_eur": frame,
        "chrom_sizes": {1: 30_000, 2: 30_000},
        "stats": {"mean_eur": 0.5, "sd_eur": 0.1, "z_threshold": 3.0},
    }
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    figure2_plots.panel_A(axis, data)
    output = tmp_path / "Figure_2_1025A.pdf"
    figure.savefig(output)
    plt.close(figure)

    with pymupdf.open(output) as document:
        assert sum(len(page.get_images(full=True)) for page in document) == 0


def test_panel_a_caps_nonsignificant_and_thins_dense_early_chromosome() -> None:
    early_count = 12_000
    late_count = 4_000
    frame = pd.DataFrame(
        {
            "chrom": np.repeat([1, 22], [early_count, late_count]),
            "pos": np.r_[
                np.arange(1, early_count + 1),
                np.arange(1, late_count + 1),
            ],
            "eur_frac": 0.5,
            "eur_z": 0.0,
            "telomere_trimmed": False,
        }
    )
    # Significant points must be retained outside the nonsignificant cap.
    frame.loc[[0, early_count], "eur_z"] = [4.0, -4.0]
    data = {
        "genome_eur": frame,
        "chrom_sizes": {1: early_count + 1, 22: late_count + 1},
        "stats": {"mean_eur": 0.5, "sd_eur": 0.1, "z_threshold": 3.0},
    }

    figure, axis = plt.subplots()
    try:
        figure2_plots.panel_A(axis, data)
        ordinary_x = axis.collections[0].get_offsets()[:, 0]
        outlier_x = axis.collections[1].get_offsets()[:, 0]
        assert len(ordinary_x) == figure2_plots.MAX_NONSIGNIFICANT_POINTS
        assert len(outlier_x) == 2
        # Sparse chr22 is preserved; the excess is removed from dense chr1.
        assert np.sum(ordinary_x >= 21.0) == late_count - 1
        assert np.sum(ordinary_x < 1.0) < early_count - 1
    finally:
        plt.close(figure)


def test_figure2_1025_standalone_panels_are_titleless() -> None:
    assert config.FIGURES["figure2"]["standalone_title_skip"] == tuple("ABCDEF")




def test_notebook_is_full_orchestrator_and_uv_only() -> None:
    notebook = json.loads(
        (config.NOTEBOOKS_DIR / "Figure_2.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "figure2_1025.run_all" in source
    assert figure2_1025.MIN_OBSERVED_MODEL_FRACTION == pytest.approx(0.10)
    requirements = (
        (config.REPO_ROOT / "requirements-figure2.txt")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "phetk==0.3.3" in requirements
    assert "numpy==1.26.4" in requirements
    implementation = (config.REPO_ROOT / "scripts" / "figure2_1025.py").read_text(
        encoding="utf-8"
    )
    assert implementation.count(
        '"min_observed_model_fraction": MIN_OBSERVED_MODEL_FRACTION'
    ) == 2
    assert '-m", "venv' not in implementation
    assert '-m", "pip' not in implementation


def test_display_region_must_contain_old_midpoint() -> None:
    matching = pd.DataFrame(
        [
            {
                "chrom": 9,
                "start": figure2_1025.OLD_REGION_MIDPOINT - 10,
                "end": figure2_1025.OLD_REGION_MIDPOINT + 10,
                "peak_z": -6.0,
                "region_id": "keep",
            },
            {"chrom": 9, "start": 1, "end": 2, "peak_z": 20.0, "region_id": "other"},
        ]
    )
    assert figure2_1025._strict_display_region(matching)["region_id"] == "keep"
    with pytest.raises(RuntimeError, match="no outlier region containing"):
        figure2_1025._strict_display_region(matching.iloc[[1]])


def test_model_requires_ten_post_filter_snvs() -> None:
    region = pd.Series({"region_id": "r1"})

    def nine(*args, **kwargs):
        return {"feature_ids": [f"v{i}" for i in range(9)]}

    with pytest.raises(ValueError, match="at least 10"):
        figure2_1025._extract_training_region_1025(nine, Path("x"), region, 1)

    def ten(*args, **kwargs):
        return {"feature_ids": [f"v{i}" for i in range(10)]}

    assert (
        len(
            figure2_1025._extract_training_region_1025(ten, Path("x"), region, 1)[
                "feature_ids"
            ]
        )
        == 10
    )


def test_shared_threshold_retains_a_19_of_119_snv_prediction() -> None:
    observed_fraction = 19 / 119
    assert observed_fraction >= figure2_1025.MIN_OBSERVED_MODEL_FRACTION


def test_snv_phewas_phecode_labels_take_priority() -> None:
    discovery = pd.DataFrame(
        {"phecode": ["SS_809.1"], "phecode_string": ["Current PheTK label"]}
    )
    with (
        patch.object(
            figure2_1025_tables.admixture_summary,
            "_phecode_string_lookup",
            return_value={"SS_809.1": "Raw PheTK label"},
        ),
        patch.object(
            figure2_1025_tables,
            "_snv_phewas_phecode_strings",
            return_value={"SS_809.1": "Acute pain"},
        ),
    ):
        lookup = figure2_1025_tables._phecode_strings(pd.DataFrame(), discovery)
    assert lookup["SS_809.1"] == "Acute pain"


def test_stage_manifest_rejects_contract_changes_and_corruption(tmp_path: Path) -> None:
    output = tmp_path / "result.tsv"
    output.write_text("a\n1\n", encoding="utf-8")
    contract = {"sample_sha256": "abc", "min_observed_model_fraction": 0.20}
    with patch.object(
        figure2_1025, "manifest_dir", return_value=tmp_path / "manifests"
    ):
        figure2_1025._write_stage_manifest("unit", contract, [output])
        assert figure2_1025._stage_current("unit", contract, [output])
        revised = {**contract, "min_observed_model_fraction": 0.10}
        assert not figure2_1025._stage_current("unit", revised, [output])
        output.write_text("a\n2\n", encoding="utf-8")
        assert not figure2_1025._stage_current("unit", contract, [output])


def test_replication_base_filters_flags_explicit_ids_and_uses_20_pcs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    ids = ["keep1", "keep2", "keep3", "flagged", "excluded-test-2"]
    covariates = pd.DataFrame(
        {
            "person_id": ids,
            "sex_at_birth": [0, 1, 0, 1, 1],
            "age_at_last_ehr_event": [40, 41, 42, 43, 44],
            "age_at_last_ehr_event_squared": [1600, 1681, 1764, 1849, 1936],
        }
    )
    cov_path = source_dir / "covariates_base.tsv"
    covariates.to_csv(cov_path, sep="\t", index=False)
    pca = pd.DataFrame({"person_id": ids})
    for index in range(1, 21):
        pca[f"PC{index}"] = np.arange(len(ids)) + index
    pca_path = source_dir / "pca.tsv"
    pca.to_csv(pca_path, sep="\t", index=False)
    phenotype_path = source_dir / "phe.tsv"
    pd.DataFrame(
        {"person_id": ids, "phecode": ["1"] * len(ids), "count": [2] * len(ids)}
    ).to_csv(phenotype_path, sep="\t", index=False)
    flagged_path = source_dir / "flagged.tsv"
    pd.DataFrame({"sample_id": ["flagged"]}).to_csv(flagged_path, sep="\t", index=False)
    relatedness_path = source_dir / "relatedness.tsv"
    pd.DataFrame(
        {"i.s": ["keep1", "outside"], "j.s": ["outside", "keep2"], "kin": [0.2, 0.2]}
    ).to_csv(relatedness_path, sep="\t", index=False)

    figure2_1025._COHORT_CACHE.clear()
    with (
        patch.dict(figure2_1025.EXPECTED_BASE_COUNTS, {"replication": 5}),
        patch.object(figure2_1025, "analysis_root", return_value=tmp_path / "analysis"),
        patch.object(
            figure2_1025.admixture_inputs,
            "stage_cohort_sources",
            return_value={
                "covariates": cov_path,
                "pca": pca_path,
                "phenotypes": phenotype_path,
            },
        ),
        patch.object(
            figure2_1025,
            "_stage_audit_sources",
            return_value={"flagged": flagged_path, "relatedness": relatedness_path},
        ),
    ):
        cov, counts = figure2_1025.load_cohort_1025("replication", force=True)
    assert set(cov.index) == {"keep1", "keep2", "keep3"}
    assert set(counts["person_id"]) == set(cov.index)
    assert [column for column in cov if column.startswith("pc")] == [
        f"pc{index}" for index in range(1, 21)
    ]
    audit = json.loads(
        (
            tmp_path / "analysis" / "cohorts" / "replication_cohort_audit_1025.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["matching_flag_overlap_before_filter"] == 1
    assert audit["exclusion_manifest_sha256"] == figure2_1025._sample_sha({"excluded-test-1", "excluded-test-2"})
    assert audit["internal_related_pairs"] == 0


def test_publication_tables_match_requested_contract(tmp_path: Path) -> None:
    lai = tmp_path / "lai"
    plrs = tmp_path / "plrs"
    phewas = tmp_path / "phewas"
    for directory in (lai, plrs, phewas):
        directory.mkdir()
    regions = pd.DataFrame(
        {
            "region_id": ["chr9_100_200", "chr1_300_400"],
            "chrom": [9, 1],
            "start": [100, 300],
            "end": [200, 400],
            "direction": ["AFR-excess", "EUR-excess"],
            "peak_z": [-15.2, 5.1],
            "n_markers": [20, 12],
        }
    )
    regions.to_csv(lai / "outlier_regions.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "region_id": ["chr9_100_200"],
            "n_snvs": [14],
            "n_components": [2],
            "n_train": [820],
            "n_test": [205],
            "n_source_samples": [1025],
            "r2_holdout": [0.61],
            "rmse_holdout": [0.1],
            "r2_null": [-0.01],
            "r2_threshold": [0.5],
            "retained_for_phewas": [True],
            "model_contract": ["unit"],
        }
    ).to_csv(plrs / "plrs_models_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "region_id": ["chr9_100_200", "chr9_100_200"],
            "phecode": ["SS_809.1", "RE_479.3"],
            "phecode_string": ["Acute pain", "Respiratory failure"],
            "p": [5e-4, 2e-4],
        }
    ).to_csv(phewas / "phewas.discovery.tsv.gz", sep="\t", index=False)
    pd.DataFrame(
        {
            "region_id": ["chr9_100_200", "chr9_100_200"],
            "phecode": ["SS_809.1", "RE_479.3"],
            "beta_discovery": [0.66, 0.7],
            "se_discovery": [0.19, 0.18],
            "p_discovery": [5e-4, 2e-4],
            "n_discovery": [8324, 8400],
            "beta_replication": [-0.03, np.nan],
            "se_replication": [0.33, np.nan],
            "p_replication": [0.93, np.nan],
            "n_replication": [27089, np.nan],
            "beta_meta": [0.49, np.nan],
            "se_meta": [0.17, np.nan],
            "p_meta": [0.003, np.nan],
            "same_direction": [False, False],
            "meta_significant": [True, False],
        }
    ).to_csv(phewas / "meta_analysis_results.tsv", sep="\t", index=False)
    genes = pd.DataFrame(
        {
            "chrom": ["chr9"],
            "start": [120],
            "end": [180],
            "gene": ["LINC00537"],
        }
    )
    output = tmp_path / "figures" / "figure2_sources"
    with (
        patch.object(figure2_1025_tables.admixture_lai, "analysis_dir", return_value=lai),
        patch.object(
            figure2_1025_tables.admixture_plrs, "analysis_dir", return_value=plrs
        ),
        patch.object(
            figure2_1025_tables.admixture_phewas, "analysis_dir", return_value=phewas
        ),
        patch.object(
            figure2_1025_tables.admixture_finemap, "_gene_table", return_value=genes
        ),
        patch.object(figure2_1025_tables, "output_dir", return_value=output),
    ):
        annotated_meta_path = figure2_1025_tables.annotate_meta_analysis_file()
        paths = figure2_1025_tables.write_tables()
    annotated_meta = pd.read_csv(annotated_meta_path, sep="\t", dtype=str)
    outliers = pd.read_csv(paths["outliers"], keep_default_na=False)
    models = pd.read_csv(paths["plrs"], keep_default_na=False)
    results = pd.read_csv(paths["phewas"], keep_default_na=False, dtype=str)
    assert len(outliers) == 2
    assert len(models) == 1
    assert models.iloc[0]["n_source_samples"] == 1025
    assert results.columns.tolist() == figure2_1025_tables.PHEWAS_COLUMNS
    assert {"genes_in_region", "phecode_string"}.issubset(annotated_meta.columns)
    annotated_hit = annotated_meta.loc[
        annotated_meta["phecode"] == "SS_809.1"
    ].iloc[0]
    assert annotated_hit["genes_in_region"] == "LINC00537"
    assert annotated_hit["phecode_string"] == "Acute pain"
    assert results.iloc[0]["genes_in_region"] == "LINC00537"
    assert results.iloc[0]["phecode_string"] == "Acute pain"
    missing = results.loc[results["phecode"] == "RE_479.3"].iloc[0]
    assert missing["beta_rep"] == "Not replicable"
    assert missing["p_meta"] == "Not replicable"
    assert set(results["meta_significant"]) == {"TRUE", "FALSE"}
    runner = _runner()
    assert all(
        runner.validate_artifact(path) == (True, "ok") for path in paths.values()
    )
