from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
import pytest
from PIL import Image

from scripts import (
    admixture_finemap as fm,
)
from scripts import (
    figure2_1025 as corrected,
)
from scripts import (
    figure2_loci as loci,
)
from scripts import figure2_loci_plots as plots
from scripts import figure2_locus_figures as pairs
from scripts import figure2_locus_phewas as locus_phewas
from scripts import run_all_figure_notebooks as runner
from scripts.common import config

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")

# Public aggregate coordinates/phecodes supplied with the request; all individual
# data and fine-map statistics below are deliberately synthetic.
ASSOCIATIONS = [
    (9, 63748785, 63837789, "SS_809.1", "Acute pain"),
    (1, 239407597, 239417310, "SS_809.31", "Pain in arm*"),
    (6, 32470563, 32481211, "RE_469", "Bronchitis"),
    (6, 32470563, 32481211, "GU_582.1", "Acute kidney failure"),
    (17, 21654025, 21724463, "CV_446", "Hypotension"),
    (17, 21654025, 21724463, "NS_324.3", "Abnormal involuntary movements"),
    (21, 8756876, 9169773, "DE_682", "Other follicular disorders"),
    (21, 8756876, 9169773, "SO_360", "Inflammation of eyelids"),
    (22, 11622874, 11940952, "RE_479.31", "Acute respiratory failure"),
    (22, 11622874, 11940952, "RE_479.3", "Respiratory failure"),
    (22, 12136917, 12223689, "EM_202.1", "Type 1 diabetes"),
    (8, 2243194, 2334803, "GU_612", "Other nonmalignant breast conditions"),
    (9, 42538337, 42732343, "GI_526", "Intestinal infection"),
]


def sources():
    publication = pd.DataFrame(ASSOCIATIONS, columns=[*loci.KEYS, "phecode_string"])
    publication["chrom"] = "chr" + publication["chrom"].astype(str)
    publication["p_disc"] = 0.0005
    publication["p_rep"] = "Not replicable"
    publication["meta_significant"] = ["TRUE"] + ["FALSE"] * 12
    meta = publication.rename(columns={"p_disc": "p_discovery"}).copy()
    meta["region_id"] = meta.apply(
        lambda row: f"{row.chrom}_{row.start}_{row.end}", axis=1
    )
    return publication, meta


def synthetic_phewas(hits):
    categories = ["Blood/Immune", "Neoplasms", "Cardiovascular", "Dermatological",
                  "Endocrine/Metab", "Genetic", "Gastrointestinal", "Genitourinary",
                  "Infectious", "Mental", "Musculoskeletal", "Neurological",
                  "Pregnancy complications", "Respiratory", "Sense organs", "Symptoms"]
    rows = []
    for region, group in hits.groupby("region_id", sort=False):
        for j, (_, hit) in enumerate(group.iterrows()):
            rows.append({"region_id": region, "phecode": hit["phecode"],
                         "phecode_name": hit["phecode_string"], "phecode_category": categories[j],
                         "p": hit["p_disc"], "beta": 0.8 if j % 2 else -0.5})
        for j in range(48):
            rows.append({"region_id": region, "phecode": f"BACKGROUND_{j}",
                         "phecode_name": f"Background phenotype {j}",
                         "phecode_category": categories[j // 3],
                         "p": 0.01 + j / 50, "beta": 0.2 if j % 2 else -0.2})
    frame = pd.DataFrame(rows)
    frame["p_value"] = frame["p"]
    frame["odds_ratio"] = np.exp(frame["beta"])
    frame["discovery_significant"] = frame["p"] < 0.001
    return frame, categories


def phewas_kwargs(hit):
    frame, categories = synthetic_phewas(loci.select_hits(*sources()))
    return {"phewas": frame.loc[frame["region_id"].eq(hit["region_id"])],
            "categories": categories,
            "phewas_parameters": locus_phewas.selection_plots.PlotParameters(phewas_y_max=5)}


def assert_category_labels_clear(axis):
    """Check rotated text rectangles, whose axis-aligned boxes may overlap."""
    from copy import copy

    rectangles = []
    for label in axis.get_xticklabels():
        assert "\n" not in label.get_text()
        assert label.get_fontsize() == 12
        plain = copy(label)
        plain.set_rotation(0)
        dimensions = plain.get_window_extent()
        bounds = label.get_window_extent()
        assert axis.figure.bbox.contains(bounds.x0, bounds.y0)
        assert axis.figure.bbox.contains(bounds.x1, bounds.y1)
        angle = np.deg2rad(label.get_rotation())
        along = dimensions.width * np.array([np.cos(angle), np.sin(angle)])
        across = dimensions.height * np.array([-np.sin(angle), np.cos(angle)])
        origin = np.array([bounds.x0 - across[0], bounds.y0])
        rectangles.append(np.array([origin, origin + along, origin + along + across, origin + across]))
    for i, left in enumerate(rectangles):
        for right in rectangles[i + 1:]:
            separated = False
            for edge in (left[1] - left[0], left[3] - left[0]):
                normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
                a, b = left @ normal, right @ normal
                if max(b.min() - a.max(), a.min() - b.max()) >= 1:
                    separated = True
            assert separated, (axis.get_xticklabels()[i].get_text(), left, right)


def frame_for(hit, *_args):
    start, end = fm.expected_window(hit)
    return pd.DataFrame(
        {
            "variant_id": [f"variant_{i}" for i in range(6)],
            "pos": np.linspace(start + 100, end - 100, 6).astype(int),
            "pip": [0.01, 0.1, 0.05, 0.8, 0.02, 0.1],
            "p_score": [0.1, 0.05, 0.02, 0.00001, 0.5, 0.02],
            "beta_score": [-0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            "se_score": [0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
            "neg_log10_p": [1, 1.3, 1.7, 5, 0.3, 1.7],
            "r2_to_min_p": [0.0, 0.2, 0.5, 1, 0.01, 0.8],
            "r2_to_lead": [0.0, 0.2, 0.5, 1, 0.01, 0.8],
            "credible_set": ["", "", "", "L1", "", ""],
            "lead_variant": [False, False, False, True, False, False],
            "lead_selection": "credible_set_max_pip",
            "mac": 60, "n_carriers": 50, "n_case_carriers": 5,
            "n_control_carriers": 45, "missing_rate": 0.01,
            "n_called_cases": 345, "n_called_controls": 7575,
            "susie_converged": True, "susie_niter": 8,
            "susie_version": "synthetic", "susie_n_active_effects": 1,
            "susie_prior_variances": "0.1,0.0",
            **fm.filter_parameters(),
            "sv_type": ["", "INS", "", "", "", "DEL"],
            "sv_length_bp": [0, 70, 0, 0, 0, 85],
            "minimum_p_variant": [False, False, False, True, False, False],
            "n_samples": 8000,
            "n_cases": 350,
            "n_pcs": 20,
            "sex_scope": "Both",
            "region_id": hit["region_id"],
            "phecode": hit["phecode"],
            "analysis_contract": fm.FINEMAP_CONTRACT,
            "window_start": start,
            "window_end": end,
        }
    )


def test_selects_all_13_without_replication_filter_and_preserves_phecode_dots():
    hits = loci.select_hits(*sources())
    assert len(hits) == 13 and hits["region_id"].nunique() == 9
    paths = [path for _, hit in hits.iterrows() for path in loci.plot_paths(hit)]
    assert len(set(paths)) == 26
    assert any(path.name.endswith("__RE_479.31.pdf") for path in paths)
    assert any(path.name.endswith("__RE_479.3.pdf") for path in paths)
    assert hits["meta_significant"].eq("TRUE").sum() == 1
    combined = [path.name for _, hit in hits.iterrows() for path in pairs.plot_paths(hit)]
    assert len(set(combined)) == 26
    assert {"chr9_63748785_63837789_acute_pain.png",
            "chr9_63748785_63837789_acute_pain.pdf",
            "chr1_239407597_239417310_pain_in_arm.png",
            "chr22_12136917_12223689_type_1_diabetes.pdf"} <= set(combined)


def test_selection_rejects_missing_duplicate_stale_or_invalid_rows():
    publication, meta = sources()
    with pytest.raises(ValueError, match="disagree"):
        loci.select_hits(publication.iloc[:-1], meta)
    with pytest.raises(ValueError, match="duplicated"):
        loci.select_hits(pd.concat([publication, publication.iloc[[0]]]), meta)
    changed = publication.copy()
    changed.loc[0, "p_disc"] = 0.0001
    with pytest.raises(ValueError, match="disagree"):
        loci.select_hits(changed, meta)
    changed.loc[0, "p_disc"] = np.nan
    with pytest.raises(ValueError, match="Invalid p_disc"):
        loci.select_hits(changed, meta)
    publication.loc[0, "p_disc"] = meta.loc[0, "p_discovery"] = 0.001
    assert len(loci.select_hits(publication, meta)) == 12


@pytest.fixture
def staged(tmp_path, monkeypatch):
    from scripts import figure2_mask_audit
    monkeypatch.setattr(figure2_mask_audit, "run", Mock(return_value={"status": "clear"}))
    monkeypatch.setattr(config, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setenv("AOU_FIG5_1025_WORK_DIR", str(tmp_path / "inputs"))
    publication, meta = sources()
    publication.to_csv(loci.source_table(), index=False)
    root = corrected.analysis_root()
    (root / "phewas").mkdir()
    meta.to_csv(root / "phewas" / "meta_analysis_results.tsv", sep="\t", index=False)
    ids = ("synthetic_1", "synthetic_2")
    audit = {
        "sample_sha256": corrected._sample_sha(ids),
        "n_pcs": 20,
        "internal_related_pairs": 0,
        "retained_after_pca_and_complete_covariates": 2,
    }
    corrected._atomic_json(root / "cohorts" / "discovery_cohort_audit_1025.json", audit)
    corrected._atomic_tsv(
        pd.DataFrame({"person_id": ids}),
        root / "cohorts" / "discovery_analysis_samples_1025.tsv",
    )
    lai_root = corrected.lai_dir()
    stats_path = lai_root / "scan_stats.json"
    corrected._atomic_json(stats_path, {"mean_eur": 0.22, "sd_eur": 0.01, "z_threshold": 4})
    outputs = [stats_path]
    scan_dir = lai_root / "chromosome_scans"
    scan_dir.mkdir()
    hits = loci.select_hits(publication, meta)
    discovery, _ = synthetic_phewas(hits)
    annotation_path = tmp_path / "annotations.tsv"
    discovery[["phecode", "phecode_name", "phecode_category"]].drop_duplicates("phecode").to_csv(
        annotation_path, sep="\t", index=False)
    monkeypatch.setenv("AOU_FIG5_PHECODE_ANNOTATIONS", str(annotation_path))
    discovery = discovery[["region_id", "phecode", "p", "beta"]].assign(
        cohort="discovery", sex_scope="Both", n=8000, n_cases=350, n_controls=7650,
        n_pcs=20, se=0.1, analysis_contract="synthetic_discovery",
        target_contract=locus_phewas.admixture_phewas._target_contract(None))
    discovery_path = root / "phewas" / "phewas.discovery.tsv.gz"
    discovery.to_csv(discovery_path, sep="\t", index=False)
    metadata_path = discovery_path.with_suffix(discovery_path.suffix + ".manifest.json")
    corrected._atomic_json(metadata_path, {key: discovery[key].iloc[0]
                                          for key in ("analysis_contract", "target_contract")})
    corrected._write_stage_manifest("discovery_phewas",
        {"cohort": corrected._cohort_contract("discovery"), "analysis_contract": "synthetic_discovery"},
        [discovery_path, metadata_path])
    for chrom, group in hits.groupby("chrom"):
        positions = np.unique(np.concatenate([
            np.linspace(*fm.expected_window(hit), 71, dtype=np.int64)
            for _, hit in group.iterrows()]))
        path = scan_dir / f"chr{chrom}.eur_fraction.npz"
        np.savez_compressed(path, pos=positions, eur_frac=np.full(len(positions), 0.19),
                            n_samples=np.array([len(ids)]))
        outputs.append(path)
    corrected._write_stage_manifest("lai", {"sample_n": len(ids),
        "sample_sha256": corrected._sample_sha(ids), "z_threshold": 4}, outputs)
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    (root / "finemap").mkdir()
    genes.to_csv(root / "finemap" / "refgene_transcripts.tsv.gz", sep="\t", index=False)
    monkeypatch.setattr(fm, "_gene_table", Mock(return_value=genes))
    loader = Mock(return_value=(pd.DataFrame(index=ids), pd.DataFrame()))
    monkeypatch.setattr(corrected, "load_cohort_1025", loader)
    staging = Mock(return_value=tmp_path / "synthetic.vcf.gz")
    monkeypatch.setattr(loci.admixture_inputs, "stage_discovery_vcf", staging)
    fit = Mock(side_effect=frame_for)
    monkeypatch.setattr(fm, "fine_map_hit", fit)

    def fake_render(frame, hit, genes, paths):
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "white").save(paths[0])
        with pymupdf.open() as document:
            document.new_page().insert_text((20, 20), "Synthetic cache fixture")
            document.save(paths[1])

    render = Mock(side_effect=fake_render)
    monkeypatch.setattr(plots, "render", render)
    monkeypatch.setattr(pairs, "render", Mock(side_effect=lambda frame, hit, markers, stats, paths,
                                             **kwargs: fake_render(frame, hit, markers, paths)))
    return ids, fit, render, loader, staging


def test_full_output_resumes_and_repairs_only_affected_products(staged):
    ids, fit, render, loader, staging = staged
    # Existing combined figures are retired only after replacement succeeds.
    hit = loci.select_hits(*sources()).iloc[0]
    legacy = [pairs.output_dir() / path.name for path in loci.plot_paths(hit)]
    legacy.extend(pairs.output_dir() / f"pain_in_arm.{ext}" for ext in ("png", "pdf"))
    legacy[0].parent.mkdir(parents=True, exist_ok=True)
    for path in legacy:
        path.write_bytes(b"previous combined plot")
    result = loci.run(ids=ids)
    assert result["n_associations"] == 13
    assert fit.call_count == render.call_count == 13
    assert loader.call_count == staging.call_count == 1
    assert len(list(loci.output_dir().glob("*.pdf"))) == 13
    assert len(list(loci.output_dir().glob("*.png"))) == 13
    assert len(list(pairs.output_dir().glob("*.png"))) == 13
    assert len(list(pairs.output_dir().glob("*.pdf"))) == 13
    assert pairs.render.call_count == 13
    assert not any(path.exists() for path in legacy)
    assert all(path.exists() for path in pairs.plot_paths(hit))
    results_path = result["two_panel_figures"]["results_table"]
    results = pd.read_csv(results_path)
    assert len(results) == 13 and results["status"].eq("ready").all()
    ancestry_path = result["two_panel_figures"]["sv_ancestry_table"]
    ancestry = pd.read_csv(ancestry_path, sep="\t")
    assert len(ancestry) == int(results["n_plotted_svs"].sum()) == 26
    assert not ancestry.duplicated(["region_id", "phecode", "variant_id"]).any()
    assert ancestry["overall_eur_frac"].between(0, 1).all()
    assert {"fine_mapped_region", "p_disc", "min_p", "min_p_variant", "lead_pip",
            "top_pip", "n_credible_sets", "min_mac", "hardmask_sha256",
            "credible_set_variants", "representative_beta_score", "representative_se_score"} <= set(results)
    members_csv = result["two_panel_figures"]["credible_set_table"]
    members_tsv = result["two_panel_figures"]["credible_set_tsv"]
    members = pd.read_csv(members_csv)
    assert len(members) == 13 and members["is_representative"].all()
    assert members["variant_id"].eq("variant_3").all()
    np.testing.assert_allclose(members["beta_score"], results["representative_beta_score"])
    pd.testing.assert_frame_equal(members, pd.read_csv(members_tsv, sep="\t"))
    pd.testing.assert_frame_equal(results, pd.read_csv(result["two_panel_figures"]["summary"], sep="\t"))
    results_path.write_bytes(b"corrupt results")
    members_csv.write_bytes(b"corrupt credible-set variants")
    members_tsv.unlink()
    ancestry_path.write_bytes(b"corrupt ancestry output")
    loci.run(ids=ids)
    assert fit.call_count == render.call_count == 13
    assert loader.call_count == staging.call_count == 1
    assert pairs.render.call_count == 13
    assert len(pd.read_csv(results_path)) == 13
    pd.testing.assert_frame_equal(ancestry, pd.read_csv(ancestry_path, sep="\t"))
    pd.testing.assert_frame_equal(members, pd.read_csv(members_csv))
    pd.testing.assert_frame_equal(members, pd.read_csv(members_tsv, sep="\t"))
    next(pairs.output_dir().glob("*.pdf")).unlink()
    loci.run(ids=ids)
    assert pairs.render.call_count == 14
    assert fit.call_count == render.call_count == 13
    summary = pd.read_csv(result["summary"], sep="\t")
    Path(summary.iloc[0]["pdf"]).unlink()
    loci.run(ids=ids)
    assert fit.call_count == 13 and render.call_count == 14
    Path(summary.iloc[1]["fine_map_table"]).write_bytes(b"corrupt gzip")
    loci.run(ids=ids)
    assert fit.call_count == 14
    assert pd.read_csv(result["summary"], sep="\t")["status"].eq("ready").all()


def test_failed_fit_is_reported_and_only_that_hit_retries(staged):
    ids, fit, render, _, _ = staged

    def fail_one(hit, *args):
        if hit["phecode"] == "RE_479.31":
            raise RuntimeError("synthetic interrupted fit")
        return frame_for(hit)

    fit.side_effect = fail_one
    with pytest.raises(RuntimeError, match="1/13"):
        loci.run(ids=ids)
    assert render.call_count == 12
    manifest = json.loads((loci.output_dir() / "run_manifest.json").read_text())
    assert manifest["completed_associations"] == 12
    fit.side_effect = frame_for
    loci.run(ids=ids)
    assert fit.call_count == 14 and render.call_count == 13


def test_existing_main_fit_is_adopted_with_its_manifest(staged):
    ids, fit, _, _, _ = staged
    hit = loci.select_hits(*sources()).query("phecode == 'SS_809.1'").iloc[0]
    with corrected.analysis_context(ids):
        table = fm.analysis_dir() / f"susie_{fm.hit_tag(hit)}.tsv.gz"
        frame_for(hit).to_csv(table, sep="\t", index=False)
        cohort = corrected._cohort_contract("discovery")
        contract = {
            "analysis_contract": fm.FINEMAP_CONTRACT,
            "discovery_cohort": cohort,
            "region_id": hit["region_id"],
            "phecode": hit["phecode"],
            "window_pad_bp": fm.WINDOW_PAD_BP,
            **fm.filter_parameters(),
        }
        # The public entry point binds this contract to discovery even without replication.
        contract["analysis_contract"] = (
            f"figure2_susie_100kb_20pc_qc_unrelated_{cohort['sample_sha256'][:12]}_v1"
        )
        frame = pd.read_csv(table, sep="\t")
        frame["analysis_contract"] = contract["analysis_contract"]
        frame.to_csv(table, sep="\t", index=False)
        corrected._write_stage_manifest("finemap", contract, [table])
    loci.run(ids=ids)
    assert fit.call_count == 12


def test_runner_verifies_every_pair(staged, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_figure_number", lambda _: 5)
    paths = runner.expected_artifacts(tmp_path / "Figure_2.ipynb")
    assert len([path for path in paths if path.parent.name == "loci"]) == 26
    combined = [path for path in paths if path.parent.name == "figures" and path.parent.parent.name == "loci"]
    assert len([path for path in combined if path.suffix in {".png", ".pdf"}]) == 26
    assert {"fine_mapping_results.csv", "figure_summary.tsv", "credible_set_variants.csv",
            "credible_set_variants.tsv"} <= {path.name for path in combined}


@pytest.mark.parametrize("parameter", ["L_SUSIE", "MIN_MAC", "MIN_CARRIERS"])
def test_changed_analysis_parameters_invalidate_fits(staged, monkeypatch, parameter):
    ids, fit, _, _, _ = staged
    loci.run(ids=ids, render=False)
    monkeypatch.setattr(fm, parameter, getattr(fm, parameter) + 1)
    loci.run(ids=ids, render=False)
    assert fit.call_count == 26


def test_corrupt_sample_manifest_fails_before_loading_controlled_inputs(staged):
    ids, fit, _, loader, staging = staged
    path = corrected.analysis_root() / "cohorts" / "discovery_analysis_samples_1025.tsv"
    path.write_text("person_id\nwrong_sample\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample manifest"):
        loci.run(ids=ids)
    assert fit.call_count == loader.call_count == staging.call_count == 0


def test_vectors_editable_text_and_dense_gene_label_layout(tmp_path):
    hit = loci.select_hits(*sources()).query("phecode == 'GU_612'").iloc[0]
    frame = frame_for(hit)
    genes = pd.DataFrame(
        [
            {
                "chrom": "chr8",
                "start": int(hit["start"]) + i * 600,
                "end": int(hit["start"]) + i * 600 + 400,
                "gene": f"LONG_GENE_NAME_{i}",
                "exons": "[]",
            }
            for i in range(12)
        ]
    )
    fig = plots.make_figure(frame, hit, genes)
    try:
        fig.canvas.draw()
        assert fig.axes[0].get_ylabel() == plots.style.P_VALUE_AXIS_LABEL
        assert fig.axes[1].get_ylabel() == "SuSiE PIP"
        assert fig.axes[0].get_position().y0 > fig.axes[1].get_position().y0
        assert fig.axes[3].get_position().height > 0
        labels = [text.get_window_extent() for text in fig.axes[2].texts]
        assert not any(
            a.overlaps(b) for i, a in enumerate(labels) for b in labels[i + 1 :]
        )
        assert all(
            not item.get_rasterized() for axis in fig.axes for item in axis.collections
        )
        pdf = tmp_path / "dense.pdf"
        fig.savefig(pdf)
        with pymupdf.open(pdf) as document:
            assert not document[0].get_images(full=True)
            assert "LONG_GENE_NAME_0" in document[0].get_text()
            assert all(font[2] != "Type3" for font in document[0].get_fonts())
    finally:
        plt.close(fig)


@pytest.mark.parametrize("all_zero", [True, False])
def test_no_credible_set_lead_is_min_p_even_with_nonzero_pips(all_zero, tmp_path):
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame["credible_set"] = ""
    frame["pip"] = 0.0 if all_zero else [0.01, 0.6, 0.05, 0.1, 0.02, 0.1]
    frame["lead_selection"] = "minimum_p_no_credible_set"
    summary = fm.summarize_fit(frame)
    assert summary["n_credible_sets"] == 0
    assert summary["lead_variant"] == "variant_3"
    assert summary["lead_selection"] == "minimum_p_no_credible_set"
    assert summary["top_pip_variant"] == ("" if all_zero else "variant_1")
    assert summary["pip_status"] == ("all_zero" if all_zero else "nonzero")
    assert loci._valid_frame(frame, hit)
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    fig = plots.make_figure(frame, hit, genes)
    try:
        assert "Lead (min p)" in [text.get_text() for text in fig.legends[0].texts]
        assert any("No credible set" in text.get_text() for text in fig.texts)
        fig.savefig(tmp_path / "no_cs.pdf")
    finally:
        plt.close(fig)


def test_old_qc_cache_and_nonconverged_results_are_rejected():
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    assert loci._valid_frame(frame, hit)
    assert not loci._valid_frame(frame.drop(columns=["qc_contract"]), hit)
    frame["susie_converged"] = False
    assert not loci._valid_frame(frame, hit)


def test_credible_set_lead_excludes_higher_pip_variants_outside_sets():
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame.loc[1, "pip"] = 0.99
    summary = fm.summarize_fit(frame)
    assert summary["top_pip_variant"] == "variant_1"
    assert summary["lead_variant"] == "variant_3"
    assert summary["lead_selection"] == "credible_set_max_pip"


def test_all_nominal_titles_legends_and_panels_fit_on_saved_page(tmp_path):
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    for i, hit in loci.select_hits(*sources()).iterrows():
        frame = frame_for(hit)
        frame["credible_set"] = ""
        frame["lead_selection"] = "minimum_p_no_credible_set"
        fig = plots.make_figure(frame, hit, genes)
        try:
            fig.canvas.draw()
            boxes = [text.get_window_extent() for text in fig.texts]
            boxes.append(fig.legends[0].get_window_extent())
            assert all(fig.bbox.contains(box.x0, box.y0) and
                       fig.bbox.contains(box.x1, box.y1) for box in boxes)
            assert not any(a.overlaps(b) for j, a in enumerate(boxes) for b in boxes[j + 1:])
            assert all(box.y0 > fig.axes[0].get_window_extent().y1 for box in boxes)
        finally:
            plt.close(fig)
        if i == 0:
            paths = [tmp_path / "locus.png", tmp_path / "locus.pdf"]
            plots.render(frame, hit, genes, paths)
            loci._validate_plots(paths)


def test_marker_sources_require_matching_samples_and_checksums(staged):
    ids, _, _, _, _ = staged
    with pytest.raises(ValueError, match="sample IDs"):
        pairs.MarkerScans(("wrong",))
    hit = loci.select_hits(*sources()).iloc[0]
    scans = pairs.MarkerScans(ids)
    markers, source = scans.region(hit)
    assert len(markers) == 71
    assert markers["in_region"].any()
    assert markers["pos"].min() == fm.expected_window(hit)[0]
    assert markers["pos"].max() == fm.expected_window(hit)[1]
    assert source["lai_sample_sha256"] == corrected._sample_sha(ids)
    Path(source["scan"]["path"]).write_bytes(b"corrupt NPZ")
    with pytest.raises(ValueError, match="corrupt corrected LAI"):
        pairs.MarkerScans(ids).region(hit)


def test_changed_marker_data_only_replots_two_panel_figures(staged):
    ids, fit, render, _, _ = staged
    loci.run(ids=ids)
    stats = corrected.lai_dir() / "scan_stats.json"
    corrected._atomic_json(stats, {"mean_eur": 0.23, "sd_eur": 0.01, "z_threshold": 4})
    manifest = json.loads(corrected._stage_manifest_path("lai").read_text())
    corrected._write_stage_manifest("lai", manifest["contract"],
                                    [Path(entry["path"]) for entry in manifest["outputs"]])
    loci.run(ids=ids)
    assert fit.call_count == render.call_count == 13
    assert pairs.render.call_count == 26


def test_phewas_uses_all_discovery_tests_and_only_annotation_labels(staged):
    annotation = Path(os.environ["AOU_FIG5_PHECODE_ANNOTATIONS"])
    labels = pd.read_csv(annotation, sep="\t")
    labels["beta"] = 99  # These unrelated effects must never reach Panel A.
    labels["p_value"] = 1e-50
    # Input annotation order must not dictate the figure's category order.
    labels.iloc[::-1].to_csv(annotation, sep="\t", index=False)
    hits = loci.select_hits(*sources())
    loaded = locus_phewas.DiscoveryPhewas(hits)
    expected, _ = synthetic_phewas(hits)
    assert len(loaded.frame) == 13 + 9 * 48
    assert loaded.frame["discovery_significant"].sum() == 13
    assert loaded.categories == list(locus_phewas.CATEGORY_ORDER)
    assert loaded.source["category_order"] == list(locus_phewas.CATEGORY_ORDER)
    keys = ["region_id", "phecode"]
    np.testing.assert_allclose(loaded.frame.sort_values(keys)["p_value"], expected.sort_values(keys)["p"])
    np.testing.assert_allclose(loaded.frame.sort_values(keys)["odds_ratio"], np.exp(expected.sort_values(keys)["beta"]))
    for _, hit in hits.iterrows():
        region = loaded.region(hit)
        assert region["region_id"].eq(hit["region_id"]).all()
        assert len(region) == 48 + int(hits["region_id"].eq(hit["region_id"]).sum())


@pytest.mark.parametrize("change", ["corrupt", "cohort", "replication", "p_disc", "missing_hit"])
def test_phewas_rejects_invalid_cache_and_wrong_analysis(staged, change):
    stage_path = corrected._stage_manifest_path("discovery_phewas")
    stage = json.loads(stage_path.read_text())
    path = Path(stage["outputs"][0]["path"])
    hits = loci.select_hits(*sources())
    if change == "corrupt":
        path.write_bytes(b"corrupt gzip")
    elif change == "cohort":
        stage["contract"]["cohort"]["sample_sha256"] = "wrong cohort"
        corrected._atomic_json(stage_path, stage)
    else:
        data = pd.read_csv(path, sep="\t")
        if change == "replication":
            data["cohort"] = "replication"
        elif change == "p_disc":
            data.loc[0, "p"] = 0.0001
        else:
            data = data.iloc[1:]
        data.to_csv(path, sep="\t", index=False)
        corrected._write_stage_manifest("discovery_phewas", stage["contract"],
                                       [Path(item["path"]) for item in stage["outputs"]])
    with pytest.raises(ValueError, match="corrupt|cohort|provenance|disagree"):
        locus_phewas.DiscoveryPhewas(hits)


def test_changed_discovery_phewas_only_redraws_combined_figures(staged):
    ids, fit, render, *_ = staged
    loci.run(ids=ids)
    stage = json.loads(corrected._stage_manifest_path("discovery_phewas").read_text())
    path = Path(stage["outputs"][0]["path"])
    data = pd.read_csv(path, sep="\t")
    data.loc[data["phecode"].eq("BACKGROUND_0"), "beta"] = 0.6
    data.to_csv(path, sep="\t", index=False)
    corrected._write_stage_manifest("discovery_phewas", stage["contract"],
                                    [Path(item["path"]) for item in stage["outputs"]])
    loci.run(ids=ids)
    assert fit.call_count == render.call_count == 13
    assert pairs.render.call_count == 26
    loci.run(ids=ids)
    assert fit.call_count == render.call_count == 13
    assert pairs.render.call_count == 26


def test_sv_positions_interpolate_and_label_boundary_fallback():
    frame = pd.DataFrame({"pos": [9, 15, 19, 25, 12],
                          "sv_type": ["INS", "DEL", "INS", "DEL", ""]})
    markers = pd.DataFrame({"pos": [10, 20], "eur_frac": [0.1, 0.3]})
    svs = pairs._sv_positions(frame, markers)
    np.testing.assert_allclose(svs["overall_eur_frac"], [0.1, 0.2, 0.28, 0.3])
    assert svs["pos"].tolist() == [9, 15, 19, 25]
    assert svs["ancestry_method"].tolist() == ["nearest_boundary", "linear_interpolation",
                                               "linear_interpolation", "nearest_boundary"]
    assert svs["left_lai_marker"].tolist() == [10, 10, 10, 20]
    assert svs["right_lai_marker"].tolist() == [10, 20, 20, 20]


def test_three_panel_vectors_exact_data_shared_axes_and_layout(tmp_path):
    for i, hit in loci.select_hits(*sources()).iterrows():
        frame = frame_for(hit)
        start, end = fm.expected_window(hit)
        markers = pd.DataFrame({"pos": np.linspace(start, end, 101).astype(int),
                                "eur_frac": 0.22 + (0.05 if i % 2 else -0.05) * np.sin(np.linspace(0, np.pi, 101))})
        markers["in_region"] = markers["pos"].between(int(hit["start"]), int(hit["end"]))
        stats = {"mean_eur": 0.22, "sd_eur": 0.01, "z_threshold": 4}
        genes = pd.DataFrame([{
            "chrom": f"chr{hit['chrom']}", "start": start + 1000, "end": end - 1000,
            "gene": "SYNTHETIC_GENE", "exons": json.dumps([[start + 1000, start + 2000],
                                                          [end - 2000, end - 1000]]),
        }])
        fig = pairs.make_figure(frame, hit, markers, stats, genes=genes, lai_sample_n=1025,
                                **phewas_kwargs(hit))
        try:
            fig.canvas.draw()
            p, b, a, g = fig.axes[:4]
            assert not fig.texts  # Titles and sample/association subtitles are omitted.
            assert len(fig.axes) == 5  # A, B, C, genes, and C's colourbar.
            assert p.get_position().y0 > b.get_position().y1 > a.get_position().y1
            assert not p.get_shared_x_axes().joined(p, b)
            expected_phewas = phewas_kwargs(hit)["phewas"]
            labels = list(p.texts)
            assert len(labels) == int(expected_phewas["discovery_significant"].sum())
            label_boxes = [text.get_window_extent() for text in labels]
            assert not any(left.overlaps(right) for j, left in enumerate(label_boxes)
                           for right in label_boxes[j + 1:])
            # Repelled text can sit exactly at an edge; tolerate pixel-roundoff.
            bounds = p.bbox.padded(0.01)
            outside = [(label.get_text(), tuple(round(float(value), 3) for value in box.extents))
                       for label, box in zip(labels, label_boxes)
                       if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))]
            assert not outside, (hit["phecode"], tuple(round(float(value), 3) for value in bounds.extents), outside)
            plotted_phewas = np.concatenate([item.get_offsets()[:, 1] for item in p.collections])
            np.testing.assert_allclose(sorted(plotted_phewas), sorted(-np.log10(expected_phewas["p"])))
            assert len(plotted_phewas) > len(expected_phewas[expected_phewas["p"] < 0.001])
            assert a.get_ylabel() == plots.style.P_VALUE_AXIS_LABEL
            assert b.get_ylabel() == "EUR-like ancestry\nfraction"
            assert fig.get_figheight() < 10.3
            assert fig.get_figwidth() == pytest.approx(8.25)
            np.testing.assert_allclose(np.diff(p.get_yticks()), 1)
            assert [label.get_text() for label in p.get_yticklabels()] == [
                f"{value:.1f}" for value in p.get_yticks()]
            assert a.get_legend() is None
            bar = fig.axes[-1]
            assert bar.get_position().width > bar.get_position().height
            assert bar.get_position().y0 > a.get_position().y1
            np.testing.assert_allclose(bar.get_xticks(), np.linspace(0, 1, 6))
            ld_label = next(text for text in a.texts if text.get_text() == pairs.LD_LABEL)
            assert not ld_label.get_window_extent().overlaps(bar.get_window_extent())
            assert ld_label.get_ha() == "right"
            assert (bar.bbox.x0 - ld_label.get_window_extent().x1) / fig.dpi == pytest.approx(0.08)
            assert (ld_label.get_window_extent().y0 + ld_label.get_window_extent().y1) / 2 == pytest.approx(
                (bar.bbox.y0 + bar.bbox.y1) / 2)
            assert b.bbox.y0 > ld_label.get_window_extent().y1
            assert a.bbox.y1 < ld_label.get_window_extent().y0
            assert a.get_xlim() == b.get_xlim() == g.get_xlim() == (start / 1e6, end / 1e6)
            assert g.get_ylabel() == "Genes"
            assert g.get_xlabel() == f"chr{hit['chrom']} position (Mb, GRCh38)"
            assert g.get_position().y1 < a.get_position().y0
            assert not b.get_xlabel() and not b.get_xticklabels()
            np.testing.assert_allclose(g.lines[0].get_xdata(), [(start + 1000) / 1e6, (end - 1000) / 1e6])
            assert [text.get_text() for text in g.texts] == ["SYNTHETIC_GENE"]
            assert len(g.patches) == 3  # Outlier shading plus the two exons.
            assert g.patches[1].get_x() == pytest.approx((start + 1000) / 1e6)
            ordinary = frame["sv_type"].eq("")
            np.testing.assert_allclose(a.collections[0].get_sizes(), [22 * pairs.POINT_AREA_SCALE])
            np.testing.assert_allclose(a.collections[1].get_sizes(), [80 * pairs.POINT_AREA_SCALE])
            np.testing.assert_allclose(a.collections[0].get_offsets(),
                np.column_stack([frame.loc[ordinary, "pos"] / 1e6, frame.loc[ordinary, "neg_log10_p"]]))
            actual = np.concatenate([item.get_offsets() for item in b.collections[:2]])
            actual = actual[np.argsort(actual[:, 0])]
            np.testing.assert_allclose(actual[:, 0], markers["pos"] / 1e6)
            np.testing.assert_allclose(actual[:, 1], markers["eur_frac"])
            np.testing.assert_allclose([line.get_ydata()[0] for line in b.lines], [0.22, 0.26, 0.18])
            assert not any(item.get_rasterized() for axis in fig.axes for item in axis.collections)
            boxes = [text.get_window_extent() for text in fig.texts]
            from matplotlib.legend import Legend
            assert "Discovery hits" not in [text.get_text() for legend in fig.findobj(match=Legend)
                                              for text in legend.get_texts()]
            assert p.get_legend().get_title().get_text() == "OR fold (log scale)"
            legend_sizes = [handle.get_markersize() ** 2 for handle in p.get_legend().legend_handles]
            parameters = phewas_kwargs(hit)["phewas_parameters"]
            np.testing.assert_allclose([legend_sizes[0], legend_sizes[-1]],
                [parameters.marker_area_min * pairs.POINT_AREA_SCALE,
                 parameters.marker_area_max * pairs.POINT_AREA_SCALE])
            boxes += [legend.get_window_extent() for legend in fig.findobj(match=Legend)]
            assert all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1) for box in boxes)
            assert not any(left.overlaps(right) for j, left in enumerate(boxes) for right in boxes[j+1:])
            assert all(not box.overlaps(axis.get_window_extent()) for box in boxes for axis in (p, a, b))
            tick_boxes = [text.get_window_extent() for text in p.get_xticklabels()]
            assert_category_labels_clear(p)
            assert [text.get_text() for text in p.get_xticklabels()] == [
                locus_phewas.CATEGORY_LABELS[category] for category in locus_phewas.CATEGORY_ORDER]
            collisions = [(p.get_xticklabels()[j].get_text(), box.bounds, legend_box.bounds)
                          for j, box in enumerate(tick_boxes) for legend_box in boxes if box.overlaps(legend_box)]
            assert not collisions, collisions
            assert not any(text.get_text() in {"A", "B", "C"} for axis in fig.axes for text in axis.texts)
            for axis in (p, a, b, g):
                box = axis.yaxis.label.get_window_extent()
                assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), box.bounds
            if i == 0:
                pdf = tmp_path / "three_panel.pdf"
                fig.savefig(pdf)
                with pymupdf.open(pdf) as document:
                    assert not document[0].get_images(full=True)
                    assert all(font[2] != "Type3" for font in document[0].get_fonts())
                    assert "EUR-like ancestry fraction" in " ".join(document[0].get_text().split())
                    assert pairs.LD_LABEL in document[0].get_text()
                    assert "Discovery hits" not in document[0].get_text()
                    assert "OR fold (log scale)" in document[0].get_text()
                    assert "cap 4" not in document[0].get_text()
                    assert "SNV / small indel" not in document[0].get_text()
                    assert "SYNTHETIC_GENE" in document[0].get_text()
                    assert "Discovery n" not in document[0].get_text()
                    assert "LAI cohort" not in document[0].get_text()
                    assert str(hit["phecode"]) not in document[0].get_text()
        finally:
            plt.close(fig)
    paths = [tmp_path / "pair.png", tmp_path / "pair.pdf"]
    pairs.render(frame, hit, markers, stats, paths, genes=genes, lai_sample_n=1025, **phewas_kwargs(hit))
    loci._validate_plots(paths)


def test_bottom_gene_tracks_pack_dense_labels_and_handle_empty_windows(tmp_path):
    hit = loci.select_hits(*sources()).iloc[0]
    start, end = fm.expected_window(hit)
    frame = frame_for(hit)
    markers = pd.DataFrame({"pos": [start, end], "eur_frac": [0.22, 0.23], "in_region": False})
    stats = {"mean_eur": 0.22, "sd_eur": 0.01, "z_threshold": 4}
    genes = pd.DataFrame([{
        "chrom": str(hit["chrom"]), "start": start + 1000 + i, "end": end - 1000 - i,
        "gene": f"LONG_SYNTHETIC_GENE_{i}", "exons": json.dumps([[start + 1000 + i, start + 2000 + i]]),
    } for i in range(16)])
    for annotations in (genes, genes.iloc[:0]):
        fig = pairs.make_figure(frame, hit, markers, stats, genes=annotations, lai_sample_n=1025,
                                **phewas_kwargs(hit))
        try:
            fig.canvas.draw()
            gene_axis = fig.axes[3]
            boxes = [text.get_window_extent() for text in gene_axis.texts]
            assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
            assert all(gene_axis.bbox.contains(box.x0, box.y0) and
                       gene_axis.bbox.contains(box.x1, box.y1) for box in boxes)
            assert all(not box.overlaps(fig.axes[1].get_window_extent()) for box in boxes)
            assert all(text.get_fontstyle() == "italic" for text in gene_axis.texts) if len(annotations) else (
                gene_axis.texts[0].get_text() == "No annotated genes in window")
            # Adding lanes preserves the physical heights of both data panels.
            for axis, height in zip(fig.axes[:3], pairs.PANEL_HEIGHTS):
                assert axis.get_position().height * fig.get_figheight() == pytest.approx(height)
            if len(annotations):
                path = tmp_path / "dense_genes.pdf"
                fig.savefig(path)
                with pymupdf.open(path) as pdf:
                    assert not pdf[0].get_images(full=True)
                    assert "LONG_SYNTHETIC_GENE_15" in pdf[0].get_text()
        finally:
            plt.close(fig)


def test_category_spacing_handles_singletons_beside_dense_groups_without_changing_effects():
    hit = loci.select_hits(*sources()).iloc[0]
    kwargs = phewas_kwargs(hit)
    frame = kwargs["phewas"].loc[lambda data: data["discovery_significant"]].copy()
    rows = []
    counts = [6, 8, 120, 40, 200, 1, 80, 90, 30, 60, 260, 2, 1, 3, 12, 5]
    for category, count in zip(locus_phewas.CATEGORY_ORDER, counts):
        for index in range(count):
            rows.append({"phecode": f"{category}_{index}", "phecode_category": category,
                         "phecode_name": f"Background {category} {index}", "p_value": 0.01 + 0.98 * index / count,
                         "p": 0.01 + 0.98 * index / count, "beta": 0.2, "odds_ratio": np.exp(0.2),
                         "discovery_significant": False})
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    before = frame.copy(deep=True)
    figure = plt.figure(figsize=(pairs.FIGURE_WIDTH, 4.5))
    axis = figure.add_axes([pairs.AXIS_LEFT, 0.43, pairs.AXIS_RIGHT - pairs.AXIS_LEFT, 0.45])
    try:
        locus_phewas.draw_panel(axis, frame, hit, kwargs["categories"][::-1], kwargs["phewas_parameters"])
        figure.canvas.draw()
        assert_category_labels_clear(axis)
        assert [label.get_text() for label in axis.get_xticklabels()] == list(locus_phewas.CATEGORY_LABELS.values())
        actual_p = np.concatenate([collection.get_offsets()[:, 1] for collection in axis.collections])
        np.testing.assert_allclose(sorted(actual_p), sorted(-np.log10(frame["p_value"])))
        # Background tests still descend by significance within every category.
        offsets = axis.collections[0].get_offsets()
        cursor = 0
        for count, tick in zip(counts, axis.get_xticks()):
            local = offsets[cursor:cursor + count]
            assert len(local) == count
            if count == 1:
                np.testing.assert_allclose(local[:, 0].mean(), tick)
            assert (np.diff(local[:, 0]) > 0).all()
            assert (np.diff(local[:, 1]) <= 0).all()
            cursor += count
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(figure)


def test_combined_ld_colors_and_outline_use_minimum_p_when_susie_lead_differs():
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame.loc[0, ["p_score", "neg_log10_p"]] = [1e-8, 8]
    frame["r2_to_min_p"] = [1, 0.6, 0.3, 0.01, 0.2, 0.1]
    before = frame.copy(deep=True)
    ordinary = frame["sv_type"].eq("")
    for reference, expected_index, r2_column in (("minimum_p", 0, "r2_to_min_p"),
                                                ("lead", 3, "r2_to_lead")):
        figure, axis = plt.subplots()
        try:
            plots.draw_variants(frame, [(axis, frame["neg_log10_p"])], ld_reference=reference)
            np.testing.assert_allclose(axis.collections[0].get_array(), frame.loc[ordinary, r2_column])
            np.testing.assert_allclose(axis.collections[1].get_offsets(),
                [[frame.loc[expected_index, "pos"], frame.loc[expected_index, "neg_log10_p"]]])
        finally:
            plt.close(figure)
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    result = pairs.association_results(frame, hit, genes)
    assert result["combined_ld_reference_variant"] == frame.loc[0, "variant_id"]
    assert result["lead_variant"] == frame.loc[3, "variant_id"]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("reference", ["minimum_p", "lead"])
@pytest.mark.parametrize("index,sv_type,marker,border,length", [
    (0, "", "o", "black", 0), (1, "INS", "^", "none", 70), (5, "DEL", "v", "none", 85)])
def test_highlighted_leads_preserve_sv_shapes_borders_and_table_status(
        tmp_path, reference, index, sv_type, marker, border, length):
    from matplotlib.markers import MarkerStyle

    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame.loc[index, ["p_score", "neg_log10_p", "r2_to_min_p"]] = [1e-8, 8, 1]
    if reference == "lead":
        frame["lead_variant"] = frame.index == index
        frame["credible_set"] = np.where(frame.index == index, "L1", "")
        frame.loc[index, "r2_to_lead"] = 1
    figure, axes = plt.subplots(2)
    try:
        _, handles = plots.draw_variants(frame, [(axes[0], frame["neg_log10_p"]),
            (axes[1], frame["pip"])], ld_reference=reference)
        expected_marker = MarkerStyle(marker)
        for axis in axes:
            highlighted = [item for item in axis.collections if item.get_zorder() == 5]
            assert len(highlighted) == 1
            np.testing.assert_allclose(highlighted[0].get_paths()[0].vertices,
                expected_marker.get_path().transformed(expected_marker.get_transform()).vertices)
            if border == "none":
                assert highlighted[0].get_edgecolors().size == 0
            else:
                np.testing.assert_allclose(highlighted[0].get_edgecolors(),
                                          [matplotlib.colors.to_rgba(border)])
            assert not highlighted[0].get_rasterized()
        assert handles[-1].get_marker() == marker
        assert handles[-1].get_markeredgecolor() == border
        path = tmp_path / "lead_markers.pdf"
        figure.savefig(path)
        with pymupdf.open(path) as pdf:
            assert not pdf[0].get_images(full=True)
    finally:
        plt.close(figure)
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    result = pairs.association_results(frame, hit, genes)
    assert result["min_p_is_sv"] == bool(sv_type)
    assert result["min_p_sv_type"] == sv_type
    assert result["min_p_sv_length_bp"] == length
    assert result["lead_is_sv"] == (bool(sv_type) if reference == "lead" else False)
    assert result["lead_sv_type"] == (sv_type if reference == "lead" else "")


def test_changed_gene_annotations_redraw_both_outputs_without_refitting(staged):
    ids, fit, original_render, *_ = staged
    loci.run(ids=ids)
    with corrected.analysis_context(ids):
        path = fm.analysis_dir() / "refgene_transcripts.tsv.gz"
        genes = pd.DataFrame([{"chrom": "chr1", "start": 239400000, "end": 239410000,
                               "gene": "UPDATED_SYNTHETIC_GENE", "exons": "[]"}])
        genes.to_csv(path, sep="\t", index=False)
        fm._gene_table.return_value = genes
    loci.run(ids=ids)
    assert fit.call_count == 13
    assert original_render.call_count == pairs.render.call_count == 26
    pd.testing.assert_frame_equal(pairs.render.call_args.kwargs["genes"], genes)
    loci.run(ids=ids)
    assert fit.call_count == 13 and pairs.render.call_count == 26


def test_mask_content_change_refits_once_even_at_same_size_and_mtime(staged, synthetic_finemap_mask):
    import os
    ids, fit, *_ = staged
    loci.run(ids=ids, render=False)
    assert fit.call_count == 13
    loci.run(ids=ids, render=False)
    assert fit.call_count == 13
    previous = synthetic_finemap_mask.stat()
    synthetic_finemap_mask.write_text("chr1\t300000002\t300000010\n", encoding="utf-8")
    os.utime(synthetic_finemap_mask, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    loci.run(ids=ids, render=False)
    assert fit.call_count == 26
    loci.run(ids=ids, render=False)
    assert fit.call_count == 26


def test_results_distinguish_minimum_p_lead_and_top_pip():
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame["p_score"] = [0.000001, 0.05, 0.02, 0.01, 0.5, 0.02]
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    result = pairs.association_results(frame, hit, genes)
    assert result["min_p"] == 0.000001
    assert result["min_p_variant"] == "variant_0" and result["min_p_pip"] == 0.01
    assert result["lead_variant"] == result["top_pip_variant"] == "variant_3"
    assert result["lead_pip"] == result["top_pip"] == 0.8
    assert result["n_credible_sets"] == result["n_credible_set_variants"] == 1
    assert result["credible_set_ids"] == "L1"
    assert result["window_start"] == fm.expected_window(hit)[0]
    assert result["n_controls"] == 7650
    frame["pip"], frame["credible_set"] = 0.0, ""
    result = pairs.association_results(frame, hit, genes)
    assert result["n_credible_sets"] == result["top_pip"] == 0
    assert result["top_pip_variant"] == ""
    assert result["lead_variant"] == result["min_p_variant"] == "variant_0"
    assert result["lead_selection"] == "minimum_p_no_credible_set"


@pytest.mark.parametrize("minimum_index", [0, 4])
def test_all_credible_variants_and_minimum_p_representative_round_trip(tmp_path, minimum_index):
    results = pairs.locus_results
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame["credible_set"] = ["L1", "L1,L2", "L2", "", "", ""]
    # A low-PIP member must not disappear; overlapping sets must retain both IDs.
    frame.loc[2, "pip"] = 0.0001
    frame.loc[1, "variant_id"] = "chr1:123:A:" + "ACGT" * 300
    frame.loc[minimum_index, "p_score"] = 1e-10
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    summary = pairs.association_results(frame, hit, genes)
    expected = frame.loc[minimum_index]
    assert summary["credible_set_variants"].split(";") == frame.loc[:2, "variant_id"].tolist()
    assert summary["n_credible_sets"] == 2 and summary["n_credible_set_variants"] == 3
    assert summary["representative_variant"] == expected["variant_id"]
    assert summary["representative_in_credible_set"] == (minimum_index == 0)
    assert summary["representative_credible_set_ids"] == ("L1" if minimum_index == 0 else "")
    for column in ("beta_score", "se_score", "p_score", "pip", "mac", "n_carriers"):
        assert summary[f"representative_{column}"] == expected[column]
    rows = results.member_rows(frame, hit, tmp_path / "source.tsv.gz")
    csv_path, tsv_path = tmp_path / "members.csv", tmp_path / "members.tsv"
    results.write_members(rows, csv_path, tsv_path)
    table = pd.read_csv(csv_path, keep_default_na=False)
    pd.testing.assert_frame_equal(table, pd.read_csv(tsv_path, sep="\t", keep_default_na=False))
    assert table["variant_id"].tolist() == frame.loc[:2, "variant_id"].tolist()
    assert table["credible_set_ids"].tolist() == ["L1", "L1,L2", "L2"]
    assert table["is_representative"].tolist() == [minimum_index == 0, False, False]
    assert table["representative_variant"].eq(expected["variant_id"]).all()
    for column in ("beta_score", "se_score", "p_score", "pip", "mac", "n_carriers"):
        np.testing.assert_allclose(table[column], frame.loc[:2, column])
    assert table.loc[1, "sv_type"] == "INS" and table.loc[1, "is_sv"]
    assert table.loc[1, "sv_length_bp"] == 70


def test_no_credible_set_saves_headers_and_keeps_representative_statistics(tmp_path):
    results = pairs.locus_results
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame["credible_set"] = ""
    frame["pip"] = 0.0
    summary = results.summary_fields(frame)
    assert summary["credible_set_variants"] == ""
    assert not summary["representative_in_credible_set"]
    assert summary["representative_variant"] == "variant_3"
    assert summary["representative_beta_score"] == 0.5
    assert summary["representative_pip"] == 0.0
    rows = results.member_rows(frame, hit, tmp_path / "source.tsv.gz")
    assert rows == []
    csv_path, tsv_path = tmp_path / "members.csv", tmp_path / "members.tsv"
    results.write_members(rows, csv_path, tsv_path)
    for path, separator in ((csv_path, ","), (tsv_path, "\t")):
        table = pd.read_csv(path, sep=separator)
        assert table.empty and list(table) == results.MEMBER_COLUMNS


def test_credible_variant_table_preserves_missing_statistics_and_rejects_duplicates(tmp_path):
    results = pairs.locus_results
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame.loc[3, "beta_score"] = np.nan
    rows = results.member_rows(frame, hit, tmp_path / "source.tsv.gz")
    csv_path, tsv_path = tmp_path / "members.csv", tmp_path / "members.tsv"
    results.write_members(rows, csv_path, tsv_path)
    assert pd.isna(pd.read_csv(csv_path).loc[0, "beta_score"])
    before = csv_path.read_bytes()
    with pytest.raises(ValueError, match="Duplicate association/variant"):
        results.write_members(rows * 2, csv_path, tsv_path)
    assert csv_path.read_bytes() == before
