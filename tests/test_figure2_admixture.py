from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf

from scripts import figure2_data, figure2_plots, figure2_sv
from scripts.common import config
from panel_fixtures import admixture_frame


def test_all_figure2_panels_output_as_native_vectors(tmp_path: Path) -> None:
    data = admixture_frame()
    for letter, panel in figure2_plots.PANELS.items():
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        panel(axis, data)
        rasterized_collections = [
            collection
            for panel_axis in figure.axes
            for collection in panel_axis.collections
            if collection.get_rasterized()
        ]
        assert not rasterized_collections, f"Figure 2{letter} rasterizes collections"
        output = tmp_path / f"Figure_2_1025{letter}.pdf"
        figure.savefig(output)
        plt.close(figure)
        with pymupdf.open(output) as document:
            image_count = sum(
                len(page.get_images(full=True)) for page in document
            )
            assert image_count == 0, (
                f"Figure 2{letter} PDF contains {image_count} image objects"
            )


def test_panel_a_uses_equal_chromosome_slots_and_reports_numeric_thresholds() -> None:
    data = admixture_frame()
    thresholds = figure2_plots.ancestry_thresholds(data)
    fig, ax = plt.subplots()
    try:
        figure2_plots.panel_A(ax, data)
        assert ax.get_xlim() == (0.0, 22.0)
        np.testing.assert_allclose(ax.get_xticks(), np.arange(22) + 0.5)
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            str(value) for value in range(1, 23)
        ]

        regular_facecolors = ax.collections[0].get_facecolors()
        expected_colors = {
            tuple(matplotlib.colors.to_rgba("#ADD8E6", alpha=0.7)),
            tuple(matplotlib.colors.to_rgba("#2C7A7B", alpha=0.7)),
        }
        assert expected_colors.issubset(map(tuple, regular_facecolors))
        outlier_facecolors = ax.collections[1].get_facecolors()
        expected_orange = matplotlib.colors.to_rgba(
            figure2_plots.style.PALETTE["orange"], alpha=0.95
        )
        assert np.allclose(outlier_facecolors[0], expected_orange)

        horizontal_values = [float(line.get_ydata()[0]) for line in ax.lines]
        np.testing.assert_allclose(
            horizontal_values,
            [thresholds["mean"], thresholds["upper"], thresholds["lower"]],
        )
        legend_text = " ".join(
            text.get_text() for text in ax.get_legend().get_texts()
        )
        assert "+/-4.42 SD =" in legend_text
        assert f"{thresholds['lower']:.3f}" in legend_text
        assert f"{thresholds['upper']:.3f}" in legend_text
    finally:
        plt.close(fig)


def test_shifted_main_panel_contract_and_supplemental_render(tmp_path: Path) -> None:
    data = admixture_frame()
    assert tuple(figure2_plots.PANELS) == ("A", "B", "C", "D", "E", "F")
    assert tuple(config.FIGURES["figure2_sources"]["panels"]) == tuple("ABCDEF")
    with patch.object(config, "figure_dir", return_value=tmp_path):
        paths = figure2_plots.render_supplemental_tracts(
            data, clean_run_plots=True
        )
    assert [path.suffix for path in paths] == [".pdf", ".png"]
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)
    assert all(path.parent.name == "panels" for path in paths)


def test_plrs_model_quality_panels_use_the_complete_summary_contract() -> None:
    data = admixture_frame()
    models = data["plrs_models"]
    assert {"region_id", "n_snvs", "r2_holdout"}.issubset(models.columns)
    assert int((models["r2_holdout"] > 0.5).sum()) == 36

    fig, axes = plt.subplots(1, 2)
    try:
        figure2_plots.panel_C(axes[0], data)
        figure2_plots.panel_D(axes[1], data)
        assert axes[0].get_xlabel() == "Held-out PLRS $R^2$"
        assert axes[0].get_ylabel() == "Number of regions"
        assert axes[0].patches
        assert any(
            np.allclose(line.get_xdata(), [0.5, 0.5])
            for line in axes[0].lines
        )
        assert axes[1].get_xlabel() == "SNVs in regional PLRS model"
        assert axes[1].get_ylabel() == "Held-out PLRS $R^2$"
        assert len(axes[1].collections) == 2
        assert any(
            text.get_text().startswith("Pearson r=") for text in axes[1].texts
        )
        assert "$R^2 > 0.5$ (n=36)" in {
            text.get_text() for text in axes[1].get_legend().get_texts()
        }
    finally:
        plt.close(fig)


def test_figure2_notebook_uses_admixture_cache_and_reports_outputs() -> None:
    notebook = json.loads(
        (config.NOTEBOOKS_DIR / "Figure_2.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "figure2_1025.run_all" in source
    assert "figure2_1025_plots.load_panel_data" in source
    assert "figure2_1025_plots.render_panel" in source
    assert "figure2_1025_plots.render_supplemental" in source
    assert "figure2_loci.run" in source


def test_locus_panels_share_100kb_window_and_show_sv_contract() -> None:
    data = admixture_frame()
    region = data["display_region"]
    expected = (
        int(region["start"]) - 100_000,
        int(region["end"]) + 100_000,
    )
    assert data["locus_window"] == expected
    assert set(data["regional_svs"]["sv_type"]) == {"INS", "DEL"}
    zoom = data["region_zoom"].set_index("pos")
    for sv in data["regional_svs"].itertuples(index=False):
        assert sv.overall_eur_frac == zoom.loc[sv.nearest_lai_marker, "eur_frac"]
        assert sv.ancestry_method == "overall_nearest_marker"

    fig = plt.figure()
    ancestry_axis = fig.add_subplot(1, 2, 1)
    finemap_parent = fig.add_subplot(1, 2, 2)
    try:
        figure2_plots.panel_E(ancestry_axis, data)
        figure2_plots.panel_F(finemap_parent, data, colorbar_side="left")
        fig.canvas.draw()
        np.testing.assert_allclose(
            ancestry_axis.get_xlim(), np.asarray(expected) / 1e6
        )
        association_axis, gene_axis, colorbar_axis = finemap_parent.child_axes
        parent_box = finemap_parent.get_position()
        association_box = association_axis.get_position()
        gene_box = gene_axis.get_position()
        colorbar_box = colorbar_axis.get_position()
        assert np.isclose(association_box.x0, parent_box.x0)
        assert np.isclose(association_box.x1, parent_box.x1)
        assert np.isclose(gene_box.x0, parent_box.x0)
        assert np.isclose(gene_box.x1, parent_box.x1)
        assert colorbar_box.x1 < parent_box.x0
        assert colorbar_axis.yaxis.get_label_position() == "left"
        np.testing.assert_allclose(association_axis.get_xlim(), expected)
        np.testing.assert_allclose(gene_axis.get_xlim(), expected)
        assert gene_axis.patches, "expected Figure 3-style exon rectangles"
        assert any(collection.get_paths() for collection in ancestry_axis.collections[2:])
        red = matplotlib.colors.to_rgba("red")
        assert any(
            len(collection.get_edgecolors())
            and np.allclose(collection.get_edgecolors()[0], red)
            for collection in association_axis.collections
        )
        lead_size = 72
        assert any(
            lead_size in collection.get_sizes()
            for collection in association_axis.collections
        )
        assert association_axis.get_legend() is None
        viridis = matplotlib.colormaps["viridis"]
        for sv in data["regional_svs"].itertuples(index=False):
            if not bool(sv.exact_finemap_match):
                continue
            expected_face = viridis(float(sv.r2_to_min_p))
            matching = [
                collection
                for collection in association_axis.collections
                if len(collection.get_offsets()) == 1
                and np.isclose(collection.get_offsets()[0, 0], sv.pos)
                and len(collection.get_edgecolors())
                and np.allclose(collection.get_edgecolors()[0], red)
            ]
            assert len(matching) == 1
            assert np.allclose(matching[0].get_facecolors()[0], expected_face)
    finally:
        plt.close(fig)


def test_panel_f_does_not_plot_unmatched_sv_as_high_association() -> None:
    data = admixture_frame()
    unmatched = data["regional_svs"].iloc[0].copy()
    unmatched["pos"] = int(data["display_region"]["end"]) + 50_000
    unmatched["variant_id"] = "unmatched-sv"
    unmatched["exact_finemap_match"] = False
    unmatched["neg_log10_p"] = np.nan
    unmatched["r2_to_min_p"] = np.nan
    data["regional_svs"] = pd.concat(
        [data["regional_svs"], unmatched.to_frame().T], ignore_index=True
    )

    fig, parent = plt.subplots()
    try:
        figure2_plots.panel_F(parent, data)
        association_axis = parent.child_axes[0]
        red = matplotlib.colors.to_rgba("red")
        red_sv_x = [
            float(collection.get_offsets()[0, 0])
            for collection in association_axis.collections
            if len(collection.get_offsets()) == 1
            and len(collection.get_edgecolors())
            and np.allclose(collection.get_edgecolors()[0], red)
        ]
        assert float(unmatched["pos"]) not in red_sv_x
    finally:
        plt.close(fig)


def test_sv_ancestry_uses_all_haplotypes_at_nearest_marker() -> None:
    result = figure2_sv.nearest_marker_overall_ancestry(
        150,
        np.asarray([100, 200]),
        np.asarray([[0, 1], [1, 1]]),
        np.asarray([[1, 0], [1, 1]]),
        eur_index=1,
        afr_index=0,
    )
    assert result["left_lai_marker"] == 100
    assert result["right_lai_marker"] == 200
    assert result["nearest_lai_marker"] == 100
    assert result["nearest_lai_distance_bp"] == 50
    assert result["n_evaluable_haplotypes"] == 4
    assert result["n_eur_haplotypes"] == 2
    assert result["n_afr_haplotypes"] == 2
    assert result["n_other_haplotypes"] == 0
    assert result["overall_eur_frac"] == 0.5
    assert result["ancestry_method"] == "overall_nearest_marker"


def test_sv_finemap_match_requires_sv_type_and_reports_effect_and_r2() -> None:
    svs = pd.DataFrame([
        {"variant_id": "sv1", "pos": 100, "sv_type": "INS", "sv_length_bp": 80},
        {"variant_id": "sv2", "pos": 200, "sv_type": "DEL", "sv_length_bp": 70},
    ])
    finemap = pd.DataFrame([
        {"variant_id": "snv_at_100", "pos": 100, "is_sv": False,
         "sv_type": "", "sv_length_bp": 0, "beta_score": 9.0, "p_score": 1e-12,
         "neg_log10_p": 12.0, "r2_to_min_p": 1.0},
        {"variant_id": "ins_at_100", "pos": 100, "is_sv": True,
         "sv_type": "INS", "sv_length_bp": 80, "beta_score": 0.7, "p_score": 1e-8,
         "neg_log10_p": 8.0, "r2_to_min_p": 0.82},
    ])
    audit = figure2_sv.match_finemap_svs(svs, finemap)
    ins = audit.set_index("variant_id").loc["sv1"]
    deletion = audit.set_index("variant_id").loc["sv2"]
    assert bool(ins["exact_finemap_match"])
    assert ins["finemap_variant_id"] == "ins_at_100"
    assert ins["beta_score"] == 0.7
    assert ins["r2_to_min_p"] == 0.82
    assert not bool(deletion["exact_finemap_match"])


def test_sv_source_contract_matches_s11_s24_shapes_and_phasing() -> None:
    assert figure2_sv.classify_sv("A", "A" * 84) == ("INS", 83)
    assert figure2_sv.classify_sv("A" * 68, "A") == ("DEL", 67)
    assert figure2_sv.classify_sv("A", "T") is None
    carriers, unphased = figure2_sv.phased_carrier_haplotypes(
        ["phased", "unphased", "homozygous"],
        [[0, 1, True], [0, 1, False], [1, 1, False]],
    )
    assert carriers == [("phased", 2), ("homozygous", 1), ("homozygous", 2)]
    assert unphased == 1


def test_sv_source_is_phased_snv_sv_panel_not_finemap_discovery_vcf() -> None:
    source = figure2_sv.sv_source_path(9)
    normalized = str(source).replace("\\", "/")
    assert normalized.endswith("SV_biallelic/chr9.biallelic.vcf.gz")
    assert "AoU12k" not in normalized


def test_figure2_runtime_is_uv_only_and_plot_focused() -> None:
    requirements = (config.REPO_ROOT / "requirements-figure2-loci.txt").read_text(
        encoding="utf-8"
    ).lower()
    active_requirements = "\n".join(
        line for line in requirements.splitlines()
        if line and not line.lstrip().startswith("#")
    )
    for package in ("numpy", "pandas", "matplotlib", "ipykernel", "nbclient"):
        assert package in active_requirements
    assert "cyvcf2" in active_requirements
    # PLSR is not refitted, but sklearn is needed to read feature metadata
    # from the existing model pickle for the upstream genomic-mask audit.
    assert "scikit-learn" in active_requirements
    for excluded in ("phetk",):
        assert excluded not in active_requirements
