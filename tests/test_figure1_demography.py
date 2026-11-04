from __future__ import annotations

import json
import logging
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.text import Text
from PIL import Image

from scripts import figure1_data, figure1_demography, figure1_plots
from scripts.common import config
from scripts import figure1_model_fits as s1


def test_main_figure_sfs_parameter_vector_is_c2_best_fit() -> None:
    c2 = s1.MODEL_SPECS["C2"]
    canonical = np.array([*c2.parameters, c2.misid])
    assert figure1_demography.BEST_FIT_MODEL_ID == "C2"
    assert np.array_equal(figure1_demography.BEST_FIT_MIG, canonical)


def test_cached_aou_sfs_is_bound_to_c2_best_fit() -> None:
    fit = figure1_demography.load_aou_fit()
    assert fit is not None
    assert set(fit["model_id"]) == {"C2"}
    assert set(fit["model_parameter_sha256"]) == {
        figure1_demography.best_fit_parameter_sha256()
    }
    assert set(fit["model_grid_points"].astype(str)) == {"35,45,55"}
    assert set(fit["model_scaling"]) == {"optimal_to_AoU_1D_SFS"}
    assert fit["model_theta"].nunique() == 1


def test_figure1e_physical_labels_match_corrected_c2_table() -> None:
    physical = figure1_demography.best_fit_physical()
    assert physical["N_ancestral_start"] == 11727
    assert physical["N_ancestral_start_ci"] == (11121, 12438)
    assert physical["N_ancestral_end"] == 18277
    assert physical["N_ancestral_end_ci"] == (17018, 19830)
    assert physical["T_ancestral_years"] == 246260
    assert physical["T_ancestral_years_ci"] == (188454, 334352)
    assert physical["T_split_years"] == 44129
    assert physical["T_split_years_ci"] == (37592, 53392)
    assert physical["N_EUR_start"] == 921
    assert physical["N_EUR_start_ci"] == (714, 1195)
    assert physical["N_EUR_end"] == 8529
    assert physical["N_EUR_end_ci"] == (7078, 10867)
    assert physical["mig_eur_to_afr"] == 7.4e-5
    assert physical["mig_eur_to_afr_ci"] == (5.8e-5, 9.4e-5)
    assert physical["mig_afr_to_eur"] == 5.2e-5
    assert physical["mig_afr_to_eur_ci"] == (4.0e-5, 6.3e-5)


def test_figure1e_uses_supplementary_c2_demes_graph() -> None:
    graph = figure1_demography.best_fit_demes_graph()
    assert [deme.name for deme in graph.demes] == ["AFR", "EUR", "AoU"]
    assert graph["AFR"].end_time == 0
    assert graph["AoU"].ancestors == ["AFR", "EUR"]
    assert graph["AoU"].start_time == s1.AOU_PLOT_START_TIME_YEARS
    assert len(graph.migrations) == 2
    migrations = {
        (migration.source, migration.dest): migration.rate
        for migration in graph.migrations
    }
    assert migrations[("EUR", "AFR")] == 7.4e-5
    assert migrations[("AFR", "EUR")] == 5.2e-5


def test_figure1e_renderer_writes_high_resolution_outputs(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="matplotlib.font_manager")
    png, pdf, manifest = figure1_demography.render_demographic_model_schematic(
        output_dir=tmp_path,
        dpi=300,
    )
    assert png.exists() and pdf.exists() and manifest.exists()
    with Image.open(png) as image:
        assert image.width >= 3500
        assert image.height >= 1900
        assert np.allclose(image.info["dpi"], (300, 300), atol=0.1)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["model_id"] == "C2"
    assert metadata["renderer"] == "demesdraw.tubes"
    assert metadata["aou_plot_start_time_years"] == 1.2
    assert metadata["requested_dpi"] == 300
    assert metadata["font_file"] == "DejaVuSans.ttf"
    assert metadata["font_weight"] == 400
    assert min(metadata["font_sizes_points"].values()) >= 20
    assert metadata["png"] == png.name
    assert metadata["pdf"] == pdf.name
    assert "Failed to find font weight" not in caplog.text


def test_panel_sfs_uses_requested_publication_labels_and_frequency_axis() -> None:
    fit = figure1_demography.load_aou_fit()
    figure, axis = plt.subplots()
    figure1_plots.panel_sfs(axis, {"sfs_aou_fit": fit})
    legend_labels = [
        text.get_text()
        for child_axis in axis.child_axes
        if child_axis.get_legend() is not None
        for text in child_axis.get_legend().get_texts()
    ]
    assert legend_labels == ["data", "model"]
    top, bottom = axis.child_axes
    legend = top.get_legend()
    assert legend is not None
    assert legend._ncols == 2
    assert legend.get_bbox_to_anchor().y0 > top.get_window_extent().y1
    assert not legend.get_frame_on()
    assert top.get_ylabel() == "Number of SNPs"
    assert bottom.get_xlabel() == "Allele Frequency"
    expected_frequency = fit.loc[
        fit["allele_count"].between(1, fit["allele_count"].max() - 1),
        "allele_count",
    ].to_numpy(dtype=float) / fit["allele_count"].max()
    assert np.allclose(top.lines[0].get_xdata(), expected_frequency)
    plt.close(figure)


def test_panel_map_uses_40_count_colorbar_tick_interval() -> None:
    figure, axis = plt.subplots()
    figure1_plots.panel_map(
        axis, {"state_counts": pd.DataFrame({"state": ["CA", "NY"], "n": [200, 300]})}
    )
    colorbar_axis = axis.child_axes[0]
    assert colorbar_axis.get_xticks().tolist() == [20.0, 60.0]
    plt.close(figure)


def test_panel_boot_size_has_opaque_cloud_without_legend() -> None:
    bootstrap = figure1_demography.load_bootstrap_fits()
    assert bootstrap is not None
    figure, axis = plt.subplots()
    figure1_plots.panel_boot_size(axis, {"bootstrap": bootstrap})
    cloud, best_fit = axis.collections
    assert cloud.get_alpha() == 0.45
    assert best_fit.get_alpha() in (None, 1.0)
    assert axis.get_legend() is None
    plt.close(figure)


def test_panel_model_draws_without_tracked_demography_asset(tmp_path) -> None:
    figure, axis = plt.subplots()
    missing_asset = tmp_path / "demographic_model_schematic.png"
    try:
        with patch.object(
            config, "DEMOGRAPHY_SCHEMATIC_PNG", missing_asset, create=True
        ):
            figure1_plots.panel_model(axis, {})
        figure.canvas.draw()
        labels = [text.get_text() for text in figure.findobj(match=Text)]
        assert not any("placeholder" in label for label in labels)
        assert any("246,260" in label for label in labels)
        assert not any(tmp_path.iterdir())
        model_axis = axis.child_axes[0]
        host_box = axis.get_position()
        model_box = model_axis.get_position()
        left_inset_fraction = (model_box.x0 - host_box.x0) / host_box.width
        assert np.isclose(left_inset_fraction, 0.075)
        schematic_axes = [model_axis, *model_axis.child_axes]
        schematic_text = [
            text
            for schematic_axis in schematic_axes
            for text in schematic_axis.findobj(match=Text)
            if text.get_text()
        ]
        assert schematic_text
        assert all(
            text.get_fontweight() in (400, "normal")
            for text in schematic_text
        )
        parameter_prefixes = ("$N_e$", "T = ", "EUR$", "AFR$", "f = ")
        parameter_text = [
            text
            for text in schematic_text
            if text.get_text().startswith(parameter_prefixes)
        ]
        assert len(parameter_text) == 10
        assert len({text.get_fontsize() for text in parameter_text}) == 1
        assert all(text.get_bbox_patch() is None for text in parameter_text)
        eur_to_afr = next(
            text
            for text in parameter_text
            if text.get_text().startswith("EUR$")
        )
        assert eur_to_afr.get_position()[1] == 1_500
    finally:
        plt.close(figure)
