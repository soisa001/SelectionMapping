from __future__ import annotations
from panel_fixtures import discovery_frame

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scripts import (
    figure4_effect_meta,
    figure4_meta,
    figure4_plots,
    figure4_s9,
    figure4_volcano,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = pd.DataFrame(
        {
            "variant_tag": [f"1_{10_000 + index}_A_G" for index in range(4)],
            "phecode": [f"P{index}" for index in range(4)],
            "phecode_string": [f"Phenotype {index}" for index in range(4)],
            "beta": [-0.8, -0.2, 0.5, 1.0],
            "se": [0.2, 0.1, 0.25, 0.3],
            "is_meta_significant": [True, False, False, False],
        }
    )
    meta = pd.DataFrame(
        {
            "variant_tag": discovery.loc[:2, "variant_tag"],
            "phecode": discovery.loc[:2, "phecode"],
            "beta_r": [-0.4, 0.1, 0.2],
            "se_r": [0.1, 0.2, 0.5],
            "meta_significant": [True, False, False],
        }
    )
    return discovery, meta


def test_volcano_thinning_preserves_all_hits_and_meta_points() -> None:
    frame = figure4_volcano._normalize(
        pd.DataFrame(
            {
                "variant_tag": [f"1_{index}_A_G" for index in range(103)],
                "phecode": [f"P{index}" for index in range(103)],
                "phecode_string": [f"Phenotype {index}" for index in range(103)],
                "beta": np.linspace(-2.0, 2.0, 103),
                "p_value": [0.5] * 100 + [1e-3, 1e-5, 1e-8],
            }
        )
    )
    meta_key = {("1_7_A_G", "P7")}

    thinned, stats = figure4_volcano._thin_volcano_background(
        frame,
        p_thresh=1e-3,
        meta_keys=meta_key,
        max_background_points=10,
        seed=17,
    )
    repeated, repeated_stats = figure4_volcano._thin_volcano_background(
        frame,
        p_thresh=1e-3,
        meta_keys=meta_key,
        max_background_points=10,
        seed=17,
    )

    assert set(frame.loc[frame["p_value"] <= 1e-3, "phecode"]) <= set(
        thinned["phecode"]
    )
    assert "P7" in set(thinned["phecode"])
    assert stats == {
        "input_points": 103,
        "significant_points": 3,
        "meta_points": 1,
        "ordinary_background_input": 99,
        "ordinary_background_plotted": 10,
        "output_points": 14,
        "seed": 17,
        "p_threshold": 1e-3,
        "background_cap": 10,
    }
    assert repeated_stats == stats
    pd.testing.assert_frame_equal(thinned, repeated)


def test_ivw_meta_is_recomputed_and_missing_replication_stays_missing() -> None:
    discovery, meta = _inputs()
    frame, audit = figure4_effect_meta.prepare_effect_meta(
        discovery,
        meta,
        expected_discovery_count=4,
        expected_meta_effect_count=3,
        expected_meta_significant_count=1,
    )

    row = frame.loc[frame["phecode"].eq("P0")].iloc[0]
    expected_beta = ((-0.8 / 0.2**2) + (-0.4 / 0.1**2)) / (
        (1 / 0.2**2) + (1 / 0.1**2)
    )
    expected_se = np.sqrt(1 / ((1 / 0.2**2) + (1 / 0.1**2)))
    assert row["beta_meta_ivw"] == pytest.approx(expected_beta)
    assert row["se_meta_ivw"] == pytest.approx(expected_se)

    unpaired = frame.loc[frame["phecode"].eq("P3")].iloc[0]
    assert not unpaired["has_meta_effect"]
    assert np.isnan(unpaired["beta_meta_ivw"])
    assert audit["counts"] == {
        "discovery_rows": 4,
        "meta_effect_rows": 3,
        "missing_meta_effect_rows": 1,
        "meta_significant_rows": 1,
        "not_meta_significant_rows": 3,
    }


def test_nonpositive_replication_se_is_not_used_as_a_meta_effect() -> None:
    discovery, meta = _inputs()
    meta.loc[2, "se_r"] = 0.0
    frame, _ = figure4_effect_meta.prepare_effect_meta(
        discovery,
        meta,
        expected_discovery_count=4,
        expected_meta_effect_count=2,
        expected_meta_significant_count=1,
    )
    invalid = frame.loc[frame["phecode"].eq("P2")].iloc[0]
    assert not invalid["has_meta_effect"]
    assert np.isnan(invalid["beta_meta_ivw"])


def test_effect_plot_uses_ivw_label_and_complete_discovery_legend_counts() -> None:
    discovery, meta = _inputs()
    frame, _ = figure4_effect_meta.prepare_effect_meta(
        discovery,
        meta,
        expected_discovery_count=4,
        expected_meta_effect_count=3,
        expected_meta_significant_count=1,
    )
    figure = figure4_effect_meta.plot_effect_meta(frame)
    try:
        axis = figure.axes[0]
        np.testing.assert_allclose(figure.get_size_inches(), [11, 8.5])
        assert axis.get_ylabel() == r"$\beta$"
        assert [label.get_text().split("\n")[0] for label in axis.get_xticklabels()] == [
            "Discovery effect",
            "IVW meta-analysis effect",
        ]
        assert [text.get_text() for text in axis.get_legend().get_texts()] == [
            "Meta-significant (1)",
            "Not meta-significant (3)",
        ]
        assert not axis.get_legend().get_frame_on()
    finally:
        plt.close(figure)


def test_forest_has_no_p_value_axis_or_header() -> None:
    frame = pd.DataFrame(
        {
            "variant_tag": ["1_100_A_G"],
            "phecode": ["P1"],
            "phecode_string": ["Phenotype one"],
            "beta_meta": [-0.2],
            "se_meta": [0.05],
            "p_meta": [1e-8],
            "meta_significant": [True],
        }
    )
    figure = figure4_meta.plot_forest(
        frame,
        show_variant=False,
        show_p_value=False,
        fixed_canvas=True,
    )
    try:
        assert len(figure.axes) == 2
        np.testing.assert_allclose(
            figure.get_size_inches(), figure4_meta.FOREST_FIGSIZE
        )
        assert all(axis.get_title() != "$p$" for axis in figure.axes)
        assert not any(text.get_text() == "$p$" for axis in figure.axes for text in axis.texts)
    finally:
        plt.close(figure)


def test_volcano_has_unboxed_labels_and_inline_beta_legend_symbol() -> None:
    frame = figure4_volcano._normalize(
        pd.DataFrame(
            {
                "variant_tag": ["1_100_A_G", "1_200_A_G"],
                "phecode": ["P1", "P2"],
                "phecode_string": ["Phenotype one", "Phenotype two"],
                "beta": [-0.5, 0.6],
                "p_value": [1e-5, 1e-6],
            }
        )
    )
    figure = figure4_volcano.plot_volcano(
        frame,
        beta_labels=True,
        legend_above=True,
        label_boxes=False,
        inline_beta_legend=True,
    )
    try:
        axis = figure.axes[0]
        assert axis.texts
        assert all(text.get_bbox_patch() is None for text in axis.texts)
        legend = axis.get_legend()
        assert legend.get_title().get_text() == ""
        assert legend.legend_handles[0].get_marker() == r"$\beta$"
        assert [text.get_text() for text in legend.get_texts()][1] == "0.5"
    finally:
        plt.close(figure)


def test_volcano_pdf_uses_vector_markers(tmp_path: Path) -> None:
    frame = figure4_volcano._normalize(
        pd.DataFrame(
            {
                "variant_tag": [f"1_{index}_A_G" for index in range(40)],
                "phecode": [f"P{index}" for index in range(40)],
                "phecode_string": [f"Phenotype {index}" for index in range(40)],
                "beta": np.linspace(-2.0, 2.0, 40),
                "p_value": np.r_[np.full(36, 0.5), [1e-3, 1e-4, 1e-5, 1e-6]],
            }
        )
    )
    figure = figure4_volcano.plot_volcano(
        frame,
        beta_labels=True,
        legend_above=True,
        label_boxes=False,
        inline_beta_legend=True,
    )
    output = tmp_path / "Figure_4A_volcano.pdf"
    figure.savefig(output)
    plt.close(figure)

    with pymupdf.open(output) as document:
        assert sum(len(page.get_images(full=True)) for page in document) == 0


def test_s9_annotations_keep_leaders_but_remove_text_boxes() -> None:
    frame = discovery_frame().iloc[[0, -21, -1]].copy()
    figure = figure4_s9.figure_s9_phewas.plot_s9(
        frame,
        marker_scale=figure4_s9.MARKER_SCALE,
        font_scale=figure4_s9.FONT_SCALE,
        legend_frame=False,
        annotation_boxes=False,
    )
    try:
        axis = figure.axes[0]
        assert len(axis.texts) == 1
        text_patch = axis.texts[0].get_bbox_patch()
        assert text_patch is not None
        assert text_patch.get_alpha() == 0
        assert text_patch.get_linewidth() == 0
        assert axis.patches
    finally:
        plt.close(figure)






@pytest.mark.parametrize("letter,module,function", [
    ("A", figure4_volcano, "render_volcano"),
    ("B", figure4_s9, "render_panel"),
    ("C", figure4_effect_meta, "render_panel"),
    ("D", figure4_meta, "render_forest"),
])
def test_individual_panel_dispatch(letter, module, function):
    expected = [Path("panel.png"), Path("panel.pdf")]
    with patch.object(module, function, return_value=expected) as render:
        assert figure4_plots.render_panel(letter, clean_run_plots=True) == expected
    render.assert_called_once()
    if letter == "D":
        assert render.call_args.kwargs["show_p_value"] is False
        assert render.call_args.kwargs["fixed_canvas"] is True
