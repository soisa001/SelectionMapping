"""Render individual panels from the corrected admixture analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import admixture_plots, figure2_1025, figure2_data, figure2_plots
from .common import panel_io, config, style

FIGURE_ID = "figure2"
SUPPLEMENTAL_STEM = "Figure_2_1025_admixture_tracts_supplemental"
PANELS = figure2_plots.PANELS


def supplemental_paths() -> list[Path]:
    stem = config.figure_dir(FIGURE_ID) / "panels" / SUPPLEMENTAL_STEM
    return [stem.with_suffix(f".{extension}") for extension in config.FIGURE_FORMATS]


def render_supplemental(data: dict) -> list[Path]:
    style.apply_style()
    figure, axis = plt.subplots(figsize=(6.0, 5.5))
    admixture_plots.panel_A(axis, data)
    figure.tight_layout(pad=0.45)
    outputs = supplemental_paths()
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        figure.savefig(output)
    plt.close(figure)
    return outputs


def load_panel_data(*, ids: tuple[str, ...] | None = None) -> dict:
    """Load the corrected analysis cache without rerunning controlled-data stages."""
    ids = ids or figure2_1025.training_ids()
    with figure2_1025.analysis_context(ids):
        data = figure2_data.build(clean_run_data=False)
    data["_figure2_training_ids"] = ids
    return data


def render_panel(letter: str, data: dict, *, clean_run_plots: bool = True) -> list[Path]:
    """Save one corrected admixture panel."""
    ids = data.get("_figure2_training_ids") or figure2_1025.training_ids()
    with figure2_1025.analysis_context(ids):
        return panel_io.render_panel(
            FIGURE_ID, letter, PANELS[letter], data, clean_run_plots=clean_run_plots,
            size=config.FIGURES[FIGURE_ID]["standalone_sizes"].get(letter, (5.0, 4.0)),
        )


def build_figure2_1025(
    clean_run_data: bool = False, clean_run_plots: bool = False,
    *, ids: tuple[str, ...] | None = None,
) -> list[Path]:
    """Save all corrected admixture panels from validated caches."""
    data = load_panel_data(ids=ids)
    with figure2_1025.analysis_context(data["_figure2_training_ids"]):
        return panel_io.render_panels(FIGURE_ID, PANELS, data, clean_run_plots=clean_run_plots or clean_run_data)
