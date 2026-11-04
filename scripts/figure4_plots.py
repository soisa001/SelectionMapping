"""Individual PheWAS and meta-analysis panel entry points."""
from __future__ import annotations

from . import figure4_effect_meta, figure4_meta, figure4_s9, figure4_volcano


def render_panel(letter: str, *, clean_run_data: bool = False, clean_run_plots: bool = False) -> list:
    """Write one panel as PNG/PDF and return its paths."""
    if letter == "A":
        return figure4_volcano.render_volcano(clean_run_data=clean_run_data, clean_run_plots=clean_run_plots)
    if letter == "B":
        return figure4_s9.render_panel()
    if letter == "C":
        return figure4_effect_meta.render_panel(clean_run_data=clean_run_data)
    if letter == "D":
        return figure4_meta.render_forest(
            clean_run_data=clean_run_data, clean_run_plots=clean_run_plots,
            show_title=False, show_variant=False, show_p_value=False,
            fixed_canvas=True,
        )
    raise ValueError(f"Unknown Figure 4 panel: {letter}")
