"""Save individual figure panels as PNG and editable PDF files."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from . import config, style

PanelFn = Callable[..., None]


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    """Write a figure to every configured format; return the paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in config.FIGURE_FORMATS:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def _panel_stem(figure_id: str, letter: str) -> Path:
    """Path stem (no extension) for a single panel image."""
    number = config.FIGURES[figure_id]["number"]
    return config.figure_dir(figure_id) / "panels" / f"Figure_{number}{letter}"


def render_panel(
    figure_id: str,
    letter: str,
    panel_fn: PanelFn,
    data: object,
    *,
    clean_run_plots: bool,
    size: tuple[float, float] = (5.0, 4.0),
) -> list[Path]:
    """Draw and save a single standalone panel, idempotently."""
    stem = _panel_stem(figure_id, letter)
    targets = [stem.with_suffix(f".{ext}") for ext in config.FIGURE_FORMATS]

    from .io import needs_build
    if not needs_build(targets, clean_run_plots):
        return targets

    style.apply_style()
    fig, ax = plt.subplots(figsize=size)
    spec = config.FIGURES[figure_id]
    panel_fn(ax, data, **spec.get("panel_kwargs", {}).get(letter, {}))
    for axis, limits in spec.get("axis_limits", {}).get(letter, {}).items():
        getattr(ax, f"set_{axis}")(*limits)
    desc = spec["panels"][letter]
    title_pad = spec.get("standalone_title_pad", {}).get(letter)
    if letter not in spec.get("standalone_title_skip", ()):
        ax.set_title(f"{letter}) {desc}", pad=title_pad)
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def panel_image_paths(figure_id: str, ext: str = "png") -> list[Path]:
    """Ordered paths to the saved individual panel images."""
    spec = config.FIGURES[figure_id]
    return [_panel_stem(figure_id, letter).with_suffix(f".{ext}") for letter in spec["panels"]]


def render_panels(
    figure_id: str, panels: dict[str, PanelFn], data: object, *, clean_run_plots: bool,
) -> list[Path]:
    """Save every available panel without arranging a combined figure."""
    spec = config.FIGURES[figure_id]
    paths = []
    for letter, draw in panels.items():
        paths.extend(render_panel(
            figure_id, letter, draw, data, clean_run_plots=clean_run_plots,
            size=spec.get("standalone_sizes", {}).get(letter, (5.0, 4.0)),
        ))
    return paths
