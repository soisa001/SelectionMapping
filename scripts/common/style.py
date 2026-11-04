"""Shared plotting style for publication-quality figures.

Kept deliberately light-touch for the skeleton: a consistent rcParams profile,
a colorblind-friendly palette, and small helpers for panel labels. When the
scientific-visualization pass happens, tune here and every figure inherits it.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Unicode labels keep subscripts/superscripts in one editable PDF text run.
P_VALUE_AXIS_LABEL = "−log₁₀(p)"
LD_COLORBAR_LABEL = "r² to min-p variant"

# Colorblind-friendly qualitative palette (Wong 2011, Nature Methods).
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Standardized AoU genetic-ancestry palette (superpopulation labels). Use this
# everywhere an ancestry appears (PCA subpops, ADMIXTURE AFR/EUR, local-ancestry
# tracts) so the whole figure set reads as one system.
ANCESTRY_PALETTE = {
    "AFR": "#D74BAE",  # magenta
    "AMR": "#4CA64C",  # green
    "EAS": "#E23B33",  # red
    "EUR": "#3C79B4",  # blue
    "MID": "#F5911E",  # orange
    "SAS": "#7E50A1",  # purple
    "OCE": "#00A6A6",  # teal (geographic Oceania, never folded into OTH)
    "OTH": "#8C5A3C",  # brown
}

# AFR/EUR for the ADMIXTURE and local-ancestry panels, drawn from the same
# palette (AoU shown as a white diamond in the PCA, so it needs no fill colour).
ANCESTRY_COLORS = {
    "AFR": ANCESTRY_PALETTE["AFR"],
    "EUR": ANCESTRY_PALETTE["EUR"],
    "AoU": PALETTE["green"],
}


def ancestry_pred_color(label: str) -> str:
    """Colour for an ancestry_pred superpopulation label (case-insensitive).

    Unknown labels fall back to the OTH colour; a neutral grey is used only for
    values that are neither a known superpop nor OTH.
    """
    s = str(label).upper()
    if s in ANCESTRY_PALETTE:
        return ANCESTRY_PALETTE[s]
    return "0.5"

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "lines.linewidth": 1.75,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent; call once per session/notebook."""
    mpl.rcParams.update(_RC)


def panel_label(ax: plt.Axes, letter: str, *, dx: float = -0.08, dy: float = 1.05) -> None:
    """Stamp a bold panel label (A, B, ...) in axes-fraction coordinates."""
    ax.text(
        dx, dy, letter,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="right",
    )
