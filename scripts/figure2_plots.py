"""Publication plots for Figure 2 admixture mapping panels A--D."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import admixture_lai, admixture_plots, admixture_summary, figure2_data
from .common import panel_io, config, style
from .common.io import needs_build

FIGURE_ID = "figure2_sources"
SUPPLEMENTAL_TRACTS_STEM = "Figure_2_sources_admixture_tracts_supplemental"
MAX_NONSIGNIFICANT_POINTS = 10_000


def ancestry_thresholds(data: dict) -> dict[str, float]:
    """Return the mean and the actual ancestry-fraction values at +/-z SD."""
    mean = float(data["stats"]["mean_eur"])
    sd = float(data["stats"]["sd_eur"])
    z = float(data["stats"]["z_threshold"])
    return {
        "mean": mean,
        "sd": sd,
        "z": z,
        "lower": mean - z * sd,
        "upper": mean + z * sd,
    }


def _equal_chromosome_slots(frame, chrom_sizes: dict[int, int]) -> np.ndarray:
    """Map every chromosome into one equal-width Manhattan-plot slot."""
    chrom = frame["chrom"].astype(int).to_numpy()
    pos = frame["pos"].astype(float).to_numpy()
    sizes = np.asarray([float(chrom_sizes[value]) for value in chrom])
    return chrom - 1 + np.clip(pos / sizes, 0.0, 1.0)


def _select_nonsignificant_points(
    frame,
    nonsignificant: np.ndarray,
    *,
    max_points: int = MAX_NONSIGNIFICANT_POINTS,
) -> np.ndarray:
    """Select a deterministic, chromosome-balanced plotting subset.

    The per-chromosome water-filling budget keeps sparse chromosomes intact and
    removes excess markers from the densest chromosomes first. Within each
    chromosome, evenly spaced rows preserve coverage across the full slot.
    Remainder slots go to later chromosomes so ties preferentially thin earlier
    chromosomes, where the controlled-data scan is densest.
    """
    nonsignificant = np.asarray(nonsignificant, dtype=bool)
    selected = np.zeros(len(frame), dtype=bool)
    candidate_indices = np.flatnonzero(nonsignificant)
    if len(candidate_indices) <= max_points:
        selected[candidate_indices] = True
        return selected

    chrom = frame["chrom"].astype(int).to_numpy()
    chromosomes = sorted(np.unique(chrom[candidate_indices]).tolist())
    counts = {
        value: int(np.sum(nonsignificant & (chrom == value)))
        for value in chromosomes
    }
    quotas: dict[int, int] = {}
    active = set(chromosomes)
    remaining = int(max_points)
    while active:
        even_share = remaining // len(active)
        sparse = [value for value in active if counts[value] <= even_share]
        if sparse:
            for value in sparse:
                quotas[value] = counts[value]
                remaining -= counts[value]
                active.remove(value)
            continue
        for value in active:
            quotas[value] = even_share
        remainder = remaining - even_share * len(active)
        for value in sorted(active, reverse=True)[:remainder]:
            quotas[value] += 1
        break

    for value in chromosomes:
        indices = np.flatnonzero(nonsignificant & (chrom == value))
        quota = quotas[value]
        if quota == len(indices):
            selected[indices] = True
        elif quota:
            offsets = np.linspace(0, len(indices) - 1, num=quota, dtype=int)
            selected[indices[offsets]] = True
    return selected


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Genome-wide EUR ancestry fraction in equal-width chromosome slots."""
    frame = data["genome_eur"].copy()
    frame["chrom"] = frame["chrom"].astype(int)
    x = _equal_chromosome_slots(frame, data["chrom_sizes"])
    trimmed_value = frame.get("telomere_trimmed", False)
    trimmed = (
        np.asarray(trimmed_value, dtype=bool)
        if not np.isscalar(trimmed_value)
        else np.full(len(frame), bool(trimmed_value))
    )
    thresholds = ancestry_thresholds(data)
    significant = frame["eur_z"].abs().to_numpy() >= thresholds["z"]
    outlier = (
        ~trimmed
        & significant
    )
    selected_nonsignificant = _select_nonsignificant_points(
        frame,
        ~significant,
    )
    # Significant telomere-trimmed markers retain their prior ordinary styling;
    # only genuinely nonsignificant markers are eligible for downsampling.
    ordinary = selected_nonsignificant | (significant & trimmed)
    # Match Figure 3A's alternating fixed chromosome slots and colours.
    ordinary_colors = np.where(
        frame.loc[ordinary, "chrom"].to_numpy() % 2 == 0,
        "#ADD8E6",
        "#2C7A7B",
    )
    ax.scatter(
        x[ordinary], frame.loc[ordinary, "eur_frac"],
        c=ordinary_colors, s=5, alpha=0.7, edgecolor="none",
        rasterized=False, zorder=1,
    )
    ax.scatter(
        x[outlier], frame.loc[outlier, "eur_frac"],
        color=style.PALETTE["orange"], s=14, alpha=0.95,
        edgecolor="none", rasterized=False, zorder=2,
        label="Outlier markers",
    )
    ax.axhline(
        thresholds["mean"], color="black", lw=1.1,
        label=f"Mean = {thresholds['mean']:.3f}",
    )
    threshold_label = (
        f"+/-{thresholds['z']:.2f} SD = "
        f"{thresholds['lower']:.3f} to {thresholds['upper']:.3f}"
    )
    ax.axhline(
        thresholds["upper"], color=style.PALETTE["orange"],
        lw=1.0, ls="--", label=threshold_label,
    )
    ax.axhline(
        thresholds["lower"], color=style.PALETTE["orange"],
        lw=1.0, ls="--",
    )
    chromosomes = tuple(sorted(int(value) for value in data["chrom_sizes"]))
    ax.set_xlim(0, len(chromosomes))
    ax.set_xticks(np.arange(len(chromosomes)) + 0.5)
    ax.set_xticklabels([str(value) for value in chromosomes], fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("EUR ancestry fraction")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
        frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.0,
    )


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Held-out PLRS calibration for the displayed region."""
    admixture_plots.panel_E(ax, data, rasterized=False)


def panel_C(ax: plt.Axes, data: dict) -> None:
    """Distribution of held-out PLRS performance across regional models."""
    admixture_summary.panel_plrs_r2_distribution(ax, data["plrs_models"])


def panel_D(ax: plt.Axes, data: dict) -> None:
    """Regional PLRS model size versus held-out performance."""
    admixture_summary.panel_plrs_model_size(ax, data["plrs_models"])


def panel_E(ax: plt.Axes, data: dict, *, sv_marker_area_scale: float = 1.0) -> None:
    """Regional LAI zoom with SVs placed at cohort-wide nearest-marker LAI."""
    admixture_plots.panel_C(ax, data, rasterized=False, sv_marker_area_scale=sv_marker_area_scale)


def panel_F(
    ax: plt.Axes,
    data: dict,
    *,
    colorbar_side: str = "right",
) -> None:
    """Regional association, SVs, and exon/intron gene track."""
    admixture_plots.panel_D(
        ax,
        data,
        matched_svs_only=True,
        emphasize_lead_snv=True,
        colorbar_side=colorbar_side,
        show_sv_legend=False,
        rasterized=False,
    )


PANELS = {
    "A": panel_A,
    "B": panel_B,
    "C": panel_C,
    "D": panel_D,
    "E": panel_E,
    "F": panel_F,
}


def supplemental_tract_paths() -> list[Path]:
    """Return the PNG/PDF targets for the former main-panel tract plot."""
    stem = config.figure_dir(FIGURE_ID) / "panels" / SUPPLEMENTAL_TRACTS_STEM
    return [stem.with_suffix(f".{ext}") for ext in config.FIGURE_FORMATS]


def render_supplemental_tracts(data: dict, *, clean_run_plots: bool) -> list[Path]:
    """Save former panel A as a supplemental-style standalone panel."""
    targets = supplemental_tract_paths()
    if not needs_build(targets, clean_run_plots):
        return targets
    style.apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    admixture_plots.panel_A(ax, data)
    fig.tight_layout(pad=0.45)
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        fig.savefig(target)
    plt.close(fig)
    return targets


def build_figure2_sources(
    clean_run_data: bool = False,
    clean_run_plots: bool = False,
) -> list[Path]:
    """Run/resume the established analysis and render its individual panels."""
    from . import recompute_admixture_reference

    if clean_run_data or not admixture_lai.real_outputs_available():
        recompute_admixture_reference.ensure_admixture_outputs(force=clean_run_data)
    clean_run_plots = clean_run_plots or clean_run_data
    data = figure2_data.build(clean_run_data=clean_run_data)
    thresholds = ancestry_thresholds(data)
    source = "controlled-data cache"
    print(
        "[figure2_sources] EUR ancestry thresholds: "
        f"source={source}; mean={thresholds['mean']:.6f}; "
        f"SD={thresholds['sd']:.6f}; +/-{thresholds['z']:.2f} SD="
        f"{thresholds['lower']:.6f} to {thresholds['upper']:.6f}"
    )
    render_supplemental_tracts(data, clean_run_plots=clean_run_plots)
    return panel_io.render_panels(
        FIGURE_ID, PANELS, data, clean_run_plots=clean_run_plots
    )
