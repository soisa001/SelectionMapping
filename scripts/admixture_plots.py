"""Publication plots for admixture reference local ancestry panels A--E."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import figure3_ehh, admixture_data, admixture_lai
from .common import panel_io, style
from .common.genome import genome_coords

FIGURE_ID = "admixture_reference"


def _region_window(region: dict, data: dict | None = None) -> tuple[int, int]:
    if data is not None and "locus_window" in data:
        return tuple(int(value) for value in data["locus_window"])
    return (
        max(1, int(region["start"]) - admixture_lai.REGION_FLANK_BP),
        int(region["end"]) + admixture_lai.REGION_FLANK_BP,
    )


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Two painted haplotypes for one representative LAI individual."""
    tracks = data["tracts"]
    masks = data["masks"]
    chromosome_order = list(range(1, 23))
    y_lookup = {chrom: len(chromosome_order) - 1 - index
                for index, chrom in enumerate(chromosome_order)}
    track_height = 0.34
    gap = 0.025
    colors = {
        "AFR": style.ANCESTRY_COLORS["AFR"],
        "EUR": style.ANCESTRY_COLORS["EUR"],
        "OTHER": "0.7",
    }
    for chrom in chromosome_order:
        y = y_lookup[chrom]
        length_mb = data["chrom_sizes"][chrom] / 1e6
        ax.add_patch(Rectangle(
            (0, y - track_height), length_mb, 2 * track_height,
            facecolor="0.96", edgecolor="0.35", linewidth=0.35, zorder=1,
        ))
        local = tracks[tracks["chrom"].astype(int) == chrom]
        for row in local.itertuples(index=False):
            y0 = y + gap if int(row.hap) == 1 else y - track_height
            ax.add_patch(Rectangle(
                (float(row.start) / 1e6, y0),
                max(float(row.end - row.start) / 1e6, 1e-6),
                track_height - gap,
                facecolor=colors.get(str(row.ancestry), "0.7"),
                edgecolor="none", zorder=2,
            ))
        for row in masks[masks["chrom"].astype(int) == chrom].itertuples(index=False):
            ax.add_patch(Rectangle(
                (float(row.start) / 1e6, y - track_height),
                max(float(row.end - row.start) / 1e6, 1e-6),
                2 * track_height,
                facecolor="0.78", edgecolor="0.35", linewidth=0.25,
                hatch="////", zorder=3,
            ))
    ax.set_xlim(0, max(data["chrom_sizes"].values()) / 1e6)
    ax.set_ylim(-1, len(chromosome_order))
    ax.set_yticks([y_lookup[chrom] for chrom in chromosome_order])
    ax.set_yticklabels([f"chr{chrom}" for chrom in chromosome_order], fontsize=7)
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.text(0.0, 1.01, "top = haplotype 1, bottom = haplotype 2",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    handles = [
        Patch(facecolor=colors["AFR"], label="AFR"),
        Patch(facecolor=colors["EUR"], label="EUR"),
    ]
    if not masks.empty:
        handles.append(Patch(facecolor="0.78", edgecolor="0.35",
                             hatch="////", label="masked ≥1 Mb"))
    ax.legend(handles=handles, loc="lower right", ncol=len(handles), fontsize=7)


def panel_B(ax: plt.Axes, data: dict) -> None:
    """Marker-level cohort EUR fraction across the autosomes."""
    frame = data["genome_eur"].copy()
    frame["chrom"] = frame["chrom"].astype(int)
    x, ticks, labels = genome_coords(frame, data["chrom_sizes"])
    trimmed = frame.get("telomere_trimmed", False)
    trimmed = np.asarray(trimmed, dtype=bool) if not np.isscalar(trimmed) \
        else np.full(len(frame), bool(trimmed))
    outlier = frame["eur_z"].abs().to_numpy() >= float(data["stats"]["z_threshold"])
    alternating = np.asarray([
        "#273746" if int(chrom) % 2 else "#95A5A6" for chrom in frame["chrom"]
    ])
    ax.scatter(x[trimmed] / 1e6, frame.loc[trimmed, "eur_frac"],
               s=1.0, color="0.88", edgecolor="none", rasterized=True)
    ordinary = ~trimmed & ~outlier
    ax.scatter(x[ordinary] / 1e6, frame.loc[ordinary, "eur_frac"],
               s=1.5, c=alternating[ordinary], edgecolor="none", rasterized=True)
    ax.scatter(x[outlier] / 1e6, frame.loc[outlier, "eur_frac"],
               s=7, color=style.PALETTE["vermillion"], edgecolor="none",
               rasterized=True, label="outlier markers")
    mean = float(data["stats"]["mean_eur"])
    sd = float(data["stats"]["sd_eur"])
    threshold = float(data["stats"]["z_threshold"])
    ax.axhline(mean, color="black", lw=0.9, label=f"mean = {mean:.3f}")
    ax.axhline(mean + threshold * sd, color=style.PALETTE["vermillion"],
               lw=0.8, ls="--", label=f"±{threshold:.2f} SD")
    ax.axhline(mean - threshold * sd, color=style.PALETTE["vermillion"],
               lw=0.8, ls="--")
    ax.set_xticks(np.asarray(ticks) / 1e6)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("EUR ancestry fraction")
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def panel_C(
    ax: plt.Axes,
    data: dict,
    *,
    rasterized: bool = True,
    sv_marker_area_scale: float = 1.0,
) -> None:
    """Marker-level zoom on the selected LAI outlier region."""
    frame = data["region_zoom"].copy().sort_values("pos")
    region = data["display_region"]
    window_start, window_end = _region_window(region, data)
    frame = frame[
        frame["pos"].between(window_start, window_end, inclusive="both")
    ].copy()
    inside = frame["in_region"].astype(bool).to_numpy()
    ax.scatter(frame.loc[~inside, "pos"] / 1e6,
               frame.loc[~inside, "eur_frac"],
               s=9, color="lightsteelblue", edgecolor="none",
               rasterized=rasterized)
    ax.scatter(frame.loc[inside, "pos"] / 1e6,
               frame.loc[inside, "eur_frac"],
               s=12, color=style.PALETTE["blue"], edgecolor="none",
               rasterized=rasterized)
    ax.axvspan(float(region["start"]) / 1e6, float(region["end"]) / 1e6,
               color=style.PALETTE["yellow"], alpha=0.18)
    mean = float(data["stats"]["mean_eur"])
    sd = float(data["stats"]["sd_eur"])
    threshold = float(data["stats"]["z_threshold"])
    ax.axhline(mean, color="black", lw=0.9)
    ax.axhline(mean + threshold * sd, color=style.PALETTE["vermillion"],
               lw=0.8, ls="--")
    ax.axhline(mean - threshold * sd, color=style.PALETTE["vermillion"],
               lw=0.8, ls="--")
    regional_svs = data.get("regional_svs", pd.DataFrame())
    if not regional_svs.empty:
        legend_handles = []
        for sv_type, marker in (("INS", "^"), ("DEL", "v")):
            subset = regional_svs[
                regional_svs["sv_type"].astype(str).str.upper().eq(sv_type)
            ].copy()
            subset["overall_eur_frac"] = pd.to_numeric(
                subset["overall_eur_frac"], errors="coerce"
            )
            subset = subset.dropna(subset=["overall_eur_frac"])
            if subset.empty:
                continue
            ax.scatter(
                pd.to_numeric(subset["pos"], errors="coerce") / 1e6,
                subset["overall_eur_frac"],
                marker=marker,
                s=68 * sv_marker_area_scale,
                facecolor=style.PALETTE["orange"],
                edgecolor="black",
                linewidth=0.9,
                zorder=8,
                clip_on=False,
            )
            legend_handles.append(Line2D(
                [], [], marker=marker, linestyle="none", markersize=7,
                markerfacecolor=style.PALETTE["orange"],
                markeredgecolor="black", label=f"{sv_type} SV position",
            ))
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                frameon=False,
                ncol=len(legend_handles),
                fontsize=7,
            )
    chrom = int(region["chrom"])
    ax.set_xlim(window_start / 1e6, window_end / 1e6)
    ax.set_xlabel(f"chr{chrom} (Mb)")
    ax.set_ylabel("EUR ancestry fraction")


def _draw_finemap_gene_track(
    ax: plt.Axes,
    genes: pd.DataFrame,
    chrom: int,
    window_start: int,
    window_end: int,
) -> None:
    """Draw the shared Figure 3 exon/intron transcript track."""
    gene_dict: dict[str, dict] = {}
    if not genes.empty:
        local = genes.copy()
        local_chrom = (
            local["chrom"].astype(str).str.lower().str.removeprefix("chr")
        )
        local["start"] = pd.to_numeric(local["start"], errors="coerce")
        local["end"] = pd.to_numeric(local["end"], errors="coerce")
        local = local[
            local_chrom.eq(str(chrom))
            & local["start"].le(window_end)
            & local["end"].ge(window_start)
            & local["gene"].notna()
        ].dropna(subset=["start", "end"])
        for row in local.itertuples(index=False):
            raw_exons = getattr(row, "exons", "[]")
            if isinstance(raw_exons, str):
                try:
                    raw_exons = json.loads(raw_exons)
                except json.JSONDecodeError:
                    raw_exons = []
            exons = [
                (int(start), int(end))
                for start, end in (raw_exons or [])
            ]
            gene_dict[str(row.gene)] = {
                "tx_start": int(row.start),
                "tx_end": int(row.end),
                "strand": str(getattr(row, "strand", "+")),
                "exons": exons or [(int(row.start), int(row.end))],
            }
    figure3_ehh.draw_gene_track(
        ax,
        gene_dict,
        window_start,
        window_end,
        gene_fontsize=10,
    )
    ax.set_ylabel("Genes", fontsize=10, labelpad=2)
    ax.set_xlabel(f"chr{chrom} (Mb)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value / 1e6:.2f}"))
    ax.tick_params(axis="x", labelsize=9.5)


def panel_D(
    ax: plt.Axes,
    data: dict,
    *,
    matched_svs_only: bool = False,
    emphasize_lead_snv: bool = False,
    colorbar_side: str = "right",
    show_sv_legend: bool = True,
    rasterized: bool = True,
) -> None:
    """Figure 4-style score-test association plot for the displayed region."""
    frame = data["finemap"].copy()
    region = data["display_region"]
    window_start, window_end = _region_window(region, data)
    chrom = int(region["chrom"])
    ax.set_axis_off()
    if frame.empty:
        ax.text(
            0.5, 0.5, "No matching 100 kb fine-map cache",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
        )
        return
    frame = frame[
        pd.to_numeric(frame["pos"], errors="coerce").between(
            window_start, window_end, inclusive="both"
        )
    ].copy()
    if frame.empty:
        ax.text(
            0.5, 0.5, "No variants in the displayed window",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
        )
        return

    # Child axes retain C and D on one mosaic row while giving D the
    # association-plus-gene-track structure used by Figure 4B/C.
    if colorbar_side == "right":
        plot_width = 0.86
        colorbar_bounds = [0.91, 0.52, 0.025, 0.35]
    elif colorbar_side == "left":
        # Keep association + gene axes at the full host width so this panel can
        # align exactly with the regional-LAI panel above. The caller reserves
        # the left margin for the scale, so it does not change plot width.
        plot_width = 1.0
        colorbar_bounds = [-0.82, 0.52, 0.025, 0.35]
    else:
        raise ValueError(f"Unsupported fine-map colorbar side: {colorbar_side!r}")
    association_ax = ax.inset_axes([0.00, 0.43, plot_width, 0.54])
    gene_ax = ax.inset_axes([0.00, 0.03, plot_width, 0.29], sharex=association_ax)
    colorbar_ax = ax.inset_axes(colorbar_bounds)

    x = pd.to_numeric(frame["pos"], errors="coerce").to_numpy()
    if "neg_log10_p" in frame:
        neg_log10_p = pd.to_numeric(
            frame["neg_log10_p"], errors="coerce"
        ).to_numpy()
    else:
        p_score = pd.to_numeric(frame["p_score"], errors="coerce").to_numpy()
        neg_log10_p = -np.log10(np.clip(p_score, 1e-300, 1.0))
    r2 = pd.to_numeric(
        frame.get("r2_to_min_p", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0).to_numpy()
    finite = np.isfinite(x) & np.isfinite(neg_log10_p)
    order = np.argsort(np.nan_to_num(r2[finite], nan=-1.0))
    norm = colors.Normalize(0, 1)
    scatter = association_ax.scatter(
        x[finite][order], neg_log10_p[finite][order],
        c=np.nan_to_num(r2[finite][order]), cmap="viridis", norm=norm,
        s=16, alpha=0.9, edgecolor="none", rasterized=rasterized,
    )
    if emphasize_lead_snv:
        minimum_p = frame.get(
            "minimum_p_variant", pd.Series(False, index=frame.index)
        ).astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        is_sv = frame.get(
            "is_sv", pd.Series(False, index=frame.index)
        ).astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        lead_snv = minimum_p.to_numpy() & ~is_sv.to_numpy() & finite
        if lead_snv.any():
            association_ax.scatter(
                x[lead_snv], neg_log10_p[lead_snv],
                c=np.nan_to_num(r2[lead_snv]), cmap="viridis", norm=norm,
                s=72, alpha=1.0, edgecolor="none",
                rasterized=rasterized, zorder=7,
            )
    association_ax.set_xlim(window_start, window_end)
    association_ax.set_ylim(bottom=0)
    association_ax.axvspan(
        float(region["start"]),
        float(region["end"]),
        color=style.PALETTE["yellow"], alpha=0.18, zorder=0,
    )
    association_ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=11)
    association_ax.tick_params(labelbottom=False, labelsize=9.5)
    association_ax.grid(axis="y", alpha=0.25, lw=0.6)
    for spine in ("top", "right"):
        association_ax.spines[spine].set_visible(False)

    regional_svs = data.get("regional_svs", pd.DataFrame())
    sv_handles = []
    if not regional_svs.empty:
        if matched_svs_only:
            exact_match = regional_svs.get(
                "exact_finemap_match",
                pd.Series(False, index=regional_svs.index),
            ).astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
            regional_svs = regional_svs[exact_match].copy()
        for sv_type, marker in (("INS", "^"), ("DEL", "v")):
            subset = regional_svs[
                regional_svs["sv_type"].astype(str).str.upper().eq(sv_type)
            ]
            if subset.empty:
                continue
            for sv in subset.itertuples(index=False):
                sv_pos = int(sv.pos)
                marker_kwargs = {
                    "marker": marker,
                    "s": 72,
                    "edgecolor": "red",
                    "linewidth": 1.5,
                    "zorder": 8,
                    "clip_on": False,
                }
                exact_match = bool(getattr(sv, "exact_finemap_match", False))
                sv_neg_log10_p = pd.to_numeric(
                    getattr(sv, "neg_log10_p", np.nan), errors="coerce"
                )
                sv_r2 = pd.to_numeric(
                    getattr(sv, "r2_to_min_p", np.nan), errors="coerce"
                )
                if exact_match and np.isfinite(sv_neg_log10_p):
                    face = plt.get_cmap("viridis")(
                        float(np.clip(sv_r2, 0, 1)) if np.isfinite(sv_r2) else 0.0
                    )
                    association_ax.scatter(
                        [sv_pos], [sv_neg_log10_p], facecolor=[face],
                        **marker_kwargs,
                    )
                else:
                    association_ax.scatter(
                        [sv_pos], [0.91], facecolor="white",
                        transform=association_ax.get_xaxis_transform(),
                        **marker_kwargs,
                    )
            sv_handles.append(Line2D(
                [], [], marker=marker, linestyle="none", markersize=7,
                markerfacecolor=plt.get_cmap("viridis")(0.5),
                markeredgecolor="red", markeredgewidth=1.3,
                label=f"{sv_type} SV",
            ))
        if sv_handles and show_sv_legend:
            association_ax.legend(
                handles=sv_handles,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                frameon=False,
                ncol=len(sv_handles),
                fontsize=7,
            )

    colorbar = ax.figure.colorbar(scatter, cax=colorbar_ax)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(rasterized)
    colorbar.set_label(r"$r^2$ to min-$p$", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    if colorbar_side == "left":
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")

    _draw_finemap_gene_track(
        gene_ax,
        data.get(
            "finemap_genes",
            pd.DataFrame(columns=["chrom", "start", "end", "gene", "strand"]),
        ),
        chrom,
        window_start,
        window_end,
    )
    gene_ax.axvspan(
        float(region["start"]),
        float(region["end"]),
        color=style.PALETTE["yellow"], alpha=0.18, zorder=0,
    )


def panel_E(
    ax: plt.Axes,
    data: dict,
    *,
    rasterized: bool = True,
) -> None:
    """Held-out PLRS observed-versus-predicted EUR dosage calibration."""
    frame = data["plrs_holdout"].copy()
    ax.set_xlabel("True EUR dosage")
    ax.set_ylabel("Predicted EUR dosage")
    ax.set_xlim(-0.03, 2.03)
    ax.set_ylim(-0.03, 2.03)
    ax.set_aspect("equal", adjustable="box")
    if frame.empty:
        ax.text(
            0.5, 0.5, "Hold-out predictions unavailable",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
        )
        return
    truth = pd.to_numeric(frame["true_eur_dosage"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(
        frame["predicted_eur_dosage"], errors="coerce"
    ).to_numpy()
    finite = np.isfinite(truth) & np.isfinite(predicted)
    truth, predicted = truth[finite], predicted[finite]
    ax.scatter(
        truth, predicted, s=13, color=style.PALETTE["blue"],
        alpha=0.65, edgecolor="white", linewidth=0.2,
        rasterized=rasterized,
    )
    ax.plot([0, 2], [0, 2], color="0.25", ls="--", lw=0.9, label="y = x")
    null_mean = float(np.mean(truth))
    ax.axhline(
        null_mean, color=style.PALETTE["vermillion"], ls=":", lw=0.8,
        alpha=0.8, label=f"hold-out mean={null_mean:.2f}",
    )
    ax.legend(loc="upper left", fontsize=6)


PANELS = {
    "A": panel_A, "B": panel_B, "C": panel_C,
    "D": panel_D, "E": panel_E,
}


def build_admixture_reference(clean_run_data: bool = False,
                  clean_run_plots: bool = False) -> list:
    from . import recompute_admixture_reference
    if clean_run_data or not admixture_lai.real_outputs_available():
        recompute_admixture_reference.ensure_admixture_outputs(force=clean_run_data)
    clean_run_plots = clean_run_plots or clean_run_data
    data = admixture_data.build(clean_run_data=clean_run_data)
    return panel_io.render_panels(
        FIGURE_ID, PANELS, data, clean_run_plots=clean_run_plots
    )
