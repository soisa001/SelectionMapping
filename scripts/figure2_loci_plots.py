"""Native-vector discovery fine maps for all nominal admixture associations."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .common import style

PLOT_CONTRACT = "figure5_nominal_score_pip_cs_or_minp_svs_no_border_vector_v4"


def _gene_layout(genes, chrom, start, end, renderer, *, width_inches=8.4):
    """Pack transcripts using both their span and measured label width."""
    from matplotlib.font_manager import FontProperties

    local = genes[
        genes["chrom"].astype(str).str.removeprefix("chr").eq(str(chrom))
        & genes["start"].le(end)
        & genes["end"].ge(start)
    ].sort_values(["start", "end", "gene"])
    rows, lane_ends = [], []
    bp_per_pixel = (end - start) / (width_inches * renderer.dpi)
    gap = (end - start) * 0.012
    for gene in local.drop_duplicates("gene").itertuples(index=False):
        width, _, _ = renderer.get_text_width_height_descent(
            str(gene.gene),
            FontProperties(size=13, style="italic"),
            False,
        )
        half = width * bp_per_pixel / 2
        x = np.clip((gene.start + gene.end) / 2, start + half + gap, end - half - gap)
        left = min(max(start, gene.start), x - half) - gap
        right = max(min(end, gene.end), x + half) + gap
        lane = next(
            (i for i, edge in enumerate(lane_ends) if edge < left), len(lane_ends)
        )
        if lane == len(lane_ends):
            lane_ends.append(right)
        else:
            lane_ends[lane] = right
        rows.append((gene, lane, x))
    return rows, max(1, len(lane_ends))


def draw_genes(axis, layout, lanes, start, end, *, position_scale=1.0):
    """Shared native-vector exon/intron tracks with measured, packed labels."""
    for gene, lane, label_x in layout:
        y = lanes - lane - 1
        axis.plot(np.array([gene.start, gene.end]) * position_scale, [y, y],
                  color="black", linewidth=1.3)
        exons = getattr(gene, "exons", "[]")
        exons = json.loads(exons) if isinstance(exons, str) else exons
        for left, right in exons or [(gene.start, gene.end)]:
            axis.add_patch(Rectangle(
                (left * position_scale, y - 0.12),
                max(right - left, (end - start) * 0.001) * position_scale,
                0.24, facecolor="black", edgecolor="none",
            ))
        axis.text(label_x * position_scale, y + 0.19, str(gene.gene),
                  ha="center", va="bottom", fontsize=13, fontstyle="italic")
    if not layout:
        axis.text(0.5, 0.5, "No annotated genes in window", transform=axis.transAxes,
                  ha="center", va="center", fontsize=13, color="0.4")
    axis.set_ylim(-0.4, lanes - 0.1)
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.set_ylabel("Genes", fontsize=16)


def draw_variants(frame, axes_values, *, position_scale=1.0, ld_reference="lead", point_area_scale=1.0):
    """Draw the same native-vector association/PIP marks in either output."""
    norm = colors.Normalize(0, 1)
    x = frame["pos"].to_numpy() * position_scale
    if ld_reference not in {"lead", "minimum_p"}:
        raise ValueError(f"Unknown LD reference: {ld_reference}")
    r2_column = "r2_to_min_p" if ld_reference == "minimum_p" else "r2_to_lead"
    r2 = pd.to_numeric(frame[r2_column], errors="coerce").fillna(0).clip(0, 1)
    sv = frame["sv_type"].fillna("").astype(str).str.upper()
    lead = (frame.index == frame["p_score"].idxmin()) if ld_reference == "minimum_p" else (
        frame["lead_variant"].astype(str).str.lower().isin({"true", "1", "1.0"}))
    fallback = ld_reference == "minimum_p" or frame["lead_selection"].iloc[0] == "minimum_p_no_credible_set"
    handles = []
    for sv_type, marker, label in (
        ("", "o", "SNV / small indel"),
        ("INS", "^", "INS SV"),
        ("DEL", "v", "DEL SV"),
    ):
        mask = sv.eq(sv_type)
        if not mask.any():
            continue
        for axis, values in axes_values:
            axis.scatter(
                x[mask],
                values[mask],
                c=r2[mask],
                cmap="viridis",
                norm=norm,
                marker=marker,
                s=(72 if sv_type else 22) * point_area_scale,
                edgecolor="none",
                linewidth=0,
                rasterized=False,
                zorder=4 if sv_type else 2,
            )
            if (mask & lead).any():
                selected = mask & lead
                axis.scatter(
                    x[selected],
                    values[selected],
                    c=r2[selected],
                    cmap="viridis",
                    norm=norm,
                    marker=marker,
                    s=(140 if sv_type else 80) * point_area_scale,
                    edgecolor="none" if sv_type else "black",
                    linewidth=0 if sv_type else 0.8,
                    rasterized=False,
                    zorder=5,
                )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markersize=8,
                markerfacecolor=plt.get_cmap("viridis")(0.5),
                markeredgecolor="none",
                label=label,
            )
        )
    handles.append(
        Patch(facecolor=style.PALETTE["yellow"], alpha=0.2, label="LAI outlier")
    )
    lead_type = sv.loc[lead].iloc[0] if np.any(lead) else ""
    handles.append(Line2D([], [], linestyle="none", marker={"INS": "^", "DEL": "v"}.get(lead_type, "o"), markersize=8,
                          markerfacecolor=plt.get_cmap("viridis")(1.0) if lead_type else "white",
                          markeredgecolor="none" if lead_type else "black",
                          label="Lead (min p)" if fallback else "Lead (PIP)"))
    return norm, handles


def make_figure(frame: pd.DataFrame, hit: pd.Series, genes: pd.DataFrame) -> plt.Figure:
    style.apply_style()
    chrom = int(str(hit["chrom"]).removeprefix("chr"))
    start, end = int(frame["window_start"].iloc[0]), int(frame["window_end"].iloc[0])
    fig = plt.figure(figsize=(11, 8.5))
    layout, lanes = _gene_layout(genes, chrom, start, end, fig.canvas.get_renderer())
    gene_height = max(1.1, 0.42 * lanes)
    height = max(8.5, 6.6 + gene_height)
    fig.set_size_inches(11, height)
    grid = fig.add_gridspec(
        3, 1, left=0.12, right=0.885, bottom=0.65 / height,
        top=1 - 1.85 / height, height_ratios=[2.1, 1.8, gene_height], hspace=0.16,
    )
    assoc_ax = fig.add_subplot(grid[0])
    pip_ax = fig.add_subplot(grid[1], sharex=assoc_ax)
    gene_ax = fig.add_subplot(grid[2], sharex=assoc_ax)
    norm, handles = draw_variants(
        frame, [(pip_ax, frame["pip"]), (assoc_ax, frame["neg_log10_p"])])
    fallback = frame["lead_selection"].iloc[0] == "minimum_p_no_credible_set"
    for axis in (pip_ax, assoc_ax, gene_ax):
        axis.axvspan(
            int(hit["start"]),
            int(hit["end"]),
            color=style.PALETTE["yellow"],
            alpha=0.18,
            zorder=0,
        )
        axis.set_xlim(start, end)
        axis.tick_params(labelsize=13)
    for axis in (pip_ax, assoc_ax):
        axis.tick_params(labelbottom=False)
        axis.grid(axis="y", alpha=0.2)
    pip_ax.set_ylim(-0.02, 1.05)
    pip_ax.set_ylabel("SuSiE PIP", fontsize=16)
    assoc_ax.set_ylim(bottom=0)
    assoc_ax.set_ylabel(style.P_VALUE_AXIS_LABEL, fontsize=16)
    draw_genes(gene_ax, layout, lanes, start, end)
    gene_ax.set_xlabel(f"chr{chrom} position (Mb, GRCh38)", fontsize=16)
    gene_ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    gene_ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value / 1e6:.2f}")
    )
    cax = fig.add_axes(
        [
            0.91,
            pip_ax.get_position().y0,
            0.018,
            assoc_ax.get_position().y1 - pip_ax.get_position().y0,
        ]
    )
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    colorbar.solids.set_rasterized(False)
    # Overlap adjacent vector faces to avoid PDF-viewer antialiasing seams.
    colorbar.solids.set_edgecolor("face")
    colorbar.set_label(r"LD to lead ($r^2$)", fontsize=13)
    colorbar.ax.tick_params(labelsize=12)
    description = str(hit.get("phecode_string", "")).strip()
    title = (
        f"{hit['phecode']}  |  {description}" if description else str(hit["phecode"])
    )
    fig.text(
        0.5,
        1 - 0.18 / height,
        textwrap.fill(title, 66),
        ha="center",
        va="top",
        fontsize=18,
    )
    fig.text(
        0.5,
        1 - 0.9 / height,
        f"chr{chrom}:{int(hit['start']):,}-{int(hit['end']):,}  |  10k discovery\n"
        f"Admixture p = {float(hit['p_disc']):.3g}  |  Fine-map n = {int(frame['n_samples'].iloc[0]):,}"
        f"  |  Cases = {int(frame['n_cases'].iloc[0]):,}"
        + ("  |  No credible set" if fallback else ""),
        ha="center",
        va="top",
        fontsize=13,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 1.42 / height),
        ncol=len(handles),
        fontsize=12,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    return fig


def render(
    frame: pd.DataFrame, hit: pd.Series, genes: pd.DataFrame, paths: list[Path]
) -> None:
    fig = make_figure(frame, hit, genes)
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + ".tmp" + path.suffix)
            with plt.rc_context({"savefig.bbox": None}):
                fig.savefig(temporary, dpi=300, bbox_inches=None)
            temporary.replace(path)
    finally:
        plt.close(fig)
