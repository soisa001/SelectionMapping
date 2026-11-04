"""Figure 3 plotting stage -- selection scan and candidate loci."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import figure3_data
from .common import config, style

FIGURE_ID = "figure3"


_SCAN_SLOT_WIDTH = 1_000_000     # matches figure3_data.SCAN_SLOT_WIDTH


_YLIM = 12.0
_HIT_LABEL_FONT_SCALE = 0.66
_RING_DASHES = (0, (2.4, 1.8))
_KNOWN_SNV_HIT_GENES = frozenset({"LCT", "SLC24A5"})
MAX_NONSIGNIFICANT_POINTS = 10_000
BACKGROUND_SAMPLE_SEED = figure3_data.SEED
BACKGROUND_EXTREMES_PER_CHROMOSOME = 10

# Collision-audited label anchors for the crowded S3 SV-gene annotations. The x
# value is a within-chromosome slot fraction (values outside 0..1 intentionally
# cross slots); y is in iHS units. Fixed anchors keep PNG/PDF layout identical.
_SV_LABEL_LAYOUT = {
    "PARK7/ERRFI1": (0.45, 7.6),
    "THRAP3": (0.45, 10.6),
    "HS2ST1": (2.20, 9.8),
    "PLEKHB2": (3.45, 6.5),
    "CDH10": (0.65, 8.4),
    "ANKRD31": (0.20, -10.0),
    "EYS": (1.00, 7.0),
    "TCP10L2": (-0.20, -7.0),
    "MUC3A": (1.50, -7.2),
    "CWF19L2": (0.60, 7.3),
    "CTNS": (0.45, 2.6),
    "RBBP8": (0.52, 7.6),
    "FRG1BP": (0.66, -0.7),
    "ZNF74": (0.42, 6.3),
}

_CUSTOM_LABEL_LAYOUT = {
    # Move LCT clear of the multi-line PARK7/ERRFI1 annotation.
    "LCT": (1.30, 5.6),
}

# Figure 3 keeps only intragenic SVs and moves labels farther from the dense
# red point bands. THRAP3 is shifted into the plot so it clears the left axis.
_GENIC_SV_LABEL_LAYOUT = {
    "THRAP3": (1.20, 10.7),
    "HS2ST1": (2.20, 9.2),
    "PLEKHB2": (3.45, 7.4),
    "CDH10": (0.65, 9.0),
    "ANKRD31": (0.35, -10.5),
    "EYS": (1.00, 7.8),
    "CTNS": (0.15, 6.4),
    "RBBP8": (0.65, 8.3),
    "FRG1BP": (0.66, -5.4),
    "ZNF74": (0.42, 7.0),
}

_KNOWN_SNV_LABEL_LAYOUT = {
    "LCT": (1.45, 6.8),
    "SLC24A5": (0.35, -6.2),
}


def _downsample_manhattan_background(
    nonsignificant,
    *,
    max_points: int = MAX_NONSIGNIFICANT_POINTS,
    seed: int = BACKGROUND_SAMPLE_SEED,
    extremes_per_chromosome: int = BACKGROUND_EXTREMES_PER_CHROMOSOME,
):
    """Deterministically thin ordinary windows while preserving the envelope."""
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if extremes_per_chromosome < 0:
        raise ValueError("extremes_per_chromosome must be non-negative")
    frame = nonsignificant.copy()
    if len(frame) <= max_points:
        return frame, {
            "input_points": len(frame),
            "output_points": len(frame),
            "background_cap": max_points,
            "seed": seed,
            "extremes_per_chromosome": extremes_per_chromosome,
        }

    chromosomes = sorted(frame["chrom_num"].astype(int).unique())
    rng = np.random.default_rng(seed)
    base_quota, remainder = divmod(max_points, len(chromosomes))
    retained: list[int] = []
    for order, chromosome in enumerate(chromosomes):
        local = frame[frame["chrom_num"].astype(int).eq(chromosome)]
        quota = min(len(local), base_quota + int(order < remainder))
        if quota == 0:
            continue
        extreme_count = min(quota, extremes_per_chromosome)
        extremes = (
            local["max_emp_ihs"].abs().nlargest(extreme_count).index.to_numpy()
        )
        retained.extend(int(index) for index in extremes)
        remaining_quota = quota - len(extremes)
        if remaining_quota:
            pool = local.index.difference(extremes).to_numpy()
            sampled = rng.choice(pool, size=remaining_quota, replace=False)
            retained.extend(int(index) for index in sampled)

    thinned = frame.loc[retained].sort_values(
        ["chrom_num", "position"], kind="stable"
    )
    return thinned, {
        "input_points": len(frame),
        "output_points": len(thinned),
        "background_cap": max_points,
        "seed": seed,
        "extremes_per_chromosome": extremes_per_chromosome,
    }


def _draw_ihs_manhattan(ax: plt.Axes, data: dict, *, fs: float = 13.0,
                        gene_fs: float | None = None, sig_s: float | None = None,
                        label_gap: float = 1.4, genic_sv_only: bool = False,
                        circle_labelled_sv_only: bool = False,
                        sv_label_layout: dict | None = None,
                        known_snv_label_layout: dict | None = None,
                        sv_leader_lw: float = 0.75,
                        known_snv_leader_lw: float = 1.2,
                        known_snv_ring_dashed: bool = True,
                        show_legend: bool = True,
                        max_background_points: int = MAX_NONSIGNIFICANT_POINTS,
                        background_sample_seed: int = BACKGROUND_SAMPLE_SEED) -> None:
    """iHS selection scan (Figure 3A): max empirical iHS per window across autosomes.

    Windows are laid out in fixed-width per-chromosome slots; non-significant
    windows are coloured by chromosome parity, candidate windows (sim p <= 0.01)
    are red and drawn large on top. Highlighted genes are labelled just beyond
    their peak with a short leader (peak position + iHS from the merged-regions
    TSV where available; ``data['gene_labels']``).
    """
    df = data["scan"]
    nonsig = df[~df["candidate"]]
    sig = df[df["candidate"]]
    nonsig, thinning = _downsample_manhattan_background(
        nonsig,
        max_points=max_background_points,
        seed=background_sample_seed,
    )
    print(
        "[figure3-manhattan] background downsampling: "
        + ", ".join(f"{key}={value}" for key, value in thinning.items()),
        flush=True,
    )
    gene_fs = gene_fs if gene_fs is not None else fs * _HIT_LABEL_FONT_SCALE
    sig_s = sig_s if sig_s is not None else fs * 1.8   # significant points: large
    tick_fs = fs * 0.90

    colors = np.where(nonsig["chrom_num"] % 2 == 0, "#ADD8E6", "#2C7A7B")
    ax.scatter(nonsig["cumulative_pos"], nonsig["max_emp_ihs"], c=colors, s=5,
               alpha=0.7, edgecolor="none", zorder=1)
    ax.scatter(sig["cumulative_pos"], sig["max_emp_ihs"], c="red", s=sig_s,
               alpha=0.9, edgecolor="none", zorder=2)

    sv_hits = data.get("sv_hits")
    plotted_sv_hits = sv_hits
    if genic_sv_only and sv_hits is not None and not sv_hits.empty:
        is_nearby = sv_hits["sv_gene_context"].str.startswith(
            "intergenic", na=False
        )
        plotted_sv_hits = sv_hits.loc[~is_nearby].copy()
    sv_label_genes = set()
    if plotted_sv_hits is not None and not plotted_sv_hits.empty:
        sv_label_genes = set(
            plotted_sv_hits.loc[
                plotted_sv_hits["label_sv"], "sv_gene_label"
            ].astype(str)
        )

    # Gene labels: sit just beyond each peak with a short leader pointing to it.
    # Keep labels inside a safe band (well clear of the +/-YLIM axis edges, so a
    # deep peak like ANKRD31 doesn't push its label onto the chromosome axis).
    label_max = _YLIM - 3.5
    labels = data.get("gene_labels")
    if labels is not None:
        known_snv_labels = labels.loc[
            labels["gene"].isin(_KNOWN_SNV_HIT_GENES)
            & ~labels["gene"].isin(sv_label_genes)
        ].copy()
        _draw_known_snv_hit_rings(
            ax,
            known_snv_labels,
            sig_s=sig_s,
            dashed=known_snv_ring_dashed,
        )
        for _, g in known_snv_labels.iterrows():
            x, y_peak = g["cumulative_pos"], g["max_emp_ihs"]
            sign = 1.0 if y_peak >= 0 else -1.0
            y_tip = float(np.clip(y_peak, -(_YLIM - 0.4), _YLIM - 0.4))
            if g["gene"] == "ANKRD31":
                # Several significant chr5 windows sit close together. Keep
                # ANKRD31 centred directly above its selected plotted point so
                # the leader cannot look like it belongs to a neighbouring dot.
                y_lab = float(np.clip(y_tip + 2.0, -label_max, label_max))
                ax.annotate(
                    g["gene"], xy=(x, y_tip), xytext=(x, y_lab),
                    fontsize=gene_fs, ha="center", va="bottom", zorder=4,
                    arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2,
                                "shrinkA": 2, "shrinkB": 1},
                    annotation_clip=False,
                )
                continue
            label_layout = known_snv_label_layout or _CUSTOM_LABEL_LAYOUT
            layout = label_layout.get(str(g["gene"]))
            if layout is None:
                x_text = x
                y_lab = float(
                    np.clip(y_tip + sign * label_gap, -label_max, label_max)
                )
            else:
                slot_fraction, y_lab = layout
                x_text = (
                    int(g["chrom_num"]) - 1 + slot_fraction
                ) * _SCAN_SLOT_WIDTH
            ax.plot(
                [x_text, x], [y_lab, y_tip], color="black",
                lw=known_snv_leader_lw, zorder=3,
            )
            ax.text(
                x_text, y_lab, g["gene"], fontsize=gene_fs, ha="center",
                va="bottom" if sign > 0 else "top", fontweight="bold",
                zorder=4,
            )

    _draw_sv_hit_annotations(
        ax,
        plotted_sv_hits,
        label_fs=gene_fs,
        sig_s=sig_s,
        label_layout=sv_label_layout,
        leader_lw=sv_leader_lw,
        circle_labelled_only=circle_labelled_sv_only,
    )

    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.axhline(2, ls="--", color="red", lw=0.8)
    ax.axhline(-2, ls="--", color="red", lw=0.8)
    ax.set_ylim(-_YLIM, _YLIM)

    n_chrom = int(df["chrom_num"].max())
    # Use the full chromosome-slot span. Matplotlib's default x margin adds
    # roughly one extra chromosome of blank canvas at each side of this panel.
    ax.set_xlim(0, n_chrom * _SCAN_SLOT_WIDTH)
    ax.set_xticks([(i - 0.5) * _SCAN_SLOT_WIDTH for i in range(1, n_chrom + 1)])
    ax.set_xticklabels([str(i) for i in range(1, n_chrom + 1)], fontsize=tick_fs)
    ax.set_xlabel("Chromosome", fontsize=fs)
    ax.set_ylabel("Max empirical iHS", fontsize=fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    if show_legend:
        # Flat single-row legend above the panel (out of the data area).
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red",
                   markersize=fs * 0.75,
                   label=r"Significant (p $\leq$ 0.01)"),
            Circle(
                (0, 0), radius=0.5, facecolor="none", edgecolor="black",
                linewidth=1.5, linestyle=_RING_DASHES,
                label="Known SNV hit",
            ),
            Circle(
                (0, 0), radius=0.5, facecolor="none", edgecolor="black",
                linewidth=1.5, label="SV or SV-linked selection hit",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=tick_fs * 0.92,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.2,
            handler_map={Circle: HandlerPatch(patch_func=_legend_circle)},
        )


def _legend_circle(
    legend,
    orig_handle,
    xdescent,
    ydescent,
    width,
    height,
    fontsize,
):
    """Render solid and dashed circular ring handles in the panel legend."""
    del legend, orig_handle, fontsize
    radius = min(width, height) / 2.0
    return Circle((xdescent + width / 2.0, ydescent + height / 2.0), radius)


def _draw_known_snv_hit_rings(
    ax: plt.Axes,
    labels,
    *,
    sig_s: float,
    dashed: bool = True,
) -> None:
    """Mark the two named known SNV hits with dashed or solid rings."""
    if labels is None or labels.empty:
        return
    ax.scatter(
        labels["cumulative_pos"],
        labels["max_emp_ihs"],
        s=max(80.0, sig_s * 3.8),
        facecolors="none",
        edgecolors="black",
        linewidths=1.45,
        linestyles=_RING_DASHES if dashed else "solid",
        zorder=5,
    )


def _draw_sv_hit_annotations(
    ax: plt.Axes,
    hits,
    *,
    label_fs: float,
    sig_s: float,
    label_layout: dict | None = None,
    leader_lw: float = 0.75,
    circle_labelled_only: bool = False,
) -> None:
    """Circle the requested S3 SV-associated hits and label unique alleles."""
    if hits is None or hits.empty:
        return

    label_rows = hits[hits["label_sv"]].copy()
    ring_rows = label_rows if circle_labelled_only else hits
    ring_size = max(80.0, sig_s * 3.8)
    ax.scatter(
        ring_rows["cumulative_pos"],
        ring_rows["max_emp_ihs"],
        s=ring_size,
        facecolors="none",
        edgecolors="black",
        linewidths=1.45,
        zorder=5,
    )

    for index, row in label_rows.reset_index(drop=True).iterrows():
        x = float(row["cumulative_pos"])
        y = float(row["max_emp_ihs"])
        layouts = label_layout or _SV_LABEL_LAYOUT
        layout = layouts.get(str(row["sv_gene_label"]))
        if layout is None:
            # Fallback for a future source-table addition; current S3 labels all
            # have explicit anchors above so their layout remains reviewable.
            x_text = x + (-0.055 if index % 2 else 0.055) * _SCAN_SLOT_WIDTH
            y_text = y + (0.85 if y < _YLIM - 2.0 else -0.85)
        else:
            slot_fraction, y_text = layout
            x_text = (int(row["chrom_num"]) - 1 + slot_fraction) * _SCAN_SLOT_WIDTH
        ax.annotate(
            str(row["display_gene_label"]),
            xy=(x, y),
            xytext=(x_text, y_text),
            fontsize=label_fs,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "black", "lw": leader_lw,
                        "alpha": 0.9, "shrinkA": 2, "shrinkB": 1},
            annotation_clip=False,
        )


def panel_A(ax: plt.Axes, data: dict) -> None:
    """Figure 3 scan with uniform solid rings and no panel legend."""
    _draw_ihs_manhattan(
        ax,
        data,
        fs=13.0,
        genic_sv_only=True,
        circle_labelled_sv_only=True,
        sv_label_layout=_GENIC_SV_LABEL_LAYOUT,
        known_snv_label_layout=_KNOWN_SNV_LABEL_LAYOUT,
        sv_leader_lw=1.15,
        known_snv_leader_lw=1.5,
        known_snv_ring_dashed=False,
        show_legend=False,
    )


def render_scan_manhattan(clean_run_data: bool = False,
                             clean_run_plots: bool = False) -> list:
    """Render panel A with uniform solid rings and no legend."""
    from .common.io import needs_build

    style.apply_style()
    data = figure3_data.build(clean_run_data=clean_run_data)
    out = config.figure_dir(FIGURE_ID) / "Figure_3A_iHS_manhattan"
    targets = [out.with_suffix(f".{ext}") for ext in config.FIGURE_FORMATS]
    if not needs_build(targets, clean_run_plots or clean_run_data):
        return targets
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _draw_ihs_manhattan(
        ax,
        data,
        fs=15.0,
        genic_sv_only=True,
        circle_labelled_sv_only=True,
        sv_label_layout=_GENIC_SV_LABEL_LAYOUT,
        known_snv_label_layout=_KNOWN_SNV_LABEL_LAYOUT,
        sv_leader_lw=1.15,
        known_snv_leader_lw=1.5,
        known_snv_ring_dashed=False,
        show_legend=False,
    )
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        fig.savefig(target)
    plt.close(fig)
    return targets
