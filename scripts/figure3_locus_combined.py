"""Combined Figure 3 showcase-locus panels for the selection composite.

Each manuscript locus is rendered as one vertically stacked panel:
EHH -> normalized |iHS| -> nucleotide diversity (pi) -> gene track / position.
The underlying calculations and visual encodings are reused from figure3_ehh.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from . import figure3_ehh
from .common import config, io

# Compact numbering after combining the formerly paired panels.
MAIN_LOCI = [
    {**figure3_ehh.MAIN_LOCI[0], "panel": "2B"},  # LCT
    {**figure3_ehh.MAIN_LOCI[1], "panel": "2C"},  # SLC24A5
    {**figure3_ehh.MAIN_LOCI[2], "panel": "2D"},  # ANKRD31
]

_FIGSIZE_COMBINED = (12, 12.5)
# Panel C is recropped after its embedded title is removed, whereas this source
# panel is cropped only once.  The 4.9 gene allocation aligns the *displayed*
# B/C gene boundary (48% of the compact lower-row cell), not merely their raw
# source-grid ratios.
# Give the three quantitative tracks enough vertical room to remain legible
# after the locus panel is reduced into the half-page supplementary cell.  The
# gene allocation is unchanged because its lane packing is already tuned for
# dense loci.
COMBINED_HEIGHT_RATIOS = (3.8, 2.0, 1.7, 4.9)

# Gene names in the composite (Figure 3B/2C) sit ~2 pt smaller than the rest of the
# locus panel: these rasters are embedded down-scaled, where the shared
# GENE_FONTSIZE reads about 2 pt larger than the neighbouring labels.
GENE_FONTSIZE_COMBINED = figure3_ehh.GENE_FONTSIZE - 6


def main_panel_image_paths(ext: str = "png") -> dict[str, object]:
    """Map compact composite letters B-D to pre-rendered stacked locus images."""
    d = config.figure_dir("figure3")
    return {
        loc["panel"][-1]: d / f"Figure_3{loc['panel'][-1]}_{loc['gene']}_combined.{ext}"
        for loc in MAIN_LOCI
    }


def plot_combined_locus(
    gene,
    chrom,
    core_pos,
    ref,
    alt,
    ihs_val,
    rs_id,
    vcf_path,
    gtf_path,
    *,
    sv_pos=None,
    sv_type=None,
    figsize=None,
):
    """Render EHH, |norm iHS|, pi and genes as one shared-x stacked panel."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    start = max(core_pos - figure3_ehh.EHH_HALF_WINDOW, 1)
    end = core_pos + figure3_ehh.EHH_HALF_WINDOW

    ehh = figure3_ehh.compute_or_load_ehh(
        chrom, core_pos, ref, alt, vcf_path, start, end
    )
    ihs_pos, ihs_norm = figure3_ehh.load_norm_ihs(chrom, start, end)
    pi_data = figure3_ehh.compute_or_load_pi(chrom, core_pos, vcf_path)
    genes_dict = figure3_ehh.parse_gtf_region(gtf_path, chrom, start, end)

    fig = plt.figure(
        figsize=_FIGSIZE_COMBINED if figsize is None else figsize,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02)
    gs = fig.add_gridspec(
        4, 1,
        # Dense supplementary loci need enough physical height for the larger
        # lane count selected by ``draw_gene_track``.
        height_ratios=COMBINED_HEIGHT_RATIOS,
        hspace=0.03,
    )
    ax_ehh = fig.add_subplot(gs[0])
    ax_ihs = fig.add_subplot(gs[1], sharex=ax_ehh)
    ax_pi = fig.add_subplot(gs[2], sharex=ax_ehh)
    ax_gene = fig.add_subplot(gs[3], sharex=ax_ehh)

    # EHH: preserve the existing reference/alternate haplotype styling.
    pl = np.asarray(ehh["pos_left"], float)
    pr = np.asarray(ehh["pos_right"], float)

    def _joined(y_left, y_right):
        yl = np.asarray(y_left, float)
        yr = np.asarray(y_right, float)
        return (
            np.concatenate([pl[::-1], [core_pos], pr]),
            np.concatenate([yl[::-1], [1.0], yr]),
        )

    xr, yr_ref = _joined(ehh["ehh0_left"], ehh["ehh0_right"])
    xa, ya_alt = _joined(ehh["ehh1_left"], ehh["ehh1_right"])
    ax_ehh.plot(xr, yr_ref, color="#2471a3", lw=4.0, label="Ref. Hap.")
    ax_ehh.plot(xa, ya_alt, color="#c0392b", lw=4.0, label="Alt. Hap.")
    ax_ehh.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_ehh.plot(core_pos, 1.0, marker="v", color="black", ms=11, zorder=5)
    ax_ehh.set_ylim(-0.02, 1.05)
    ax_ehh.set_ylabel("EHH", fontsize=figure3_ehh.FONTSIZE)
    ax_ehh.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=figure3_ehh.LEGEND_FONTSIZE,
        handlelength=2.2,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    # Normalized |iHS|.
    if len(ihs_pos):
        abs_ihs = np.abs(ihs_norm)
        ax_ihs.vlines(
            ihs_pos, 0, abs_ihs,
            colors="#5b2c6f", alpha=0.35, linewidth=1.0,
        )
        ax_ihs.scatter(
            ihs_pos, abs_ihs,
            s=28, color="#5b2c6f", alpha=0.75,
            zorder=3, edgecolors="none",
        )
    ax_ihs.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_ihs.set_ylim(bottom=0)
    ax_ihs.set_ylabel("|iHS|", fontsize=figure3_ehh.FONTSIZE)

    # Nucleotide diversity pi.
    pi_mids = pi_data["pi_mids"]
    pi_vals = pi_data["pi_vals"]
    valid = ~np.isnan(pi_vals)
    if valid.sum():
        ax_pi.vlines(
            pi_mids[valid], 0, pi_vals[valid],
            colors="#117864", alpha=0.35, linewidth=1.0,
        )
        ax_pi.scatter(
            pi_mids[valid], pi_vals[valid],
            s=40, color="#117864", zorder=3,
            edgecolors="white", linewidths=0.8,
        )
    ax_pi.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_pi.set_ylim(bottom=0)
    ax_pi.set_ylabel("π", fontsize=figure3_ehh.FONTSIZE)

    # Shared gene track and genomic-position axis at the bottom.
    figure3_ehh.draw_gene_track(
        ax_gene,
        genes_dict,
        start,
        end,
        highlight_gene=gene,
        sv_pos=sv_pos,
        sv_type=sv_type,
        gene_fontsize=GENE_FONTSIZE_COMBINED,
    )
    ax_gene.set_xticks([start, core_pos, end])
    ax_gene.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1e6:.2f} Mb")
    )

    for ax in (ax_ehh, ax_ihs, ax_pi):
        ax.set_xlim(start, end)
        ax.grid(axis="y", alpha=0.3, lw=0.8)
        ax.tick_params(labelbottom=False, labelsize=figure3_ehh.TICK_FONTSIZE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax_gene.tick_params(labelsize=figure3_ehh.TICK_FONTSIZE)

    return fig


def render_main_panels(
    clean_run_plots: bool = False,
    panel_letters: tuple[str, ...] = ("B", "C", "D"),
) -> list:
    """Render stacked panels B-D for LCT, SLC24A5 and ANKRD31."""
    import matplotlib.pyplot as plt

    selected = [loc for loc in MAIN_LOCI if loc["panel"][-1] in panel_letters]
    fig_dir = config.figure_dir("figure3")
    stems = [
        fig_dir / f"Figure_3{loc['panel'][-1]}_{loc['gene']}_combined"
        for loc in selected
    ]
    targets = [
        stem.with_suffix(f".{extension}")
        for stem in stems
        for extension in config.FIGURE_FORMATS
    ]
    if not io.needs_build(targets, clean_run_plots):
        print(
            "[figure3-locus] reusing cached combined panels: "
            + ", ".join(panel_letters),
            flush=True,
        )
        return [stem.with_suffix(".png") for stem in stems]

    if not figure3_ehh.on_workbench():
        raise FileNotFoundError("Required phased VCF inputs are unavailable; stage them on Workbench.")
    try:
        import allel  # noqa: F401
    except ImportError:
        warnings.warn(
            "[figure3] scikit-allel not installed -- run `pip install scikit-allel`.",
            stacklevel=2,
        )
        return []

    saved = []
    with plt.rc_context(figure3_ehh._RC):
        for loc in selected:
            gene = loc["gene"]
            chrom = loc["chrom"]
            core_pos = loc["core_pos"]
            vcf_path = figure3_ehh.VCF_PATTERN.format(chrom=chrom)
            if not os.path.exists(vcf_path):
                warnings.warn(
                    f"[figure3] VCF not found for {gene}: {vcf_path}",
                    stacklevel=2,
                )
                continue

            stem = fig_dir / f"Figure_3{loc['panel'][-1]}_{gene}_combined"
            try:
                fig = plot_combined_locus(
                    gene,
                    chrom,
                    core_pos,
                    loc["ref"],
                    loc["alt"],
                    loc["ihs"],
                    loc["rs"],
                    vcf_path,
                    figure3_ehh.GTF_PATH,
                    sv_pos=loc.get("sv_pos"),
                )
                for ext in config.FIGURE_FORMATS:
                    fig.savefig(
                        stem.with_suffix(f".{ext}"),
                        **figure3_ehh._SAVEFIG_KW,
                    )
                plt.close(fig)
                saved.append(stem.with_suffix(".png"))
            except Exception as exc:
                warnings.warn(
                    f"[figure3] {loc['panel']} combined ({gene}) failed: {exc}",
                    stacklevel=2,
                )
    return saved
