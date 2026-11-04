"""Discovery PheWAS/ancestry/fine-map figures with aligned exon/intron tracks."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D

from . import admixture_finemap as fm, figure2_1025 as corrected
from . import figure2_loci_plots as locus_plots, figure2_plots
from . import figure2_locus_phewas as locus_phewas
from . import figure2_locus_results as locus_results
from .common import style

PLOT_CONTRACT = "nominal_discovery_eur_like_smaller_svs_no_red_finemap_v13"
LD_LABEL = "r² to most significant variant"
PANEL_HEIGHTS = (2.05, 1.55, 2.25)
FIGURE_WIDTH = 8.25
AXIS_LEFT = 1.4 / FIGURE_WIDTH
AXIS_RIGHT = 0.97
POINT_AREA_SCALE = 1.5
ANCESTRY_SV_AREA_SCALE = 0.75
GENE_TRACK_GAP_INCHES = 0.30


def output_dir() -> Path:
    from . import figure2_loci
    return figure2_loci.output_dir() / "figures"


def _phenotype_stem(hit) -> str:
    description = hit.get("phecode_string")
    if pd.isna(description):
        raise ValueError(f"Missing phecode_string for {hit['phecode']}")
    stem = re.sub(r"[^a-z0-9]+", "_", str(description).lower()).strip("_")
    if not stem:
        raise ValueError(f"Empty phenotype filename for {hit['phecode']}")
    return stem


def plot_paths(hit) -> list[Path]:
    chrom = int(str(hit["chrom"]).removeprefix("chr"))
    stem = f"chr{chrom}_{int(hit['start'])}_{int(hit['end'])}_{_phenotype_stem(hit)}"
    return [output_dir() / f"{stem}.{ext}" for ext in ("png", "pdf")]


class MarkerScans:
    """Verify saved LAI sample provenance; load each needed chromosome once.

    No raw VCF staging or analysis is performed by this plotting phase. The
    genome-wide downsampled plotting table and chr9-only zoom cannot substitute
    for the full chromosome marker scans.
    """

    def __init__(self, ids):
        self.sample_n = len(ids)
        self.sample_sha256 = corrected._sample_sha(ids)
        self.root = corrected.lai_dir()
        manifest_path = corrected._stage_manifest_path("lai")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = self.manifest["contract"]
        if (contract.get("sample_n") != self.sample_n or
                contract.get("sample_sha256") != self.sample_sha256):
            raise ValueError("LAI scan manifest does not match the selected LAI sample IDs")
        self.stats_path = self.root / "scan_stats.json"
        self.stats_source = self._verify(self.stats_path)
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        mean, sd, z = (float(self.stats[key]) for key in
                       ("mean_eur", "sd_eur", "z_threshold"))
        if not (np.isfinite([mean, sd, z]).all() and 0 <= mean <= 1 and sd > 0 and z > 0):
            raise ValueError("Invalid saved marker-level ancestry thresholds")
        if z != contract.get("z_threshold"):
            raise ValueError("LAI scan statistics and manifest thresholds disagree")
        self.chrom = None
        self.sources = {}

    def _verify(self, path):
        if not corrected._stage_current("lai", self.manifest["contract"], [path]):
            raise ValueError(f"Missing or corrupt corrected LAI source: {path}. "
                             "Repair the Figure_2_1025 LAI products before plotting.")
        return next(entry for entry in self.manifest["outputs"] if entry["path"] == str(path))

    def region(self, hit) -> tuple[pd.DataFrame, dict]:
        chrom = int(str(hit["chrom"]).removeprefix("chr"))
        if chrom != self.chrom:
            path = self.root / "chromosome_scans" / f"chr{chrom}.eur_fraction.npz"
            self.sources[chrom] = self._verify(path)
            with np.load(path, allow_pickle=False) as saved:
                pos, fraction, sample_n = saved["pos"], saved["eur_frac"], saved["n_samples"]
            if (pos.ndim != 1 or fraction.shape != pos.shape or not len(pos)
                    or not np.issubdtype(pos.dtype, np.integer)
                    or (pos < 1).any() or (np.diff(pos) < 0).any()
                    or not np.isfinite(fraction).all()
                    or ((fraction < 0) | (fraction > 1)).any()
                    or sample_n.shape != (1,) or sample_n[0] != self.sample_n):
                raise ValueError(f"Invalid marker coordinates, fractions or sample count: {path}")
            self.chrom, self.pos, self.fraction = chrom, pos, fraction
        start, end = fm.expected_window(hit)
        left = np.searchsorted(self.pos, start, side="left")
        right = np.searchsorted(self.pos, end, side="right")
        if left == right:
            raise ValueError(f"No LAI markers in chr{chrom}:{start}-{end}")
        frame = pd.DataFrame({"pos": self.pos[left:right], "eur_frac": self.fraction[left:right]})
        frame["in_region"] = frame["pos"].between(int(hit["start"]), int(hit["end"]))
        provenance = {
            "scan": self.sources[chrom], "statistics": self.stats_source,
            "lai_sample_n": self.sample_n, "lai_sample_sha256": self.sample_sha256,
            "window_start": start, "window_end": end, "n_markers": len(frame),
            "sv_positions": "QC-passing fine-map SVs: linear interpolation of cohort EUR fraction; nearest boundary outside marker span",
        }
        return frame, provenance


def _sv_positions(frame, markers):
    columns = [key for key in ("variant_id", "pos", "sv_type", "sv_length_bp", "mac") if key in frame]
    svs = frame.loc[frame["sv_type"].fillna("").ne(""), columns].copy()
    for key in ("overall_eur_frac", "left_lai_marker", "right_lai_marker", "ancestry_method"):
        svs[key] = pd.Series(index=svs.index, dtype=object if key == "ancestry_method" else float)
    if svs.empty:
        return svs
    track = markers.groupby("pos", as_index=False)["eur_frac"].mean().sort_values("pos")
    if track.empty or not np.isfinite(track.to_numpy()).all() or not track["eur_frac"].between(0, 1).all():
        raise ValueError("SV ancestry interpolation requires finite LAI marker fractions in [0, 1]")
    positions = track["pos"].to_numpy()
    target = svs["pos"].to_numpy()
    right = np.searchsorted(positions, target, side="left").clip(0, len(positions) - 1)
    left = (np.searchsorted(positions, target, side="right") - 1).clip(0, len(positions) - 1)
    svs["overall_eur_frac"] = np.interp(target, positions, track["eur_frac"].to_numpy())
    svs["left_lai_marker"], svs["right_lai_marker"] = positions[left], positions[right]
    svs["ancestry_method"] = np.where(
        (target < positions[0]) | (target > positions[-1]), "nearest_boundary",
        np.where(positions[left] == positions[right], "observed_marker", "linear_interpolation"))
    return svs


def make_figure(frame, hit, markers, stats, *, genes, lai_sample_n,
                phewas, categories, phewas_parameters):
    """Stack region PheWAS, LAI, fine-map p-values and genes without panel letters."""
    style.apply_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH, 8.5))
    start, end = fm.expected_window(hit)
    chrom = int(str(hit["chrom"]).removeprefix("chr"))
    layout, lanes = locus_plots._gene_layout(
        genes, chrom, start, end, fig.canvas.get_renderer(),
        width_inches=FIGURE_WIDTH * (AXIS_RIGHT - AXIS_LEFT))
    gene_height = max(0.6, 0.34 * lanes)
    # Keep data panels and text at fixed physical sizes; dense genes get
    # extra room instead of squeezing the labels or the ancestry panel.
    # Clear the association axis's zero tick even for a single gene lane,
    # where the vertical Genes label extends slightly above its short axis.
    heights = [PANEL_HEIGHTS[0], 1.4, PANEL_HEIGHTS[1], 0.50,
               PANEL_HEIGHTS[2], GENE_TRACK_GAP_INCHES, gene_height]
    height = 0.58 + 0.62 + sum(heights)
    fig.set_size_inches(FIGURE_WIDTH, height)
    grid = fig.add_gridspec(7, 1, left=AXIS_LEFT, right=AXIS_RIGHT, bottom=0.62 / height,
                            top=1 - 0.58 / height, height_ratios=heights, hspace=0)
    phewas_axis = fig.add_subplot(grid[0])
    ancestry = fig.add_subplot(grid[2])
    assoc = fig.add_subplot(grid[4], sharex=ancestry)
    gene_axis = fig.add_subplot(grid[6], sharex=ancestry)
    norm, _ = locus_plots.draw_variants(
        frame, [(assoc, frame["neg_log10_p"])], position_scale=1e-6,
        ld_reference="minimum_p", point_area_scale=POINT_AREA_SCALE)
    assoc.axvspan(int(hit["start"]) / 1e6, int(hit["end"]) / 1e6,
                  color=style.PALETTE["yellow"], alpha=0.18, zorder=0)
    assoc.set_ylim(bottom=0)
    assoc.set_ylabel(style.P_VALUE_AXIS_LABEL, fontsize=20)
    assoc.tick_params(labelbottom=False)
    assoc.grid(axis="y", alpha=0.2)
    data = {"region_zoom": markers, "display_region": {**dict(hit), "chrom": chrom},
            "locus_window": (start, end), "stats": stats,
            "regional_svs": _sv_positions(frame, markers)}
    figure2_plots.panel_E(ancestry, data, sv_marker_area_scale=ANCESTRY_SV_AREA_SCALE)
    ancestry.set_ylabel("EUR-like ancestry\nfraction", fontsize=18)
    ancestry.set_xlabel("")
    ancestry.tick_params(labelbottom=False)
    for collection in ancestry.collections:
        collection.set_rasterized(False)
        sizes = collection.get_sizes()
        collection.set_sizes(sizes * 2 * POINT_AREA_SCALE)
    for line in ancestry.lines:
        line.set_linewidth(1.3)
    legend = ancestry.get_legend()
    ancestry_handles = list(legend.legend_handles) if legend else []
    if legend:
        legend.remove()
    ancestry_handles.extend([
        Line2D([], [], color="black", label="Genome-wide mean"),
        Line2D([], [], color=style.PALETTE["vermillion"], linestyle="--",
               label=f"Mean ± {float(stats['z_threshold']):.2f} SD"),
    ])
    ancestry.legend(handles=ancestry_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=len(ancestry_handles), fontsize=11, columnspacing=0.5, handletextpad=0.5)
    for axis in (assoc, ancestry):
        axis.set_xlim(start / 1e6, end / 1e6)
        axis.tick_params(labelsize=16)
    gene_axis.axvspan(int(hit["start"]) / 1e6, int(hit["end"]) / 1e6,
                     color=style.PALETTE["yellow"], alpha=0.18, zorder=0)
    locus_plots.draw_genes(gene_axis, layout, lanes, start, end, position_scale=1e-6)
    gene_axis.set_xlim(start / 1e6, end / 1e6)
    gene_axis.set_ylabel("Genes", fontsize=16)
    gene_axis.set_xlabel(f"chr{chrom} position (Mb, GRCh38)", fontsize=20)
    gene_axis.tick_params(labelsize=16)
    gene_axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
    gene_axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:.2f}"))
    # Draw after the other axes so label repulsion sees the final canvas size.
    enlarged_parameters = replace(phewas_parameters,
        marker_area_min=phewas_parameters.marker_area_min * POINT_AREA_SCALE,
        marker_area_max=phewas_parameters.marker_area_max * POINT_AREA_SCALE)
    locus_phewas.draw_panel(phewas_axis, phewas, hit, categories, enlarged_parameters)
    # Allocate only the space actually needed by A's rotated categories and
    # B's header. Physical panel/text sizes are unchanged by this final resize.
    fig.canvas.draw()
    category_bottom = min(label.get_window_extent().y0 for label in phewas_axis.get_xticklabels())
    b_header_top = ancestry.get_legend().get_window_extent().y1
    heights[1] = ((phewas_axis.bbox.y0 - category_bottom + b_header_top - ancestry.bbox.y1)
                  / fig.dpi + 0.06)
    height = 0.58 + 0.62 + sum(heights)
    fig.set_size_inches(FIGURE_WIDTH, height)
    grid.set_height_ratios(heights)
    grid.update(bottom=0.62 / height, top=1 - 0.58 / height)
    position = assoc.get_position()
    cax = fig.add_axes([position.x0 + position.width * 0.56, position.y1 + 0.24 / height,
                       position.width * 0.44, 0.13 / height])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                           orientation="horizontal", ticks=np.linspace(0, 1, 6))
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.outline.set_visible(False)
    colorbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(labelsize=13, pad=2)
    label_x = 0.56 - 0.08 / (position.width * FIGURE_WIDTH)
    assoc.text(label_x, 1 + 0.305 / PANEL_HEIGHTS[2], LD_LABEL, transform=assoc.transAxes,
               fontsize=15, va="center", ha="right")
    return fig


def render(frame, hit, markers, stats, paths, *, genes, lai_sample_n,
           phewas, categories, phewas_parameters):
    fig = make_figure(frame, hit, markers, stats, genes=genes, lai_sample_n=lai_sample_n,
                      phewas=phewas, categories=categories, phewas_parameters=phewas_parameters)
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + ".tmp" + path.suffix)
            with plt.rc_context({"savefig.bbox": None}):
                fig.savefig(temporary, dpi=300, bbox_inches=None)
            temporary.replace(path)
    finally:
        plt.close(fig)


def association_results(frame, hit, genes):
    """Keep the minimum-p variant, plotted lead and maximum PIP distinct."""
    result = fm.summarize_fit(frame)
    minimum = frame.loc[frame["p_score"].idxmin()]
    lead, _ = fm.select_lead(frame)
    start, end = fm.expected_window(hit)
    chrom = str(hit["chrom"]).removeprefix("chr")
    local = genes.loc[
        genes["chrom"].astype(str).str.removeprefix("chr").eq(chrom)
        & genes["start"].le(end) & genes["end"].ge(start), "gene"]
    members = frame["credible_set"].fillna("").astype(str).str.strip()
    result.update(
        fine_mapped_region=f"chr{chrom}:{start}-{end}", genome_build="GRCh38",
        window_start=start, window_end=end, window_pad_bp=fm.WINDOW_PAD_BP,
        p_disc=float(hit["p_disc"]),
        min_p=float(minimum["p_score"]), min_p_variant=minimum["variant_id"],
        min_p_pos=int(minimum["pos"]), min_p_pip=float(minimum["pip"]),
        combined_ld_reference_variant=minimum["variant_id"],
        combined_ld_reference_pos=int(minimum["pos"]),
        min_p_mac=int(minimum["mac"]), min_p_n_carriers=int(minimum["n_carriers"]),
        credible_set_ids=",".join(sorted({name for value in members for name in value.split(",") if name})),
        n_credible_set_variants=int(members.ne("").sum()),
        genes_in_finemap_window=",".join(sorted(local.dropna().astype(str).unique())),
        n_controls=result["n_samples"] - result["n_cases"],
        n_pcs=int(frame["n_pcs"].iloc[0]), sex_scope=str(frame["sex_scope"].iloc[0]),
        analysis_contract=str(frame["analysis_contract"].iloc[0]),
    )
    # Report the exact cached classification used for marker shapes; long VCF
    # REF/ALT strings alone do not establish a structural insertion/deletion.
    for prefix, variant in (("min_p", minimum), ("lead", lead)):
        sv_type = "" if pd.isna(variant["sv_type"]) else str(variant["sv_type"]).strip().upper()
        length = variant.get("sv_length_bp", 0)
        result.update({f"{prefix}_is_sv": sv_type in {"INS", "DEL"},
                       f"{prefix}_sv_type": sv_type,
                       f"{prefix}_sv_length_bp": 0 if pd.isna(length) else int(length)})
    result.update(locus_results.summary_fields(frame))
    return result


def _write_results(rows, summary_path, results_path):
    """Checkpoint both formats; validate the CSV before replacing it."""
    frame = pd.DataFrame(rows)
    corrected._atomic_tsv(frame, summary_path)
    temporary = results_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    checked = pd.read_csv(temporary, dtype={"region_id": str, "phecode": str})
    keys = ["region_id", "phecode"]
    if (list(checked.columns) != list(frame.columns) or len(checked) != len(frame)
            or not checked[keys].equals(frame[keys].astype(str))):
        raise ValueError(f"Results CSV failed read-back validation: {temporary}")
    temporary.replace(results_path)


def _write_sv_ancestry(frame, path):
    """Check the complete round trip before replacing an ancestry report."""
    temporary = path.with_suffix(".tsv.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    checked = pd.read_csv(temporary, sep="\t", float_precision="round_trip",
                          dtype={"phecode": str, "region_id": str, "variant_id": str})
    pd.testing.assert_frame_equal(checked, frame.reset_index(drop=True), check_dtype=False,
                                  rtol=1e-12, atol=0)
    temporary.replace(path)


def save_hits(hits, ids, *, force=False, panel=None):
    """Separate plotting phase, called inside the corrected analysis context."""
    from . import figure2_loci as loci

    started = time.perf_counter()
    summary_path = output_dir() / "figure_summary.tsv"
    results_path = output_dir() / "fine_mapping_results.csv"
    members_csv = output_dir() / "credible_set_variants.csv"
    members_tsv = output_dir() / "credible_set_variants.tsv"
    manifest_path = output_dir() / "run_manifest.json"
    manifest = {"plot_contract": PLOT_CONTRACT, "results_contract": locus_results.RESULTS_CONTRACT, "status": "running",
                "variant_set": panel.variant_set if panel is not None else "all_variants",
                "started_utc": corrected._utc_now(), "expected_associations": len(hits),
                "completed_associations": 0, "output_directory": str(output_dir()),
                "results_table": str(results_path),
                "sv_ancestry_table": str(output_dir() / "sv_ancestry.tsv"),
                "sv_ancestry_definition": (
                    "Linear interpolation of cohort-wide EUR fraction between adjacent plotted LAI markers; "
                    "observed_marker for exact matches; nearest_boundary outside marker span. "
                    "Display values only, not SV genotypes or carrier-specific ancestry."),
                "credible_set_tables": [str(members_csv), str(members_tsv)],
                "results_columns": {
                    "p_disc": "Region-level discovery admixture association p-value",
                    "min_p": "Minimum per-variant logistic score-test p-value in the fine map",
                    "credible_set_variants": "All distinct SuSiE credible-set members, semicolon separated, in genomic order",
                    "representative_selection": "Minimum score-test p-value across the whole fine-map window, including when outside a credible set",
                    "representative_beta_score": "Cached covariate-adjusted logistic score beta for the representative ALT-dosage variant",
                    "representative_se_score": "Cached score-test SE for the same representative variant",
                    "credible_set_tables": "One row per association/variant; credible_set_ids retains every membership; no rows for a fit without credible sets",
                    "lead_pip": "PIP of the fine-map lead in the original locus/PIP plots; selection rule is in lead_selection",
                    "combined_ld_reference_variant": "Minimum score-test p-value variant used for Panel C LD colours and outline",
                    "min_p_is_sv": "Whether the minimum-p variant is classified as an INS/DEL SV in the cached fine map",
                    "min_p_sv_type": "INS or DEL for an SV; empty for SNVs/small indels",
                    "min_p_sv_length_bp": "Cached SV length in bp (zero for a non-SV, not the small-indel length)",
                    "lead_is_sv": "SV status of the fine-map lead used in the original locus/PIP plots",
                    "top_pip": "Maximum variant PIP; all-zero fits have no top_pip_variant",
                    "fine_mapped_region": "Outlier boundaries plus the configured flanks, GRCh38, 1-based inclusive",
                },
                "panels": {"A": "All discovery admixture PheWAS tests for the outlier region; OR = exp(discovery beta)",
                           "B": "Full marker-level EUR fractions from the corrected LAI cohort",
                           "C": "Discovery per-variant logistic score-test p-values in the fine-mapped window",
                           "bottom": "RefGene exon/intron tracks, longest transcript per gene"},
                "parameters": {"window_pad_bp": fm.WINDOW_PAD_BP, "png_dpi": 300,
                               "ancestry_sv_area_scale": ANCESTRY_SV_AREA_SCALE,
                               "figure_width_inches": FIGURE_WIDTH, "point_area_scale": POINT_AREA_SCALE,
                               "panel_heights_inches": PANEL_HEIGHTS,
                               "ld_reference": "minimum_score_p", "ld_scale": "horizontal_above_C",
                               "ld_label_alignment": "right_aligned_adjacent_to_colorbar",
                               "panel_letters": False, "or_legend_title": "OR fold (log scale)",
                               "phewas_tick_step": 1.0, "phewas_tick_format": "%.1f",
                               "title_and_subtitle": False,
                               "filename": "chr_chrom_outlier_start_outlier_end_lowercase_phecode_string",
                               "pdf_vectors": True, "pdf_fonttype": 42,
                               "seed": int(os.environ.get("AOU_FIGURE_SEED", "20260715")),
                               "threads": os.environ.get("OMP_NUM_THREADS", "4")}}
    corrected._atomic_json(manifest_path, manifest)
    try:
        names = [path.name for _, hit in hits.iterrows() for path in plot_paths(hit)]
        if len(names) != len(set(names)):
            raise ValueError("Region and phenotype descriptions produce duplicate filenames; refusing to overwrite distinct associations")
        scans = MarkerScans(ids)
        phewas = locus_phewas.DiscoveryPhewas(hits)
        manifest["phewas_source"] = phewas.source
        genes = fm._gene_table(force=False)
        gene_path = fm.gene_table_path()
        if not gene_path.is_file() or not {"chrom", "start", "end", "gene", "exons"}.issubset(genes):
            raise ValueError(f"Missing or invalid gene annotation cache: {gene_path}")
        gene_source = corrected._output_fingerprint(gene_path)
        manifest["gene_source"] = gene_source
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), completed_utc=corrected._utc_now())
        corrected._atomic_json(manifest_path, manifest)
        raise
    print(f"[figure2_sources][locus-figures] phase=plot; associations={len(hits)}; "
          f"LAI samples={len(ids)}; marker sources={scans.root}; output={output_dir()}", flush=True)
    rows, credible_members, failures, sv_ancestry_rows = [], [], [], []
    sv_ancestry_path = output_dir() / "sv_ancestry.tsv"
    sv_ancestry_columns = [*loci.KEYS, "region_id", "variant_id", "pos", "sv_type", "sv_length_bp", "mac",
                           "overall_eur_frac", "left_lai_marker", "right_lai_marker", "ancestry_method"]
    for _, hit in hits.iterrows():
        paths = plot_paths(hit)
        table = fm.analysis_dir() / f"susie_{fm.hit_tag(hit)}.tsv.gz"
        row = {key: hit[key] for key in [*loci.KEYS, "region_id"]}
        row.update(phecode_string=str(hit["phecode_string"]),
                   png=str(paths[0]), pdf=str(paths[1]), fine_map_table=str(table))
        try:
            frame = fm.load_current_cache(table, hit)
            if not loci._valid_frame(frame, hit, panel=panel):
                raise ValueError(f"Missing or invalid discovery fine map: {table}")
            row.update(association_results(frame, hit, genes))
            credible_members.extend(locus_results.member_rows(frame, hit, table))
            markers, source = scans.region(hit)
            sv_ancestry = _sv_positions(frame, markers)
            for key in [*loci.KEYS, "region_id"]:
                sv_ancestry[key] = hit[key]
            sv_ancestry_rows.extend(sv_ancestry.to_dict("records"))
            row["n_plotted_svs"] = len(sv_ancestry)
            region_phewas = phewas.region(hit)
            row.update(n_phewas_tests=len(region_phewas),
                       n_phewas_discovery_hits=int(region_phewas["discovery_significant"].sum()),
                       phewas_effect_basis=locus_phewas.EFFECT_BASIS,
                       phewas_source=phewas.source["discovery"]["path"])
            contract = {"plot_contract": PLOT_CONTRACT, "fine_map": corrected._output_fingerprint(table),
                        "phewas_source": phewas.source,
                        "marker_source": source, "gene_source": gene_source, "p_disc": float(hit["p_disc"]),
                        "phecode_string": str(hit.get("phecode_string", ""))}
            stage = loci._stage_name("two_panel", hit)
            if force or not corrected._stage_current(stage, contract, paths):
                render(frame, hit, markers, scans.stats, paths, genes=genes, lai_sample_n=scans.sample_n,
                       phewas=region_phewas, categories=phewas.categories, phewas_parameters=phewas.parameters)
                loci._validate_plots(paths)
                corrected._write_stage_manifest(stage, contract, paths)
            # Migrate only this hit's former combined-figure filenames, after
            # their replacements have been saved and verified successfully.
            legacy_paths = [output_dir() / original.name for original in loci.plot_paths(hit)]
            legacy_paths.extend(output_dir() / f"{_phenotype_stem(hit)}.{ext}" for ext in ("png", "pdf"))
            for legacy in legacy_paths:
                if legacy not in paths:
                    legacy.unlink(missing_ok=True)
            row.update(status="ready", n_markers=len(markers), lai_sample_n=scans.sample_n,
                       n_samples=int(frame["n_samples"].iloc[0]),
                       n_cases=int(frame["n_cases"].iloc[0]), error="")
            print(f"[figure2_sources][locus-figures] {fm.hit_tag(hit)}: markers={len(markers)}; "
                  f"saved/verified PNG+PDF={paths[0].parent / paths[0].stem}", flush=True)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures.append(f"{fm.hit_tag(hit)}: {exc}")
            print(f"[figure2_sources][locus-figures] FAILED {failures[-1]}", flush=True)
        rows.append(row)
        _write_results(rows, summary_path, results_path)
        locus_results.write_members(credible_members, members_csv, members_tsv)
        _write_sv_ancestry(pd.DataFrame(sv_ancestry_rows, columns=sv_ancestry_columns), sv_ancestry_path)
        manifest.update(completed_associations=len(rows) - len(failures), failures=failures,
                        n_credible_set_variants=len(credible_members),
                        duration_seconds=round(time.perf_counter() - started, 3))
        corrected._atomic_json(manifest_path, manifest)
    manifest.update(status="failed" if failures else "ready", completed_utc=corrected._utc_now(),
                    result_outputs=[corrected._output_fingerprint(path) for path in (summary_path, results_path, members_csv, members_tsv, sv_ancestry_path)])
    corrected._atomic_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"{len(failures)}/{len(hits)} locus figures failed; " + "; ".join(failures))
    print(f"[figure2_sources][locus-figures] results saved/verified: {results_path}", flush=True)
    print(f"[figure2_sources][locus-figures] credible-set variants={len(credible_members)}; "
          f"saved/verified: {members_csv}; {members_tsv}", flush=True)
    return {"directory": output_dir(), "summary": summary_path,
            "sv_ancestry_table": sv_ancestry_path,
            "credible_set_table": members_csv, "credible_set_tsv": members_tsv,
            "results_table": results_path, "n_associations": len(hits)}
