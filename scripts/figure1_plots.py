"""Individual cohort and demographic-history panels for Figure 1."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from . import figure1_data, figure1_demography
from .common import panel_io, config, geo, style

FIGURE_ID = "figure1"

# lightblue -> blue, matching the R usmap starter's fill scale.
_AOU_BLUES = LinearSegmentedColormap.from_list("aou_blues", ["#ADD8E6", "#0000CD"])


def panel_pca(ax: plt.Axes, data: dict) -> None:
    """PCA (PC1 vs PC2): AoU samples highlighted over ancestry_pred references.

    Non-AoU reference samples are drawn small and very low-opacity (dense
    background clouds), coloured by their (uppercased) ancestry_pred label; AoU
    samples are drawn larger, opaque, with a black outline so they stand out.
    """
    df = data["pca"]
    non_aou = df[~df["is_aou"]]
    aou = df[df["is_aou"]]

    labels = sorted(non_aou["label"].unique())
    for label in labels:
        sub = non_aou[non_aou["label"] == label]
        ax.scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.15, linewidths=0,
                   color=style.ancestry_pred_color(label))
    # AoU: small white diamonds with a black border, drawn on top.
    ax.scatter(aou["PC1"], aou["PC2"], s=12, marker="D",
               facecolors="white", edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Legend above the plot, with larger full-opacity proxy handles (the
    # plotted reference points are faint) so the swatches read clearly.
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                      markerfacecolor=style.ancestry_pred_color(l),
                      markeredgecolor="none", label=str(l)) for l in labels]
    handles.append(Line2D([0], [0], marker="D", linestyle="", markersize=8,
                          markerfacecolor="white", markeredgecolor="black", label="AOU"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, fontsize=11, frameon=False, handletextpad=0.3,
              columnspacing=0.9, borderaxespad=0.1)


def _admix_colors(data: dict) -> dict:
    """Map comp1/comp2 -> ancestry colour, anchored by the AFR component."""
    afr_comp = data.get("admixture_afr_comp", "comp1")
    eur_comp = "comp2" if afr_comp == "comp1" else "comp1"
    return {
        afr_comp: (style.ANCESTRY_COLORS["AFR"], "AFR"),
        eur_comp: (style.ANCESTRY_COLORS["EUR"], "EUR"),
    }


def _stacked_admixture(ax, df, colors, *, sort_desc="comp1", x0=0, label=True):
    """Draw a stacked k=2 admixture block: comp1 on the BOTTOM, comp2 on top."""
    order = df.sort_values(sort_desc, ascending=False).reset_index(drop=True)
    x = np.arange(len(order)) + x0
    bottom = np.zeros(len(order))
    for comp in ("comp1", "comp2"):  # comp1 first -> bottom
        color, name = colors[comp]
        ax.bar(x, order[comp].to_numpy(), width=1.0, bottom=bottom, color=color,
               label=name if label else None)
        bottom = bottom + order[comp].to_numpy()
    return x[-1] + 1 if len(x) else x0


def panel_admixture(ax: plt.Axes, data: dict) -> None:
    """AoU-only ADMIXTURE (k=2), with complementary component medians."""
    colors = _admix_colors(data)
    df = data["admixture"]
    _stacked_admixture(ax, df, colors, sort_desc="comp1", x0=0)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("individuals")
    ax.set_ylabel("ancestry fraction")
    # Because k=2 components sum to one, a single horizontal line at the
    # bottom component's median identifies both complementary medians.
    bottom_median = float(df["comp1"].median())
    ax.axhline(bottom_median, color="black", lw=1.8, ls=(0, (5, 4)),
               alpha=0.95, zorder=5)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=11, borderaxespad=0.1)


def plot_admixture_with_reference(data: dict, *, clean_run_plots: bool) -> list:
    """Standalone AoU + 1kg AFR/EUR admixture plot (grouped), saved to figures/.

    Groups left->right as 1kg AFR | AoU | 1kg EUR, each sorted by component 1,
    with separators and group labels. Returns the written paths.
    """
    number = config.FIGURES[FIGURE_ID]["number"]
    # ADMIXTURE is panel C, so the reference variant is Figure_<N>C_AoU_plus_1kg.
    # Always redrawn (like the panels) so it never drifts from the current data.
    stem = config.figure_dir(FIGURE_ID) / f"Figure_{number}C_AoU_plus_1kg"

    colors = _admix_colors(data)
    ref = data["admixture_ref"]
    blocks = [
        ("1kg AFR", ref[ref["group"] == "AFR"]),
        ("AoU", ref[ref["cohort"] == "AoU"]),
        ("1kg EUR", ref[ref["group"] == "EUR"]),
    ]
    style.apply_style()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x0, ticks, seps = 0, [], []
    for name, block in blocks:
        if not len(block):
            continue
        start = x0
        x0 = _stacked_admixture(ax, block, colors, sort_desc="comp1", x0=x0, label=False)
        ticks.append(((start + x0) / 2, name))
        seps.append(x0)
    for s in seps[:-1]:
        ax.axvline(s - 0.5, color="white", lw=1.5)
    ax.set_xlim(0, x0 - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ancestry fraction")
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c][0]) for c in ("comp1", "comp2")]
    ax.legend(handles, [colors[c][1] for c in ("comp1", "comp2")], ncol=2, loc="upper center")
    ax.set_title(f"Figure {number}C (AoU + 1000G reference)")
    out = []
    for ext in config.FIGURE_FORMATS:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def panel_map(ax: plt.Axes, data: dict) -> None:
    """Choropleth of AoU sample counts per US state (Alaska skipped).

    Counts are disclosure-safe (AoU rule: <20 rounded up to 20; see
    figure1_data). States with no sampled participants render white.
    """
    states = geo.load_us_states(exclude=("AK", "HI"))  # skip Alaska (+ Hawaii for extent)
    # Disclosure: floor any 1-19 count to 20 for display (raw kept on disk).
    counts = figure1_data.display_state_counts(data["state_counts"])
    merged = states.merge(counts, on="state", how="left")
    # Start the colour scale + legend at the disclosure floor (20), so the
    # lowest tick is 20 rather than an auto value like 50.
    vmin = figure1_data.AOU_MIN_COUNT
    vmax = max(int(counts["n"].max()), vmin)
    step = 40
    ticks = list(range(vmin, vmax + 1, step)) or [vmin]
    merged.plot(
        column="n",
        ax=ax,
        cmap=_AOU_BLUES,
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        missing_kwds={"color": "white", "edgecolor": "black", "linewidth": 0.3},
    )
    # A compact colour bar above the map leaves the panel itself full-width.
    # Keeping the legend attached to the panel also preserves it in the
    # standalone Figure_1A output.
    cax = ax.inset_axes([0.18, 1.05, 0.68, 0.035])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=_AOU_BLUES)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("count", fontsize=11, labelpad=2)
    cbar.ax.tick_params(labelsize=10, length=3, pad=1)
    cbar.outline.set_linewidth(0.5)
    ax.set_axis_off()
    # Equal aspect for a projected (Albers) CRS; approximate the mid-US
    # latitude correction when we fell back to lon/lat degrees.
    if states.crs is not None and states.crs.is_geographic:
        ax.set_aspect(1.0 / math.cos(math.radians(39.5)))
    else:
        ax.set_aspect("equal")


def _place_labels(sources, sample_xy, extent):
    """Place source-population codes in deterministic low-density positions."""
    lon0, lon1, lat0, lat1 = extent
    lon_span, lat_span = lon1 - lon0, lat1 - lat0
    sample_lon = np.asarray(sample_xy[:, 0])
    sample_lat = np.asarray(sample_xy[:, 1])
    crowd_radius_squared = 0.055**2
    label_radius = 0.115
    angles = np.deg2rad(np.arange(0, 360, 15))
    placed, output = [], []
    for lon, lat, code in sources:
        best, best_score = None, None
        for multiplier in (1.4, 2.0, 2.8, 3.6, 4.4):
            radius = 0.045 * lat_span * multiplier
            for angle in angles:
                candidate_lon = lon + radius * math.cos(angle) * 1.2
                candidate_lat = lat + radius * math.sin(angle)
                if not (
                    lon0 + 1.5 < candidate_lon < lon1 - 1.5
                    and lat0 + 1.5 < candidate_lat < lat1 - 1.5
                ):
                    continue
                crowd = int(
                    np.count_nonzero(
                        ((sample_lon - candidate_lon) / lon_span) ** 2
                        + ((sample_lat - candidate_lat) / lat_span) ** 2
                        < crowd_radius_squared
                    )
                )
                label_penalty = sum(
                    22
                    for placed_lon, placed_lat in placed
                    if ((placed_lon - candidate_lon) / lon_span) ** 2
                    + ((placed_lat - candidate_lat) / lat_span) ** 2
                    < label_radius**2
                )
                score = crowd * 6 + label_penalty + multiplier * 0.5
                if best_score is None or score < best_score:
                    best_score = score
                    best = (candidate_lon, candidate_lat)
        if best is None:
            best = (lon, lat + 0.05 * lat_span)
        placed.append(best)
        output.append((best[0], best[1], code, lon, lat))
    return output


def panel_idw(ax: plt.Axes, data: dict) -> None:
    """Phase 1 IDW origins: AoU diamonds over hollow 1KG AFR/EUR anchors."""
    import matplotlib.patheffects as path_effects

    longitude_min, longitude_max, latitude_min, latitude_max = config.AFR_EUR_EXTENT
    sources = data["idw_sources"]
    samples_frame = data["idw_samples"]
    regions = set(sources["region"].astype(str).str.upper())
    if regions != {"AFR", "EUR"}:
        raise ValueError(f"Figure 1D source regions must be exactly AFR/EUR; found {regions}")

    ax.set_facecolor("white")
    geo.load_world().plot(
        ax=ax,
        facecolor="white",
        edgecolor="#999FA5",
        linewidth=0.5,
        antialiased=True,
        zorder=1,
    )
    source_points = ax.scatter(
        sources["lon"],
        sources["lat"],
        s=100,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        zorder=6,
        label="Source pops (1KG)",
    )
    aou_points = ax.scatter(
        samples_frame["pred_longitude"],
        samples_frame["pred_latitude"],
        s=22,
        marker="D",
        facecolors="white",
        edgecolors="black",
        linewidths=0.65,
        alpha=0.82,
        zorder=7,
        label="AoU IDW predictions",
    )

    sample_xy = samples_frame[["pred_longitude", "pred_latitude"]].to_numpy(float)
    labels = list(zip(sources["lon"], sources["lat"], sources["pop_code"]))
    for lon, lat, code, anchor_lon, anchor_lat in _place_labels(
        labels,
        sample_xy,
        (longitude_min, longitude_max, latitude_min, latitude_max),
    ):
        ax.plot(
            [anchor_lon, lon],
            [anchor_lat, lat],
            color="black",
            linewidth=0.6,
            alpha=0.55,
            zorder=8,
        )
        ax.text(
            lon,
            lat,
            code,
            fontsize=11,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=9,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
        )

    ax.set_xlim(longitude_min, longitude_max)
    ax.set_ylim(latitude_min, latitude_max)
    ax.set_aspect(1.0 / math.cos(math.radians(20)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        [aou_points, source_points],
        ["AoU IDW predictions", "Source pops (1KG)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=9.5,
        framealpha=0.93,
        columnspacing=1.1,
        handletextpad=0.45,
    )


def panel_model(ax: plt.Axes, data: dict) -> None:
    """Best-fit demographic model (dadi 3D-SFS fit): labelled schematic.

    Renders the manuscript schematic (AFR/EUR/AoU split-admixture model with
    fitted sizes, times, migration rates and admixture fractions + CIs). The
    parameters behind the labels come from the committed demes fit
    (``figure1_demography.best_fit_physical``); the fit itself is not re-run.
    """
    # Draw into a child axes so DemesDraw's y label and size-scale inset remain
    # inside panel E instead of extending into the neighboring F/G row. The
    # host axes remains the stable target for the composite's panel letter.
    model_ax = ax.inset_axes([0.075, 0.23, 0.915, 0.73])
    figure1_demography.render_demographic_model_schematic(ax=model_ax)
    ax.set_axis_off()


def panel_sfs(ax: plt.Axes, data: dict) -> None:
    """Fig 1F: AoU-marginal SFS -- observed vs. C2 best fit + residuals.

    The 3D data spectrum and the best-fit model are marginalized to the AoU
    population; the top strip overlays the two SFS (log y), the bottom strip
    shows Poisson residuals ``(model - data)/sqrt(model)``. Monomorphic bins
    (0 and 2N) are dropped.
    """
    fit = data["sfs_aou_fit"]
    kmax = int(fit["allele_count"].max())
    sub = fit[(fit["allele_count"] >= 1) & (fit["allele_count"] <= kmax - 1)]
    x = sub["allele_count"].to_numpy(dtype=float) / kmax

    # Use the host cell purely as a container for two stacked sub-axes.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    top = ax.inset_axes([0, 0.30, 1, 0.70])
    bot = ax.inset_axes([0, 0, 1, 0.22], sharex=top)

    top.plot(x, sub["data"], marker="o", ms=3, lw=1.0, color="black", label="data")
    top.plot(
        x,
        sub["model"],
        lw=1.6,
        color=style.PALETTE["vermillion"],
        label="model",
    )
    top.set_yscale("log")
    top.set_ylabel("Number of SNPs", labelpad=0)
    top.legend(
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    top.tick_params(labelbottom=False)

    bot.axhline(0, color="0.6", lw=0.8)
    bot.bar(
        x,
        sub["poisson_resid"],
        color=style.PALETTE["blue"],
        width=0.8 / kmax,
    )
    bot.set_ylabel("resid.", fontsize=11)
    bot.set_xlabel("Allele Frequency")


def _boot_scatter(ax, x, y, x_best, y_best, *, cloud_alpha=0.25):
    """Bootstrap cloud (black, low alpha) with the red best-fit dot on top."""
    ax.scatter(
        x,
        y,
        s=18,
        color="black",
        alpha=cloud_alpha,
        linewidths=0,
        zorder=2,
    )
    ax.scatter([x_best], [y_best], s=36, color="red", zorder=5, label="best fit")


def panel_boot_size(ax: plt.Axes, data: dict) -> None:
    """Bootstrap (Fig 1G): initial vs. final EUR size (physical units).

    Each point is one of the 100 bootstrap dadi fits; the red dot is the best fit.
    """
    df = data["bootstrap"]
    bf = figure1_demography.best_fit_point()
    _boot_scatter(
        ax,
        df["N_EUR_initial"],
        df["N_EUR_final"],
        bf["N_EUR_initial"],
        bf["N_EUR_final"],
        cloud_alpha=0.45,
    )
    ax.set_xlabel("Initial European Size")
    ax.set_ylabel("Final European Size")


def panel_boot_admix(ax: plt.Axes, data: dict) -> None:
    """Bootstrap (Fig 1H): AFR admixture fraction vs. final EUR size (physical)."""
    df = data["bootstrap"]
    bf = figure1_demography.best_fit_point()
    _boot_scatter(ax, df["f_AFR"], df["N_EUR_final"], bf["f_AFR"], bf["N_EUR_final"])
    ax.set_xlabel("Admixture Fraction")
    ax.set_ylabel("Final European Size")
    ax.legend()


# Panel letters -> content. A=sampling map, B=PCA, C=ADMIXTURE, D=IDW geographic
# origins; E-H are the demographic-inference panels (E spans two columns in the
# composite; F-H occupy the final row in left-to-right reading order).
PANELS = {
    "A": panel_map, "B": panel_pca, "C": panel_admixture, "D": panel_idw,
    "E": panel_model, "F": panel_sfs, "G": panel_boot_size, "H": panel_boot_admix,
}


# --- Bootstrap correlogram (QC / debug) ---------------------------------------
# Interpretable physical bootstrap parameters, in display order, with labels.
_CORRELOGRAM_PARAMS = [
    ("N_AFR_postsplit", "AFR size"),
    ("N_EUR_initial", "EUR initial size"),
    ("N_EUR_final", "EUR final size"),
    ("f_AFR", "AFR admix. frac."),
    ("T_split_years", "AFR/EUR split (yr)"),
    ("T_ancestral_years", "anc. growth (yr)"),
    ("m_EUR_to_AFR", "mig EUR→AFR"),
    ("m_AFR_to_EUR", "mig AFR→EUR"),
]

# S2 is an eight-by-eight matrix that is normally reduced to letter-page width.
# These deliberately large native sizes remain legible after that reduction.
_CORRELOGRAM_CELL_INCHES = 2.25
_CORRELOGRAM_AXIS_LABEL_SIZE = 22
_CORRELOGRAM_VALUE_SIZE = 21
_CORRELOGRAM_COLORBAR_SIZE = 20
_CORRELOGRAM_TITLE_SIZE = 26


def qc_image_paths(ext: str = "png") -> list:
    """Path(s) to the Figure 1 QC images (currently the bootstrap correlogram)."""
    return [config.figure_dir(FIGURE_ID) / "qc" / f"bootstrap_correlogram.{ext}"]


def s2_image_paths(ext: str = "png") -> list:
    """Publication paths for Supplementary Figure S2."""
    stem = (
        config.figure_dir(FIGURE_ID)
        / "bootstrap"
        / "Figure_S2_Bootstrap_Correlogram"
    )
    return [stem.with_suffix(f".{ext}")]


def plot_bootstrap_correlogram(data: dict, *, supplementary: bool = False) -> list:
    """Corner/pairs plot of the physical bootstrap parameters (100 dadi fits).

    A debug/QC view of parameter identifiability. Lower triangle: the actual
    scatter for each parameter pair (black cloud + red best fit) so trade-offs
    are visible directly. Upper triangle: the Pearson r for that pair, coloured.
    Diagonal: each parameter's bootstrap histogram with the best fit marked.
    By default, written to ``figures/figure1/qc/``. With
    ``supplementary=True``, written to the publication-facing S2 directory.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    df = data["bootstrap"]
    params = [(c, lab) for c, lab in _CORRELOGRAM_PARAMS if c in df.columns]
    cols = [c for c, _ in params]
    labels = [lab for _, lab in params]
    n = len(cols)
    corr = df[cols].corr().to_numpy()
    bf = figure1_demography.best_fit_point()

    norm = Normalize(-1, 1)
    cmap = plt.get_cmap("RdBu_r")
    style.apply_style()
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(_CORRELOGRAM_CELL_INCHES * n, _CORRELOGRAM_CELL_INCHES * n),
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:  # diagonal: bootstrap histogram + best fit
                ax.hist(
                    df[cols[i]], bins=18, color="0.65", edgecolor="white",
                    linewidth=0.5,
                )
                if cols[i] in bf:
                    ax.axvline(bf[cols[i]], color="red", lw=2.6)
            elif i > j:  # lower triangle: the actual scatter (x=col j, y=col i)
                ax.scatter(
                    df[cols[j]], df[cols[i]], s=22, color="black",
                    alpha=0.28, linewidths=0,
                )
                if cols[j] in bf and cols[i] in bf:
                    ax.scatter(
                        [bf[cols[j]]], [bf[cols[i]]], s=72, color="red",
                        zorder=5,
                    )
            else:  # upper triangle: coloured Pearson r
                r = corr[i, j]
                ax.set_facecolor(cmap(norm(r)))
                ax.text(0.5, 0.5, f"{r:.2f}", transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=_CORRELOGRAM_VALUE_SIZE + abs(r) * 7,
                        color="white" if abs(r) > 0.5 else "black")
            for spine in ax.spines.values():
                spine.set_edgecolor("0.85")
            # Parameter names only on the outer edges (corner-plot convention).
            if j == 0:
                ax.set_ylabel(
                    labels[i], fontsize=_CORRELOGRAM_AXIS_LABEL_SIZE,
                    rotation=45, ha="right", va="center", labelpad=8,
                )
            if i == n - 1:
                ax.set_xlabel(
                    labels[j], fontsize=_CORRELOGRAM_AXIS_LABEL_SIZE,
                    rotation=45, ha="right", labelpad=8,
                )

    # Reserve title space before creating the colorbar; adjusting subplots after
    # the colorbar would move the matrix back underneath it.
    fig.subplots_adjust(top=0.94)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label(
        "Pearson r", fontsize=_CORRELOGRAM_COLORBAR_SIZE, labelpad=14,
    )
    cbar.ax.tick_params(labelsize=_CORRELOGRAM_COLORBAR_SIZE - 2)
    fig.suptitle(
        f"Bootstrap parameter correlogram (n={len(df)} fits)",
        fontsize=_CORRELOGRAM_TITLE_SIZE,
    )
    out = []
    paths = s2_image_paths if supplementary else qc_image_paths
    for e in config.FIGURE_FORMATS:
        p = paths(e)[0]
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def render_qc(clean_run_data: bool = False, clean_run_plots: bool = False) -> list:
    """Build Fig 1 data and (re)draw the QC plots; return the image paths."""
    data = figure1_data.build(clean_run_data=clean_run_data)
    plot_bootstrap_correlogram(data)
    return qc_image_paths()


def render_s2(clean_run_data: bool = False, clean_run_plots: bool = False) -> list:
    """Build committed bootstrap data and render publication Supplementary S2."""
    del clean_run_data, clean_run_plots  # kept for notebook-runner compatibility
    bootstrap = figure1_demography.load_bootstrap_fits()
    if bootstrap is None:
        raise FileNotFoundError(
            "The committed demographic bootstrap-fit archive is unavailable."
        )
    plot_bootstrap_correlogram({"bootstrap": bootstrap}, supplementary=True)
    return [
        path
        for extension in config.FIGURE_FORMATS
        for path in s2_image_paths(extension)
    ]


def _idw_render_provenance(data: dict) -> dict[str, object]:
    """Audit the Figure 1D inputs against the validated IDW cache."""
    from . import figure1_idw

    samples_frame = data["idw_samples"]
    sources_frame = data["idw_sources"]
    cache_valid, cache_reason = figure1_idw.cache_status()
    if not cache_valid:
        raise RuntimeError(f"Figure 1D requires a validated IDW cache: {cache_reason}")
    source = "checksum- and sample-manifest-validated IDW cache"
    geography_summary = figure1_idw.european_continent_summary(
        samples_frame,
        write_cache=bool(
            cache_valid and figure1_idw.IDW_SAMPLES_CSV.is_file()
        ),
    )
    return {
        "source": source,
        "sample_count": len(samples_frame),
        "source_location_count": len(sources_frame),
        "europe_count": geography_summary["n_europe"],
        "europe_percent": geography_summary["percent_europe"],
        "cache_valid": cache_valid,
        "cache_reason": cache_reason,
    }


def build_figure1(clean_run_data: bool = False, clean_run_plots: bool = False) -> list:
    """End-to-end: build data, render individual panels and return their paths.

    Also emits the standalone AoU + 1000G reference admixture plot.
    """
    # A clean data run invalidates the plots that depend on it.
    clean_run_plots = clean_run_plots or clean_run_data
    data = figure1_data.build(clean_run_data=clean_run_data)
    idw_provenance = _idw_render_provenance(data)
    print(
        "[figure1] Figure 1D render input: "
        f"{idw_provenance['source']}; "
        f"samples={idw_provenance['sample_count']}; "
        f"source_locations={idw_provenance['source_location_count']}; "
        f"Europe_continent={idw_provenance['europe_count']} "
        f"({idw_provenance['europe_percent']:.2f}%)"
    )
    plot_admixture_with_reference(data, clean_run_plots=clean_run_plots)
    return panel_io.render_panels(FIGURE_ID, PANELS, data, clean_run_plots=clean_run_plots)
