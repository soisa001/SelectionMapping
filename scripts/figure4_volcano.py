"""Figure 4A discovery volcano with meta-significant associations outlined."""
from __future__ import annotations

import re
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import figure4_meta
from .common import config, style

FIGURE_ID = "figure4"
SEED = 3

# Nominal-significance threshold (p <= 1e-3).
P_THRESH = 1e-3

# Volcano geometry / marker knobs (from the notebook volcano cell).
VOLCANO_XLIM = 10.0
VOLCANO_YLIM = 12.0
ALPHA_HIT = 0.85
ALPHA_BG = 0.35

# Marker area scales with |effect size|, on the same log scale and caps as the
# per-variant PheWAS Manhattan plots (notebook ``CappedPlot``): the area for a
# test with log-odds |beta| is SIZE_FLOOR + SIZE_SCALE * ln(min(|beta|, BETA_CAP)
# + 1). Keeping BETA_CAP/SIZE_FLOOR/SIZE_SCALE identical makes the volcano and
# Manhattan markers directly comparable.
BETA_CAP = 5.0
SIZE_FLOOR = 40.0
SIZE_SCALE = 120.0

POS_COLOR = "indianred"   # positive log-OR hits
NEG_COLOR = "darkcyan"    # negative log-OR hits
BG_COLOR = "0.7"          # non-significant tests (gray)

VOLCANO_FONT = 22
VOLCANO_LABEL = 16
FONT_SCALE = 1.35
# Panel shrunk 25% from the original (14, 11).
VOLCANO_FIGSIZE = (10.5, 8.25)
N_LABEL_PER_SIDE = 4  # top hits (by p) to annotate on each effect-size side
MAX_BACKGROUND_POINTS = 10_000
BACKGROUND_SAMPLE_SEED = SEED
EXPECTED_GENIC_SENTINELS = 963
EXPECTED_NOMINAL_HITS = 446

# Canonical (variant, phecode) columns as written by the PheWAS notebook.
_VARIANT_COLS = ("variant_tag", "snp", "SNP", "variant_id", "tag")
_PHECODE_COLS = ("phecode", "Phecode", "phenotype")
_PHESTR_COLS = ("phecode_string", "phecode_description", "description", "meaning")
_BETA_COLS = ("beta", "Beta", "BETA", "b", "log_odds", "beta_ind")
_P_COLS = ("p_value", "p", "P", "pval", "pvalue", "p_ind")


def _col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


# --- Data loading -------------------------------------------------------------


def load_phewas_results(clean_run_data: bool = False) -> pd.DataFrame:
    """Load the filtered PheWAS frame and retain the 963 genic sentinels."""
    path = figure4_meta._stage_gcs_file(
        config.PHEWAS_FILTERED_TSV,
        config.PHEWAS_FILTERED_GS,
        force=clean_run_data,
    )
    if Path(path).exists():
        sentinel_path = figure4_meta._stage_gcs_file(
            config.PHEWAS_GENIC_SENTINELS_TSV,
            config.PHEWAS_GENIC_SENTINELS_GS,
            force=clean_run_data,
        )
        if not Path(sentinel_path).is_file():
            raise FileNotFoundError(
                "[figure4] the real volcano source requires the 963-genic "
                f"sentinel table at {sentinel_path}; set "
                "AOU_PHEWAS_GENIC_SENTINELS_TSV or AOU_S9_SENTINELS_TSV"
            )
        df = pd.read_csv(path, sep="\t", dtype={"phecode": str}, low_memory=False)
        sentinels = pd.read_csv(sentinel_path, sep="\t", low_memory=False)
        return _filter_to_genic_sentinels(_normalize(df), sentinels)
    raise FileNotFoundError(f"Filtered PheWAS results missing: {path}")


def _filter_to_genic_sentinels(
    phewas: pd.DataFrame,
    sentinels: pd.DataFrame,
    *,
    expected_genic_sentinels: int = EXPECTED_GENIC_SENTINELS,
    expected_nominal_hits: int = EXPECTED_NOMINAL_HITS,
) -> pd.DataFrame:
    """Restrict the full PheWAS frame to S9's genic sentinel contract."""
    # Panels A and B use the same sentinel schema.
    from . import figure4_discovery as figure_s9_phewas

    sentinel_contract = figure_s9_phewas._sentinel_contract(sentinels)
    genic_tags = set(
        sentinel_contract.loc[sentinel_contract["is_genic"], "variant_tag"]
        .astype(str)
        .str.strip()
    )
    if len(genic_tags) != expected_genic_sentinels:
        raise ValueError(
            "[figure4] expected "
            f"{expected_genic_sentinels} unique genic sentinels but found "
            f"{len(genic_tags)}"
        )

    input_rows = len(phewas)
    input_hits = int(phewas["p_value"].le(P_THRESH).sum())
    normalized_tags = phewas["variant_tag"].astype(str).str.strip()
    filtered = phewas.loc[normalized_tags.isin(genic_tags)].copy().reset_index(drop=True)
    represented_tags = int(filtered["variant_tag"].nunique())
    filtered_hits = int(filtered["p_value"].le(P_THRESH).sum())
    print(
        "[figure4-volcano] genic filter: "
        f"sentinels={len(genic_tags)}, represented_sentinels={represented_tags}, "
        f"input_rows={input_rows}, output_rows={len(filtered)}, "
        f"input_p_le_1e-3={input_hits}, output_p_le_1e-3={filtered_hits}"
    )
    if filtered.empty:
        raise ValueError(
            "[figure4] no PheWAS rows matched the 963-genic sentinel variant tags"
        )
    if filtered_hits != expected_nominal_hits:
        raise ValueError(
            "[figure4] expected "
            f"{expected_nominal_hits} genic PheWAS associations at p <= "
            f"{P_THRESH:g} but found {filtered_hits}"
        )
    return filtered


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with canonical beta / p_value / variant_tag / phecode."""
    beta = _col(df, _BETA_COLS)
    p = _col(df, _P_COLS)
    var = _col(df, _VARIANT_COLS)
    phe = _col(df, _PHECODE_COLS)
    phestr = _col(df, _PHESTR_COLS)
    if beta is None or p is None:
        raise ValueError(f"[figure4] volcano needs beta + p columns; "
                         f"have {list(df.columns)}")
    out = pd.DataFrame({
        "beta": pd.to_numeric(df[beta], errors="coerce"),
        "p_value": pd.to_numeric(df[p], errors="coerce"),
        "variant_tag": df[var].astype(str) if var else "",
        "phecode": df[phe].astype(str) if phe else "",
        "phecode_string": df[phestr].astype(str) if phestr else "",
    })
    out = out.dropna(subset=["beta", "p_value"])
    out = out[(out["p_value"] > 0) & (out["p_value"] <= 1)].reset_index(drop=True)
    out["neg_log_p"] = -np.log10(out["p_value"].clip(lower=1e-300))
    return out


def _meta_sig_keys(clean_run_data: bool = False) -> set[tuple[str, str]]:
    """Set of meta-significant (variant_tag, phecode) pairs for meta-significance outlines."""
    meta = figure4_meta.load_meta_results(clean_run_data=clean_run_data)
    var = _col(meta, _VARIANT_COLS)
    phe = _col(meta, _PHECODE_COLS)
    if var is None or phe is None:
        return set()
    if "meta_significant" in meta.columns:
        keep = meta["meta_significant"].astype(str).str.upper().isin(["TRUE", "1"])
        meta = meta[keep]
    return set(zip(meta[var].astype(str), meta[phe].astype(str)))


# --- Rendering ----------------------------------------------------------------
def _size_from_beta(beta: np.ndarray) -> np.ndarray:
    """Marker area from |effect size|, log-scaled and capped like the Manhattans.

    Matches the notebook ``CappedPlot``: area = SIZE_FLOOR + SIZE_SCALE *
    ln(min(|beta|, BETA_CAP) + 1), so a large log-odds saturates at the same
    marker size used in the per-variant PheWAS Manhattan panels.
    """
    b = np.clip(np.abs(np.asarray(beta, dtype=float)), 0, BETA_CAP)
    return SIZE_FLOOR + SIZE_SCALE * np.log(b + 1.0)


def _thin_volcano_background(
    frame: pd.DataFrame,
    *,
    p_thresh: float,
    meta_keys: set[tuple[str, str]] | None,
    max_background_points: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Deterministically cap ordinary non-significant points.

    All nominally significant and meta-highlighted rows are retained. The cap
    applies only to the remaining background cloud.
    """
    if max_background_points < 0:
        raise ValueError("max_background_points must be non-negative")

    is_hit = frame["p_value"].to_numpy() <= p_thresh
    if meta_keys:
        keys = zip(
            frame["variant_tag"].to_numpy(),
            frame["phecode"].to_numpy(),
        )
        is_meta = np.fromiter(
            (key in meta_keys for key in keys),
            dtype=bool,
            count=len(frame),
        )
    else:
        is_meta = np.zeros(len(frame), dtype=bool)

    ordinary_background = np.flatnonzero(~is_hit & ~is_meta)
    retained_background = ordinary_background
    if len(ordinary_background) > max_background_points:
        rng = np.random.default_rng(seed)
        retained_background = np.sort(
            rng.choice(
                ordinary_background,
                size=max_background_points,
                replace=False,
            )
        )

    keep = is_hit | is_meta
    keep[retained_background] = True
    thinned = frame.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
    stats = {
        "input_points": len(frame),
        "significant_points": int(is_hit.sum()),
        "meta_points": int(is_meta.sum()),
        "ordinary_background_input": len(ordinary_background),
        "ordinary_background_plotted": len(retained_background),
        "output_points": len(thinned),
        "seed": seed,
        "p_threshold": p_thresh,
        "background_cap": max_background_points,
    }
    return thinned, stats


def plot_volcano(
    df: pd.DataFrame,
    *,
    circle_meta: bool = False,
    meta_keys: set[tuple[str, str]] | None = None,
    p_thresh: float = P_THRESH,
    shorten_labels: bool = False,
    x_limit: float = VOLCANO_XLIM,
    y_limit: float = VOLCANO_YLIM,
    beta_labels: bool = False,
    legend_above: bool = False,
    font_scale: float = 1.0,
    legend_frame: bool = True,
    label_boxes: bool = True,
    inline_beta_legend: bool = False,
    max_background_points: int = MAX_BACKGROUND_POINTS,
    background_sample_seed: int = BACKGROUND_SAMPLE_SEED,
):
    """Render the PheWAS volcano (Figure 4A). Returns the Figure.

    Non-significant tests are gray; nominally significant hits are coloured by
    effect direction. With ``circle_meta`` the meta-significant hits in
    ``meta_keys`` are outlined with a bold black open circle.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    if not np.isfinite(font_scale) or font_scale <= 0:
        raise ValueError("font_scale must be a finite positive number")
    axis_font = VOLCANO_FONT * font_scale
    label_font = VOLCANO_LABEL * font_scale
    neg_log_thresh = -np.log10(p_thresh)

    res = df[df["beta"].abs() < x_limit].copy()
    res, thinning = _thin_volcano_background(
        res,
        p_thresh=p_thresh,
        meta_keys=meta_keys,
        max_background_points=max_background_points,
        seed=background_sample_seed,
    )
    print(
        "[figure4-volcano] thinning: "
        + ", ".join(f"{key}={value}" for key, value in thinning.items())
    )
    beta = res["beta"].to_numpy()
    negp = res["neg_log_p"].to_numpy()
    negp_y = np.minimum(negp, y_limit)
    sizes = _size_from_beta(beta)   # marker area from |effect size|
    is_hit = res["p_value"].to_numpy() <= p_thresh
    is_pos = beta >= 0

    colors = np.where(is_pos, POS_COLOR, NEG_COLOR)
    colors[~is_hit] = BG_COLOR   # non-significant -> gray

    fig, ax = plt.subplots(figsize=VOLCANO_FIGSIZE)

    # Background (non-significant) first, then the coloured hits on top.
    ax.scatter(beta[~is_hit], negp_y[~is_hit], s=sizes[~is_hit],
               c=colors[~is_hit], alpha=ALPHA_BG, edgecolors="none", linewidths=0)
    ax.scatter(beta[is_hit], negp_y[is_hit], s=sizes[is_hit],
               c=colors[is_hit], alpha=ALPHA_HIT, edgecolors="none", linewidths=0)

    # outline meta-significant (variant x phecode) hits.
    if circle_meta and meta_keys:
        keys = list(zip(res["variant_tag"].to_numpy(), res["phecode"].to_numpy()))
        is_meta = np.array([k in meta_keys for k in keys])
        if is_meta.any():
            ax.scatter(beta[is_meta], negp_y[is_meta],
                       s=sizes[is_meta] * 1.6, facecolors="none",
                       edgecolors="black", linewidths=2.2, zorder=6)

    ax.axhline(neg_log_thresh, ls="--", color="gray", lw=1.2, zorder=1)

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.2, y_limit)
    ax.set_xlabel(
        r"$\beta$" if beta_labels else r"$\log$(OR)",
        fontsize=axis_font,
    )
    ax.set_ylabel(r"$-\log_{10}$(p-value)", fontsize=axis_font)
    ax.tick_params(labelsize=axis_font)

    # Label the top hits by p-value, taken separately for each effect-size
    # direction (top N with negative log-OR + top N with positive log-OR).
    hits_df = res[is_hit].copy()
    hits_df = hits_df[hits_df["phecode_string"].astype(str).str.len() > 0]
    neg_top = hits_df[hits_df["beta"] < 0].nsmallest(N_LABEL_PER_SIDE, "p_value")
    pos_top = hits_df[hits_df["beta"] >= 0].nsmallest(N_LABEL_PER_SIDE, "p_value")
    top = pd.concat([neg_top, pos_top])
    if legend_above:
        _annotate_upper_lanes(
            ax,
            top,
            y_limit=y_limit,
            fontsize=label_font,
            shorten_labels=shorten_labels,
        )
    else:
        texts = []
        for _, r in top.iterrows():
            col = POS_COLOR if r["beta"] >= 0 else NEG_COLOR
            texts.append(ax.text(
                r["beta"], min(r["neg_log_p"], y_limit),
                (
                    _display_phecode_name(r["phecode_string"])
                    if shorten_labels
                    else str(r["phecode_string"])
                ),
                fontsize=label_font,
                color=col,
                ha="center",
                va="center",
                bbox=(
                    dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7)
                    if label_boxes
                    else None
                ),
            ))
    if not legend_above and texts:
        try:
            import adjustText
            adjustText.adjust_text(
                texts, ax=ax,
                expand=(1.8, 2.2), force_text=(0.8, 1.2), force_static=(0.6, 1.0),
                force_pull=(0.02, 0.02),
                arrowprops=dict(arrowstyle="-", color="gray", lw=1.4, alpha=0.85),
                max_move=80, iter_lim=2000,
            )
        except ImportError:
            warnings.warn(
                "[figure4] adjustText not installed; volcano labels are not "
                "de-collided (pip install adjustText).", stacklevel=2)

    _add_size_legend(
        ax,
        circle_meta=circle_meta,
        beta_labels=beta_labels,
        legend_above=legend_above,
        font_scale=font_scale,
        frameon=legend_frame,
        inline_beta_label=inline_beta_legend,
    )
    if legend_above:
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.11, top=0.82)
    else:
        fig.subplots_adjust(left=0.09, right=0.82, bottom=0.09, top=0.97)
    return fig


def _annotate_upper_lanes(
    ax,
    hits: pd.DataFrame,
    *,
    y_limit: float,
    fontsize: float,
    shorten_labels: bool,
) -> list[object]:
    """Place hit labels in deterministic, non-crossing upper-side lanes."""
    from matplotlib.text import Text

    annotations = []
    placed_boxes = []
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    top_margin = 0.035 * axes_box.height
    vertical_gap = 0.018 * axes_box.height
    top_clipped_targets = 0
    for positive, x_position, alignment, line_origin in (
        (False, 0.015, "left", (1.0, 0.5)),
        (True, 0.985, "right", (0.0, 0.5)),
    ):
        side = hits.loc[(hits["beta"] >= 0).eq(positive)].sort_values(
            ["neg_log_p", "beta"], ascending=[False, positive]
        )
        next_top = axes_box.y1 - top_margin
        for _, row in side.iterrows():
            raw_label = (
                _display_phecode_name(row["phecode_string"])
                if shorten_labels
                else str(row["phecode_string"])
            )
            label = textwrap.fill(
                raw_label,
                width=22,
                break_long_words=False,
                break_on_hyphens=False,
            )
            color = POS_COLOR if positive else NEG_COLOR
            annotation = ax.annotate(
                label,
                xy=(float(row["beta"]), min(float(row["neg_log_p"]), y_limit)),
                xycoords="data",
                xytext=(x_position, 0.5),
                textcoords="axes fraction",
                ha=alignment,
                va="center",
                fontsize=fontsize,
                color=color,
                annotation_clip=False,
                clip_on=False,
                zorder=8,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.48",
                    "linewidth": 2.4,
                    "shrinkA": 7,
                    "shrinkB": 5,
                    "connectionstyle": "arc3,rad=0",
                    "relpos": line_origin,
                },
            )
            annotation.arrow_patch.set_clip_on(False)
            annotation.arrow_patch.set_capstyle("round")

            target_axes_y = ax.transAxes.inverted().transform(
                ax.transData.transform(annotation.xy)
            )[1]
            annotation.update_positions(renderer)
            provisional = Text.get_window_extent(annotation, renderer)
            center_display_y = next_top - provisional.height / 2.0
            lane = ax.transAxes.inverted().transform(
                (axes_box.x0, center_display_y)
            )[1]
            if lane <= target_axes_y + 0.055:
                # A target clipped against the fixed y ceiling cannot have a
                # label above it. Keep the deterministic top lane.
                top_clipped_targets += 1
            annotation.xyann = (x_position, float(lane))
            annotation.update_positions(renderer)
            box = Text.get_window_extent(annotation, renderer).expanded(1.02, 1.10)
            if box.y0 < axes_box.y0 or box.y1 > axes_box.y1:
                raise RuntimeError(
                    f"[figure4] upper annotation lane is outside the axes: {raw_label}"
                )
            if any(box.overlaps(placed) for placed in placed_boxes):
                raise RuntimeError(
                    "[figure4] upper annotation lanes cannot place all labels "
                    f"without overlap: {raw_label}"
                )
            placed_boxes.append(box)
            next_top = box.y0 - vertical_gap
            annotations.append(annotation)
    print(
        "[figure4-volcano] annotation layout: method=upper_two_column_lanes, "
        f"labels={len(annotations)}, leader_linewidth=2.4, wrap_width=22, "
        f"minimum_target_clearance=0.055, top_clipped_targets={top_clipped_targets}"
    )
    return annotations


def _display_phecode_name(value: object) -> str:
    """Return the concise Figure 4 panel-A display name."""
    text = str(value).strip()
    manual_names = {
        "Regular menstrual cycle/bleeding": "Regular menstrual cycle",
        "Irregular menstrual cycle/bleeding": "Irregular menstrual cycle",
        "Patient's noncompliance with medical treatment and regimen": (
            "Patient noncompliance with medical treatment"
        ),
        "Disorders of mineral metabolism and mineral deficiencies": (
            "Disorders of mineral metabolism"
        ),
        "Degenerative disc disease": "Degenerative disc disease",
        "Pulmonary insufficiency and acute respiratory distress syndrome": (
            "Pulmonary insufficiency"
        ),
    }
    text = manual_names.get(text, text)
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    return " ".join(text.split())


def _add_size_legend(
    ax,
    *,
    circle_meta: bool,
    beta_labels: bool = False,
    legend_above: bool = False,
    font_scale: float = 1.0,
    frameon: bool = True,
    inline_beta_label: bool = False,
) -> None:
    """Right-hand proxy legend for marker size (|effect size|) + meta outline.

    Marker area encodes |log-OR| on the capped log scale used by the PheWAS
    Manhattans. The ``meta-significant`` outline entry is shown only when
    ``circle_meta`` is on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend_beta = [0.5, 1.0, 2.0, BETA_CAP]
    handles = [
        plt.scatter([], [], s=_size_from_beta(np.array([v])), c="gray",
                    alpha=0.7, edgecolors="none",
                    label=(f"$\\geq$ {v:g}" if v == BETA_CAP else f"{v:g}"))
        for v in legend_beta
    ]
    if inline_beta_label:
        handles.insert(
            0,
            Line2D(
                [],
                [],
                linestyle="none",
                marker=r"$\beta$",
                markersize=VOLCANO_LABEL * font_scale,
                color="black",
                label="",
            ),
        )
    if circle_meta:
        handles.append(plt.scatter(
            [], [], s=200, facecolors="none", edgecolors="black", linewidths=2.2,
            label="meta-significant"))
    legend_kwargs = {
        "handles": handles,
        "title": (
            None
            if inline_beta_label
            else (r"$|\beta|$" if beta_labels else r"$|\log$(OR)$|$")
        ),
        "fontsize": VOLCANO_LABEL * font_scale,
        "title_fontsize": VOLCANO_LABEL * font_scale,
        "frameon": frameon,
        "borderpad": 0.7 if legend_above else 1.0,
        "handletextpad": 0.7 if legend_above else 1.2,
        "scatterpoints": 1,
    }
    if legend_above:
        compact_spacing = font_scale > 1.0
        legend_kwargs.update(
            loc="lower center",
            bbox_to_anchor=(0.46 if compact_spacing else 0.5, 1.015),
            ncol=len(handles),
            columnspacing=0.7 if compact_spacing else 1.0,
            handletextpad=0.5 if compact_spacing else 0.7,
            labelspacing=0.6,
        )
    else:
        legend_kwargs.update(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            labelspacing=1.6,
        )
    ax.legend(**legend_kwargs)


def volcano_image_path(ext: str = "png") -> Path:
    """Return the discovery volcano output path."""
    return config.figure_dir(FIGURE_ID) / f"Figure_4A_volcano.{ext}"


def render_volcano(
    *, clean_run_data: bool = False, clean_run_plots: bool = False,
) -> list[Path]:
    """Render the discovery volcano with meta-significant hits outlined."""
    import matplotlib.pyplot as plt

    df = load_phewas_results(clean_run_data=clean_run_data)
    meta_keys = _meta_sig_keys(clean_run_data=clean_run_data)
    print(f"[figure4-volcano] rows={len(df)}, x_limit=2, y_limit=10, font_scale={FONT_SCALE}")
    fig = plot_volcano(
        df, circle_meta=True, meta_keys=meta_keys, shorten_labels=True,
        x_limit=2.0, y_limit=10.0, beta_labels=True, legend_above=True,
        font_scale=FONT_SCALE, legend_frame=False, label_boxes=False,
        inline_beta_legend=True,
    )
    fig.set_size_inches(11, 8.5, forward=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.84)
    stem = volcano_image_path().with_suffix("")
    outputs = []
    for ext in config.FIGURE_FORMATS:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=150, bbox_inches=fig.bbox_inches)
        outputs.append(path)
    plt.close(fig)
    return outputs
