"""Figure 4 -- PheWAS meta-analysis figures (forest plot = Figure 4A).

Parses the outputs of the meta-analysis notebook (discovery PheWAS x replication
ACAF, inverse-variance meta-analysis, Li-Ji M_eff significance) and regenerates
its figures. The headline figure is the **forest plot of meta-significant hits**
(Figure 4A): for each meta-significant (variant x phecode) cell, only the combined
meta-analysis effect and 95% CI are shown.

Data source: ``config.PHEWAS_META_RESULTS_TSV`` -- the pipeline's
``meta_analysis_results_genic_meff.tsv`` (one row per sentinel x phecode with
``beta_d/se_d``, ``beta_r/se_r``, ``beta_meta/se_meta``, ``p_meta``,
``meta_significant``, ``phecode``, ``phecode_string``). Workbench working-dir
file, ``$HOME``-anchored and env-overridable (``AOU_META_RESULTS_TSV``).

Other notebook figures (MAF vs effect, PheWAS Manhattan, p-value histograms) are
ported in ``render_all`` and use the discovery results table
(``config.PHEWAS_DISCOVERY_TSV``) when present.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs, style

FIGURE_ID = "figure4"
GENIC_LABEL = "genic"
FOREST_FIGSIZE = (11.0, 8.5)
FOREST_LEFT = 0.06

_META = {"color": "#c0392b", "marker": "D"}


def _col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def on_workbench() -> bool:
    """True when a local or workspace-bucket meta-analysis source is available."""
    return os.path.exists(config.PHEWAS_META_RESULTS_TSV) or bool(
        config.PHEWAS_META_RESULTS_GS and gcs.available()
    )


def _stage_gcs_file(local_path: str, gs_path: str, *, force: bool = False) -> Path:
    """Copy a Figure 4 source from the study bucket into its local staging path."""
    local = Path(local_path)
    if local.exists() and not force:
        return local
    if not gs_path or not gcs.available():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    result = gcs.gcs_cp(gs_path, local)
    if result.returncode != 0:
        import warnings
        warnings.warn(
            f"[figure4] could not stage {gs_path} -> {local}: "
            f"{result.stderr.strip()[:300]}",
            stacklevel=2,
        )
    return local


# --- Data loading -------------------------------------------------------------


def load_meta_results(clean_run_data: bool = False) -> pd.DataFrame:
    """Stage and load the meta-analysis results table."""
    path = _stage_gcs_file(config.PHEWAS_META_RESULTS_TSV, config.PHEWAS_META_RESULTS_GS, force=clean_run_data)
    return pd.read_csv(path, sep="\t", dtype={"phecode": str})


def load_discovery_results(clean_run_data: bool = False) -> pd.DataFrame | None:
    """Load the discovery PheWAS results table, or None if absent."""
    path = _stage_gcs_file(
        config.PHEWAS_DISCOVERY_TSV,
        config.PHEWAS_DISCOVERY_GS,
        force=clean_run_data,
    )
    if path.exists():
        return pd.read_csv(path, sep="\t", dtype={"phecode": str})
    return None


# --- Figure 4A: forest plot ------------------------------------------------------
def plot_forest(
    meta_df: pd.DataFrame,
    *,
    genic_label: str = GENIC_LABEL,
    show_title: bool = True,
    show_variant: bool = True,
    show_p_value: bool = True,
    fixed_canvas: bool = False,
):
    """Forest plot of meta-significant hits (Figure 4A).

    Each row is presented as a compact table: phenotype and beta with its 95%
    CI, with an optional meta-analysis p-value column. ``show_variant=True``
    appends a readable variant to the phenotype. Alternating bands retain the
    variant grouping without putting identifiers inside the effect axis.

    ``show_title=False`` omits the ``Meta-significant hits`` suptitle (used when
    the panel is embedded title-less in the Figure 4/v3/v4 composites).
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    df = meta_df.copy()
    if "meta_significant" in df.columns:
        df = df[df["meta_significant"].astype(str).str.upper().isin(["TRUE", "1"])
                | (df["meta_significant"] == True)].copy()  # noqa: E712
    snp = _col(df, ["variant_tag", "snp", "SNP", "variant_id", "tag"])
    phe = _col(df, ["phecode", "Phecode", "phenotype"])
    desc = _col(df, ["phecode_string", "phecode_description", "description", "meaning"])
    b_m, s_m = _col(df, ["beta_meta", "beta_m"]), _col(df, ["se_meta", "se_m"])
    p_m = _col(df, ["p_meta", "P_meta", "pval_meta"])
    for name, c in [("beta_meta", b_m), ("se_meta", s_m)]:
        if c is None:
            raise ValueError(f"[figure4] forest plot needs a '{name}' column; "
                             f"have {list(df.columns)}")
    for c in (b_m, s_m, p_m):
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_p"] = df[p_m] if p_m else np.nan
    var_order = df.groupby(snp)["_p"].min().sort_values().index.tolist()
    z95 = stats.norm.ppf(0.975)

    # Assign a y for each (variant, phecode) row, top-to-bottom, grouped by variant.
    rows, groups = [], []
    n_total = len(df)
    y = n_total - 1
    for v in var_order:
        sub = df[df[snp] == v].sort_values("_p")
        y_top = y
        for _, r in sub.iterrows():
            phenotype = str(r[desc]) if desc else str(r[phe])
            label = (
                f"{phenotype} ({_display_variant(v)})"
                if show_variant
                else phenotype
            )
            rows.append((y, r, label))
            y -= 1
        groups.append((str(v), y_top, y + 1))

    style.apply_style()
    # Keep the forest compact: the previous per-row height made panel A nearly
    # three times taller than the locus panels and introduced a large composite
    # gap. Labels remain 14 pt, while rows use a tighter publication-style pitch.
    fig_h = FOREST_FIGSIZE[1] if fixed_canvas else max(5.2, 0.28 * n_total + 1.0)
    fig_w = FOREST_FIGSIZE[0] if fixed_canvas else 12.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        1,
        3 if show_p_value else 2,
        width_ratios=(5.7, 1.35, 3.7) if show_p_value else (6.5, 4.1),
        wspace=0.025,
    )
    ax_label = fig.add_subplot(grid[0])
    ax_p = fig.add_subplot(grid[1], sharey=ax_label) if show_p_value else None
    ax = fig.add_subplot(grid[2 if show_p_value else 1], sharey=ax_label)

    table_axes = (ax_label, ax, *((ax_p,) if ax_p is not None else ()))
    text_axes = (ax_label, *((ax_p,) if ax_p is not None else ()))
    row_font = 19.0 if fixed_canvas else 13.2
    title_font = 21.0 if fixed_canvas else 15.0
    x_label_font = 23.0 if fixed_canvas else 16.0
    tick_font = 19.0 if fixed_canvas else 14.0
    marker_size = 12 if fixed_canvas else 9
    error_linewidth = 2.8 if fixed_canvas else 2.4
    capsize = 6 if fixed_canvas else 5

    for gi, (v, y_top, y_bot) in enumerate(groups):
        if gi % 2 == 0:
            for table_ax in table_axes:
                table_ax.axhspan(y_bot - 0.5, y_top + 0.5, color="0.93", zorder=0)

    for y, r, label in rows:
        ax_label.text(0.0, y, label, ha="left", va="center", fontsize=row_font)
        if ax_p is not None:
            ax_p.text(
                0.5,
                y,
                _display_p_value(r["_p"]),
                ha="center",
                va="center",
                fontsize=row_font,
            )
        ax.errorbar(float(r[b_m]), y, xerr=z95 * float(r[s_m]),
                    fmt=_META["marker"], color=_META["color"], ms=marker_size,
                    lw=error_linewidth, capsize=capsize, zorder=4)

    for table_ax in table_axes:
        table_ax.set_ylim(-0.7, n_total - 0.3)
    for text_ax in text_axes:
        text_ax.set_xlim(0, 1)
        text_ax.set_xticks([])
        text_ax.set_yticks([])
        for spine in text_ax.spines.values():
            spine.set_visible(False)

    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.set_yticks([])
    ax.set_xlabel(r"Meta-analysis $\beta$ (log-odds)", fontsize=x_label_font)
    ax.tick_params(axis="x", labelsize=tick_font)
    ax_label.set_title(
        "Phecode (Variant)" if show_variant else "Phecode",
        loc="left",
        fontsize=title_font,
        pad=8,
    )
    if ax_p is not None:
        ax_p.set_title(r"$p$", fontsize=title_font, pad=8)
    ax.set_title(r"$\beta$ and 95% CI", fontsize=title_font, pad=8)
    if show_title:
        fig.suptitle(f"Meta-significant hits ({genic_label})", fontsize=18, y=0.995)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if fixed_canvas:
        fig.subplots_adjust(
            left=FOREST_LEFT,
            right=0.985,
            bottom=0.13,
            top=0.91,
        )
    else:
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.14, top=0.86)
    return fig


def _display_variant(value) -> str:
    """Convert ``4_99103864_C_T`` to the compact ``4:99,103,864 C>T``."""
    parts = str(value).strip().split("_", 3)
    if len(parts) != 4:
        return str(value)
    chrom, pos, ref, alt = parts
    try:
        pos_text = f"{int(pos):,}"
    except ValueError:
        return str(value)
    return f"{chrom.removeprefix('chr')}:{pos_text} {ref}>{alt}"


def _display_p_value(value) -> str:
    """Return a compact scientific-notation p-value for the table column."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return "—"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.1f}\times 10^{{{exponent}}}$"


def render_forest(
    clean_run_data: bool = False,
    clean_run_plots: bool = False,
    *,
    show_title: bool = True,
    show_variant: bool = True,
    show_p_value: bool = True,
    image_path: Path | None = None,
    fixed_canvas: bool = False,
) -> list:
    """Render Figure 4A (forest plot) to figures/figure4/; return the image paths.

    ``show_title=False`` with ``image_path`` renders a title-less copy to a
    distinct path without disturbing the canonical titled forest used by the
    main Figure 4. Figure 4 also sets ``show_variant=False``.
    """
    import matplotlib.pyplot as plt
    meta = load_meta_results(clean_run_data=clean_run_data)
    fig = plot_forest(
        meta,
        genic_label=GENIC_LABEL,
        show_title=show_title,
        show_variant=show_variant,
        show_p_value=show_p_value,
        fixed_canvas=fixed_canvas,
    )
    stem = (image_path or forest_image_path()).with_suffix("")
    out = []
    for ext in config.FIGURE_FORMATS:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(
            p,
            dpi=150,
            bbox_inches=(
                fig.bbox_inches
                if fixed_canvas
                else "tight"
            ),
        )
        out.append(p)
    plt.close(fig)
    return out


def forest_image_path(ext: str = "png") -> Path:
    """Fixed-canvas, p-free forest image embedded as Figure 4 panel D."""
    return config.figure_dir(FIGURE_ID) / f"Figure_4D_forest.{ext}"


# --- Other notebook figures (ported; need the discovery results table) --------
def plot_maf_vs_effect(disc: pd.DataFrame, meta: pd.DataFrame):
    """MAF vs |effect size| for all discovery tests, meta-significant highlighted."""
    import matplotlib.pyplot as plt
    maf = _col(disc, ["maf", "MAF", "af", "AF", "allele_freq", "EAF", "eaf"])
    beta = _col(disc, ["beta_ind", "beta", "Beta", "BETA", "b", "log_odds"])
    snp = _col(disc, ["snp", "SNP", "variant_id", "variant_tag", "tag"])
    phe = _col(disc, ["phecode", "Phecode", "phenotype"])
    if maf is None or beta is None:
        return None
    x = disc[maf].astype(float).to_numpy()
    yv = np.abs(disc[beta].astype(float).to_numpy())
    sig = np.zeros(len(disc), dtype=bool)
    m_snp = _col(meta, ["variant_tag", "snp", "SNP", "variant_id", "tag"])
    m_phe = _col(meta, ["phecode", "Phecode", "phenotype"])
    if snp and phe and m_snp and m_phe and "meta_significant" in meta.columns:
        keys = set(zip(meta.loc[meta["meta_significant"] == True, m_snp],  # noqa: E712
                       meta.loc[meta["meta_significant"] == True, m_phe].astype(str)))
        sig = np.array([k in keys for k in zip(disc[snp], disc[phe].astype(str))])
    style.apply_style()
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(x[~sig], yv[~sig], s=16, alpha=0.35, color="#888888", edgecolor="none",
               rasterized=True, label=f"not meta-sig ({int((~sig).sum()):,})")
    if sig.sum():
        ax.scatter(x[sig], yv[sig], s=90, alpha=0.9, color="#c0392b", edgecolor="black",
                   lw=0.6, zorder=5, label=f"meta-significant ({int(sig.sum())})")
    ax.set_xlabel("Minor allele frequency", fontsize=16)
    ax.set_ylabel("|β| (log-odds)", fontsize=16)
    ax.set_title(f"MAF vs effect size — discovery (n={len(disc):,})", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_pval_histograms(disc: pd.DataFrame):
    """Discovery p-value distribution histogram."""
    import matplotlib.pyplot as plt
    pcol = _col(disc, ["p", "P", "pval", "pvalue", "p_value", "p_ind"])
    if pcol is None:
        return None
    p = pd.to_numeric(disc[pcol], errors="coerce").dropna()
    p = p[(p > 0) & (p <= 1)]
    style.apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(p, bins=50, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("discovery p-value", fontsize=14)
    ax.set_ylabel("count", fontsize=14)
    ax.set_title(f"Discovery p-value distribution (n={len(p):,})", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig


def render_all(clean_run_data: bool = False, clean_run_plots: bool = False) -> list:
    """Generate every available Figure 4 meta-analysis figure.

    Always renders Figure 4A (forest plot). The discovery-based figures (MAF vs
    effect, p-value histogram) are added when the discovery results table is
    present. Returns the saved PNG paths.
    """
    import matplotlib.pyplot as plt
    out_dir = config.figure_dir(FIGURE_ID)
    saved = list(render_forest(clean_run_data, clean_run_plots))
    meta = load_meta_results()
    disc = load_discovery_results(clean_run_data=clean_run_data)
    extras = []
    if disc is not None:
        extras = [("maf_vs_effect", lambda: plot_maf_vs_effect(disc, meta)),
                  ("pval_histogram", lambda: plot_pval_histograms(disc))]
    for name, fn in extras:
        try:
            fig = fn()
        except Exception as exc:
            import warnings
            warnings.warn(f"[figure4] {name} failed: {exc}", stacklevel=2)
            continue
        if fig is None:
            continue
        p = out_dir / f"Figure_4_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved
