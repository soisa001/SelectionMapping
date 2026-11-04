"""Permutation enrichment of iHS-selected windows for LD-linked GWAS, eQTL, and sQTL overlap.

Load the per-window linkage cache or regenerate it from Workbench source data.
The fixed random seed preserves the original null permutation distribution."""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs

FIGURE_ID = "figure3"

# --- analysis contract (must match the source notebook cell exactly) ----------
SIG_PVAL = 0.01          # selected windows: simulation p <= 0.01
R2_LINKED = 0.5          # a variant is "LD-linked" to the lead at r^2 >= this
LEAD_R2_TOL = 0.999      # LD panel must self-match the lead at ~r^2 == 1
WINDOW_BP = 100_000      # +/-100 kb LD window when computing from the VCF
N_PERMS = 1000           # null draws
RANDOM_SEED = 1          # fixes both the null draws and their reproduction

# Annotation order + colours (identical to the notebook figure).
ANNOTATIONS = ("GWAS", "eQTL", "sQTL")
PALETTE = {"GWAS": "#377EB8", "eQTL": "#E41A1C", "sQTL": "#FF7F00"}
# Figure 3 v5 panel D pastel null-distribution colours, sampled from the
# supplied reference swatches. The standalone enrichment strip remains
# monochrome unless this palette is passed explicitly.
D_NULL_PALETTE = {
    "GWAS": "#A4AECA",
    "eQTL": "#B2C8A0",
    "sQTL": "#ECC593",
}
_LINK_COL = {"GWAS": "has_gwas", "eQTL": "has_eqtl", "sQTL": "has_sqtl"}


def _adir() -> Path:
    return config.analysis_dir(FIGURE_ID)


def _stage_dir() -> Path:
    """Local staging root for bucket copies + recomputed cache (gitignored)."""
    d = _adir() / "enrichment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def on_workbench() -> bool:
    """True when the enrichment bucket is reachable (a GCS CLI is on PATH)."""
    return gcs.available()


# --------------------------------------------------------------------------
# Idempotent bucket staging
# --------------------------------------------------------------------------
def _stage(remote_url: str, local_path: Path, *, recursive: bool = False) -> bool:
    """Copy ``remote_url`` -> ``local_path`` once. Returns True if present after.

    Idempotent: an existing local file (or non-empty directory) is left as-is, so
    re-running the pipeline never re-downloads valid staged inputs.
    """
    if recursive:
        if local_path.exists() and any(local_path.iterdir()):
            return True
    elif local_path.exists() and local_path.stat().st_size > 0:
        return True
    if not remote_url or not gcs.available():
        return local_path.exists()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if recursive:
        local_path.mkdir(parents=True, exist_ok=True)
        proj = gcs.billing_project()
        import shutil
        import subprocess
        if shutil.which("gcloud") is not None:
            cmd = ["gcloud", "storage", "cp", "-r"]
            if proj:
                cmd.append(f"--billing-project={proj}")
            cmd += [remote_url.rstrip("/") + "/*", str(local_path)]
        else:
            cmd = ["gsutil"] + (["-u", proj] if proj else []) + [
                "-m", "cp", "-r", remote_url.rstrip("/") + "/*", str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            warnings.warn(f"[figure3_enrichment] staging {remote_url} failed: "
                          f"{result.stderr.strip()}", stacklevel=2)
        return local_path.exists() and any(local_path.iterdir())
    result = gcs.gcs_cp(remote_url, str(local_path))
    if result.returncode != 0:
        warnings.warn(f"[figure3_enrichment] staging {remote_url} failed: "
                      f"{result.stderr.strip()}", stacklevel=2)
    return local_path.exists()


# --------------------------------------------------------------------------
# Windows + linkage cache
# --------------------------------------------------------------------------
def _make_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{int(pos)}:{ref}:{alt}"


def _load_windows() -> pd.DataFrame:
    """All iHS windows with their lead variant + selection flag.

    Uses the tracked Figure 3A windows TSV. Columns needed: chr, pos, ref, alt,
    sim_pval. Row order is preserved so the seeded permutation draws line up with
    the linkage cache built on the Workbench.
    """
    win = pd.read_csv(config.IHS_WINDOWS_TSV, sep="\t", compression="infer",
                      dtype={"chr": str}, low_memory=False)
    win["chr"] = win["chr"].where(win["chr"].str.startswith("chr"), "chr" + win["chr"])
    win["pos"] = pd.to_numeric(win["pos"], errors="coerce").astype("Int64")
    win["sim_pval"] = pd.to_numeric(win["sim_pval"], errors="coerce")
    win = win.dropna(subset=["pos"]).reset_index(drop=True)
    win["pos"] = win["pos"].astype(np.int64)
    win["selected"] = win["sim_pval"] <= SIG_PVAL
    win["_key"] = [
        _make_key(c, p, r, a)
        for c, p, r, a in zip(win["chr"], win["pos"], win["ref"], win["alt"])
    ]
    return win


def _resolve_linkage_cache() -> Path | None:
    """Locate the per-window linkage cache, staging from the bucket if needed.

    Order: tracked asset -> previously staged copy -> ``gcloud cp`` from the
    enrichment bucket (Workbench only). Returns None when no cache is reachable.
    """
    asset = config.IHS_ENRICHMENT_LINKAGE_TSV
    if asset.exists():
        return asset
    staged = _stage_dir() / "permutation_pool_linkage.tsv"
    if staged.exists():
        return staged
    remote = config.enrichment_bucket_path("enrichment_results/permutation_pool_linkage.tsv")
    if _stage(remote, staged):
        return staged
    return None


def _load_linkage_lookup(path: Path) -> dict[str, dict]:
    """key -> {ld_ok, has_gwas, has_eqtl, has_sqtl} from the linkage cache TSV."""
    df = pd.read_csv(path, sep="\t", compression="infer", dtype={"chr": str})
    bool_cols = ["ld_ok", "has_gwas", "has_eqtl", "has_sqtl"]
    for c in bool_cols:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "t", "yes"])
    return {
        r["key"]: {c: bool(r[c]) for c in bool_cols}
        for _, r in df.iterrows()
    }


# --------------------------------------------------------------------------
# Permutation test (fast array indexing over the linkage booleans)
# --------------------------------------------------------------------------
def _permutation(win: pd.DataFrame, lookup: dict[str, dict]) -> dict:
    """Observed hit rate + seeded null distribution per annotation.

    A window not present in the linkage cache defaults to no LD -> no hit, exactly
    as in the source cell.
    """
    n_total = len(win)
    keys = win["_key"].to_numpy()
    link = {a: np.zeros(n_total, dtype=bool) for a in ANNOTATIONS}
    for i, k in enumerate(keys):
        rec = lookup.get(k)
        if rec is None:
            continue
        for a in ANNOTATIONS:
            link[a][i] = rec[_LINK_COL[a]]

    sel_indices = np.where(win["selected"].to_numpy())[0]
    n_selected = len(sel_indices)

    rng = np.random.default_rng(RANDOM_SEED)
    perm_draws = np.empty((N_PERMS, n_selected), dtype=np.int64)
    for p in range(N_PERMS):
        perm_draws[p] = rng.choice(n_total, size=n_selected, replace=False)

    observed, null, stats = {}, {}, {}
    for a in ANNOTATIONS:
        arr = link[a]
        obs = float(arr[sel_indices].mean()) if n_selected else float("nan")
        dist = np.array([arr[perm_draws[p]].sum() / n_selected
                         for p in range(N_PERMS)], dtype=float)
        n_ge = int((dist >= obs).sum())
        mean = float(dist.mean())
        sd = float(dist.std())
        observed[a] = obs
        null[a] = dist
        stats[a] = {
            "observed_pct": round(100 * obs, 2),
            "perm_mean_pct": round(100 * mean, 2),
            "perm_sd_pct": round(100 * sd, 2),
            "fold_enrichment": round(obs / mean, 2) if mean > 0 else float("inf"),
            "z_score": round((obs - mean) / sd, 2) if sd > 0 else float("inf"),
            "n_perm_ge_obs": n_ge,
            "n_perms": N_PERMS,
            "perm_pval": round((n_ge + 1) / (N_PERMS + 1), 4),
        }
    return {
        "annotations": list(ANNOTATIONS),
        "observed": observed,
        "null": null,
        "stats": stats,
        "n_selected": n_selected,
        "n_total": n_total,
        "r2_linked": R2_LINKED,
        "n_perms": N_PERMS,
        "source": "cache",
    }


# --------------------------------------------------------------------------
# Public build (cache-first)
# --------------------------------------------------------------------------
def build(clean_run_data: bool = False) -> dict:
    """Load the linkage cache or regenerate it from the Workbench source inputs."""
    if clean_run_data:
        regenerate_cache()
    win = _load_windows()
    cache = _resolve_linkage_cache()
    if cache is None:
        regenerate_cache()
        cache = _resolve_linkage_cache()
    if cache is None:
        raise FileNotFoundError(f"Enrichment linkage cache missing: {config.IHS_ENRICHMENT_LINKAGE_TSV}")
    return _permutation(win, _load_linkage_lookup(cache))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def draw_panels(axes, enr: dict, *, fs: float = 13.0,
                show_titles: bool = True,
                null_colors: dict[str, str] | None = None,
                show_y_grid: bool = True) -> None:
    """Draw the three permutation-distribution histograms into ``axes``.

    By default, GWAS/eQTL/sQTL read as one monochrome series: black null
    histogram + dashed observed line (no shaded tail). ``null_colors`` can map
    each annotation to a distinct histogram colour while the observed line
    remains black. The observed percentage is annotated inline at the dashed
    line; a single shared legend is added above the row by the caller (see
    ``add_shared_legend``).

    With ``show_titles=False`` (the composite enrichment row) the titles are
    dropped and the annotation set moves into a two-line x-axis label instead, e.g.
    "% windows with GWAS / LD-linked hit".
    """
    tick_fs = fs * 0.85
    for ax, label in zip(axes, enr["annotations"]):
        dist = enr["null"][label] * 100.0
        obs = enr["observed"][label] * 100.0
        hist_clr = null_colors.get(label, "black") if null_colors else "black"
        hist_alpha = 1.0 if null_colors else 0.45
        clr = "black"
        st = enr["stats"][label]
        pval = st["perm_pval"]
        fold = st["fold_enrichment"]

        ax.hist(dist, bins=25, color=hist_clr, alpha=hist_alpha,
                edgecolor="white",
                lw=1.0, zorder=3)
        ax.axvline(obs, color=clr, lw=2.5, ls="--", zorder=5)
        # Keep the numeric observed rate on the panel now that the legend moved
        # out: a compact label just inside the top, at the observed line.
        ax.annotate(f"{obs:.2f}%", xy=(obs, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="top", fontsize=tick_fs, color=clr)

        star = ("***" if pval < 0.001 else "**" if pval < 0.01
                else "*" if pval < 0.05 else "ns")
        if show_titles:
            # Compact second line so adjacent panel titles don't collide once the
            # H row is narrowed.
            ax.set_title(f"{label}\np = {pval:.4f}{star} ({fold:.1f}×)",
                         fontsize=fs, fontweight="bold", pad=6)
            ax.set_xlabel("% windows with LD-linked hit", fontsize=fs * 0.92)
        else:
            # Titleless variant (composite panel H): the annotation set moves
            # into the x-axis label, wrapped onto two lines to keep the three
            # narrow panels from colliding.
            ax.set_xlabel(f"% windows with {label}\nLD-linked hit",
                          fontsize=fs * 0.92)
        ax.set_ylabel("# permutations", fontsize=fs * 0.92)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_y_grid:
            ax.grid(axis="y", alpha=0.2, zorder=0)
        else:
            ax.yaxis.grid(False)


def add_shared_legend(fig, ax_center, *, fs: float = 13.0, y: float = 1.12,
                      null_colors: dict[str, str] | None = None) -> None:
    """One shared legend above the enrichment row."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if null_colors:
        permutation_handle = tuple(
            Patch(facecolor=null_colors[label], edgecolor="white")
            for label in ANNOTATIONS
        )
        handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.1)}
    else:
        permutation_handle = Patch(
            facecolor="black", alpha=0.45, edgecolor="white"
        )
        handler_map = None
    handles = [
        permutation_handle,
        Line2D([0], [0], color="black", lw=2.5, ls="--"),
    ]
    ax_center.legend(
        handles=handles, labels=["Permutations", "Observed"],
        loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=2, frameon=False, fontsize=fs * 0.9,
        handlelength=2.0, columnspacing=1.6, borderaxespad=0.0,
        handler_map=handler_map,
    )


def render_enrichment_strip(clean_run_data: bool = False,
                            clean_run_plots: bool = False) -> list[Path]:
    """Standalone 3-panel enrichment figure (the Figure 3 enrichment content)."""
    import matplotlib.pyplot as plt

    from .common import style
    from .common.io import needs_build

    style.apply_style()
    stem = config.figure_dir(FIGURE_ID) / "panels" / "Figure_3E_enrichment"
    targets = [stem.with_suffix(f".{e}") for e in config.FIGURE_FORMATS]
    if not needs_build(targets, clean_run_plots or clean_run_data):
        return targets

    enr = build(clean_run_data=clean_run_data)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    draw_panels(axes, enr)
    # Header band, top to bottom: suptitle, shared legend, per-panel titles.
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    add_shared_legend(fig, axes[1], y=1.22)
    fig.suptitle(
        f"Permutation test: {enr['n_selected']:,} selected windows vs "
        f"{enr['n_perms']} random draws (LD r² ≥ {enr['r2_linked']})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for t in targets:
        fig.savefig(t)
        saved.append(t)
    plt.close(fig)
    return saved


# --------------------------------------------------------------------------
# Source pipeline (Workbench) -- regenerate the linkage cache from the bucket
# --------------------------------------------------------------------------
def _normalize_variant(chrom, pos, ref, alt):
    pos = int(pos); ref = str(ref).upper(); alt = str(alt).upper()
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]; pos += 1
    return chrom, pos, ref, alt


def _ld_dir() -> Path:
    return _stage_dir() / "ihs_output_v4" / "ld_cache"


def _ld_meta_path(chrom, pos):
    hits = glob.glob(str(_ld_dir() / chrom / f"{chrom}_{pos}_*.snplist.tsv"))
    return hits[0] if hits else None


def _load_existing_ld(chrom, pos):
    """Read a cached LD panel (signed r to lead) -> lo/hi span + r^2 lookups."""
    meta_path = _ld_meta_path(chrom, pos)
    if meta_path is None:
        return "no_ld_cache", None
    meta = pd.read_csv(meta_path, sep="\t", dtype={"variant_id": str, "ref": str, "alt": str})
    r2 = meta["signed_r_lead"].to_numpy(dtype=np.float64) ** 2
    lead_idx = int(np.argmax(r2))
    if r2[lead_idx] < LEAD_R2_TOL:
        return "no_self_match", None
    if int(meta["pos"].iloc[lead_idx]) != pos:
        pm = np.where(meta["pos"].to_numpy() == pos)[0]
        if pm.size == 0:
            return "lead_pos_mismatch", None
    pp = meta["pos"].to_numpy(dtype=np.int64)
    pr = meta["ref"].astype(str).str.upper().to_numpy()
    pa = meta["alt"].astype(str).str.upper().to_numpy()
    by_key = {(int(pp[i]), pr[i], pa[i]): float(r2[i]) for i in range(len(meta))}
    pos_to_r2 = {}
    for i in range(len(meta)):
        p = int(pp[i])
        if p not in pos_to_r2 or r2[i] > pos_to_r2[p]:
            pos_to_r2[p] = float(r2[i])
    return "ok", {"lo": int(pp.min()), "hi": int(pp.max()),
                  "by_key": by_key, "pos_to_r2": pos_to_r2}


def _compute_ld_from_vcf(chrom, pos, ref, alt):
    """Compute an LD panel from the staged biallelic VCF (cyvcf2); cache it."""
    try:
        from cyvcf2 import VCF as cyVCF
    except ImportError:
        return "no_cyvcf2", None
    vcf_path = _stage_dir() / "SV_biallelic" / f"{chrom}.biallelic.vcf.gz"
    if not vcf_path.exists():
        remote = config.enrichment_bucket_path(f"SV_biallelic/{chrom}.biallelic.vcf.gz")
        if not _stage(remote, vcf_path):
            return "no_vcf", None
        _stage(remote + ".tbi", vcf_path.with_suffix(vcf_path.suffix + ".tbi"))
    ref = str(ref).upper(); alt = str(alt).upper()
    region = f"{chrom}:{max(1, pos - WINDOW_BP)}-{pos + WINDOW_BP}"
    vcf = cyVCF(str(vcf_path))
    positions, refs, alts, dosages = [], [], [], []
    for v in vcf(region):
        if len(v.ALT) != 1:
            continue
        gt = v.gt_types.astype(np.float64)
        gt[v.gt_types == 3] = np.nan
        positions.append(v.POS); refs.append(v.REF.upper())
        alts.append(v.ALT[0].upper()); dosages.append(gt)
    vcf.close()
    if not positions:
        return "empty_region", None
    positions = np.array(positions, dtype=np.int64)
    refs_arr = np.array(refs); alts_arr = np.array(alts)
    lead_mask = (positions == pos) & (refs_arr == ref) & (alts_arr == alt)
    if not lead_mask.any():
        _, npos, nref, nalt = _normalize_variant(chrom, pos, ref, alt)
        lead_mask = (positions == npos) & (refs_arr == nref) & (alts_arr == nalt)
    if not lead_mask.any():
        return "lead_not_in_vcf", None
    lead_gt = dosages[int(np.where(lead_mask)[0][0])]
    valid_lead = np.isfinite(lead_gt)
    if np.nanstd(lead_gt) == 0 or valid_lead.sum() < 20:
        return "lead_monomorphic", None
    n_vars = len(positions)
    signed_r = np.full(n_vars, np.nan)
    for i in range(n_vars):
        valid = valid_lead & np.isfinite(dosages[i])
        if valid.sum() < 20:
            continue
        xi, yi = lead_gt[valid], dosages[i][valid]
        if np.std(yi) == 0:
            signed_r[i] = 0.0
            continue
        with np.errstate(invalid="ignore"):
            signed_r[i] = np.corrcoef(xi, yi)[0, 1]
    r2 = signed_r ** 2
    by_key, pos_to_r2 = {}, {}
    for i in range(n_vars):
        if np.isfinite(r2[i]):
            by_key[(int(positions[i]), refs_arr[i], alts_arr[i])] = float(r2[i])
            p = int(positions[i])
            if p not in pos_to_r2 or r2[i] > pos_to_r2[p]:
                pos_to_r2[p] = float(r2[i])
    return "ok", {"lo": int(positions.min()), "hi": int(positions.max()),
                  "by_key": by_key, "pos_to_r2": pos_to_r2}


def _get_ld(chrom, pos, ref, alt):
    status, ld = _load_existing_ld(chrom, pos)
    if status == "ok":
        return ld
    _, ld2 = _compute_ld_from_vcf(chrom, pos, ref, alt)
    return ld2


def _load_gwas_positions() -> dict:
    path = _stage_dir() / "GWAS_catalog" / "gwas_catalog_associations.tsv"
    _stage(config.enrichment_bucket_path("GWAS_catalog/gwas_catalog_associations.tsv"), path)
    gwas = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    gwas["_p"] = pd.to_numeric(gwas["P-VALUE"], errors="coerce")
    gwas["_pos"] = pd.to_numeric(gwas["CHR_POS"], errors="coerce")
    gwas = gwas.dropna(subset=["_p", "_pos"])
    gwas = gwas[gwas["_p"] <= 5e-8].copy()
    gwas["_chr"] = gwas["CHR_ID"].astype(str).str.strip()
    gwas["_chr"] = gwas["_chr"].where(gwas["_chr"].str.startswith("chr"), "chr" + gwas["_chr"])
    gwas["_pos"] = gwas["_pos"].astype(np.int64)
    return {c: np.sort(sub["_pos"].unique().astype(np.int64))
            for c, sub in gwas.groupby("_chr")}


def _load_qtl(remote_rel: str, local_name: str, *, qval_col=None, qval_thresh=0.05):
    path = _stage_dir() / "ihs_output_v4" / local_name
    _stage(config.enrichment_bucket_path(remote_rel), path)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["variantChrom"] = df["variantChrom"].astype(str)
    df["variantChrom"] = df["variantChrom"].where(
        df["variantChrom"].str.startswith("chr"), "chr" + df["variantChrom"])
    df["variantPosition"] = pd.to_numeric(df["variantPosition"], errors="coerce")
    df = df.dropna(subset=["variantPosition"])
    df["variantPosition"] = df["variantPosition"].astype(np.int64)
    df["variantRef"] = df["variantRef"].astype(str).str.upper()
    df["variantAlt"] = df["variantAlt"].astype(str).str.upper()
    if qval_col and qval_col in df.columns:
        df = df[pd.to_numeric(df[qval_col], errors="coerce") < qval_thresh].copy()
    by_chrom = {}
    for c, sub in df.groupby("variantChrom"):
        keys = set(zip(sub["variantPosition"].astype(int),
                       sub["variantRef"], sub["variantAlt"]))
        spos = np.sort(sub["variantPosition"].unique().astype(np.int64))
        by_chrom[c] = (spos, keys)
    return by_chrom


def _qtl_linked(qtl_chrom_data, ld) -> bool:
    if qtl_chrom_data is None or ld is None:
        return False
    spos, keys = qtl_chrom_data
    lo_i = np.searchsorted(spos, ld["lo"], side="left")
    hi_i = np.searchsorted(spos, ld["hi"], side="right")
    if lo_i >= hi_i:
        return False
    range_pos = set(spos[lo_i:hi_i].tolist())
    for key in keys:
        if key[0] in range_pos:
            r2 = ld["by_key"].get(key)
            if r2 is not None and r2 >= R2_LINKED:
                return True
    return False


def regenerate_cache(*, save_reference_cache: bool = True) -> Path:
    """Rerun the source pipeline and rewrite the linkage cache (Workbench).

    Idempotently stages the LD cache, GWAS Catalog and MAGE QTL tables (and, only
    where an LD panel must be computed, the biallelic VCFs) from
    ``config.ENRICHMENT_BUCKET_GS``, recomputes each window's GWAS/eQTL/sQTL LD
    linkage over the seeded permutation hitlist, and writes
    ``permutation_pool_linkage.tsv`` + ``permutation_results.tsv``. When
    ``save_reference_cache`` is set, the linkage cache is also written (gzipped) to the
    local reference-cache path for subsequent runs.
    """
    if not gcs.available():
        raise RuntimeError("regenerate_cache requires a GCS CLI (Workbench).")

    # Stage the whole LD cache once (large; recursive).
    _stage(config.enrichment_bucket_path("ihs_output_v4/ld_cache"),
           _ld_dir(), recursive=True)

    win = _load_windows()
    n_total = len(win)
    sel_indices = np.where(win["selected"].to_numpy())[0]
    n_selected = len(sel_indices)

    # The seeded draws define the exact hitlist we must resolve.
    rng = np.random.default_rng(RANDOM_SEED)
    perm_draws = np.empty((N_PERMS, n_selected), dtype=np.int64)
    for p in range(N_PERMS):
        perm_draws[p] = rng.choice(n_total, size=n_selected, replace=False)
    hit_idx = np.unique(np.concatenate([sel_indices, perm_draws.ravel()]))

    gwas_by_chrom = _load_gwas_positions()
    eqtl_by_chrom = _load_qtl(
        "ihs_output_v4/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt.gz",
        "mage_eQTL.txt.gz", qval_col="qval", qval_thresh=0.05)
    sqtl_by_chrom = _load_qtl(
        "ihs_output_v4/sQTL_finemapping.geneMerged.leadVariant.MAGE.v1.0.txt.gz",
        "mage_sQTL.txt.gz")

    rows = []
    hit = win.iloc[hit_idx]
    for _, r in hit.iterrows():
        chrom, pos = str(r["chr"]), int(r["pos"])
        ref, alt = str(r["ref"]).upper(), str(r["alt"]).upper()
        ld = _get_ld(chrom, pos, ref, alt)
        rec = {"key": r["_key"], "chr": chrom, "pos": pos, "ref": r["ref"],
               "alt": r["alt"], "ld_ok": ld is not None,
               "has_gwas": False, "has_eqtl": False, "has_sqtl": False}
        if ld is not None:
            gpos = gwas_by_chrom.get(chrom)
            if gpos is not None:
                lo_i = np.searchsorted(gpos, ld["lo"], side="left")
                hi_i = np.searchsorted(gpos, ld["hi"], side="right")
                for gp in gpos[lo_i:hi_i]:
                    r2 = ld["pos_to_r2"].get(int(gp))
                    if r2 is not None and r2 >= R2_LINKED:
                        rec["has_gwas"] = True
                        break
            rec["has_eqtl"] = _qtl_linked(eqtl_by_chrom.get(chrom) if eqtl_by_chrom else None, ld)
            rec["has_sqtl"] = _qtl_linked(sqtl_by_chrom.get(chrom) if sqtl_by_chrom else None, ld)
        rows.append(rec)

    linkage = pd.DataFrame(rows)
    out = _stage_dir() / "permutation_pool_linkage.tsv"
    linkage.to_csv(out, sep="\t", index=False)

    # Results summary + tracked asset.
    enr = _permutation(win, _load_linkage_lookup(out))
    results = pd.DataFrame([{"annotation": a, **enr["stats"][a]} for a in ANNOTATIONS])
    results.to_csv(_stage_dir() / "permutation_results.tsv", sep="\t", index=False)
    if save_reference_cache:
        config.IHS_ENRICHMENT_LINKAGE_TSV.parent.mkdir(parents=True, exist_ok=True)
        linkage.to_csv(config.IHS_ENRICHMENT_LINKAGE_TSV, sep="\t", index=False,
                       compression="gzip")
        results.to_csv(config.IHS_ENRICHMENT_RESULTS_TSV, sep="\t", index=False)
    return out
