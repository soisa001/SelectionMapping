"""Figure 4 panel C: discovery effects versus IVW meta-analysis effects.

The discovery universe is the audited 446-row Figure S9 source-data table.
Only associations with finite discovery and replication standard errors and
effects receive a fixed-effect inverse-variance-weighted meta estimate. Rows
without a replication test remain visible on the discovery side and missing on
the meta-analysis side; they are never replaced by zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from . import figure4_meta, figure4_s9
from .common import config, style

FIGURE_ID = "figure4"
FIGURE_STEM = "Figure_4C_Discovery_Meta"
EXPECTED_DISCOVERY_COUNT = 446
EXPECTED_META_EFFECT_COUNT = 437
EXPECTED_META_SIGNIFICANT_COUNT = 17
META_SIGNIFICANT_COLOR = "#B94F3A"
SOURCE_COLUMNS = (
    "variant_tag",
    "phecode",
    "phecode_string",
    "beta_discovery",
    "se_discovery",
    "beta_replication",
    "se_replication",
    "beta_meta_ivw",
    "se_meta_ivw",
    "has_meta_effect",
    "meta_significant",
)


class Figure3EffectMetaContractError(ValueError):
    """Raised when panel C inputs violate the audited row/effect contract."""


def panel_image_path(ext: str = "png") -> Path:
    """Return the standalone panel-C output path."""
    return config.figure_dir(FIGURE_ID) / f"{FIGURE_STEM}.{ext}"


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _required_column(
    frame: pd.DataFrame, names: tuple[str, ...], label: str
) -> str:
    column = _first_column(frame, names)
    if column is None:
        raise Figure3EffectMetaContractError(f"Table is missing {label}")
    return column


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES"})


def _check_identifiers(frame: pd.DataFrame, *, label: str) -> None:
    invalid = frame["variant_tag"].isin({"", "nan", "None"}) | frame[
        "phecode"
    ].isin({"", "nan", "None"})
    if invalid.any():
        raise Figure3EffectMetaContractError(
            f"{label} has {int(invalid.sum())} rows with empty identifiers"
        )
    duplicates = frame.duplicated(["variant_tag", "phecode"], keep=False)
    if duplicates.any():
        raise Figure3EffectMetaContractError(
            f"{label} is not unique by variant_tag x phecode"
        )


def _prepare_discovery(discovery: pd.DataFrame) -> pd.DataFrame:
    variant = _required_column(
        discovery, ("variant_tag", "snp", "SNP", "variant_id", "tag"), "variant_tag"
    )
    phecode = _required_column(discovery, ("phecode", "Phecode"), "phecode")
    beta = _required_column(
        discovery, ("beta", "beta_d", "beta_discovery", "beta_disc"), "discovery beta"
    )
    se = _required_column(
        discovery, ("se", "se_d", "se_discovery", "se_disc"), "discovery SE"
    )
    meta_flag = _first_column(
        discovery, ("is_meta_significant", "meta_significant")
    )
    category = _first_column(discovery, ("category",))
    if meta_flag is None and category is None:
        raise Figure3EffectMetaContractError(
            "Discovery table needs is_meta_significant, meta_significant, or category"
        )
    description = _first_column(
        discovery,
        ("phecode_string", "phecode_description", "description", "meaning"),
    )

    out = pd.DataFrame(index=discovery.index)
    out["variant_tag"] = discovery[variant].astype(str).str.strip()
    out["phecode"] = discovery[phecode].astype(str).str.strip()
    out["phecode_string"] = (
        discovery[description].fillna("").astype(str).str.strip()
        if description
        else out["phecode"]
    )
    out["beta_discovery"] = pd.to_numeric(discovery[beta], errors="coerce")
    out["se_discovery"] = pd.to_numeric(discovery[se], errors="coerce")
    if meta_flag:
        out["meta_significant"] = _truthy(discovery[meta_flag])
    else:
        out["meta_significant"] = discovery[category].astype(str).str.strip().eq(
            "meta-significant"
        )
    _check_identifiers(out, label="Discovery table")
    invalid = (
        ~np.isfinite(out["beta_discovery"])
        | ~np.isfinite(out["se_discovery"])
        | out["se_discovery"].le(0)
    )
    if invalid.any():
        raise Figure3EffectMetaContractError(
            f"Discovery table has {int(invalid.sum())} invalid beta/SE rows"
        )
    return out


def _prepare_replication(meta: pd.DataFrame) -> pd.DataFrame:
    variant = _required_column(
        meta, ("variant_tag", "snp", "SNP", "variant_id", "tag"), "variant_tag"
    )
    phecode = _required_column(meta, ("phecode", "Phecode"), "phecode")
    beta_r = _required_column(
        meta, ("beta_r", "beta_replication", "beta_rep"), "replication beta"
    )
    se_r = _required_column(
        meta, ("se_r", "se_replication", "se_rep"), "replication SE"
    )
    out = pd.DataFrame(index=meta.index)
    out["variant_tag"] = meta[variant].astype(str).str.strip()
    out["phecode"] = meta[phecode].astype(str).str.strip()
    out["beta_replication"] = pd.to_numeric(meta[beta_r], errors="coerce")
    out["se_replication"] = pd.to_numeric(meta[se_r], errors="coerce")
    meta_flag = _first_column(meta, ("meta_significant", "is_meta_significant"))
    if meta_flag:
        out["meta_significant_meta"] = _truthy(meta[meta_flag])
    provided_beta = _first_column(meta, ("beta_meta", "beta_m"))
    if provided_beta:
        out["beta_meta_provided"] = pd.to_numeric(meta[provided_beta], errors="coerce")
    _check_identifiers(out, label="Meta-analysis table")
    return out


def prepare_effect_meta(
    discovery: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    expected_discovery_count: int | None = EXPECTED_DISCOVERY_COUNT,
    expected_meta_effect_count: int | None = EXPECTED_META_EFFECT_COUNT,
    expected_meta_significant_count: int | None = EXPECTED_META_SIGNIFICANT_COUNT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute fixed-effect IVW estimates for all valid discovery/replication pairs."""
    discovery_frame = _prepare_discovery(discovery)
    replication_frame = _prepare_replication(meta)
    discovery_keys = set(
        zip(discovery_frame["variant_tag"], discovery_frame["phecode"])
    )
    replication_keys = set(
        zip(replication_frame["variant_tag"], replication_frame["phecode"])
    )
    extra = replication_keys - discovery_keys
    if extra:
        raise Figure3EffectMetaContractError(
            f"Meta-analysis table has {len(extra)} rows outside the discovery universe"
        )

    frame = discovery_frame.merge(
        replication_frame,
        on=["variant_tag", "phecode"],
        how="left",
        validate="one_to_one",
    )
    if "meta_significant_meta" in frame:
        overlap = frame["meta_significant_meta"].notna()
        disagreement = overlap & (
            frame["meta_significant"].astype(bool)
            != frame["meta_significant_meta"].astype(bool)
        )
        if disagreement.any():
            raise Figure3EffectMetaContractError(
                f"Meta-significance flags disagree for {int(disagreement.sum())} rows"
            )
        frame = frame.drop(columns="meta_significant_meta")

    frame["has_meta_effect"] = (
        np.isfinite(frame["beta_discovery"])
        & np.isfinite(frame["se_discovery"])
        & frame["se_discovery"].gt(0)
        & np.isfinite(frame["beta_replication"])
        & np.isfinite(frame["se_replication"])
        & frame["se_replication"].gt(0)
    )
    frame["beta_meta_ivw"] = np.nan
    frame["se_meta_ivw"] = np.nan
    paired = frame["has_meta_effect"]
    weight_d = 1.0 / frame.loc[paired, "se_discovery"].pow(2)
    weight_r = 1.0 / frame.loc[paired, "se_replication"].pow(2)
    total_weight = weight_d + weight_r
    frame.loc[paired, "beta_meta_ivw"] = (
        weight_d * frame.loc[paired, "beta_discovery"]
        + weight_r * frame.loc[paired, "beta_replication"]
    ) / total_weight
    frame.loc[paired, "se_meta_ivw"] = np.sqrt(1.0 / total_weight)

    missing_meta_significant = frame["meta_significant"] & ~frame["has_meta_effect"]
    if missing_meta_significant.any():
        raise Figure3EffectMetaContractError(
            "Meta-significant rows must have finite discovery and replication beta/SE"
        )

    discovery_count = len(frame)
    meta_effect_count = int(frame["has_meta_effect"].sum())
    meta_significant_count = int(frame["meta_significant"].sum())
    checks = (
        ("discovery associations", expected_discovery_count, discovery_count),
        ("IVW meta effects", expected_meta_effect_count, meta_effect_count),
        (
            "meta-significant associations",
            expected_meta_significant_count,
            meta_significant_count,
        ),
    )
    for label, expected, observed in checks:
        if expected is not None and observed != expected:
            raise Figure3EffectMetaContractError(
                f"Expected {expected} {label}; found {observed}"
            )

    provided_difference = None
    if "beta_meta_provided" in frame:
        compare = frame["has_meta_effect"] & np.isfinite(frame["beta_meta_provided"])
        if compare.any():
            provided_difference = float(
                (frame.loc[compare, "beta_meta_ivw"] - frame.loc[compare, "beta_meta_provided"])
                .abs()
                .max()
            )

    paired_frame = frame.loc[frame["has_meta_effect"]]
    audit: dict[str, object] = {
        "status": "complete",
        "row_unit": "one unique audited discovery variant_tag x phecode association",
        "meta_effect_formula": (
            "beta_meta=(beta_d/se_d^2 + beta_r/se_r^2)/(1/se_d^2 + 1/se_r^2); "
            "se_meta=sqrt(1/(1/se_d^2 + 1/se_r^2))"
        ),
        "missing_replication_policy": "leave IVW meta effect missing; never impute zero",
        "counts": {
            "discovery_rows": discovery_count,
            "meta_effect_rows": meta_effect_count,
            "missing_meta_effect_rows": discovery_count - meta_effect_count,
            "meta_significant_rows": meta_significant_count,
            "not_meta_significant_rows": discovery_count - meta_significant_count,
        },
        "effect_summary": {
            "median_abs_beta_discovery_all": float(
                frame["beta_discovery"].abs().median()
            ),
            "median_abs_beta_meta_paired": float(
                paired_frame["beta_meta_ivw"].abs().median()
            ),
            "max_abs_difference_from_provided_beta_meta": provided_difference,
        },
    }
    frame = frame.loc[:, SOURCE_COLUMNS].sort_values(
        ["variant_tag", "phecode"], kind="stable"
    )
    return frame.reset_index(drop=True), audit


def plot_effect_meta(frame: pd.DataFrame) -> plt.Figure:
    """Draw paired discovery and computed-IVW meta-analysis effect violins."""
    style.apply_style()
    discovery = frame["beta_discovery"].to_numpy(dtype=float)
    paired_frame = frame.loc[frame["has_meta_effect"]]
    meta_effect = paired_frame["beta_meta_ivw"].to_numpy(dtype=float)
    discovery_significant = frame["meta_significant"].to_numpy(dtype=bool)
    paired_significant = paired_frame["meta_significant"].to_numpy(dtype=bool)
    n_discovery = len(frame)
    n_paired = len(paired_frame)
    n_significant = int(discovery_significant.sum())

    with mpl.rc_context(
        {
            "font.size": 21,
            "axes.labelsize": 30,
            "xtick.labelsize": 20,
            "ytick.labelsize": 21,
            "legend.fontsize": 20,
        }
    ):
        figure, axis = plt.subplots(figsize=(11, 8.5))
        parts = axis.violinplot(
            [discovery, meta_effect],
            positions=[0.0, 1.0],
            widths=0.62,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(
            parts["bodies"], (style.PALETTE["blue"], style.PALETTE["orange"])
        ):
            body.set_facecolor(color)
            body.set_edgecolor(style.PALETTE["black"])
            body.set_alpha(0.28)
            body.set_linewidth(1.7)

        paired_discovery = paired_frame["beta_discovery"].to_numpy(dtype=float)
        segments = np.stack(
            [
                np.column_stack([np.zeros(n_paired), paired_discovery]),
                np.column_stack([np.ones(n_paired), meta_effect]),
            ],
            axis=1,
        )
        axis.add_collection(
            LineCollection(
                segments[~paired_significant],
                colors="0.58",
                alpha=0.26,
                linewidths=1.0,
                zorder=2,
            )
        )
        if paired_significant.any():
            axis.add_collection(
                LineCollection(
                    segments[paired_significant],
                    colors=META_SIGNIFICANT_COLOR,
                    alpha=0.82,
                    linewidths=2.4,
                    zorder=3,
                )
            )

        axis.scatter(
            np.zeros(n_discovery)[~discovery_significant],
            discovery[~discovery_significant],
            s=34,
            color="0.58",
            edgecolor="0.25",
            linewidth=0.35,
            alpha=0.74,
            zorder=4,
        )
        axis.scatter(
            np.ones(n_paired)[~paired_significant],
            meta_effect[~paired_significant],
            s=34,
            color="0.58",
            edgecolor="0.25",
            linewidth=0.35,
            alpha=0.74,
            zorder=4,
        )
        if n_significant:
            axis.scatter(
                np.zeros(n_discovery)[discovery_significant],
                discovery[discovery_significant],
                s=68,
                color=META_SIGNIFICANT_COLOR,
                edgecolor=style.PALETTE["black"],
                linewidth=0.7,
                alpha=0.96,
                zorder=5,
            )
            axis.scatter(
                np.ones(n_paired)[paired_significant],
                meta_effect[paired_significant],
                s=68,
                color=META_SIGNIFICANT_COLOR,
                edgecolor=style.PALETTE["black"],
                linewidth=0.7,
                alpha=0.96,
                zorder=5,
            )

        for x, values in ((0.0, discovery), (1.0, meta_effect)):
            axis.hlines(
                float(np.median(values)),
                x - 0.24,
                x + 0.24,
                colors=style.PALETTE["black"],
                linewidth=3.0,
                zorder=6,
            )

        all_values = np.concatenate([discovery, meta_effect])
        span = float(np.ptp(all_values))
        padding = max(0.14 * span, 0.1)
        axis.set_ylim(float(np.min(all_values)) - padding, float(np.max(all_values)) + padding)
        axis.axhline(0, color="0.45", linewidth=1.3, linestyle=":", zorder=1)
        axis.set_xlim(-0.55, 1.55)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(
            [
                (
                    "Discovery effect\n"
                    f"median |β| = {np.median(np.abs(discovery)):.3f}"
                ),
                (
                    "IVW meta-analysis effect\n"
                    f"median |β| = {np.median(np.abs(meta_effect)):.3f}"
                ),
            ]
        )
        axis.set_ylabel(r"$\beta$")
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=META_SIGNIFICANT_COLOR,
                    linewidth=3,
                    marker="o",
                    markersize=9,
                    label=f"Meta-significant ({n_significant})",
                ),
                Line2D(
                    [0],
                    [0],
                    color="0.58",
                    linewidth=2,
                    marker="o",
                    markersize=8,
                    label=f"Not meta-significant ({n_discovery - n_significant})",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=2,
            columnspacing=1.2,
            handlelength=2.2,
            frameon=False,
        )
        axis.tick_params(axis="x", pad=12)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.84)
    return figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def render_panel(*, clean_run_data: bool = False) -> list[Path]:
    """Prepare, render, verify, and provenance-track Figure 4 panel C."""
    discovery, discovery_source = figure4_s9.load_source_data()
    meta = figure4_meta.load_meta_results(clean_run_data=clean_run_data)
    meta_path = Path(config.PHEWAS_META_RESULTS_TSV)
    meta_source = {
        "kind": "meta-analysis results",
        "path": str(meta_path.resolve()) if meta_path.exists() else None,
        "sha256": _sha256(meta_path) if meta_path.exists() else None,
    }
    frame, audit = prepare_effect_meta(discovery, meta)
    print(
        "[figure4c] parameters: "
        f"discovery_rows={len(frame)}, meta_effect_rows={int(frame['has_meta_effect'].sum())}, "
        f"meta_significant_rows={int(frame['meta_significant'].sum())}, "
        "method=fixed-effect inverse-variance weighting"
    )
    figure = plot_effect_meta(frame)
    stem = panel_image_path().with_suffix("")
    outputs: list[Path] = []
    for ext in config.FIGURE_FORMATS:
        path = stem.with_suffix(f".{ext}")
        figure.savefig(path, dpi=300, facecolor="white", bbox_inches=figure.bbox_inches)
        outputs.append(path)
        print(f"[figure4c] wrote: {path}")
    plt.close(figure)

    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise Figure3EffectMetaContractError(f"Invalid PNG signature: {png}")
    pdf_bytes = pdf.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-2048:]:
        raise Figure3EffectMetaContractError(f"Invalid PDF structure: {pdf}")

    analysis_dir = config.analysis_dir(FIGURE_ID)
    source_path = analysis_dir / "figure4c_discovery_meta_v6_source_data.tsv"
    audit_path = analysis_dir / "figure4c_discovery_meta_v6_audit.json"
    manifest_path = analysis_dir / "figure4c_discovery_meta_v6_manifest.json"
    frame.to_csv(source_path, sep="\t", index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "row_unit": audit["row_unit"],
        "discovery_source": discovery_source,
        "meta_source": meta_source,
        "parameters": {
            "expected_discovery_count": EXPECTED_DISCOVERY_COUNT,
            "expected_meta_effect_count": EXPECTED_META_EFFECT_COUNT,
            "expected_meta_significant_count": EXPECTED_META_SIGNIFICANT_COUNT,
            "meta_method": "fixed-effect inverse-variance weighting",
        },
        "output_sha256": {
            str(path.resolve()): _sha256(path)
            for path in (*outputs, source_path, audit_path)
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[figure4c] manifest: {manifest_path}")
    return outputs
