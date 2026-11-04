"""Figure 1 reference PCA panels comparing cohort membership and centroids."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable

from .common import samples
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# Workbench can inherit an inline backend that is unavailable in batch jobs.
os.environ["MPLBACKEND"] = "Agg"
plt = import_module("matplotlib.pyplot")


EXPECTED_PCS = 20
EXPECTED_AOU = 1025
ID_CANDIDATES = ("person_id", "sample_id", "research_id", "s", "iid")
AOU_COLOR = "#CC79A7"          # colorblind-safe pink
TEN_K_COLOR = "#0072B2"        # colorblind-safe blue
THIRTY_K_COLOR = "#E69F00"     # colorblind-safe orange
S10_CONTRACT = "figure_s10_pca_cohort_comparison_v1"


def _compression(path: Path) -> str | None:
    return "gzip" if path.name.lower().endswith((".gz", ".bgz")) else None


def _separator(path: Path) -> str:
    name = path.name.lower()
    while name.endswith((".gz", ".bgz")):
        name = name.rsplit(".", 1)[0]
    return "," if name.endswith(".csv") else "\t"


def _open_text(path: Path):
    if _compression(path) == "gzip":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def _normalize_ids(values: Iterable) -> pd.Series:
    ids = pd.Series(values, dtype="string").str.strip()
    # Pandas or an upstream output can turn numeric IDs into strings like 1.0.
    return ids.str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def _id_column(columns: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    return next((lookup[name] for name in ID_CANDIDATES if name in lookup), None)


def _read_ids(path: Path) -> set[str]:
    """Read either a headered cohort table or a headerless one-column list."""
    path = path.resolve()
    with _open_text(path) as handle:
        first = next((line.rstrip("\r\n") for line in handle if line.strip()), "")
    if not first:
        raise ValueError(f"ID file is empty: {path}")

    separator = _separator(path)
    first_fields = [field.strip().strip('"').lower() for field in first.split(separator)]
    has_header = any(field in ID_CANDIDATES for field in first_fields)
    frame = pd.read_csv(
        path,
        sep=separator,
        compression=_compression(path),
        dtype="string",
        header=0 if has_header else None,
        low_memory=False,
    )
    id_column = _id_column(frame.columns) if has_header else frame.columns[0]
    if id_column is None:
        raise ValueError(
            f"Could not find an ID column in {path}; columns={list(frame.columns)[:10]}"
        )
    ids = set(_normalize_ids(frame[id_column]).dropna())
    ids.discard("")
    if not ids:
        raise ValueError(f"No sample IDs found in {path}")
    return ids


def canonical_aou_ids(ids: set[str]) -> set[str]:
    """Return the README-defined 1,025-person AoU analysis set."""
    canonical = set(ids).difference(samples.load_excluded_sample_ids())
    if len(canonical) != EXPECTED_AOU:
        raise ValueError(
            "The canonical AoU list must contain exactly "
            f"{EXPECTED_AOU:,} unique IDs after cohort exclusions; found {len(canonical):,}"
        )
    return canonical


def _split_score_array(values: pd.Series) -> pd.DataFrame:
    cleaned = (
        values.astype("string")
        .str.strip()
        .str.strip("[]()")
        .str.replace(r"\s+", ",", regex=True)
        .str.replace(r",+", ",", regex=True)
        .str.strip(",")
    )
    return cleaned.str.split(",", expand=True)


def _wide_pc_columns(columns: Iterable[str], expected_pcs: int) -> list[str] | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    zero_indexed_scores = "scores_0" in lookup
    zero_indexed_features = "pca_features_0" in lookup
    found: list[str] = []
    for number in range(1, expected_pcs + 1):
        candidates = [
            f"pc{number}",
            f"pca{number}",
            f"component_{number}",
            f"pca_feature_{number}",
        ]
        candidates.append(
            f"scores_{number - 1}" if zero_indexed_scores else f"scores_{number}"
        )
        candidates.append(
            f"pca_features_{number - 1}"
            if zero_indexed_features
            else f"pca_features_{number}"
        )
        column = next((lookup[name] for name in candidates if name in lookup), None)
        if column is None:
            return None
        found.append(column)
    return found


def load_new_pca(path: Path, expected_pcs: int = EXPECTED_PCS) -> pd.DataFrame:
    """Return unique ``person_id, PC1, PC2, PC3`` rows from a >=20-PC file."""
    path = path.resolve()
    raw = pd.read_csv(
        path,
        sep=_separator(path),
        compression=_compression(path),
        dtype="string",
        low_memory=False,
    )
    id_column = _id_column(raw.columns)
    if id_column is None:
        raise ValueError(
            f"Could not find a sample ID column in {path}; "
            f"columns={list(raw.columns)[:10]}"
        )

    column_lookup = {str(column).strip().lower(): str(column) for column in raw.columns}
    array_column = next(
        (column_lookup[name] for name in ("pca_features", "scores")
         if name in column_lookup),
        None,
    )
    if array_column is not None:
        scores = _split_score_array(raw[array_column])
        if scores.shape[1] < expected_pcs:
            raise ValueError(
                f"{path} has only {scores.shape[1]} PCA values in {array_column!r}; "
                f"the new {expected_pcs}-PC input is required"
            )
        pc1 = pd.to_numeric(scores.iloc[:, 0], errors="coerce")
        pc2 = pd.to_numeric(scores.iloc[:, 1], errors="coerce")
        pc3 = pd.to_numeric(scores.iloc[:, 2], errors="coerce")
    else:
        pc_columns = _wide_pc_columns(raw.columns, expected_pcs)
        if pc_columns is None:
            raise ValueError(
                f"Could not identify {expected_pcs} PC columns in {path}; "
                "expected a scores/pca_features array or PC1..PC20-like columns"
            )
        pc1 = pd.to_numeric(raw[pc_columns[0]], errors="coerce")
        pc2 = pd.to_numeric(raw[pc_columns[1]], errors="coerce")
        pc3 = pd.to_numeric(raw[pc_columns[2]], errors="coerce")

    pca = pd.DataFrame(
        {
            "person_id": _normalize_ids(raw[id_column]),
            "PC1": pc1,
            "PC2": pc2,
            "PC3": pc3,
        }
    ).dropna(subset=["person_id", "PC1", "PC2", "PC3"])
    duplicate_ids = pca.loc[pca["person_id"].duplicated(keep=False), "person_id"]
    if not duplicate_ids.empty:
        raise ValueError("PCA input contains duplicate sample IDs; inspect the source in Workbench.")
    if pca.empty:
        raise ValueError(f"No complete PC1/PC2/PC3 rows found in {path}")
    return pca.reset_index(drop=True)


def select_cohorts(
    pca: pd.DataFrame,
    *,
    discovery_ids: set[str],
    replication_ids: set[str] | None = None,
    replication_exclude_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Select disjoint discovery/replication sets in one PCA coordinate system."""
    numeric_aou = pca[pca["person_id"].str.fullmatch(r"\d+").fillna(False)].copy()
    discovery = numeric_aou[numeric_aou["person_id"].isin(discovery_ids)].copy()
    excluded = set(discovery_ids)
    if replication_exclude_ids is not None:
        excluded.update(replication_exclude_ids)
    if replication_ids is None:
        replication_target = set(numeric_aou["person_id"]).difference(excluded)
    else:
        # Earlier cohorts win if an explicit table accidentally overlaps them.
        replication_target = replication_ids.difference(excluded)
    replication = numeric_aou[
        numeric_aou["person_id"].isin(replication_target)
    ].copy()
    if discovery.empty:
        raise ValueError("None of the 10k discovery IDs were found in the PCA input")
    if replication.empty:
        raise ValueError("No 30k replication IDs were found in the PCA input")

    counts = {
        "discovery_requested": len(discovery_ids),
        "discovery_plotted": discovery["person_id"].nunique(),
        "discovery_missing_pca": len(
            discovery_ids.difference(set(discovery["person_id"]))
        ),
        "replication_requested": len(replication_target),
        "replication_plotted": replication["person_id"].nunique(),
        "replication_missing_pca": len(
            replication_target.difference(set(replication["person_id"]))
        ),
        "overlap_plotted": len(
            set(discovery["person_id"]).intersection(replication["person_id"])
        ),
    }
    return (
        discovery.reset_index(drop=True),
        replication.reset_index(drop=True),
        counts,
    )


def _cohort_stats(frame: pd.DataFrame, x_pc: str, y_pc: str) -> dict[str, float]:
    """Return unweighted coordinate means and distances for plotted rows.

    ``load_new_pca`` has already removed incomplete PC rows, so the centroid is
    the ordinary arithmetic mean of the same participants visible in the
    panel. It is not weighted by ancestry, cohort size, or distance.
    """
    centroid = frame[[x_pc, y_pc]].mean().to_numpy(float)
    radii = np.linalg.norm(frame[[x_pc, y_pc]].to_numpy(float) - centroid, axis=1)
    return {
        "n": len(frame),
        "x_centroid": float(centroid[0]),
        "y_centroid": float(centroid[1]),
        "mean_distance_to_centroid": float(np.mean(radii)),
        "median_distance_to_centroid": float(np.median(radii)),
        "rms_distance_to_centroid": float(np.sqrt(np.mean(np.square(radii)))),
    }


def _limits(
    first: pd.DataFrame,
    second: pd.DataFrame,
    x_pc: str,
    y_pc: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    both = pd.concat([first[[x_pc, y_pc]], second[[x_pc, y_pc]]])
    x_min, x_max = float(both[x_pc].min()), float(both[x_pc].max())
    y_min, y_max = float(both[y_pc].min()), float(both[y_pc].max())
    x_pad = max((x_max - x_min) * 0.04, 1e-6)
    y_pad = max((y_max - y_min) * 0.04, 1e-6)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    color: str,
    label: str,
    size: float,
    alpha: float,
    zorder: int,
    x_pc: str,
    y_pc: str,
) -> None:
    ax.scatter(
        frame[x_pc],
        frame[y_pc],
        s=size,
        color=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        label=label,
        zorder=zorder,
    )


def _centroid(
    ax: plt.Axes,
    stats: dict[str, float],
    *,
    color: str,
    label: str,
    zorder: int = 8,
) -> None:
    ax.scatter(
        [stats["x_centroid"]],
        [stats["y_centroid"]],
        marker="X",
        s=155,
        color=color,
        edgecolor="black",
        linewidth=0.9,
        label=label,
        zorder=zorder,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Place a publication panel label in the upper-left margin."""
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def render_figure_s10(
    discovery: pd.DataFrame,
    replication: pd.DataFrame,
    *,
    output_stem: Path,
    summary_path: Path | None = None,
    show: bool = False,
) -> tuple[list[Path], Path, dict[str, dict[str, float]]]:
    """Render the two-panel PC1/PC2 Figure S10 and centroid source table."""
    discovery_stats = _cohort_stats(discovery, "PC1", "PC2")
    replication_stats = _cohort_stats(replication, "PC1", "PC2")
    xlim, ylim = _limits(discovery, replication, "PC1", "PC2")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), squeeze=False)
    panels = (
        (
            "A",
            axes[0, 0],
            discovery,
            discovery_stats,
            TEN_K_COLOR,
            "Discovery 10k set",
            "Discovery 10k centroid",
            6,
            0.28,
        ),
        (
            "B",
            axes[0, 1],
            replication,
            replication_stats,
            THIRTY_K_COLOR,
            "Replication 30k set",
            "Replication 30k centroid",
            5,
            0.18,
        ),
    )
    for panel, ax, frame, stats, color, cohort_label, centroid_label, size, alpha in panels:
        _scatter(
            ax,
            frame,
            color=color,
            label=cohort_label,
            size=size,
            alpha=alpha,
            zorder=3,
            x_pc="PC1",
            y_pc="PC2",
        )
        _centroid(ax, stats, color=color, label=centroid_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if panel == "A" else "")
        ax.legend(loc="lower left", markerscale=1.15)
        _panel_label(ax, panel)

    # Fixed margins keep A/B outside the axes without relying on a backend-
    # dependent tight-layout adjustment. Both panels retain identical limits.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.90, wspace=0.16)

    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for extension in ("png", "pdf"):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(path)
    if show:
        plt.show()
    plt.close(fig)

    summary_path = (
        summary_path.resolve()
        if summary_path is not None
        else output_stem.with_name(output_stem.name + ".centroids.tsv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        (
            {
                "panel": "A",
                "cohort": "discovery_10k",
                "n": int(discovery_stats["n"]),
                "pc1_centroid": discovery_stats["x_centroid"],
                "pc2_centroid": discovery_stats["y_centroid"],
            },
            {
                "panel": "B",
                "cohort": "replication_30k",
                "n": int(replication_stats["n"]),
                "pc1_centroid": replication_stats["x_centroid"],
                "pc2_centroid": replication_stats["y_centroid"],
            },
        )
    )
    summary["centroid_definition"] = (
        "unweighted arithmetic mean of PC1 and PC2 across plotted participants"
    )
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.10g")
    return (
        figure_paths,
        summary_path,
        {"discovery_10k": discovery_stats, "replication_30k": replication_stats},
    )


def render_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_label: str,
    second_label: str,
    first_summary_name: str,
    second_summary_name: str,
    figure_title: str,
    first_color: str,
    second_color: str,
    output_stem: Path,
    show: bool = False,
) -> tuple[list[Path], Path, dict[str, float]]:
    """Render PNG/PDF and write a centroid/radius summary TSV."""
    pairs = (("PC1", "PC2"), ("PC2", "PC3"))
    pair_stats: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    centroid_distances: dict[str, float] = {}
    for x_pc, y_pc in pairs:
        first_stats = _cohort_stats(first, x_pc, y_pc)
        second_stats = _cohort_stats(second, x_pc, y_pc)
        pair_stats[(x_pc, y_pc)] = {
            "first": first_stats,
            "second": second_stats,
        }
        first_centroid = np.array(
            [first_stats["x_centroid"], first_stats["y_centroid"]]
        )
        second_centroid = np.array(
            [second_stats["x_centroid"], second_stats["y_centroid"]]
        )
        centroid_distances[f"{x_pc.lower()}_{y_pc.lower()}"] = float(
            np.linalg.norm(first_centroid - second_centroid)
        )

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), squeeze=False)
    column_titles = (
        first_label,
        second_label,
        f"{first_label} and {second_label}",
    )
    first_centroid_label = first_label.removesuffix(" set") + " centroid"
    second_centroid_label = second_label.removesuffix(" set") + " centroid"

    for row, (x_pc, y_pc) in enumerate(pairs):
        row_axes = axes[row]
        first_stats = pair_stats[(x_pc, y_pc)]["first"]
        second_stats = pair_stats[(x_pc, y_pc)]["second"]
        first_centroid = np.array(
            [first_stats["x_centroid"], first_stats["y_centroid"]]
        )
        second_centroid = np.array(
            [second_stats["x_centroid"], second_stats["y_centroid"]]
        )
        xlim, ylim = _limits(first, second, x_pc, y_pc)

        _scatter(
            row_axes[0], first, color=first_color,
            label=first_label, size=6, alpha=0.28, zorder=3,
            x_pc=x_pc, y_pc=y_pc,
        )
        _centroid(
            row_axes[0], first_stats, color=first_color,
            label=first_centroid_label,
        )
        row_axes[0].legend(loc="best")

        _scatter(
            row_axes[1], second, color=second_color,
            label=second_label, size=5, alpha=0.18, zorder=2,
            x_pc=x_pc, y_pc=y_pc,
        )
        _centroid(
            row_axes[1], second_stats, color=second_color,
            label=second_centroid_label,
        )
        row_axes[1].legend(loc="best")

        _scatter(
            row_axes[2], second, color=second_color,
            label=second_label, size=5, alpha=0.13, zorder=1,
            x_pc=x_pc, y_pc=y_pc,
        )
        _scatter(
            row_axes[2], first, color=first_color,
            label=first_label, size=6, alpha=0.28, zorder=3,
            x_pc=x_pc, y_pc=y_pc,
        )
        _centroid(
            row_axes[2], second_stats, color=second_color,
            label=second_centroid_label,
        )
        _centroid(
            row_axes[2], first_stats, color=first_color,
            label=first_centroid_label, zorder=9,
        )
        row_axes[2].plot(
            [second_centroid[0], first_centroid[0]],
            [second_centroid[1], first_centroid[1]],
            color="black",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=7,
        )
        row_axes[2].legend(loc="best", fontsize=9, markerscale=1.2)

        for column, ax in enumerate(row_axes):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(x_pc)
            ax.set_ylabel(y_pc if column == 0 else "")
            ax.set_title(f"{column_titles[column]}: {x_pc} vs {y_pc}")
            panel_index = row * 3 + column
            ax.text(
                -0.10, 1.04, chr(ord("A") + panel_index), transform=ax.transAxes,
                fontsize=17, fontweight="bold", ha="right", va="top",
            )
    fig.suptitle(figure_title, y=1.01)
    fig.tight_layout()

    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for extension in ("png", "pdf"):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(path)
    if show:
        plt.show()
    plt.close(fig)

    summary_path = output_stem.with_name(output_stem.name + ".centroids.tsv")
    rows = []
    for cohort, frame, stats_key in (
        (first_summary_name, first, "first"),
        (second_summary_name, second, "second"),
    ):
        stats_12 = pair_stats[("PC1", "PC2")][stats_key]
        stats_23 = pair_stats[("PC2", "PC3")][stats_key]
        rows.append(
            {
                "cohort": cohort,
                "n": len(frame),
                "pc1_centroid": stats_12["x_centroid"],
                "pc2_centroid": stats_12["y_centroid"],
                "pc3_centroid": stats_23["y_centroid"],
                "mean_distance_to_centroid": stats_12["mean_distance_to_centroid"],
                "median_distance_to_centroid": stats_12["median_distance_to_centroid"],
                "rms_distance_to_centroid": stats_12["rms_distance_to_centroid"],
                "mean_distance_to_centroid_pc2_pc3": stats_23[
                    "mean_distance_to_centroid"
                ],
                "median_distance_to_centroid_pc2_pc3": stats_23[
                    "median_distance_to_centroid"
                ],
                "rms_distance_to_centroid_pc2_pc3": stats_23[
                    "rms_distance_to_centroid"
                ],
            }
        )
    summary = pd.DataFrame(rows)
    summary["centroid_distance_pc1_pc2"] = centroid_distances["pc1_pc2"]
    summary["centroid_distance_pc2_pc3"] = centroid_distances["pc2_pc3"]
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.10g")
    return figure_paths, summary_path, centroid_distances


def render_three_cohorts(
    aou: pd.DataFrame,
    ten_k: pd.DataFrame,
    thirty_k: pd.DataFrame,
    *,
    output_stem: Path,
    show: bool = False,
) -> tuple[list[Path], Path]:
    """Render a single-row PC1/PC2 figure with one panel per cohort."""
    cohorts = (
        ("AoU 1k set", "AoU 1k centroid", "aou_1k", aou, AOU_COLOR, 6, 0.28),
        ("10k set", "10k centroid", "discovery_10k", ten_k, TEN_K_COLOR, 5, 0.18),
        (
            "30k set",
            "30k centroid",
            "replication_30k",
            thirty_k,
            THIRTY_K_COLOR,
            4,
            0.13,
        ),
    )
    combined = pd.concat([entry[3][["PC1", "PC2"]] for entry in cohorts])
    x_min, x_max = float(combined["PC1"].min()), float(combined["PC1"].max())
    y_min, y_max = float(combined["PC2"].min()), float(combined["PC2"].max())
    x_pad = max((x_max - x_min) * 0.04, 1e-6)
    y_pad = max((y_max - y_min) * 0.04, 1e-6)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), squeeze=False)
    summary_rows: list[dict[str, float | int | str]] = []
    for panel, (
        label,
        centroid_label,
        summary_name,
        frame,
        color,
        size,
        alpha,
    ) in enumerate(cohorts):
        ax = axes[0, panel]
        stats_12 = _cohort_stats(frame, "PC1", "PC2")
        stats_23 = _cohort_stats(frame, "PC2", "PC3")
        _scatter(
            ax,
            frame,
            color=color,
            label=label,
            size=size,
            alpha=alpha,
            zorder=3,
            x_pc="PC1",
            y_pc="PC2",
        )
        _centroid(ax, stats_12, color=color, label=centroid_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if panel == 0 else "")
        ax.set_title(f"{label}: PC1 vs PC2")
        ax.legend(loc="best")
        ax.text(
            -0.10,
            1.04,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            ha="right",
            va="top",
        )
        summary_rows.append(
            {
                "cohort": summary_name,
                "n": len(frame),
                "pc1_centroid": stats_12["x_centroid"],
                "pc2_centroid": stats_12["y_centroid"],
                "pc3_centroid": stats_23["y_centroid"],
            }
        )

    fig.suptitle("AoU 1k, 10k, and 30k sets in the new 20-PC PCA", y=1.02)
    fig.tight_layout()
    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for extension in ("png", "pdf"):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(path)
    if show:
        plt.show()
    plt.close(fig)

    summary_path = output_stem.with_name(output_stem.name + ".centroids.tsv")
    pd.DataFrame(summary_rows).to_csv(
        summary_path, sep="\t", index=False, float_format="%.10g"
    )
    return figure_paths, summary_path


def _stage(remote: str, local: Path, *, refresh: bool, gcs_module) -> Path:
    local = local.resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    if refresh and local.exists():
        local.unlink()
    if local.exists():
        return local
    result = gcs_module.gcs_cp(remote, local)
    if result.returncode != 0 or not local.exists():
        raise FileNotFoundError(
            f"Could not stage {remote} to {local}: {result.stderr.strip()}"
        )
    return local


def _repo_defaults(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None, Path]:
    # When this helper lives outside the checkout, Python otherwise adds the
    # helper's directory rather than the current repository root to sys.path.
    repo_root = next(
        (
            candidate
            for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (candidate / "scripts" / "common" / "config.py").exists()
        ),
        None,
    )
    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts.common import config, gcs, samples
    except ImportError as exc:
        raise RuntimeError(
            "Default inputs require running from the AllofUsPhase1Companion "
            "repository root. Alternatively, pass --pca, --aou-ids, "
            "--discovery-ids, and --output-stem; --replication-ids is optional."
        ) from exc

    cache = config.analysis_dir("figure1") / "pca_10k_vs_30k_inputs"
    pca_path = args.pca or _stage(
        config.phewas_sv_path("pca_scores_afr.tsv.bgz"),
        cache / "pca_scores_afr.tsv.bgz",
        refresh=args.refresh_inputs,
        gcs_module=gcs,
    )
    aou_path = args.aou_ids or _stage(
        samples.AOU_SAMPLE_LIST_GS,
        cache / "AOU_sample_list.txt",
        refresh=args.refresh_inputs,
        gcs_module=gcs,
    )
    discovery_path = args.discovery_ids or _stage(
        config.phewas_sv_path("covariates_base.tsv"),
        cache / "covariates_base.tsv",
        refresh=args.refresh_inputs,
        gcs_module=gcs,
    )
    replication_path = (
        Path(args.replication_ids) if args.replication_ids is not None else None
    )
    output_stem = args.output_stem or (
        config.figure_dir("figure1") / "Figure_1_PCA"
    )
    return (
        Path(pca_path),
        Path(aou_path),
        Path(discovery_path),
        replication_path,
        Path(output_stem),
    )


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None, Path]:
    complete_local = all(
        path is not None
        for path in (args.pca, args.aou_ids, args.discovery_ids, args.output_stem)
    )
    if complete_local:
        return (
            Path(args.pca),
            Path(args.aou_ids),
            Path(args.discovery_ids),
            Path(args.replication_ids) if args.replication_ids else None,
            Path(args.output_stem),
        )
    return _repo_defaults(args)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _valid_figure(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(8)
    if path.suffix.lower() == ".png":
        return prefix == b"\x89PNG\r\n\x1a\n"
    if path.suffix.lower() == ".pdf":
        return prefix.startswith(b"%PDF-")
    return False


def _manifest_reusable(
    manifest_path: Path,
    *,
    cache_key: dict[str, object],
    expected_outputs: list[Path],
) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    if manifest.get("cache_key") != cache_key:
        return False
    recorded = manifest.get("outputs", {})
    for path in expected_outputs:
        if path.suffix.lower() in {".png", ".pdf"}:
            valid = _valid_figure(path)
        else:
            valid = path.is_file() and path.stat().st_size > 0
        if not valid or recorded.get(str(path.resolve())) != _sha256(path):
            return False
    return True


def _s10_default_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None, Path, Path, Path, Path]:
    from scripts.common import config, gcs, samples

    analysis_dir = config.ANALYSIS_DIR / "supplementary" / "s10_pca"
    input_dir = analysis_dir / "inputs"
    figure_dir = config.figure_dir("figure1") / "pca"
    input_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    def resolve_input(explicit: Path | None, remote: str, filename: str) -> Path:
        if explicit is not None:
            return explicit.resolve()
        local = input_dir / filename
        if local.exists() and not args.refresh_inputs:
            return local.resolve()
        if not args.download_missing:
            raise FileNotFoundError(
                f"Missing cached input {local}. Pass --download-missing on "
                "Workbench or supply all three local input paths."
            )
        return _stage(remote, local, refresh=args.refresh_inputs, gcs_module=gcs)

    pca_path = resolve_input(
        args.pca,
        config.phewas_sv_path("pca_scores_afr.tsv.bgz"),
        "pca_scores_afr.tsv.bgz",
    )
    aou_path = resolve_input(
        args.aou_ids,
        samples.AOU_SAMPLE_LIST_GS,
        "AOU_sample_list.txt",
    )
    discovery_path = resolve_input(
        args.discovery_ids,
        config.phewas_sv_path("covariates_base.tsv"),
        "covariates_base.tsv",
    )
    replication_path = (
        args.replication_ids.resolve() if args.replication_ids is not None else None
    )
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else figure_dir / "Figure_S10_PCA_10k_vs_30k"
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else analysis_dir / "Figure_S10_PCA_10k_vs_30k.centroids.tsv"
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else analysis_dir / "Figure_S10_PCA_10k_vs_30k.manifest.json"
    )
    return (
        pca_path,
        aou_path,
        discovery_path,
        replication_path,
        output_stem,
        summary_path,
        manifest_path,
        config.REPO_ROOT,
    )


def _s10_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pca",
        type=Path,
        default=os.environ.get("AOU_S10_PCA"),
        help="New >=20-PC score TSV/BGZ",
    )
    parser.add_argument(
        "--aou-ids",
        type=Path,
        default=os.environ.get("AOU_S10_AOU_IDS"),
        help="Canonical AoU 1k list used to exclude Phase 1 participants",
    )
    parser.add_argument(
        "--discovery-ids",
        type=Path,
        default=os.environ.get("AOU_S10_DISCOVERY_IDS"),
        help="10k discovery cohort table/sample list",
    )
    parser.add_argument(
        "--replication-ids",
        type=Path,
        default=os.environ.get("AOU_S10_REPLICATION_IDS"),
        help=(
            "Optional exact replication list; default: all other numeric AoU "
            "IDs in the PCA after excluding the 1k and discovery sets"
        ),
    )
    parser.add_argument("--output-stem", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--expected-pcs",
        type=int,
        default=EXPECTED_PCS,
        help=f"Require this many PCs in the input (default: {EXPECTED_PCS})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Recorded workflow thread budget (default: 4)",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing default Workbench inputs into the analysis cache",
    )
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="Redownload default Workbench inputs; requires --download-missing",
    )
    parser.add_argument("--force", action="store_true", help="Ignore a valid cache")
    parser.add_argument("--show", action="store_true", help="Also call plt.show()")
    return parser


def figure_s10_main(argv: list[str] | None = None) -> list[Path]:
    """Run the Figure S10 input, cohort, cache, and rendering contract."""
    args = _s10_parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.refresh_inputs and not args.download_missing:
        raise ValueError("--refresh-inputs requires --download-missing")
    explicit_inputs = (args.pca, args.aou_ids, args.discovery_ids)
    if any(path is not None for path in explicit_inputs) and not all(
        path is not None for path in explicit_inputs
    ):
        raise ValueError(
            "Supply --pca, --aou-ids, and --discovery-ids together, or omit all "
            "three to use the Workbench cache."
        )

    (
        pca_path,
        aou_path,
        discovery_path,
        replication_path,
        output_stem,
        summary_path,
        manifest_path,
        repo_root,
    ) = _s10_default_paths(args)
    for label, path in (
        ("PCA", pca_path),
        ("AoU 1k IDs", aou_path),
        ("Discovery IDs", discovery_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} input is missing or empty: {path}")
    if replication_path is not None and (
        not replication_path.is_file() or replication_path.stat().st_size == 0
    ):
        raise FileNotFoundError(
            f"Replication ID input is missing or empty: {replication_path}"
        )

    input_hashes = {
        "pca": _sha256(pca_path),
        "aou_ids": _sha256(aou_path),
        "discovery_ids": _sha256(discovery_path),
        "replication_ids": _sha256(replication_path) if replication_path else None,
    }
    cache_key: dict[str, object] = {
        "contract": S10_CONTRACT,
        "input_sha256": input_hashes,
        "expected_pcs": args.expected_pcs,
        "cohort_rule": (
            "discovery IDs excluding AoU 1k; replication IDs or remaining numeric "
            "PCA IDs excluding AoU 1k and discovery"
        ),
        "centroid_rule": "unweighted coordinate-wise arithmetic mean of PC1 and PC2",
    }
    expected_outputs = [
        output_stem.with_suffix(".png").resolve(),
        output_stem.with_suffix(".pdf").resolve(),
        summary_path.resolve(),
    ]
    if not args.force and _manifest_reusable(
        manifest_path, cache_key=cache_key, expected_outputs=expected_outputs
    ):
        print(f"Reusing valid Figure S10 cache: {manifest_path.resolve()}")
        return [*expected_outputs, manifest_path.resolve()]

    print(f"Figure S10 contract: {S10_CONTRACT}")
    print(f"Repository: {repo_root.resolve()}")
    print(f"Git revision: {_git_revision(repo_root) or 'unavailable'}")
    print(f"Thread budget: {args.threads}")
    for name, path in (
        ("PCA", pca_path),
        ("AoU 1k IDs", aou_path),
        ("Discovery IDs", discovery_path),
        ("Replication IDs", replication_path),
    ):
        if path is not None:
            print(f"{name}: {path.resolve()} sha256={_sha256(path)}")

    pca = load_new_pca(pca_path, expected_pcs=args.expected_pcs)
    aou_ids = canonical_aou_ids(_read_ids(aou_path))
    discovery_ids = _read_ids(discovery_path).difference(
        aou_ids.union(samples.load_excluded_sample_ids())
    )
    if not discovery_ids:
        raise ValueError("No discovery IDs remain after excluding the AoU 1k set")
    replication_ids = (
        _read_ids(replication_path) if replication_path is not None else None
    )
    discovery, replication, counts = select_cohorts(
        pca,
        discovery_ids=discovery_ids,
        replication_ids=replication_ids,
        replication_exclude_ids=aou_ids.union(samples.load_excluded_sample_ids()),
    )
    figure_paths, written_summary, centroid_stats = render_figure_s10(
        discovery,
        replication,
        output_stem=output_stem,
        summary_path=summary_path,
        show=args.show,
    )
    outputs = [*figure_paths, written_summary]
    for path in figure_paths:
        if not _valid_figure(path):
            raise RuntimeError(f"Generated figure failed integrity checks: {path}")
    if not written_summary.is_file() or written_summary.stat().st_size == 0:
        raise RuntimeError(f"Generated centroid table is missing or empty: {written_summary}")

    manifest = {
        "contract": S10_CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(repo_root),
        "thread_budget": args.threads,
        "paths": {
            "pca": str(pca_path.resolve()),
            "aou_ids": str(aou_path.resolve()),
            "discovery_ids": str(discovery_path.resolve()),
            "replication_ids": (
                str(replication_path.resolve()) if replication_path is not None else None
            ),
        },
        "cache_key": cache_key,
        "cohort_counts": counts,
        "centroids": centroid_stats,
        "outputs": {str(path.resolve()): _sha256(path) for path in outputs},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Figure S10 rows: "
        f"Discovery 10k={counts['discovery_plotted']:,}/"
        f"{counts['discovery_requested']:,}; "
        f"Replication 30k={counts['replication_plotted']:,}/"
        f"{counts['replication_requested']:,}; "
        f"overlap={counts['overlap_plotted']:,}"
    )
    for cohort, stats in centroid_stats.items():
        print(
            f"{cohort} centroid: PC1={stats['x_centroid']:.10g}; "
            f"PC2={stats['y_centroid']:.10g}; n={int(stats['n']):,}"
        )
    for path in (*outputs, manifest_path):
        print(f"Wrote {path.resolve()}")
    return [*outputs, manifest_path.resolve()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", type=Path, help="New >=20-PC score TSV/BGZ")
    parser.add_argument(
        "--aou-ids", "--one-k-ids", dest="aou_ids",
        type=Path,
        help="Canonical AoU 1k sample list/table",
    )
    parser.add_argument(
        "--discovery-ids", "--ten-k-ids", dest="discovery_ids",
        type=Path,
        help="10k discovery cohort table/sample list",
    )
    parser.add_argument(
        "--replication-ids",
        type=Path,
        help=(
            "Optional exact 30k replication cohort table/sample list; default: "
            "all numeric AoU PCA IDs outside the 1k and 10k sets"
        ),
    )
    parser.add_argument(
        "--expected-pcs", type=int, default=EXPECTED_PCS,
        help=f"Require this many PCs in the input (default: {EXPECTED_PCS})",
    )
    parser.add_argument(
        "--output-stem", type=Path,
        help=(
            "Common output prefix; appends _1k_vs_10k, _10k_vs_30k, and "
            "_1k_10k_30k and writes PNG, PDF, and centroid TSV for each figure"
        ),
    )
    parser.add_argument(
        "--refresh-inputs", action="store_true",
        help="Restage the default cached Workbench inputs",
    )
    parser.add_argument("--show", action="store_true", help="Also call plt.show()")
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = _parser().parse_args(argv)
    (
        pca_path,
        aou_path,
        discovery_path,
        replication_path,
        output_stem,
    ) = _resolve_paths(args)

    pca = load_new_pca(pca_path, expected_pcs=args.expected_pcs)
    aou_ids = canonical_aou_ids(_read_ids(aou_path))
    # Keep the three plotted cohorts disjoint even if an input table contains
    # an accidental cross-cohort duplicate.
    discovery_ids = _read_ids(discovery_path).difference(
        aou_ids.union(samples.load_excluded_sample_ids())
    )
    if not discovery_ids:
        raise ValueError("No 10k discovery IDs remain after excluding the AoU 1k set")
    replication_ids = (
        _read_ids(replication_path) if replication_path is not None else None
    )

    aou, ten_k, counts_1k_10k = select_cohorts(
        pca,
        discovery_ids=aou_ids,
        replication_ids=discovery_ids,
    )
    discovery, replication, counts_10k_30k = select_cohorts(
        pca,
        discovery_ids=discovery_ids,
        replication_ids=replication_ids,
        replication_exclude_ids=aou_ids.union(samples.load_excluded_sample_ids()),
    )

    output_stem = output_stem.resolve()
    stem_1k_10k = output_stem.with_name(output_stem.name + "_1k_vs_10k")
    stem_10k_30k = output_stem.with_name(output_stem.name + "_10k_vs_30k")
    stem_three = output_stem.with_name(output_stem.name + "_1k_10k_30k")
    paths_1k_10k, summary_1k_10k, distances_1k_10k = render_comparison(
        aou,
        ten_k,
        first_label="AoU 1k set",
        second_label="Replication 10k set",
        first_summary_name="aou_1k",
        second_summary_name="replication_10k",
        figure_title="AoU 1k set versus Replication 10k set in the new 20-PC PCA",
        first_color=AOU_COLOR,
        second_color=TEN_K_COLOR,
        output_stem=stem_1k_10k,
        show=args.show,
    )
    paths_10k_30k, summary_10k_30k, distances_10k_30k = render_comparison(
        discovery,
        replication,
        first_label="Discovery 10k set",
        second_label="Replication 30k set",
        first_summary_name="discovery_10k",
        second_summary_name="replication_30k",
        figure_title=(
            "Discovery 10k set versus Replication 30k set in the new 20-PC PCA"
        ),
        first_color=TEN_K_COLOR,
        second_color=THIRTY_K_COLOR,
        output_stem=stem_10k_30k,
        show=args.show,
    )
    paths_three, summary_three = render_three_cohorts(
        aou,
        discovery,
        replication,
        output_stem=stem_three,
        show=args.show,
    )

    print(
        "1k vs 10k PCA rows: "
        f"AoU 1k={counts_1k_10k['discovery_plotted']:,}/"
        f"{counts_1k_10k['discovery_requested']:,} "
        f"({counts_1k_10k['discovery_missing_pca']:,} missing PCA); "
        f"Replication 10k={counts_1k_10k['replication_plotted']:,}/"
        f"{counts_1k_10k['replication_requested']:,} "
        f"({counts_1k_10k['replication_missing_pca']:,} missing PCA); "
        f"overlap={counts_1k_10k['overlap_plotted']:,}"
    )
    print(
        "10k vs 30k PCA rows: "
        f"Discovery 10k={counts_10k_30k['discovery_plotted']:,}/"
        f"{counts_10k_30k['discovery_requested']:,} "
        f"({counts_10k_30k['discovery_missing_pca']:,} missing PCA); "
        f"Replication 30k={counts_10k_30k['replication_plotted']:,}/"
        f"{counts_10k_30k['replication_requested']:,} "
        f"({counts_10k_30k['replication_missing_pca']:,} missing PCA); "
        f"overlap={counts_10k_30k['overlap_plotted']:,}"
    )
    print(
        "1k vs 10k centroid distances: "
        f"PC1-PC2={distances_1k_10k['pc1_pc2']:.10g}; "
        f"PC2-PC3={distances_1k_10k['pc2_pc3']:.10g}"
    )
    print(
        "10k vs 30k centroid distances: "
        f"PC1-PC2={distances_10k_30k['pc1_pc2']:.10g}; "
        f"PC2-PC3={distances_10k_30k['pc2_pc3']:.10g}"
    )
    all_paths = [
        *paths_1k_10k,
        summary_1k_10k,
        *paths_10k_30k,
        summary_10k_30k,
        *paths_three,
        summary_three,
    ]
    for path in all_paths:
        print(f"Wrote {path}")
    return all_paths


if __name__ == "__main__":
    figure_s10_main()
