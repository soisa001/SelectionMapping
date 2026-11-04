"""Render one vector PheWAS Manhattan panel per discovered admixture region.

The Figure 2 admixture workflow stores one row per region x Phecode test in
``phewas.discovery.tsv.gz``.  This plot stage retains every tested Phecode for
each region, but emits a panel only when at least one test has discovery
``p < 1e-3``.  Phecode labels and categories are recovered from the canonical
SNV-PheWAS tables so these panels use the same category layout and rendering as
the existing supplementary locus composites.

Outputs are restartable and integrity checked.  Cache validity depends on the
discovery table, annotation tables, plot parameters, and output hashes rather
than the repository commit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from PIL import Image

from scripts.common import config, style
from scripts import figure5_locus_composites as composites

ANALYSIS_CONTRACT = "supp_admixture_discovery_phewas_v1"
DEFAULT_THRESHOLD = 1e-3
DEFAULT_DPI = 300
ANNOTATION_CHUNK_ROWS = 250_000
DEFAULT_FIGSIZE = (11.0, 6.3)
DEFAULT_OUTPUT_DIR = (
    config.figure_dir("figure2") / "discovery_phewas"
)
DISCOVERY_RELATIVE_PATH = (
    Path("analysis") / "figure2" / "phewas" / "phewas.discovery.tsv.gz"
)
DEFAULT_DISCOVERY_PATH = config.REPO_ROOT / DISCOVERY_RELATIVE_PATH
LOGGER = logging.getLogger("admixture_discovery_phewas")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_discovery_path(requested: Path) -> Path:
    """Find a completed discovery cache, including above an accidental nested clone."""
    expanded = requested.expanduser()
    candidates: list[Path] = [expanded.resolve()]
    if expanded.is_absolute():
        parts = expanded.parts
        analysis_indices = [index for index, value in enumerate(parts) if value == "analysis"]
        relative = Path(*parts[analysis_indices[-1]:]) if analysis_indices else None
    else:
        relative = expanded
    if relative is not None:
        for root in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
            candidate = (root / relative).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            if candidate != candidates[0]:
                LOGGER.warning(
                    "Requested discovery cache was absent; recovered ancestor cache: %s",
                    candidate,
                )
            return candidate
    searched = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Discovery PheWAS cache is missing or empty. Searched:\n  - " + searched
    )


def resolve_output_dir(requested: Path, discovery_path: Path) -> Path:
    """Keep outputs beside the recovered outer repo rather than in a nested clone."""
    expanded = requested.expanduser().resolve()
    if expanded != DEFAULT_OUTPUT_DIR.expanduser().resolve():
        return expanded
    for parent in discovery_path.resolve().parents:
        if parent.name == "analysis":
            return (
                parent.parent / "figures" / "figure2"
                / "discovery_phewas"
            ).resolve()
    return expanded


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() in {".gz", ".bgz"}:
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("rt", encoding="utf-8-sig", errors="replace")


def _table_layout(path: Path, required: set[str]) -> tuple[int, str]:
    with _open_text(path) as stream:
        for row_number, line in enumerate(stream):
            delimiter = "\t" if line.count("\t") >= line.count(",") else ","
            columns = {value.strip() for value in line.rstrip("\r\n").split(delimiter)}
            if required <= columns:
                return row_number, delimiter
            if row_number >= 100:
                break
    raise ValueError(f"No header containing {sorted(required)} in {path}")


def _read_table(path: Path, *, required: set[str]) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Input table is missing or empty: {path}")
    skiprows, delimiter = _table_layout(path, required)
    return pd.read_csv(
        path,
        sep=delimiter,
        skiprows=skiprows,
        dtype=str,
        low_memory=False,
    )


def _first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def load_annotations(paths: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a unique Phecode label/category map, preserving source category order."""
    if not paths:
        raise ValueError("At least one Phecode annotation table is required")
    rows: list[pd.DataFrame] = []
    sources = []
    offset = 0
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Input table is missing or empty: {path}")
        skiprows, delimiter = _table_layout(path, {"phecode"})
        header = pd.read_csv(path, sep=delimiter, skiprows=skiprows, nrows=0)
        name_column = _first_column(
            list(header.columns),
            ("phecode_string", "phecode_name", "phecode_description", "description", "meaning"),
        )
        category_column = _first_column(
            list(header.columns), ("phecode_category", "category")
        )
        usecols = ["phecode", *[value for value in (name_column, category_column) if value]]
        source_rows = 0
        for frame in pd.read_csv(
            path,
            sep=delimiter,
            skiprows=skiprows,
            dtype=str,
            usecols=usecols,
            chunksize=ANNOTATION_CHUNK_ROWS,
            low_memory=False,
        ):
            selected = pd.DataFrame(
                {"phecode": frame["phecode"].astype(str).str.strip()}
            )
            selected["phecode_name"] = (
                frame[name_column].fillna("").astype(str).str.strip()
                if name_column
                else ""
            )
            selected["phecode_category"] = (
                frame[category_column].fillna("").astype(str).str.strip()
                if category_column
                else ""
            )
            selected["_annotation_order"] = (
                np.arange(len(selected), dtype=int) + offset + source_rows
            )
            source_rows += len(selected)
            selected = selected[selected["phecode"].ne("")].drop_duplicates(
                ["phecode", "phecode_name", "phecode_category"], keep="first"
            )
            rows.append(selected)
        offset += source_rows
        sources.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "name_column": name_column,
                "category_column": category_column,
                "rows_scanned": source_rows,
            }
        )
    combined = pd.concat(rows, ignore_index=True)
    conflicts = combined[combined["phecode_category"].ne("")].groupby("phecode")[
        "phecode_category"
    ].nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.astype(str).tolist()[:10]
        raise ValueError(f"Conflicting Phecode categories in annotation tables: {bad}")

    def first_nonempty(values: pd.Series) -> str:
        values = values[values.astype(str).str.strip().ne("")]
        return str(values.iloc[0]) if len(values) else ""

    annotations = (
        combined.sort_values("_annotation_order", kind="mergesort")
        .groupby("phecode", sort=False, as_index=False)
        .agg(
            phecode_name=("phecode_name", first_nonempty),
            phecode_category=("phecode_category", first_nonempty),
            _annotation_order=("_annotation_order", "min"),
        )
    )
    return annotations, {"sources": sources, "unique_phecodes": len(annotations)}


def prepare_plot_data(
    discovery: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and annotate the discovery results used by every panel."""
    required = {"region_id", "phecode", "p", "beta"}
    missing = required.difference(discovery.columns)
    if missing:
        raise ValueError(f"Discovery table is missing columns: {sorted(missing)}")
    if not 0 < threshold < 1:
        raise ValueError("Discovery threshold must be between zero and one")
    frame = discovery.copy()
    frame["region_id"] = frame["region_id"].astype(str).str.strip()
    frame["phecode"] = frame["phecode"].astype(str).str.strip()
    frame["p_value"] = pd.to_numeric(frame["p"], errors="coerce")
    frame["beta"] = pd.to_numeric(frame["beta"], errors="coerce")
    invalid = (
        frame["region_id"].eq("")
        | frame["phecode"].eq("")
        | ~np.isfinite(frame["p_value"])
        | ~np.isfinite(frame["beta"])
        | frame["p_value"].le(0)
        | frame["p_value"].gt(1)
    )
    if invalid.any():
        raise ValueError(f"Discovery table contains {int(invalid.sum()):,} invalid rows")
    duplicated = frame.duplicated(["region_id", "phecode"], keep=False)
    if duplicated.any():
        conflicts = (
            frame.loc[duplicated]
            .groupby(["region_id", "phecode"])
            .agg(p_values=("p_value", "nunique"), betas=("beta", "nunique"))
        )
        if ((conflicts["p_values"] > 1) | (conflicts["betas"] > 1)).any():
            raise ValueError("Conflicting duplicate region-Phecode discovery tests")
        frame = frame.drop_duplicates(["region_id", "phecode"], keep="first")
    frame = frame.merge(annotations, on="phecode", how="left", validate="many_to_one")
    missing_category = frame["phecode_category"].fillna("").astype(str).str.strip().eq("")
    if missing_category.any():
        codes = sorted(frame.loc[missing_category, "phecode"].unique())[:20]
        raise ValueError(f"Missing Phecode categories for plotted tests: {codes}")
    significant = frame["p_value"] < threshold
    if not significant.any():
        raise ValueError(f"No discovery association passed p < {threshold:.0e}")
    missing_names = (
        frame.loc[significant, "phecode_name"].fillna("").astype(str).str.strip().eq("")
    )
    if missing_names.any():
        codes = sorted(frame.loc[significant].loc[missing_names, "phecode"].unique())
        raise ValueError(f"Missing names for discovery-significant Phecodes: {codes}")
    frame["phecode_name"] = frame["phecode_name"].fillna(frame["phecode"])
    frame["odds_ratio"] = np.exp(frame["beta"].clip(-700, 700))
    frame["discovery_significant"] = significant
    significant_regions = list(dict.fromkeys(frame.loc[significant, "region_id"]))
    frame = frame[frame["region_id"].isin(significant_regions)].copy()
    category_order = (
        frame[["phecode_category", "_annotation_order"]]
        .sort_values("_annotation_order", kind="mergesort")["phecode_category"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    return frame.reset_index(drop=True), category_order


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.") or "region"


def _output_stem(index: int, region_id: str) -> str:
    return f"Supplementary_Admixture_Discovery_Phewas_{index:02d}_{_safe_slug(region_id)}"


def _validate_png(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"PNG missing or empty: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width < 1000 or height < 500:
        raise ValueError(f"PNG dimensions are unexpectedly small: {width}x{height}")
    return {
        "path": str(path), "size_bytes": path.stat().st_size,
        "width": width, "height": height, "sha256": _sha256_file(path),
    }


def _validate_pdf(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1000:
        raise FileNotFoundError(f"PDF missing, empty, or truncated: {path}")
    content = path.read_bytes()
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-1024:]:
        raise ValueError(f"Invalid PDF structure: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_is_current(manifest_path: Path, contract: dict[str, Any]) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("contract") != contract:
        return False
    for output in manifest.get("outputs", []):
        path = Path(output["path"])
        if (
            not path.is_file()
            or path.stat().st_size != output.get("size_bytes")
            or _sha256_file(path) != output.get("sha256")
        ):
            return False
    return bool(manifest.get("outputs"))


def _render_region(
    region_frame: pd.DataFrame,
    *,
    index: int,
    category_order: Sequence[str],
    parameters: composites.PlotParameters,
    output_dir: Path,
    threshold: float,
) -> tuple[Path, Path]:
    region_id = str(region_frame["region_id"].iloc[0])
    hits = region_frame[region_frame["discovery_significant"]].sort_values(
        ["p_value", "phecode"], kind="mergesort"
    )
    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)
    composites.draw_phewas_panel(
        axis,
        region_frame,
        meta_hits=region_frame.iloc[0:0],
        target_hit=hits.iloc[0],
        category_order=category_order,
        parameters=parameters,
        highlight_label=None,
        nominal_label=f"Discovery p < {threshold:.0e}",
        threshold_inclusive=False,
    )
    axis.set_ylabel(r"$-\log_{10}(p)$", fontsize=14)
    axis.tick_params(axis="x", labelsize=9.5, pad=2)
    axis.tick_params(axis="y", labelsize=11)
    figure.suptitle(
        f"Admixture outlier {region_id}: discovery PheWAS",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.935,
        f"{len(hits)} discovery-significant Phecode{'s' if len(hits) != 1 else ''}; "
        f"{len(region_frame):,} tests shown",
        ha="center",
        va="top",
        fontsize=12,
    )
    for legend in figure.findobj(match=Legend):
        for label in legend.get_texts():
            label.set_fontsize(9)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(9)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.77)
    stem = _output_stem(index, region_id)
    pdf = output_dir / "pdf" / f"{stem}.pdf"
    png = output_dir / "png" / f"{stem}.png"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = pdf.with_suffix(pdf.suffix + ".partial")
    temporary_png = png.with_suffix(png.suffix + ".partial")
    figure.savefig(temporary_pdf, format="pdf", dpi=parameters.dpi, facecolor="white")
    figure.savefig(temporary_png, format="png", dpi=parameters.dpi, facecolor="white")
    plt.close(figure)
    temporary_pdf.replace(pdf)
    temporary_png.replace(png)
    _validate_pdf(pdf)
    _validate_png(png)
    return pdf, png


def render_all(
    discovery_path: Path = DEFAULT_DISCOVERY_PATH,
    annotation_paths: Sequence[Path] = (),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    force: bool = False,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Any]:
    """Render and audit all discovery-significant admixture-region panels."""
    discovery_path = resolve_discovery_path(discovery_path)
    output_dir = resolve_output_dir(output_dir, discovery_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery = _read_table(
        discovery_path, required={"region_id", "phecode", "p", "beta"}
    )
    annotations, annotation_metadata = load_annotations(annotation_paths)
    plotted, category_order = prepare_plot_data(
        discovery, annotations, threshold=threshold
    )
    region_ids = list(dict.fromkeys(plotted["region_id"].astype(str)))
    scales = composites.resolve_shared_phewas_scales(
        plotted.rename(columns={"p_value": "p_value"}),
        odds_ratio_cap=composites.DEFAULT_OR_CAP,
    )
    parameters = composites.PlotParameters(
        odds_ratio_cap=composites.DEFAULT_OR_CAP,
        marker_area_min=composites.DEFAULT_MARKER_MIN,
        marker_area_max=composites.DEFAULT_MARKER_MAX,
        phewas_threshold=threshold,
        phewas_y_max=scales["phewas_y_max"],
        dpi=dpi,
    )
    expected_plots = [
        output_dir / extension / f"{_output_stem(index, region_id)}.{extension}"
        for index, region_id in enumerate(region_ids, start=1)
        for extension in ("pdf", "png")
    ]
    source_path = output_dir / "admixture_discovery_phewas_source_data.tsv.gz"
    regions_path = output_dir / "admixture_discovery_phewas_regions.tsv"
    manifest_path = output_dir / "admixture_discovery_phewas_manifest.json"
    contract = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "discovery": {
            "path": str(discovery_path),
            "size_bytes": discovery_path.stat().st_size,
            "sha256": _sha256_file(discovery_path),
        },
        "annotations": annotation_metadata["sources"],
        "threshold_rule": f"p < {threshold:.17g}",
        "parameters": {
            "threshold": threshold,
            "dpi": dpi,
            "figsize_inches": list(DEFAULT_FIGSIZE),
            "odds_ratio_cap": parameters.odds_ratio_cap,
            "marker_area_min": parameters.marker_area_min,
            "marker_area_max": parameters.marker_area_max,
            "phewas_y_max": parameters.phewas_y_max,
            "pdf_fonttype": 42,
            "rasterized": False,
        },
        "category_order": category_order,
        "regions": region_ids,
        "expected_plots": [str(path) for path in expected_plots],
    }
    if not force and _cache_is_current(manifest_path, contract):
        LOGGER.info("Verified cached panels for %d admixture regions", len(region_ids))
        return {
            "status": "cached", "regions": region_ids, "plots": expected_plots,
            "source_data": source_path, "region_manifest": regions_path,
            "manifest": manifest_path,
        }

    style.apply_style()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    records = []
    plot_paths = []
    for index, region_id in enumerate(region_ids, start=1):
        region_frame = plotted[plotted["region_id"].eq(region_id)].copy()
        hits = region_frame[region_frame["discovery_significant"]].sort_values(
            ["p_value", "phecode"], kind="mergesort"
        )
        LOGGER.info(
            "Rendering region %d/%d %s (%d tests; %d hits)",
            index, len(region_ids), region_id, len(region_frame), len(hits),
        )
        pdf, png = _render_region(
            region_frame,
            index=index,
            category_order=category_order,
            parameters=parameters,
            output_dir=output_dir,
            threshold=threshold,
        )
        plot_paths.extend([pdf, png])
        records.append(
            {
                "figure_index": index,
                "region_id": region_id,
                "n_tests": len(region_frame),
                "n_discovery_significant": len(hits),
                "minimum_p": float(hits["p_value"].min()),
                "significant_phecodes": ";".join(hits["phecode"].astype(str)),
                "significant_phecode_names": ";".join(hits["phecode_name"].astype(str)),
                "pdf": str(pdf),
                "png": str(png),
            }
        )
    source_columns = [
        "region_id", "phecode", "phecode_name", "phecode_category", "beta",
        "odds_ratio", "p_value", "discovery_significant",
    ]
    source_tmp = source_path.with_suffix(source_path.suffix + ".partial")
    plotted[source_columns].to_csv(source_tmp, sep="\t", index=False, compression="gzip")
    source_tmp.replace(source_path)
    regions_tmp = regions_path.with_suffix(regions_path.suffix + ".partial")
    pd.DataFrame(records).to_csv(regions_tmp, sep="\t", index=False)
    regions_tmp.replace(regions_path)
    outputs = [
        *[_validate_pdf(path) if path.suffix == ".pdf" else _validate_png(path) for path in plot_paths],
        {"path": str(source_path), "size_bytes": source_path.stat().st_size, "sha256": _sha256_file(source_path)},
        {"path": str(regions_path), "size_bytes": regions_path.stat().st_size, "sha256": _sha256_file(regions_path)},
    ]
    _atomic_json(
        manifest_path,
        {
            "contract": contract,
            "created_utc": _utc_now(),
            "outputs": outputs,
            "summary": {
                "input_rows": len(discovery),
                "plotted_rows": len(plotted),
                "regions": len(region_ids),
                "significant_tests": int(plotted["discovery_significant"].sum()),
            },
        },
    )
    return {
        "status": "rendered", "regions": region_ids, "plots": plot_paths,
        "source_data": source_path, "region_manifest": regions_path,
        "manifest": manifest_path,
    }


def _default_annotation_paths() -> list[Path]:
    candidates = [
        Path(config.PHEWAS_FILTERED_TSV),
        Path(config.PHEWAS_META_RESULTS_TSV),
        Path(config.PHEWAS_DISCOVERY_TSV),
    ]
    return [path for path in candidates if path.expanduser().is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY_PATH)
    parser.add_argument(
        "--annotations", type=Path, action="append", default=None,
        help="Phecode annotation TSV/CSV; repeat to supply multiple sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    annotation_paths = args.annotations or _default_annotation_paths()
    if not annotation_paths:
        raise FileNotFoundError(
            "No local Phecode annotation table found; pass --annotations PATH"
        )
    result = render_all(
        args.discovery,
        annotation_paths,
        args.output_dir,
        threshold=args.threshold,
        force=args.force,
        dpi=args.dpi,
    )
    LOGGER.info(
        "%s %d region panels; manifest=%s",
        result["status"], len(result["regions"]), result["manifest"],
    )


if __name__ == "__main__":
    main()
