"""Figure 4 panel B from the audited 446-row discovery association source table."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config
from . import figure4_discovery as figure_s9_phewas

FIGURE_ID = "figure4"
FIGURE_STEM = "Figure_4B_MAF_Effect"
MARKER_SCALE = 1.2
FONT_SCALE = 1.35
SOURCE_ENV = "AOU_S9_SOURCE_DATA_TSV"
EXPECTED_ROWS = figure_s9_phewas.EXPECTED_HIT_COUNT
REQUIRED_COLUMNS = {
    "variant_tag",
    "phecode",
    "phecode_string",
    "maf",
    "abs_beta",
    "category",
    "sv_annotation",
}


def source_data_path() -> Path:
    """Return the audited S9 source-data path used by panel B."""
    default = (
        config.ANALYSIS_DIR
        / "supplementary"
        / figure_s9_phewas.FIGURE_ID
        / "figure_s9_source_data.tsv"
    )
    return Path(os.environ.get(SOURCE_ENV, default))


def panel_image_path(ext: str = "png") -> Path:
    """Return the standalone panel-B output path."""
    return config.figure_dir(FIGURE_ID) / f"{FIGURE_STEM}.{ext}"


def _validate_source_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data is missing: " + ", ".join(missing)
        )
    if len(frame) != EXPECTED_ROWS:
        raise figure_s9_phewas.S9ContractError(
            f"Figure 4B expected {EXPECTED_ROWS} S9 rows but found {len(frame)}"
        )
    out = frame.copy()
    if out.duplicated(["variant_tag", "phecode"]).any():
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data has duplicate variant_tag x phecode rows"
        )
    out["maf"] = pd.to_numeric(out["maf"], errors="coerce")
    out["abs_beta"] = pd.to_numeric(out["abs_beta"], errors="coerce")
    invalid = (
        ~np.isfinite(out["maf"])
        | out["maf"].le(0)
        | out["maf"].gt(0.5)
        | ~np.isfinite(out["abs_beta"])
        | out["abs_beta"].lt(0)
    )
    if invalid.any():
        raise figure_s9_phewas.S9ContractError(
            f"Figure 4B S9 source data has {int(invalid.sum())} invalid plot rows"
        )
    unknown = sorted(
        set(out["category"].astype(str)).difference(figure_s9_phewas.CATEGORY_ORDER)
    )
    if unknown:
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data has unknown categories: " + ", ".join(unknown)
        )
    names = out["phecode_string"].fillna("").astype(str).str.strip()
    if names.eq("").any() or names.eq(out["phecode"].astype(str).str.strip()).any():
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data needs phenotype string names, not phecode IDs"
        )
    sv_rows = out["category"].eq("SV-linked hit")
    missing_annotations = out.loc[sv_rows, "sv_annotation"].fillna("").str.strip().eq("")
    if missing_annotations.any():
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data has an SV-linked row without sv_annotation"
        )
    missing_sizes = ~out.loc[sv_rows, "sv_annotation"].str.contains(
        r"\([0-9][0-9,]* bp [^)]+\)$", regex=True
    )
    if missing_sizes.any():
        raise figure_s9_phewas.S9ContractError(
            "Figure 4B S9 source data needs size-aware SV annotations"
        )
    return out


def load_source_data() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate the audited discovery association source table."""
    path = source_data_path()
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(
            f"Figure 4B requires audited source data at {path}. "
            f"Run scripts.figure4_discovery first or set {SOURCE_ENV}."
        )
    frame = _validate_source_data(pd.read_csv(path, sep="\t", dtype={"phecode": str}, low_memory=False))
    return frame, {"kind": "audited S9 source data", "path": str(path.resolve()), "sha256": _sha256(path)}


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


def render_panel() -> list[Path]:
    """Render panel B to paired PNG/PDF files and record provenance."""
    import matplotlib.pyplot as plt

    frame, source = load_source_data()
    print(f"[figure4b] source: {json.dumps(source, sort_keys=True)}")
    marker_scale = MARKER_SCALE
    font_scale = FONT_SCALE
    legend_frame = False
    annotation_boxes = False
    print(
        "[figure4b] parameters: "
        f"rows={len(frame)}, x=maf(log), y=abs_beta, "
        f"marker_scale={marker_scale}, font_scale={font_scale}, "
        f"legend_frame={legend_frame}, annotation_boxes={annotation_boxes}, "
        f"label_seed={figure_s9_phewas.SV_LABEL_SEED}"
    )
    figure = figure_s9_phewas.plot_s9(
        frame,
        marker_scale=marker_scale,
        font_scale=font_scale,
        legend_frame=legend_frame,
        annotation_boxes=annotation_boxes,
    )
    stem = panel_image_path().with_suffix("")
    outputs = []
    for ext in config.FIGURE_FORMATS:
        path = stem.with_suffix(f".{ext}")
        # ``savefig.bbox`` is globally ``"tight"``.  Override it explicitly
        # so B retains the same fixed 11 x 8.5 canvas as panel A.
        save_kwargs = {"bbox_inches": figure.bbox_inches}
        figure.savefig(path, dpi=300, facecolor="white", **save_kwargs)
        outputs.append(path)
        print(f"[figure4b] wrote: {path}")
    plt.close(figure)

    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise figure_s9_phewas.S9ContractError(f"Invalid PNG signature: {png}")
    pdf_bytes = pdf.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-2048:]:
        raise figure_s9_phewas.S9ContractError(f"Invalid PDF structure: {pdf}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "row_unit": "unique variant_tag x phecode association",
        "source": source,
        "parameters": {
            "expected_rows": EXPECTED_ROWS,
            "x": "minor allele frequency",
            "x_scale": "log",
            "y": "absolute discovery beta (log-odds)",
            "marker_area_scale": marker_scale,
            "font_scale": font_scale,
            "legend_frame": legend_frame,
            "annotation_boxes": annotation_boxes,
            "label_seed": figure_s9_phewas.SV_LABEL_SEED,
        },
        "category_counts": frame["category"].value_counts().sort_index().to_dict(),
        "output_sha256": {str(path.resolve()): _sha256(path) for path in outputs},
    }
    manifest_name = "figure4b_manifest.json"
    manifest_path = config.analysis_dir(FIGURE_ID) / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[figure4b] manifest: {manifest_path}")
    return outputs
