"""Marker-level local-ancestry scan and Figure 4A-C source products.

This module is the clean replacement for the exploratory LAI notebook.  It
starts from the 22 indexed FLARE ``*.anc.vcf.gz`` files beneath the normalized
``gs://.../lai_masked`` prefix, calculates the cohort EUR fraction at every
marker, calls sign-pure outlier regions, and extracts one individual's painted
haplotypes.  Expensive chromosome products are cached independently so a
restarted Workbench run resumes rather than rescanning completed chromosomes.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs
from .common.genome import CHROM_SIZES


FIGURE_ID = "admixture_reference"
CHROMS = tuple(range(1, 23))
Z_THRESHOLD = float(os.environ.get("AOU_FIG4_Z_THRESHOLD", "4.42"))
REGION_FLANK_BP = 50_000
TELOMERE_PAD_BP = int(os.environ.get("AOU_FIG4_TELOMERE_PAD_BP", "2000000"))
REGION_GAP_BP = int(os.environ.get("AOU_FIG4_REGION_GAP_BP", "200000"))
VALLEY_MIN_MARKERS = int(os.environ.get("AOU_FIG4_VALLEY_MIN_MARKERS", "10"))
CENTROMERE_MAX_OVERLAP = float(
    os.environ.get("AOU_FIG4_CENTROMERE_MAX_OVERLAP", "0.5")
)
KARYOGRAM_STRIDE_BP = int(
    os.environ.get("AOU_FIG4_KARYOGRAM_STRIDE_BP", "100000")
)
KARYOGRAM_MASK_MIN_BP = int(
    os.environ.get("AOU_FIG4_KARYOGRAM_MASK_MIN_BP", "1000000")
)
DEFAULT_DISPLAY_REGION = os.environ.get(
    "AOU_FIG4_REGION", "chr9:63748785-63837789"
)


def analysis_dir() -> Path:
    path = config.analysis_dir(FIGURE_ID) / "lai"
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_lai_dir() -> Path:
    path = Path(config.LAI_MASKED_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_file(remote: str, local: Path, *, force: bool = False,
                required: bool = True) -> Path | None:
    if local.exists() and not force:
        return local
    if not remote or not gcs.available():
        if required:
            raise FileNotFoundError(
                f"Required admixture reference input is absent locally and cannot be staged: {remote}"
            )
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    result = gcs.gcs_cp(remote, local)
    if result.returncode != 0:
        if required:
            raise FileNotFoundError(
                f"Could not stage {remote}: {result.stderr.strip()}"
            )
        warnings.warn(
            f"[admixture_reference] optional input could not be staged from {remote}: "
            f"{result.stderr.strip()[:300]}",
            stacklevel=2,
        )
        return None
    return local


def stage_lai_chromosome(chrom: int, *, force: bool = False) -> Path:
    """Stage one FLARE ancestry VCF and its tabix index."""
    name = f"chr{int(chrom)}.anc.vcf.gz"
    remote = f"{config.LAI_MASKED_GS_DIR.rstrip('/')}/{name}"
    local = local_lai_dir() / name
    _stage_file(remote, local, force=force, required=True)
    _stage_file(remote + ".tbi", Path(str(local) + ".tbi"),
                force=force, required=True)
    return local


def stage_lai_inputs(*, force: bool = False) -> dict[int, Path]:
    """Stage all autosomal marker-level LAI VCFs under the bucket prefix."""
    return {chrom: stage_lai_chromosome(chrom, force=force) for chrom in CHROMS}


def stage_reference_inputs(*, force: bool = False) -> dict[str, Path | None]:
    cache = Path(config.ADMIXTURE_WORK_DIR).expanduser() / "workspace_inputs"
    hardmask = _stage_file(
        config.ADMIXTURE_HARDMASK_GS,
        cache / "hardmask.hg38.v4.over99.bed",
        force=force,
        required=False,
    )
    gtf = _stage_file(
        config.ADMIXTURE_GTF_GS,
        cache / "hg38.refGene.gtf",
        force=force,
        required=False,
    )
    return {"hardmask": hardmask, "gtf": gtf}


def _cyvcf2():
    try:
        from cyvcf2 import VCF
    except ImportError as exc:  # pragma: no cover - Workbench dependency path
        raise RuntimeError(
            "admixture reference LAI scanning requires cyvcf2; run "
            "`python -m scripts.recompute_admixture_reference` to install dependencies"
        ) from exc
    return VCF


def parse_ancestry_header(vcf_path: str | os.PathLike) -> dict[str, int]:
    """Return the FLARE ``##ANCESTRY`` name-to-index mapping."""
    VCF = _cyvcf2()
    vcf = VCF(str(vcf_path))
    try:
        header = vcf.raw_header
    finally:
        vcf.close()
    for line in header.splitlines():
        if not line.startswith("##ANCESTRY="):
            continue
        body = line.split("=<", 1)[-1].strip().rstrip(">")
        mapping = {}
        for item in body.split(","):
            name, value = item.split("=", 1)
            mapping[name.strip()] = int(value)
        if mapping:
            return mapping
    raise ValueError(f"No ##ANCESTRY mapping found in {vcf_path}")


def ancestry_indices(mapping: dict[str, int]) -> tuple[int, int, str, str]:
    """Resolve EUR/NFE and AFR indices without assuming FLARE column order."""
    lower = {str(name).lower(): (name, value) for name, value in mapping.items()}
    eur_key = next((key for key in ("eur", "nfe") if key in lower), None)
    afr_key = next((key for key in ("afr", "african") if key in lower), None)
    if eur_key is None or afr_key is None:
        raise ValueError(f"Expected EUR/NFE and AFR ancestry labels; found {mapping}")
    eur_name, eur_idx = lower[eur_key]
    afr_name, afr_idx = lower[afr_key]
    return int(eur_idx), int(afr_idx), str(eur_name), str(afr_name)


def _scan_cache(chrom: int) -> Path:
    path = analysis_dir() / "chromosome_scans"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"chr{chrom}.eur_fraction.npz"


def scan_chromosome(vcf_path: str | os.PathLike, eur_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream a FLARE VCF and return marker position and cohort EUR fraction."""
    VCF = _cyvcf2()
    vcf = VCF(str(vcf_path))
    positions: list[int] = []
    fractions: list[float] = []
    n_samples = len(vcf.samples)
    try:
        for record in vcf:
            an1 = record.format("AN1")
            an2 = record.format("AN2")
            if an1 is None or an2 is None:
                continue
            calls = np.concatenate([np.asarray(an1).reshape(-1),
                                    np.asarray(an2).reshape(-1)])
            valid = calls >= 0
            if not valid.any():
                continue
            positions.append(int(record.POS))
            fractions.append(float(np.mean(calls[valid] == eur_idx)))
    finally:
        vcf.close()
    return (np.asarray(positions, dtype=np.int64),
            np.asarray(fractions, dtype=np.float64), n_samples)


def build_chromosome_scans(vcf_paths: dict[int, Path], eur_idx: int,
                           *, force: bool = False) -> dict[int, Path]:
    """Create one resumable compressed marker scan per chromosome."""
    outputs: dict[int, Path] = {}
    for chrom in CHROMS:
        output = _scan_cache(chrom)
        if output.exists() and not force:
            outputs[chrom] = output
            continue
        pos, frac, n_samples = scan_chromosome(vcf_paths[chrom], eur_idx)
        tmp = output.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pos=pos, eur_frac=frac,
                            n_samples=np.asarray([n_samples], dtype=np.int64))
        tmp.replace(output)
        outputs[chrom] = output
        print(f"[admixture_reference] chr{chrom}: scanned {len(pos):,} LAI markers")
    return outputs


def _load_chromosome_scan(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as data:
        return (data["pos"].astype(np.int64),
                data["eur_frac"].astype(float), int(data["n_samples"][0]))


def trimmed_scan_statistics(scan_paths: dict[int, Path]) -> dict[str, float]:
    """Compute the marker-level mean/SD after the fixed 2-Mb end trim."""
    total = total_sq = 0.0
    n = 0
    for chrom, path in scan_paths.items():
        pos, frac, _ = _load_chromosome_scan(path)
        chrom_size = int(CHROM_SIZES[chrom])
        keep = ((pos >= TELOMERE_PAD_BP) &
                (pos <= chrom_size - TELOMERE_PAD_BP) & np.isfinite(frac))
        values = frac[keep]
        total += float(values.sum())
        total_sq += float(np.square(values).sum())
        n += int(len(values))
    if n < 2:
        raise ValueError("Too few telomere-trimmed LAI markers to estimate mean/SD")
    mean = total / n
    variance = max((total_sq - n * mean * mean) / (n - 1), 0.0)
    sd = math.sqrt(variance)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Invalid LAI marker SD: {sd}")
    return {"mean_eur": mean, "sd_eur": sd, "n_trimmed_markers": n}


def merge_regions_signed_valley(
    marker_df: pd.DataFrame,
    *,
    z_threshold: float = Z_THRESHOLD,
    gap_bp: int = REGION_GAP_BP,
    valley_min_markers: int = VALLEY_MIN_MARKERS,
) -> pd.DataFrame:
    """Merge significant markers independently by sign and split deep valleys."""
    required = {"chrom", "pos", "eur_frac", "eur_z"}
    missing = required.difference(marker_df.columns)
    if missing:
        raise ValueError(f"marker scan missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    ordered = marker_df.sort_values(["chrom", "pos"]).reset_index(drop=True)
    for direction, sign in (("EUR-excess", 1), ("AFR-excess", -1)):
        for chrom, group in ordered.groupby("chrom", sort=False):
            tail = (group["eur_z"] >= z_threshold if sign > 0
                    else group["eur_z"] <= -z_threshold)
            significant = group.loc[tail].reset_index(drop=True)
            if significant.empty:
                continue
            non_tail_pos = group.loc[~tail, "pos"].to_numpy(dtype=np.int64)
            starts = [0]
            for idx in range(1, len(significant)):
                previous = int(significant.loc[idx - 1, "pos"])
                current = int(significant.loc[idx, "pos"])
                left = np.searchsorted(non_tail_pos, previous, side="right")
                right = np.searchsorted(non_tail_pos, current, side="left")
                if current - previous > gap_bp or right - left >= valley_min_markers:
                    starts.append(idx)
            starts.append(len(significant))
            for start_idx, end_idx in zip(starts[:-1], starts[1:]):
                chunk = significant.iloc[start_idx:end_idx]
                peak_index = chunk["eur_z"].abs().idxmax()
                start = int(chunk["pos"].min())
                end = int(chunk["pos"].max())
                rows.append({
                    "region_id": f"chr{int(chrom)}_{start}_{end}",
                    "chrom": int(chrom),
                    "start": start,
                    "end": end,
                    "span_kb": (end - start) / 1000.0,
                    "n_markers": int(len(chunk)),
                    "peak_pos": int(chunk.loc[peak_index, "pos"]),
                    "peak_z": float(chunk.loc[peak_index, "eur_z"]),
                    "mean_z": float(chunk["eur_z"].mean()),
                    "mean_eur_frac": float(chunk["eur_frac"].mean()),
                    "direction": direction,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "region_id", "chrom", "start", "end", "span_kb", "n_markers",
            "peak_pos", "peak_z", "mean_z", "mean_eur_frac", "direction",
        ])
    return (pd.DataFrame(rows)
            .sort_values("peak_z", key=lambda values: values.abs(), ascending=False)
            .reset_index(drop=True))


def _centromere_cache() -> Path:
    return analysis_dir() / "hg38_centromeres.bed"


def load_hg38_centromeres(*, force: bool = False) -> pd.DataFrame:
    """Load merged UCSC hg38 acen bands, cached as a three-column BED."""
    cache = _centromere_cache()
    if cache.exists() and not force:
        return pd.read_csv(cache, sep="\t", names=["chrom", "start", "end"])
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = gzip.decompress(response.read()).decode("utf-8")
        records = [line.split("\t") for line in raw.splitlines()
                   if line and line.split("\t")[4] == "acen"]
        frame = pd.DataFrame(
            records, columns=["chrom", "start", "end", "band", "stain"]
        )
        frame[["start", "end"]] = frame[["start", "end"]].astype(int)
        frame["chrom"] = frame["chrom"].str.replace("chr", "", regex=False).astype(int)
        merged = (frame.groupby("chrom", as_index=False)
                  .agg(start=("start", "min"), end=("end", "max")))
        merged.to_csv(cache, sep="\t", header=False, index=False)
        return merged
    except Exception as exc:
        warnings.warn(
            f"[admixture_reference] could not load hg38 centromeres; continuing without "
            f"centromere filtering: {exc}", stacklevel=2,
        )
        return pd.DataFrame(columns=["chrom", "start", "end"])


def filter_centromeric_regions(regions: pd.DataFrame,
                               centromeres: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        out = regions.copy()
        out["centromere_overlap_fraction"] = pd.Series(dtype=float)
        return out
    lookup = {int(row.chrom): (int(row.start), int(row.end))
              for row in centromeres.itertuples(index=False)}
    fractions = []
    for row in regions.itertuples(index=False):
        centro_start, centro_end = lookup.get(int(row.chrom), (0, 0))
        overlap = max(0, min(int(row.end), centro_end) -
                      max(int(row.start), centro_start))
        span = max(int(row.end) - int(row.start), 1)
        fractions.append(overlap / span)
    out = regions.copy()
    out["centromere_overlap_fraction"] = fractions
    return out[out["centromere_overlap_fraction"] < CENTROMERE_MAX_OVERLAP].reset_index(drop=True)


def _parse_requested_region(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"(?:chr)?(\d+):(\d+)-(\d+)", str(value).replace(",", ""))
    if not match:
        return None
    chrom, start, end = map(int, match.groups())
    return chrom, min(start, end), max(start, end)


def choose_display_region(regions: pd.DataFrame,
                          requested: str = DEFAULT_DISPLAY_REGION) -> pd.Series:
    """Prefer the requested chr9 region; otherwise use the strongest outlier."""
    parsed = _parse_requested_region(requested)
    if parsed is not None and not regions.empty:
        chrom, start, end = parsed
        overlap = regions[(regions["chrom"] == chrom) &
                          (regions["start"] <= end) &
                          (regions["end"] >= start)]
        if not overlap.empty:
            return overlap.sort_values("peak_z", key=lambda x: x.abs(),
                                       ascending=False).iloc[0]
    if regions.empty:
        raise ValueError("No LAI outlier region is available for admixture referenceC")
    return regions.sort_values("peak_z", key=lambda x: x.abs(), ascending=False).iloc[0]


def _downsample_for_plot(frame: pd.DataFrame, max_background: int = 5000) -> pd.DataFrame:
    significant = frame[frame["eur_z"].abs() >= Z_THRESHOLD]
    background = frame[frame["eur_z"].abs() < Z_THRESHOLD]
    stride = max(1, int(math.ceil(len(background) / max_background)))
    return (pd.concat([background.iloc[::stride], significant], ignore_index=True)
            .drop_duplicates(["chrom", "pos"]).sort_values(["chrom", "pos"]))


def build_outlier_products(scan_paths: dict[int, Path], *, force: bool = False) -> dict:
    """Call outliers and write compact inputs for admixture referenceB/C."""
    out_dir = analysis_dir()
    regions_path = out_dir / "outlier_regions.tsv"
    plot_path = out_dir / "genome_scan_plot.tsv.gz"
    zoom_path = out_dir / "region_zoom.tsv.gz"
    stats_path = out_dir / "scan_stats.json"
    region_path = out_dir / "display_region.json"
    if all(path.exists() for path in
           (regions_path, plot_path, zoom_path, stats_path, region_path)) and not force:
        return {
            "regions": regions_path, "genome_plot": plot_path,
            "region_zoom": zoom_path, "stats": stats_path,
            "display_region": region_path,
        }

    stats = trimmed_scan_statistics(scan_paths)
    mean_eur, sd_eur = stats["mean_eur"], stats["sd_eur"]
    region_frames = []
    plot_frames = []
    for chrom, path in scan_paths.items():
        pos, fraction, _ = _load_chromosome_scan(path)
        trim = ((pos >= TELOMERE_PAD_BP) &
                (pos <= int(CHROM_SIZES[chrom]) - TELOMERE_PAD_BP))
        z = np.where(trim, (fraction - mean_eur) / sd_eur, np.nan)
        frame = pd.DataFrame({
            "chrom": chrom, "pos": pos, "eur_frac": fraction,
            "eur_z": z, "telomere_trimmed": ~trim,
        })
        region_frames.append(merge_regions_signed_valley(frame[trim]))
        plot_frames.append(_downsample_for_plot(frame))

    prefilter = pd.concat(region_frames, ignore_index=True)
    regions = filter_centromeric_regions(prefilter, load_hg38_centromeres(force=force))
    regions.to_csv(regions_path, sep="\t", index=False)
    pd.concat(plot_frames, ignore_index=True).to_csv(
        plot_path, sep="\t", index=False, compression="gzip"
    )

    display = choose_display_region(regions)
    chrom = int(display["chrom"])
    pos, fraction, _ = _load_chromosome_scan(scan_paths[chrom])
    pad = REGION_FLANK_BP
    keep = ((pos >= int(display["start"]) - pad) &
            (pos <= int(display["end"]) + pad))
    zoom = pd.DataFrame({
        "chrom": chrom,
        "pos": pos[keep],
        "eur_frac": fraction[keep],
        "eur_z": (fraction[keep] - mean_eur) / sd_eur,
    })
    zoom["in_region"] = ((zoom["pos"] >= int(display["start"])) &
                         (zoom["pos"] <= int(display["end"])))
    zoom.to_csv(zoom_path, sep="\t", index=False, compression="gzip")
    display_dict = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in display.to_dict().items()
    }
    region_path.write_text(json.dumps(display_dict, indent=2), encoding="utf-8")
    stats.update({
        "z_threshold": Z_THRESHOLD,
        "telomere_pad_bp": TELOMERE_PAD_BP,
        "region_gap_bp": REGION_GAP_BP,
        "valley_min_markers": VALLEY_MIN_MARKERS,
        "region_flank_bp": REGION_FLANK_BP,
        "n_outlier_regions": int(len(regions)),
    })
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[admixture_reference] retained {len(regions):,} sign/valley-aware outlier regions")
    return {
        "regions": regions_path, "genome_plot": plot_path,
        "region_zoom": zoom_path, "stats": stats_path,
        "display_region": region_path,
    }


def _detect_chrom_token(vcf, chrom: int) -> str | None:
    seqnames = set(vcf.seqnames)
    for token in (f"chr{chrom}", str(chrom)):
        if token in seqnames:
            return token
    return None


def _calls_to_segments(pos: np.ndarray, calls: np.ndarray, chrom_size: int,
                       eur_idx: int, afr_idx: int) -> list[dict]:
    if len(pos) == 0:
        return []
    labels = np.where(calls == eur_idx, "EUR",
                      np.where(calls == afr_idx, "AFR", "OTHER"))
    rows = []
    segment_start = 1
    current = labels[0]
    for idx in range(1, len(pos)):
        if labels[idx] == current:
            continue
        midpoint = int((int(pos[idx - 1]) + int(pos[idx])) // 2)
        rows.append({"start": segment_start, "end": midpoint,
                     "ancestry": str(current)})
        segment_start = midpoint
        current = labels[idx]
    rows.append({"start": segment_start, "end": chrom_size,
                 "ancestry": str(current)})
    return rows


def _sample_chromosome(vcf_path: Path, chrom: int, sample_id: str,
                       eur_idx: int, afr_idx: int, stride_bp: int) -> list[dict]:
    VCF = _cyvcf2()
    vcf = VCF(str(vcf_path), samples=[sample_id])
    token = _detect_chrom_token(vcf, chrom)
    if token is None:
        vcf.close()
        return []
    positions: list[int] = []
    hap1: list[int] = []
    hap2: list[int] = []
    chrom_size = int(CHROM_SIZES[chrom])
    try:
        for start in range(1, chrom_size + 1, stride_bp):
            end = min(start + stride_bp - 1, chrom_size)
            record = next(iter(vcf(f"{token}:{start}-{end}")), None)
            if record is None:
                continue
            an1, an2 = record.format("AN1"), record.format("AN2")
            if an1 is None or an2 is None:
                continue
            positions.append(int(record.POS))
            hap1.append(int(np.asarray(an1).reshape(-1)[0]))
            hap2.append(int(np.asarray(an2).reshape(-1)[0]))
    finally:
        vcf.close()
    pos = np.asarray(positions, dtype=np.int64)
    rows = []
    for hap, calls in ((1, hap1), (2, hap2)):
        for segment in _calls_to_segments(
                pos, np.asarray(calls, dtype=np.int16), chrom_size,
                eur_idx, afr_idx):
            rows.append({"sample_id": sample_id, "chrom": chrom, "hap": hap,
                         **segment})
    return rows


def _load_mask_segments(path: Path | None) -> pd.DataFrame:
    columns = ["chrom", "start", "end"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, sep="\t", comment="#", header=None,
                        usecols=[0, 1, 2], names=columns)
    frame["chrom"] = (frame["chrom"].astype(str).str.replace("chr", "", regex=False))
    frame = frame[frame["chrom"].str.fullmatch(r"\d+")].copy()
    frame["chrom"] = frame["chrom"].astype(int)
    frame = frame[frame["chrom"].isin(CHROMS)]
    return frame[(frame["end"] - frame["start"]) >= KARYOGRAM_MASK_MIN_BP]


def build_karyogram(vcf_paths: dict[int, Path], eur_idx: int, afr_idx: int,
                     hardmask_path: Path | None, *, force: bool = False) -> dict:
    """Extract two painted haplotypes for one representative LAI sample."""
    out_dir = analysis_dir()
    segments_path = out_dir / "karyogram_segments.tsv.gz"
    masks_path = out_dir / "karyogram_masks.tsv.gz"
    if segments_path.exists() and masks_path.exists() and not force:
        return {"segments": segments_path, "masks": masks_path}
    VCF = _cyvcf2()
    header = VCF(str(vcf_paths[1]))
    try:
        available = list(header.samples)
    finally:
        header.close()
    requested = os.environ.get("AOU_FIG4_SAMPLE")
    sample_id = requested if requested in available else available[0]
    if requested and requested not in available:
        warnings.warn(
            f"[admixture_reference] requested karyogram sample {requested!r} was not found; "
            f"using {sample_id!r}", stacklevel=2,
        )
    rows: list[dict] = []
    workers = min(8, len(CHROMS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_sample_chromosome, vcf_paths[chrom], chrom, sample_id,
                        eur_idx, afr_idx, KARYOGRAM_STRIDE_BP): chrom
            for chrom in CHROMS
        }
        for future in as_completed(futures):
            chrom = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"Karyogram extraction failed on chr{chrom}") from exc
    pd.DataFrame(rows).sort_values(["chrom", "hap", "start"]).to_csv(
        segments_path, sep="\t", index=False, compression="gzip"
    )
    _load_mask_segments(hardmask_path).to_csv(
        masks_path, sep="\t", index=False, compression="gzip"
    )
    return {"segments": segments_path, "masks": masks_path}


def run(*, force: bool = False) -> dict:
    """Run the complete marker-level LAI stage and return output paths."""
    vcf_paths = stage_lai_inputs(force=False)
    references = stage_reference_inputs(force=False)
    mapping = parse_ancestry_header(vcf_paths[1])
    eur_idx, afr_idx, eur_name, afr_name = ancestry_indices(mapping)
    scans = build_chromosome_scans(vcf_paths, eur_idx, force=force)
    outputs = build_outlier_products(scans, force=force)
    outputs.update(build_karyogram(
        vcf_paths, eur_idx, afr_idx, references["hardmask"], force=force
    ))
    outputs["ancestry_mapping"] = {
        "eur_name": eur_name, "eur_index": eur_idx,
        "afr_name": afr_name, "afr_index": afr_idx,
    }
    return outputs


def real_outputs_available() -> bool:
    out = analysis_dir()
    required = [
        out / "karyogram_segments.tsv.gz",
        out / "genome_scan_plot.tsv.gz",
        out / "region_zoom.tsv.gz",
        out / "outlier_regions.tsv",
        out / "scan_stats.json",
        out / "display_region.json",
    ]
    return all(path.exists() for path in required)

