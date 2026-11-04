"""Figure 1D -- Phase 1 AFR/EUR IDW geographic-origin projection.

The AoU ancestry scores and HGDP+1KG training scores share a PC coordinate
system.  Figure 1D intentionally uses only the 1000 Genomes AFR and EUR
reference samples as geographic anchors.

The derived predictions, source-population anchors, and a checksum/provenance
manifest are cached under ``analysis/figure1``.  Cache validation enforces the
Phase 1 sample manifest and the exact AFR/EUR + 1000 Genomes source contract.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs, geo, samples

IDW_SAMPLES_CSV = config.ANALYSIS_DIR / "figure1" / "idw_phase1_predictions.csv"
IDW_SOURCES_CSV = config.ANALYSIS_DIR / "figure1" / "idw_source_pops.csv"
IDW_MANIFEST_JSON = config.ANALYSIS_DIR / "figure1" / "idw_manifest.json"
IDW_GEOGRAPHY_SUMMARY_JSON = (
    config.ANALYSIS_DIR / "figure1" / "idw_geography_summary.json"
)
_RAW_DIR = config.RAW_DATA_DIR / "idw"

# Version 3 changes the semantic contract from the accidentally introduced
# global HGDP+1KG cache back to the intended Phase 1 1KG AFR/EUR reference.
_CACHE_SCHEMA_VERSION = 3
_GEOGRAPHY_AUDIT_SCHEMA_VERSION = 1
EXPECTED_SOURCE_REGIONS = frozenset({"AFR", "EUR"})

_GENETIC_REGION_MAP = {
    "AFR": "AFR",
    "AMR": "AMR",
    "EAS": "EAS",
    "EUR": "EUR",
    "CSA": "SAS",
    "SAS": "SAS",
    "MID": "MID",
    "OCE": "OCE",
}
_INFERENCE_REGION_MAP = {
    "afr": "AFR",
    "amr": "AMR",
    "eas": "EAS",
    "eur": "EUR",
    "nfe": "EUR",
    "fin": "EUR",
    "sas": "SAS",
    "mid": "MID",
}
_SAMPLE_COLUMNS = {"sample_id", "pred_longitude", "pred_latitude", "region"}
_SOURCE_COLUMNS = {
    "pop_code",
    "lon",
    "lat",
    "region",
    "source_project",
    "n_reference",
}


def on_workbench() -> bool:
    return gcs.available()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_hash(values) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _is_1kg_project(values: pd.Series) -> pd.Series:
    projects = values.astype(str).str.casefold()
    return (
        projects.str.contains("1000", regex=False)
        | projects.str.contains("1kg", regex=False)
        | projects.str.contains("tgp", regex=False)
    )


def _projection_frame_sha256(samples_frame: pd.DataFrame) -> str:
    """Hash the participant IDs and projected coordinates deterministically."""
    required = ["sample_id", "pred_longitude", "pred_latitude"]
    missing = set(required) - set(samples_frame.columns)
    if missing:
        raise ValueError(f"IDW sample frame is missing audit columns: {sorted(missing)}")
    normalized = samples_frame[required].copy()
    normalized["sample_id"] = normalized["sample_id"].astype(str).str.strip()
    normalized["pred_longitude"] = pd.to_numeric(
        normalized["pred_longitude"], errors="raise"
    )
    normalized["pred_latitude"] = pd.to_numeric(
        normalized["pred_latitude"], errors="raise"
    )
    normalized = normalized.sort_values("sample_id").reset_index(drop=True)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def european_continent_summary(
    samples_frame: pd.DataFrame | None = None,
    *,
    write_cache: bool = True,
) -> dict[str, object]:
    """Count predicted points covered by Natural Earth Europe polygons.

    Country boundaries are taken from the same vendored map drawn in Figure
    1D.  ``covers`` includes points exactly on a coastline or international
    boundary.  The cached audit contains aggregate counts and hashes only; it
    never writes controlled participant identifiers.
    """
    import geopandas as gpd

    if samples_frame is None:
        if not IDW_SAMPLES_CSV.is_file():
            raise FileNotFoundError(f"IDW predictions are unavailable: {IDW_SAMPLES_CSV}")
        samples_frame = pd.read_csv(IDW_SAMPLES_CSV)
    required = {"sample_id", "pred_longitude", "pred_latitude"}
    missing = required - set(samples_frame.columns)
    if missing:
        raise ValueError(f"IDW sample frame is missing geography columns: {sorted(missing)}")
    if samples_frame.empty:
        raise ValueError("IDW sample frame has no projected participants")

    longitude = pd.to_numeric(samples_frame["pred_longitude"], errors="coerce")
    latitude = pd.to_numeric(samples_frame["pred_latitude"], errors="coerce")
    if not (
        np.isfinite(longitude).all()
        and np.isfinite(latitude).all()
        and longitude.between(-180.0, 180.0).all()
        and latitude.between(-90.0, 90.0).all()
    ):
        raise ValueError("IDW sample frame has invalid projected coordinates")

    world_path = Path(config.WORLD_COUNTRIES_GEOJSON)
    world_sha256 = _sha256(world_path)
    projection_frame_sha256 = _projection_frame_sha256(samples_frame)
    sample_id_sha256 = _sample_id_hash(samples_frame["sample_id"])
    persistent_projection_sha256 = (
        _sha256(IDW_SAMPLES_CSV) if IDW_SAMPLES_CSV.is_file() else None
    )
    expected_provenance = {
        "schema_version": _GEOGRAPHY_AUDIT_SCHEMA_VERSION,
        "prediction_output_sha256": persistent_projection_sha256,
        "projection_frame_sha256": projection_frame_sha256,
        "sample_id_sha256": sample_id_sha256,
        "world_polygons_sha256": world_sha256,
        "n_projected": len(samples_frame),
    }
    if write_cache and IDW_GEOGRAPHY_SUMMARY_JSON.is_file():
        try:
            cached = json.loads(
                IDW_GEOGRAPHY_SUMMARY_JSON.read_text(encoding="utf-8")
            )
            if all(cached.get(key) == value for key, value in expected_provenance.items()):
                return cached
        except (OSError, ValueError, TypeError):
            pass

    world = geo.load_world()
    if "CONTINENT" not in world.columns:
        raise ValueError("Vendored world polygons have no CONTINENT field")
    europe = world.loc[world["CONTINENT"].astype(str).eq("Europe")].copy()
    if europe.empty:
        raise ValueError("Vendored world polygons contain no Europe features")
    if world.crs is None:
        world = world.set_crs(epsg=4326)
        europe = europe.set_crs(world.crs)
    points = gpd.GeoSeries(
        gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    )
    if points.crs != europe.crs:
        points = points.to_crs(europe.crs)
    europe_geometry = (
        europe.geometry.union_all()
        if hasattr(europe.geometry, "union_all")
        else europe.geometry.unary_union
    )
    in_europe = np.fromiter(
        (bool(europe_geometry.covers(point)) for point in points),
        dtype=bool,
        count=len(points),
    )
    n_europe = int(in_europe.sum())
    result = {
        **expected_provenance,
        "definition": (
            "IDW-predicted lon/lat point covered by a vendored Natural Earth "
            "country polygon with CONTINENT == Europe"
        ),
        "boundary_rule": "covers (country boundaries and coastlines included)",
        "continent_field": "CONTINENT",
        "continent_value": "Europe",
        "europe_polygon_count": len(europe),
        "world_polygons_path": str(world_path.resolve()),
        "prediction_output_path": (
            str(IDW_SAMPLES_CSV.resolve()) if IDW_SAMPLES_CSV.is_file() else None
        ),
        "n_europe": n_europe,
        "n_not_europe": int(len(samples_frame) - n_europe),
        "percent_europe": round(100.0 * n_europe / len(samples_frame), 6),
    }
    if write_cache:
        IDW_GEOGRAPHY_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        IDW_GEOGRAPHY_SUMMARY_JSON.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def cache_status() -> tuple[bool, str]:
    """Validate exact Phase 1 sample, source, schema, and checksum contracts."""
    required = (IDW_SAMPLES_CSV, IDW_SOURCES_CSV, IDW_MANIFEST_JSON)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing cache file(s): {', '.join(missing)}"
    try:
        manifest = json.loads(IDW_MANIFEST_JSON.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return False, "cache schema version changed"

        expected_hashes = manifest.get("output_sha256", {})
        for path in (IDW_SAMPLES_CSV, IDW_SOURCES_CSV):
            if expected_hashes.get(path.name) != _sha256(path):
                return False, f"checksum mismatch: {path.name}"

        sample_header = set(pd.read_csv(IDW_SAMPLES_CSV, nrows=0).columns)
        source_header = set(pd.read_csv(IDW_SOURCES_CSV, nrows=0).columns)
        if not _SAMPLE_COLUMNS.issubset(sample_header):
            return False, "sample cache is missing Phase 1 IDW columns"
        if not _SOURCE_COLUMNS.issubset(source_header):
            return False, "source cache is missing Phase 1 source-population columns"

        sample_cache = pd.read_csv(
            IDW_SAMPLES_CSV,
            usecols=["sample_id", "pred_longitude", "pred_latitude", "region"],
        )
        if sample_cache.empty:
            return False, "sample cache has no projected participants"
        cached_ids = sample_cache["sample_id"].astype(str).str.strip()
        if cached_ids.eq("").any() or cached_ids.duplicated().any():
            return False, "sample cache has blank or duplicate participant IDs"
        longitude = pd.to_numeric(sample_cache["pred_longitude"], errors="coerce")
        latitude = pd.to_numeric(sample_cache["pred_latitude"], errors="coerce")
        if not (
            np.isfinite(longitude).all()
            and np.isfinite(latitude).all()
            and longitude.between(-180.0, 180.0).all()
            and latitude.between(-90.0, 90.0).all()
        ):
            return False, "sample cache has invalid projected coordinates"
        if not sample_cache["region"].astype(str).eq("AoU").all():
            return False, "sample cache contains non-AoU projected rows"

        source_cache = pd.read_csv(IDW_SOURCES_CSV)
        source_longitude = pd.to_numeric(source_cache["lon"], errors="coerce")
        source_latitude = pd.to_numeric(source_cache["lat"], errors="coerce")
        if source_cache.empty or not (
            np.isfinite(source_longitude).all()
            and np.isfinite(source_latitude).all()
            and source_longitude.between(-180.0, 180.0).all()
            and source_latitude.between(-90.0, 90.0).all()
        ):
            return False, "source cache has invalid geographic coordinates"
        regions = set(source_cache["region"].astype(str).str.upper())
        if regions != set(EXPECTED_SOURCE_REGIONS):
            return False, f"source cache regions must be exactly AFR/EUR; found {sorted(regions)}"
        if not _is_1kg_project(source_cache["source_project"]).all():
            return False, "source cache contains a non-1000 Genomes anchor"

        parameters = manifest.get("parameters", {})
        if parameters.get("source_regions") != sorted(EXPECTED_SOURCE_REGIONS):
            return False, "manifest source-region contract is not AFR/EUR"
        if parameters.get("reference_panel") != "1000 Genomes only":
            return False, "manifest reference-panel contract is not 1000 Genomes only"

        sample_manifest = manifest.get("sample_manifest", {})
        if int(sample_manifest.get("n_projected", -1)) != len(sample_cache):
            return False, "sample cache row count does not match its manifest"
        if _sample_id_hash(cached_ids) != sample_manifest.get("sample_id_sha256"):
            return False, "sample cache IDs do not match their manifest"

        current_ids = samples.load_aou_sample_list(download=False)
        if not current_ids:
            return False, "current AoU sample manifest is unavailable on Workbench"
        if current_ids and _sample_id_hash(current_ids) != sample_manifest.get(
            "sample_id_sha256"
        ):
            return False, "cached projections do not match the current sample manifest"
    except Exception as exc:  # noqa: BLE001 - status probes must return, not fail
        return False, f"cache validation failed: {exc}"
    return True, "Phase 1 sample manifest, 1KG AFR/EUR sources, and checksums valid"


def outputs_exist() -> bool:
    return cache_status()[0]


def _pc_columns(
    frame: pd.DataFrame, array_col: str = "pca_features"
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize supported PCA layouts to ``PC1..PCn`` columns."""
    feature_columns = sorted(
        [column for column in frame.columns if column.startswith("pca_features_")],
        key=lambda value: int(re.search(r"(\d+)$", value).group(1)),
    )
    if feature_columns:
        frame = frame.rename(
            columns={column: f"PC{index + 1}" for index, column in enumerate(feature_columns)}
        )
        return frame, [f"PC{index + 1}" for index in range(len(feature_columns))]
    if "PC1" in frame.columns:
        columns = sorted(
            [column for column in frame.columns if re.fullmatch(r"PC\d+", column)],
            key=lambda value: int(value[2:]),
        )
        return frame, columns
    if array_col in frame.columns:
        values = frame[array_col].apply(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
        matrix = pd.DataFrame(values.tolist(), index=frame.index)
        matrix.columns = [f"PC{index + 1}" for index in range(matrix.shape[1])]
        frame = pd.concat([frame.drop(columns=[array_col]), matrix], axis=1)
        return frame, list(matrix.columns)
    raise ValueError(f"Cannot find PC columns in {list(frame.columns)[:10]}")


def _column(meta: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in meta.columns:
            return candidate
    raise ValueError(f"gnomAD metadata has no {label} column; tried {candidates}")


def _excluded_geography(longitude: pd.Series, latitude: pd.Series) -> pd.Series:
    """Flag US/Canada and Caribbean relocation sites (ASW/ACB/CEU)."""
    north_america = longitude.between(-170, -30) & latitude.between(25, 85)
    caribbean = longitude.between(-90, -58) & latitude.between(10, 27)
    return north_america | caribbean


def _reference_panel(training_local: str, meta_local: str) -> pd.DataFrame:
    """Join PCA and geography; retain only 1KG AFR/EUR source samples."""
    training = pd.read_csv(training_local, sep="\t")
    training, _ = _pc_columns(training, array_col="scores")
    if "s" not in training.columns:
        raise ValueError("training PCA table must contain sample column 's'")

    meta = pd.read_csv(meta_local, sep="\t", low_memory=False)
    id_col = _column(meta, ("s", "sample_id"), "sample ID")
    lat_col = _column(meta, ("hgdp_tgp_meta.Latitude", "latitude"), "latitude")
    lon_col = _column(meta, ("hgdp_tgp_meta.Longitude", "longitude"), "longitude")
    population_col = _column(
        meta, ("hgdp_tgp_meta.Population", "population"), "source population"
    )
    project_col = _column(
        meta,
        ("hgdp_tgp_meta.Project", "project_meta.research_project", "source"),
        "source project",
    )
    genetic_col = _column(
        meta,
        ("hgdp_tgp_meta.Genetic.region", "population_inference.pop"),
        "genetic region",
    )

    selected = meta[
        [id_col, lat_col, lon_col, population_col, project_col, genetic_col]
    ].copy()
    selected.columns = [
        "gnomad_s",
        "latitude",
        "longitude",
        "pop_code",
        "source_project",
        "source_region",
    ]
    selected["gnomad_s"] = selected["gnomad_s"].astype(str)
    selected["latitude"] = pd.to_numeric(selected["latitude"], errors="coerce")
    selected["longitude"] = pd.to_numeric(selected["longitude"], errors="coerce")

    ref = training.merge(selected, left_on="s", right_on="gnomad_s", how="inner")
    pc_columns = [column for column in ref.columns if re.fullmatch(r"PC\d+", column)]
    joined_rows = len(ref)
    ref = ref.dropna(subset=["latitude", "longitude", "pop_code", *pc_columns]).copy()
    missing_rows = joined_rows - len(ref)
    raw_region = ref["source_region"].astype(str).str.upper()
    ref["genetic_region"] = raw_region.map(_GENETIC_REGION_MAP)
    if genetic_col == "population_inference.pop":
        ref["genetic_region"] = (
            ref["source_region"].astype(str).str.lower().map(_INFERENCE_REGION_MAP)
        )

    ref["source_project"] = ref["source_project"].astype(str).str.strip()
    ref["pop_code"] = ref["pop_code"].astype(str).str.strip()
    # Match the last known-good Phase 1 implementation's canonical 1KG sample
    # IDs and also require the metadata project label.  Either check alone is
    # too permissive for a file that contains both HGDP and 1KG samples.
    canonical_1kg_id = ref["s"].astype(str).str.match(r"^(HG|NA)\d")
    candidate = (
        canonical_1kg_id
        & _is_1kg_project(ref["source_project"])
        & ref["genetic_region"].isin(EXPECTED_SOURCE_REGIONS)
    )
    candidate_rows = ref.loc[candidate].copy()
    excluded = _excluded_geography(
        candidate_rows["longitude"], candidate_rows["latitude"]
    )
    excluded_rows = candidate_rows.loc[excluded]
    ref = candidate_rows.loc[~excluded].copy()
    if ref.empty:
        raise ValueError("No 1000 Genomes AFR/EUR references remain after filtering")
    observed_regions = set(ref["genetic_region"].astype(str))
    if observed_regions != set(EXPECTED_SOURCE_REGIONS):
        raise ValueError(
            "1000 Genomes reference must contain both AFR and EUR; "
            f"found {sorted(observed_regions)}"
        )

    ref.attrs["source_audit"] = {
        "training_rows": len(training),
        "joined_rows": joined_rows,
        "missing_coordinate_or_pc_rows": missing_rows,
        "candidate_1kg_afr_eur_rows": len(candidate_rows),
        "excluded_us_canada_or_caribbean_rows": len(excluded_rows),
        "excluded_populations": sorted(excluded_rows["pop_code"].unique()),
        "retained_rows": len(ref),
        "retained_population_codes": sorted(ref["pop_code"].unique()),
        "retained_region_counts": {
            str(key): int(value)
            for key, value in ref["genetic_region"].value_counts().sort_index().items()
        },
    }
    return ref


def _ensure_input(gs_src: str, local: Path) -> str:
    """Download a GCS input into ``data/raw/idw`` only when absent."""
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        result = gcs.gcs_cp(gs_src, str(local))
        if result.returncode != 0:
            raise RuntimeError(
                f"gcloud storage cp failed for {gs_src}: {result.stderr.strip()[:300]}"
            )
    return str(local)


def _input_fingerprint(path: str) -> dict[str, object]:
    local = Path(path)
    stat = local.stat()
    return {
        "path": str(local.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(local),
    }


def build_idw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache Phase 1 projections from 1KG AFR/EUR anchors."""
    from scipy.spatial import cKDTree

    predictions_local = _ensure_input(
        config.ANCESTRY_PREDS_GS, _RAW_DIR / "ancestry_preds.tsv"
    )
    training_local = _ensure_input(
        config.TRAINING_PCA_GS, _RAW_DIR / "training_pca.tsv"
    )
    eigenvalues_local = _ensure_input(
        config.EIGENVALUES_GS, _RAW_DIR / "eigenvalues.txt"
    )
    metadata_local = _ensure_input(
        config.HGDP_1KG_META_TSV, _RAW_DIR / "gnomad_meta.tsv"
    )

    aou, pc_columns = _pc_columns(pd.read_csv(predictions_local, sep="\t"))
    reference = _reference_panel(training_local, metadata_local)
    source_audit = dict(reference.attrs.get("source_audit", {}))
    eigenvalues = np.atleast_1d(np.loadtxt(eigenvalues_local)).astype(float)

    id_col = "research_id" if "research_id" in aou.columns else (
        "person_id" if "person_id" in aou.columns else aou.columns[0]
    )
    aou[id_col] = aou[id_col].astype(str).str.strip()
    phase1_ids = samples.load_aou_sample_list(download=True)
    if not phase1_ids:
        raise RuntimeError("The authoritative Phase 1 sample manifest is unavailable")
    normalized_ids = {str(value).strip() for value in phase1_ids if str(value).strip()}
    aou = aou[aou[id_col].isin(normalized_ids)].copy()
    observed_ids = set(aou[id_col])
    if observed_ids != normalized_ids:
        raise ValueError(
            "AoU ancestry predictions do not exactly match the Phase 1 manifest; "
            "inspect the ancestry input and sample manifest in Workbench."
        )
    if aou[id_col].duplicated().any():
        raise ValueError(f"AoU PCA input has duplicate {id_col} values")

    n_pcs = min(config.IDW_N_PCS, len(pc_columns), len(eigenvalues))
    if n_pcs < 2:
        raise ValueError(f"IDW requires at least two PCs/eigenvalues; found {n_pcs}")
    pcs = [f"PC{index + 1}" for index in range(n_pcs)]
    for frame_name, frame in (("AoU", aou), ("reference", reference)):
        numeric = frame[pcs].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{frame_name} PC matrix contains non-finite values")
        frame.loc[:, pcs] = numeric
    if not np.isfinite(eigenvalues[:n_pcs]).all() or (eigenvalues[:n_pcs] <= 0).any():
        raise ValueError("IDW eigenvalues must be finite and positive")

    corrected = aou[pcs].copy()
    for pc in pcs:
        reference_sd = float(reference[pc].std())
        ratio = float(aou[pc].std()) / reference_sd if reference_sd > 0 else 1.0
        if np.isfinite(ratio) and 0 < ratio < 0.9:
            corrected[pc] = aou[pc] / ratio
    scale = np.sqrt(eigenvalues[:n_pcs] / eigenvalues[:n_pcs].sum())[np.newaxis, :]

    reference_pcs = reference[pcs].to_numpy(float) * scale
    reference_coords = reference[["latitude", "longitude"]].to_numpy(float)
    query = corrected[pcs].to_numpy(float) * scale
    k_nearest = min(config.IDW_K, len(reference))
    distances, indices = cKDTree(reference_pcs).query(query, k=k_nearest)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    sigma = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-12)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weight_sum = weights.sum(axis=1)
    aou["pred_latitude"] = (
        weights * reference_coords[indices, 0]
    ).sum(axis=1) / weight_sum
    aou["pred_longitude"] = (
        weights * reference_coords[indices, 1]
    ).sum(axis=1) / weight_sum

    samples_frame = aou[[id_col, "pred_longitude", "pred_latitude"]].rename(
        columns={id_col: "sample_id"}
    )
    samples_frame["region"] = "AoU"
    sources_frame = (
        reference.groupby(["pop_code", "source_project", "genetic_region"])
        .agg(
            lon=("longitude", "mean"),
            lat=("latitude", "mean"),
            n_reference=("gnomad_s", "size"),
        )
        .reset_index()
        .rename(columns={"genetic_region": "region"})
        .sort_values(["region", "pop_code"])
        .reset_index(drop=True)
    )

    IDW_SAMPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    samples_frame.to_csv(IDW_SAMPLES_CSV, index=False)
    sources_frame.to_csv(IDW_SOURCES_CSV, index=False)
    geography_audit = european_continent_summary(samples_frame, write_cache=True)
    manifest = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "analysis": "Phase 1 Gaussian IDW in shrinkage-corrected eigenvalue-weighted PC space",
        "parameters": {
            "n_pcs": int(n_pcs),
            "k_nearest": int(k_nearest),
            "requested_n_pcs": int(config.IDW_N_PCS),
            "requested_k_nearest": int(config.IDW_K),
            "reference_panel": "1000 Genomes only",
            "source_regions": sorted(EXPECTED_SOURCE_REGIONS),
            "excluded_geography": "North America and Caribbean relocation boxes",
        },
        "inputs": {
            "ancestry_predictions": _input_fingerprint(predictions_local),
            "training_pca": _input_fingerprint(training_local),
            "eigenvalues": _input_fingerprint(eigenvalues_local),
            "gnomad_metadata": _input_fingerprint(metadata_local),
        },
        "sample_manifest": {
            "n_projected": len(samples_frame),
            "sample_id_sha256": _sample_id_hash(samples_frame["sample_id"]),
        },
        "source_audit": source_audit,
        "geography_audit": geography_audit,
        "outputs": {
            "source_populations": len(sources_frame),
            "source_region_counts": {
                str(key): int(value)
                for key, value in sources_frame["region"].value_counts().sort_index().items()
            },
        },
        "output_sha256": {
            IDW_SAMPLES_CSV.name: _sha256(IDW_SAMPLES_CSV),
            IDW_SOURCES_CSV.name: _sha256(IDW_SOURCES_CSV),
        },
    }
    IDW_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return samples_frame, sources_frame


def load_idw_frames() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return checksum-validated cached frames, or ``None`` when invalid."""
    valid, _reason = cache_status()
    if not valid:
        return None
    return pd.read_csv(IDW_SAMPLES_CSV), pd.read_csv(IDW_SOURCES_CSV)
