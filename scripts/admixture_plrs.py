"""PLS imputation of region-level EUR local-ancestry dosage.

Each sign/valley-aware LAI outlier region receives an independent PLS model.
The marker genotypes in ``lai_masked/*.anc.vcf.gz`` are predictors and the
length-weighted FLARE EUR ancestry dosage (0--2) is the target.  Models are
retained for downstream PheWAS only when their held-out :math:`R^2` exceeds
0.5, as requested for the new admixture reference analysis.
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import admixture_inputs, admixture_lai
from .common import samples as sample_set


MODEL_CONTRACT = "figure4_plrs_real_diploids_unrelated1025_v2"
R2_MIN = float(os.environ.get("AOU_FIG4_PLRS_R2_MIN", "0.5"))
MIN_MAF = float(os.environ.get("AOU_FIG4_PLRS_MIN_MAF", "0.01"))
TEST_FRACTION = float(os.environ.get("AOU_FIG4_PLRS_TEST_FRACTION", "0.2"))
RANDOM_STATE = int(os.environ.get("AOU_FIG4_PLRS_RANDOM_STATE", "42"))
MAX_COMPONENTS = int(os.environ.get("AOU_FIG4_PLRS_MAX_COMPONENTS", "20"))
MIN_OBSERVED_FEATURE_FRACTION = float(
    os.environ.get("AOU_FIG4_MIN_OBSERVED_FEATURE_FRACTION", "0.1")
)
PREDICTION_COLUMNS = [
    "person_id", "region_id", "eur_dosage", "n_model_snvs",
    "n_observed_snvs", "observed_snv_fraction", "plrs_r2_holdout",
    "plrs_model_contract", "plrs_training_n",
]


def _progress(message: str, *, stage: int = 2) -> None:
    """Emit an immediately visible workflow diagnostic in notebooks."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[admixture_reference][stage{stage} {stamp}] {message}", flush=True)


@lru_cache(maxsize=1)
def _unrelated_training_ids() -> frozenset[str]:
    """Load and validate the canonical unrelated Phase 1 training panel."""
    unrelated_ids = sample_set.load_aou_sample_list(download=True)
    if unrelated_ids is None:
        raise FileNotFoundError(
            "The canonical 1,025-person AoU sample list could not be loaded from "
            f"{sample_set.AOU_SAMPLE_LIST_GS}"
        )
    unrelated_ids = set(map(str, unrelated_ids)).difference(
        sample_set.load_excluded_sample_ids()
    )
    if len(unrelated_ids) != sample_set.N_AOU:
        raise ValueError(
            f"Canonical AoU sample list must contain {sample_set.N_AOU} unrelated "
            f"individuals; found {len(unrelated_ids)}"
        )
    return frozenset(unrelated_ids)


def analysis_dir() -> Path:
    path = admixture_lai.analysis_dir().parent / "plrs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    path = analysis_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(region_id: str) -> Path:
    return model_dir() / f"{region_id}.pkl"


def _sklearn():
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import KFold, train_test_split
    except ImportError as exc:  # pragma: no cover - Workbench dependency path
        raise RuntimeError(
            "admixture reference PLRS requires scikit-learn; run "
            "`python -m scripts.recompute_admixture_reference`"
        ) from exc
    return PLSRegression, mean_squared_error, r2_score, KFold, train_test_split


def _marker_widths(positions: np.ndarray, start: int, end: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 1:
        return np.asarray([max(end - start + 1, 1)], dtype=float)
    midpoints = (positions[:-1] + positions[1:]) / 2.0
    edges = np.concatenate([[float(start)], midpoints, [float(end + 1)]])
    return np.maximum(np.diff(edges), 1.0)


def extract_training_region(vcf_path: str | os.PathLike, region: pd.Series,
                            eur_idx: int) -> dict:
    """Extract genotype predictors and true length-weighted EUR dosage."""
    try:
        from cyvcf2 import VCF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cyvcf2 is required for admixture reference PLRS") from exc
    chrom = int(region["chrom"])
    start, end = int(region["start"]), int(region["end"])
    unrelated_ids = _unrelated_training_ids()
    header = VCF(str(vcf_path))
    try:
        selected_samples = [
            str(sample) for sample in header.samples
            if str(sample) in unrelated_ids
        ]
    finally:
        header.close()
    if len(selected_samples) != sample_set.N_AOU:
        raise ValueError(
            f"PLRS requires exactly {sample_set.N_AOU} unrelated LAI samples; "
            f"found {len(selected_samples)} in {vcf_path}"
        )
    vcf = VCF(str(vcf_path), samples=selected_samples)
    samples = np.asarray(vcf.samples, dtype=str)
    token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
    if token is None:
        vcf.close()
        raise ValueError(f"Chromosome {chrom} is absent from {vcf_path}")
    feature_ids: list[str] = []
    positions: list[int] = []
    dosage_columns: list[np.ndarray] = []
    ancestry1: list[np.ndarray] = []
    ancestry2: list[np.ndarray] = []
    try:
        for record in vcf(f"{token}:{start}-{end}"):
            if len(record.ALT or []) != 1 or len(record.REF) != 1 or len(record.ALT[0]) != 1:
                continue
            dosage = admixture_inputs.genotype_dosage(record)
            called = np.isfinite(dosage)
            if called.sum() < 2:
                continue
            af = float(np.nanmean(dosage) / 2.0)
            maf = min(af, 1.0 - af)
            if not np.isfinite(maf) or maf < MIN_MAF:
                continue
            an1, an2 = record.format("AN1"), record.format("AN2")
            if an1 is None or an2 is None:
                continue
            call1 = np.asarray(an1).reshape(-1)
            call2 = np.asarray(an2).reshape(-1)
            if len(call1) != len(samples) or len(call2) != len(samples):
                continue
            feature_ids.append(admixture_inputs.variant_id(
                record.CHROM, record.POS, record.REF, record.ALT[0]
            ))
            positions.append(int(record.POS))
            dosage_columns.append(dosage)
            ancestry1.append(call1)
            ancestry2.append(call2)
    finally:
        vcf.close()
    if not dosage_columns:
        raise ValueError(f"No eligible SNVs in {region['region_id']}")
    order = np.argsort(positions)
    feature_ids = [feature_ids[index] for index in order]
    positions_array = np.asarray(positions, dtype=np.int64)[order]
    X = np.column_stack(dosage_columns)[:, order].astype(float)
    an1 = np.column_stack(ancestry1)[:, order]
    an2 = np.column_stack(ancestry2)[:, order]
    weights = _marker_widths(positions_array, start, end)
    valid1, valid2 = an1 >= 0, an2 >= 0
    denominator1 = valid1 @ weights
    denominator2 = valid2 @ weights
    hap1 = np.divide((an1 == eur_idx) @ weights, denominator1,
                     out=np.full(len(samples), np.nan), where=denominator1 > 0)
    hap2 = np.divide((an2 == eur_idx) @ weights, denominator2,
                     out=np.full(len(samples), np.nan), where=denominator2 > 0)
    y = hap1 + hap2
    return {
        "samples": samples, "feature_ids": feature_ids,
        "positions": positions_array, "X": X, "y": y,
    }


def _fit_best_pls(X_train: np.ndarray, y_train: np.ndarray):
    PLSRegression, _, _, KFold, _ = _sklearn()
    max_components = max(1, min(MAX_COMPONENTS, X_train.shape[1],
                                X_train.shape[0] - 1))
    candidates = list(range(1, max_components + 1))
    n_splits = min(5, max(2, X_train.shape[0] // 50), X_train.shape[0])
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    losses = []
    for components in candidates:
        fold_losses = []
        for train_index, validation_index in folds.split(X_train):
            usable = min(components, len(train_index) - 1, X_train.shape[1])
            model = PLSRegression(n_components=max(1, usable), scale=True)
            model.fit(X_train[train_index], y_train[train_index])
            prediction = np.asarray(model.predict(X_train[validation_index])).reshape(-1)
            fold_losses.append(float(np.mean(
                np.square(y_train[validation_index] - prediction)
            )))
        losses.append(float(np.mean(fold_losses)))
    best_components = candidates[int(np.argmin(losses))]
    model = PLSRegression(n_components=best_components, scale=True)
    model.fit(X_train, y_train)
    return model, best_components, dict(zip(candidates, losses))


def fit_region_model(region: pd.Series, vcf_path: Path, eur_idx: int) -> tuple[dict, pd.DataFrame]:
    _, mean_squared_error, r2_score, _, train_test_split = _sklearn()
    extracted = extract_training_region(vcf_path, region, eur_idx)
    X, y = extracted["X"], extracted["y"]
    usable = np.isfinite(y)
    X, y = X[usable], y[usable]
    samples = extracted["samples"][usable]
    if len(y) < 30:
        raise ValueError(f"Only {len(y)} usable LAI samples in {region['region_id']}")
    train_index, test_index = train_test_split(
        np.arange(len(y)), test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )
    feature_means = np.nanmean(X[train_index], axis=0)
    feature_means = np.where(np.isfinite(feature_means), feature_means, 0.0)
    X_filled = np.where(np.isfinite(X), X, feature_means)
    model, components, cv_losses = _fit_best_pls(X_filled[train_index], y[train_index])
    prediction = np.clip(
        np.asarray(model.predict(X_filled[test_index])).reshape(-1), 0.0, 2.0
    )
    null_prediction = np.full(len(test_index), float(np.mean(y[train_index])))
    r2 = float(r2_score(y[test_index], prediction))
    rmse = float(math.sqrt(mean_squared_error(y[test_index], prediction)))
    null_r2 = float(r2_score(y[test_index], null_prediction))
    payload = {
        "contract": MODEL_CONTRACT,
        "region": {key: (value.item() if isinstance(value, np.generic) else value)
                   for key, value in region.to_dict().items()},
        "feature_ids": extracted["feature_ids"],
        "feature_positions": extracted["positions"],
        "feature_means": feature_means,
        "model": model,
        "n_components": components,
        "cv_mse": cv_losses,
        "r2_holdout": r2,
        "rmse_holdout": rmse,
        "r2_null": null_r2,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "n_source_samples": int(len(extracted["samples"])),
        "r2_threshold": R2_MIN,
    }
    holdout = pd.DataFrame({
        "region_id": str(region["region_id"]),
        "sample_id": samples[test_index],
        "true_eur_dosage": y[test_index],
        "predicted_eur_dosage": prediction,
    })
    return payload, holdout


def reconstruct_holdout_predictions(region_id: str) -> pd.DataFrame:
    """Recreate a region's deterministic 20% holdout from its cached model."""
    payload = admixture_inputs.pickle_load(model_path(str(region_id)))
    if payload.get("contract") != MODEL_CONTRACT:
        raise ValueError(f"PLRS model cache is stale for {region_id}")
    region = pd.Series(payload["region"])
    vcf_path = admixture_lai.stage_lai_chromosome(int(region["chrom"]))
    mapping = admixture_lai.parse_ancestry_header(vcf_path)
    eur_idx, _, _, _ = admixture_lai.ancestry_indices(mapping)
    extracted = extract_training_region(vcf_path, region, eur_idx)
    source_index = {
        variant_id: index
        for index, variant_id in enumerate(extracted["feature_ids"])
    }
    expected = list(payload["feature_ids"])
    missing = [variant_id for variant_id in expected if variant_id not in source_index]
    if missing:
        raise ValueError(
            f"Cannot reconstruct {region_id}: {len(missing)} model SNVs are absent"
        )
    X = extracted["X"][:, [source_index[variant_id] for variant_id in expected]]
    y = extracted["y"]
    usable = np.isfinite(y)
    X, y = X[usable], y[usable]
    samples = extracted["samples"][usable]
    _, _, _, _, train_test_split = _sklearn()
    _, test_index = train_test_split(
        np.arange(len(y)), test_size=TEST_FRACTION, random_state=RANDOM_STATE,
    )
    feature_means = np.asarray(payload["feature_means"], dtype=float)
    X_filled = np.where(np.isfinite(X), X, feature_means)
    prediction = np.clip(
        np.asarray(payload["model"].predict(X_filled[test_index])).reshape(-1),
        0.0,
        2.0,
    )
    return pd.DataFrame({
        "region_id": str(region_id),
        "sample_id": samples[test_index],
        "true_eur_dosage": y[test_index],
        "predicted_eur_dosage": prediction,
    })


def fit_plrs_models(*, force: bool = False) -> pd.DataFrame:
    """Fit/resume every outlier-region model and write the QC summary."""
    regions_path = admixture_lai.analysis_dir() / "outlier_regions.tsv"
    if not regions_path.exists():
        raise FileNotFoundError("Run the admixture reference LAI stage before PLRS fitting")
    regions = pd.read_csv(regions_path, sep="\t")
    training_ids = _unrelated_training_ids()
    _progress(f"loaded canonical unrelated PLRS training panel: n={len(training_ids):,}")
    _progress(f"checking {len(regions):,} region-specific PLRS models")
    model_started = time.perf_counter()
    first_vcf = admixture_lai.stage_lai_chromosome(1)
    mapping = admixture_lai.parse_ancestry_header(first_vcf)
    eur_idx, _, _, _ = admixture_lai.ancestry_indices(mapping)
    rows = []
    holdout_frames = []
    cached_count = 0
    fitted_count = 0
    for region_number, region in enumerate(regions.itertuples(index=False), start=1):
        region_series = pd.Series(region._asdict())
        path = model_path(str(region_series["region_id"]))
        payload = None
        if path.exists() and not force:
            payload = admixture_inputs.pickle_load(path)
            if payload.get("contract") != MODEL_CONTRACT:
                payload = None
            else:
                cached_count += 1
        if payload is None:
            try:
                payload, holdout = fit_region_model(
                    region_series,
                    admixture_lai.stage_lai_chromosome(int(region_series["chrom"])),
                    eur_idx,
                )
            except Exception as exc:
                print(f"[admixture_reference] PLRS skipped {region_series['region_id']}: {exc}")
                continue
            admixture_inputs.atomic_pickle_dump(payload, path)
            holdout_frames.append(holdout)
            fitted_count += 1
        rows.append({
            **payload["region"],
            "n_snvs": len(payload["feature_ids"]),
            "n_components": int(payload["n_components"]),
            "n_train": int(payload["n_train"]),
            "n_test": int(payload["n_test"]),
            "n_source_samples": int(payload["n_source_samples"]),
            "r2_holdout": float(payload["r2_holdout"]),
            "rmse_holdout": float(payload["rmse_holdout"]),
            "r2_null": float(payload["r2_null"]),
            "r2_threshold": R2_MIN,
            "retained_for_phewas": float(payload["r2_holdout"]) > R2_MIN,
            "model_contract": MODEL_CONTRACT,
        })
        if region_number == 1 or region_number % 10 == 0 or region_number == len(regions):
            _progress(
                f"PLRS model {region_number:,}/{len(regions):,}: "
                f"{region_series['region_id']} "
                f"({cached_count:,} cached, {fitted_count:,} fitted this run)"
            )
    summary = pd.DataFrame(rows)
    summary_path = analysis_dir() / "plrs_models_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    if holdout_frames:
        pd.concat(holdout_frames, ignore_index=True).to_csv(
            analysis_dir() / "plrs_holdout_predictions.tsv.gz",
            sep="\t", index=False, compression="gzip",
        )
    retained = int(summary.get("retained_for_phewas", pd.Series(dtype=bool)).sum())
    _progress(
        f"PLRS retained {retained}/{len(summary)} regions at R² > {R2_MIN}; "
        f"model check completed in {time.perf_counter() - model_started:,.1f} s"
    )
    return summary


def _cohort_ids(covariates_path: Path) -> list[str]:
    header = pd.read_csv(covariates_path, sep="\t", nrows=1)
    id_column = next((name for name in
                      ("person_id", "sample_id", "research_id", "s", "IID")
                      if name in header.columns), header.columns[0])
    values = pd.read_csv(covariates_path, sep="\t", usecols=[id_column])[id_column]
    ids = values.dropna().astype(str).drop_duplicates()
    return ids[~ids.isin(sample_set.load_excluded_sample_ids())].tolist()


def _fill_feature_matrix(vcf_paths: list[Path], chrom: int, start: int, end: int,
                         feature_ids: list[str], allowed_samples: list[str]) -> tuple[np.ndarray, np.ndarray]:
    try:
        from cyvcf2 import VCF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cyvcf2 is required for admixture reference PLRS prediction") from exc
    feature_lookup = {value: index for index, value in enumerate(feature_ids)}
    allowed_lookup = {sample: index for index, sample in enumerate(allowed_samples)}
    X = np.full((len(allowed_samples), len(feature_ids)), np.nan, dtype=float)
    observed_features = np.zeros(len(feature_ids), dtype=bool)
    for path in vcf_paths:
        # ACAF shards can contain hundreds of thousands of samples.  Read the
        # header first and reopen with only the cohort samples so genotype
        # decoding and memory scale with the analysis cohort, not the shard.
        header = VCF(str(path))
        try:
            available = set(map(str, header.samples))
        finally:
            header.close()
        shared_samples = [sample for sample in allowed_samples
                          if sample in available]
        if not shared_samples:
            continue
        vcf = VCF(str(path), samples=shared_samples)
        token = admixture_inputs.detect_vcf_chrom(vcf, chrom)
        if token is None:
            vcf.close()
            continue
        sample_pairs = [(vcf_index, allowed_lookup[sample])
                        for vcf_index, sample in enumerate(map(str, vcf.samples))]
        vcf_indices = np.asarray([pair[0] for pair in sample_pairs], dtype=int)
        output_indices = np.asarray([pair[1] for pair in sample_pairs], dtype=int)
        try:
            for record in vcf(f"{token}:{int(start)}-{int(end)}"):
                if len(record.ALT or []) != 1:
                    continue
                identifier = admixture_inputs.variant_id(
                    record.CHROM, record.POS, record.REF, record.ALT[0]
                )
                feature_index = feature_lookup.get(identifier)
                if feature_index is None:
                    continue
                dosage = admixture_inputs.genotype_dosage(record)
                X[output_indices, feature_index] = dosage[vcf_indices]
                observed_features[feature_index] = True
        finally:
            vcf.close()
    return X, observed_features


def _predict_payload(payload: dict, X: np.ndarray,
                     samples: list[str]) -> pd.DataFrame:
    feature_means = np.asarray(payload["feature_means"], dtype=float)
    observed = np.isfinite(X)
    fraction = observed.mean(axis=1)
    filled = np.where(observed, X, feature_means)
    prediction = np.clip(
        np.asarray(payload["model"].predict(filled)).reshape(-1), 0.0, 2.0
    )
    prediction[fraction < MIN_OBSERVED_FEATURE_FRACTION] = np.nan
    return pd.DataFrame({
        "person_id": samples,
        "region_id": str(payload["region"]["region_id"]),
        "eur_dosage": prediction,
        "n_model_snvs": len(payload["feature_ids"]),
        "n_observed_snvs": observed.sum(axis=1),
        "observed_snv_fraction": fraction,
        "plrs_r2_holdout": float(payload["r2_holdout"]),
        "plrs_model_contract": str(payload["contract"]),
        "plrs_training_n": int(payload["n_source_samples"]),
    })


def _retained_payloads(summary: pd.DataFrame, *, region_ids=None) -> list[dict]:
    retained = summary[summary["retained_for_phewas"].astype(bool)]
    if region_ids is not None:
        requested = {str(region_id) for region_id in region_ids}
        retained = retained[retained["region_id"].astype(str).isin(requested)]
    return [admixture_inputs.pickle_load(model_path(str(region_id)))
            for region_id in retained["region_id"]]


def _prediction_cache_current(path: Path) -> bool:
    """Accept predictions only when built from the unrelated-1025 models."""
    if not path.exists():
        return False
    try:
        contract = pd.read_csv(
            path, sep="\t", usecols=["plrs_model_contract"], nrows=10
        )["plrs_model_contract"]
    except (ValueError, KeyError, pd.errors.EmptyDataError, OSError):
        return False
    return not contract.empty and contract.astype(str).eq(MODEL_CONTRACT).all()


def predict_discovery(summary: pd.DataFrame, *, force: bool = False) -> Path:
    output = analysis_dir() / "eur_dosage_predictions.discovery.tsv.gz"
    requested = set(summary.loc[
        summary["retained_for_phewas"].astype(bool), "region_id"
    ].astype(str))
    if not force and _prediction_cache_current(output):
        cached_regions = set(pd.read_csv(
            output, sep="\t", usecols=["region_id"], dtype=str,
        )["region_id"].dropna().astype(str))
        if cached_regions == requested:
            _progress(f"discovery predictions already cached: {output}")
            return output
    _progress("discovery imputation: staging cohort files")
    sources = admixture_inputs.stage_cohort_sources("discovery")
    samples = _cohort_ids(sources["covariates"])
    _progress(f"discovery imputation: loaded {len(samples):,} cohort sample IDs")
    _progress("discovery imputation: staging indexed imputed VCF")
    vcf_path = admixture_inputs.stage_discovery_vcf()
    rows = []
    payloads = _retained_payloads(summary)
    _progress(f"discovery imputation: predicting {len(payloads):,} retained regions")
    for region_number, payload in enumerate(payloads, start=1):
        region = payload["region"]
        started = time.perf_counter()
        _progress(
            f"discovery region {region_number}/{len(payloads)}: "
            f"{region['region_id']} ({len(payload['feature_ids']):,} model SNVs)"
        )
        X, observed_features = _fill_feature_matrix(
            [vcf_path], int(region["chrom"]), int(region["start"]), int(region["end"]),
            list(payload["feature_ids"]), samples,
        )
        rows.append(_predict_payload(payload, X, samples))
        _progress(
            f"discovery region {region_number}/{len(payloads)} complete: "
            f"{int(observed_features.sum()):,}/{len(observed_features):,} SNVs observed "
            f"in {time.perf_counter() - started:,.1f} s"
        )
    frame = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=PREDICTION_COLUMNS))
    _progress(f"discovery imputation: writing {len(frame):,} rows to {output}")
    frame.to_csv(output, sep="\t", index=False, compression="gzip")
    _progress("discovery imputation complete")
    return output


def predict_replication(summary: pd.DataFrame, *, region_ids,
                        force: bool = False) -> Path:
    """Impute replication dosage only for discovery-significant regions."""
    output = analysis_dir() / "eur_dosage_predictions.replication.tsv.gz"
    requested = {str(region_id) for region_id in region_ids}
    if not requested:
        raise ValueError("Replication prediction requires at least one discovery region")
    if not force and _prediction_cache_current(output):
        cached_regions = set(pd.read_csv(
            output, sep="\t", usecols=["region_id"], dtype=str,
        )["region_id"].dropna().astype(str))
        if cached_regions == requested:
            _progress(f"replication predictions already cached: {output}", stage=4)
            return output
    _progress("replication imputation: staging cohort files and shard index", stage=4)
    sources = admixture_inputs.stage_cohort_sources("replication")
    samples = _cohort_ids(sources["covariates"])
    shard_index = admixture_inputs.load_replication_shard_index()
    rows = []
    payloads = _retained_payloads(summary, region_ids=requested)
    available = {str(payload["region"]["region_id"]) for payload in payloads}
    missing = requested.difference(available)
    if missing:
        raise ValueError(
            "Discovery-significant regions lack retained PLRS models: "
            + ", ".join(sorted(missing))
        )
    _progress(
        f"replication imputation: loaded {len(samples):,} sample IDs; "
        f"predicting {len(payloads):,} discovery-significant regions",
        stage=4,
    )
    for region_number, payload in enumerate(payloads, start=1):
        region = payload["region"]
        started = time.perf_counter()
        remote_shards = admixture_inputs.shards_for_region(
            shard_index, int(region["chrom"]), int(region["start"]), int(region["end"])
        )
        if not remote_shards:
            _progress(
                f"replication region {region_number}/{len(payloads)}: "
                f"no ACAF shard for {region['region_id']}; skipped",
                stage=4,
            )
            continue
        _progress(
            f"replication region {region_number}/{len(payloads)}: "
            f"{region['region_id']} ({len(remote_shards)} shard(s), "
            f"{len(payload['feature_ids']):,} model SNVs)",
            stage=4,
        )
        local_shards = [admixture_inputs.stage_acaf_shard(path) for path in remote_shards]
        X, observed_features = _fill_feature_matrix(
            local_shards, int(region["chrom"]), int(region["start"]), int(region["end"]),
            list(payload["feature_ids"]), samples,
        )
        rows.append(_predict_payload(payload, X, samples))
        _progress(
            f"replication region {region_number}/{len(payloads)} complete: "
            f"{int(observed_features.sum()):,}/{len(observed_features):,} SNVs observed "
            f"in {time.perf_counter() - started:,.1f} s",
            stage=4,
        )
    frame = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=PREDICTION_COLUMNS))
    _progress(
        f"replication imputation: writing {len(frame):,} rows to {output}",
        stage=4,
    )
    frame.to_csv(output, sep="\t", index=False, compression="gzip")
    _progress("replication imputation complete", stage=4)
    return output


def predict_replication_regions(region_ids, *, force: bool = False) -> Path:
    """Load the Stage 2 summary and predict selected replication regions."""
    summary_path = analysis_dir() / "plrs_models_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError("Run admixture reference Stage 2 before replication prediction")
    summary = pd.read_csv(summary_path, sep="\t")
    return predict_replication(summary, region_ids=region_ids, force=force)


def run_discovery(*, force: bool = False) -> dict[str, Path]:
    """Fit all region models and generate discovery-cohort EUR dosages."""
    summary_path = analysis_dir() / "plrs_models_summary.tsv"
    predictions_path = analysis_dir() / "eur_dosage_predictions.discovery.tsv.gz"
    if not force and summary_path.exists() and _prediction_cache_current(
        predictions_path
    ):
        try:
            summary = pd.read_csv(summary_path, sep="\t")
            retained = set(summary.loc[
                summary["retained_for_phewas"].astype(bool), "region_id"
            ].astype(str))
            predicted = set(pd.read_csv(
                predictions_path, sep="\t", usecols=["region_id"], dtype=str,
            )["region_id"].dropna().astype(str))
        except (KeyError, OSError, ValueError, pd.errors.EmptyDataError):
            retained, predicted = set(), set()
        if retained and predicted == retained:
            _progress(
                "Stage 2 summary and discovery dosage are complete; "
                "skipping model pickle deserialization"
            )
            return {
                "summary": summary_path,
                "discovery_predictions": predictions_path,
            }

    summary = fit_plrs_models(force=force)
    if summary.empty or not summary["retained_for_phewas"].astype(bool).any():
        print(f"[admixture_reference] no PLRS regions exceeded R² > {R2_MIN}; PheWAS will be empty")
        return {"summary": summary_path}
    return {
        "summary": summary_path,
        "discovery_predictions": predict_discovery(summary, force=force),
    }


def run(*, force: bool = False) -> dict[str, Path]:
    """Backward-compatible entry point for the discovery-generation stage."""
    return run_discovery(force=force)
