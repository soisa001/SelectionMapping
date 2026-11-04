"""Staged discovery and targeted replication PheWAS for LAI dosage."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import admixture_inputs, admixture_lai, admixture_plrs
from .common import config, samples as sample_set


ANALYSIS_CONTRACT = "figure4_lai_dosage_phetk_logit_20pc_sexaware_unrelated_v6"
N_PCS = 20
MIN_CASES = 200
MIN_PHECODE_COUNT = int(os.environ.get("AOU_FIG4_MIN_PHECODE_COUNT", "2"))
DISCOVERY_P_THRESHOLD = 1e-3
PHETK_METHOD = "logit"
PHETK_VERSION = "0.3.3"
PHETK_NUMPY_VERSION = "1.26.4"
PHETK_WORKERS = int(os.environ.get(
    "AOU_FIG4_PHETK_WORKERS", str(max(1, min(4, os.cpu_count() or 1)))
))
COVARIATE_COLUMNS = [
    "age_at_last_ehr_event",
    "age_at_last_ehr_event_squared",
    "sex_at_birth",
    *[f"pc{index}" for index in range(1, N_PCS + 1)],
]
ASSOCIATION_COLUMNS = [
    "cohort", "region_id", "phecode", "sex_scope", "beta", "se", "p",
    "n", "n_cases", "n_controls", "n_pcs", "analysis_contract",
    "target_contract",
]


def _progress(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[admixture_reference][phetk {stamp}] {message}", flush=True)


def analysis_dir() -> Path:
    path = admixture_lai.analysis_dir().parent / "phewas"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _id_column(columns) -> str:
    return next((name for name in
                 ("person_id", "sample_id", "research_id", "s", "IID")
                 if name in columns), list(columns)[0])


def harmonize_pca(raw: pd.DataFrame, n_pcs: int = N_PCS) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.rename(columns={_id_column(frame.columns): "person_id"})
    frame["person_id"] = frame["person_id"].astype(str)
    array_column = next((name for name in ("scores", "pca_features")
                         if name in frame.columns), None)
    if array_column is not None:
        expanded = (frame[array_column].astype(str).str.strip()
                    .str.strip("[]").str.split(",", expand=True))
        if expanded.shape[1] < n_pcs:
            raise ValueError(f"PCA score array has {expanded.shape[1]} entries; need {n_pcs}")
        for index in range(n_pcs):
            frame[f"pc{index + 1}"] = pd.to_numeric(
                expanded[index], errors="coerce"
            )
    else:
        upper = {str(column).upper(): column for column in frame.columns}
        remap = {}
        for index in range(1, n_pcs + 1):
            source = next((upper[name] for name in
                           (f"PC{index}", f"SCORES_{index - 1}",
                            f"PCA_FEATURE_{index}", f"COMPONENT_{index}")
                           if name in upper), None)
            if source is not None:
                remap[source] = f"pc{index}"
        frame = frame.rename(columns=remap)
    required = ["person_id", *[f"pc{index}" for index in range(1, n_pcs + 1)]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"PCA input missing required columns: {missing}")
    return frame[required]


def _normalize_sex(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower().replace({
        "male": "1", "m": "1", "man": "1", "male at birth": "1",
        "sex at birth: male": "1",
        "female": "0", "f": "0", "woman": "0", "female at birth": "0",
        "sex at birth: female": "0",
    })
    numeric = pd.to_numeric(text, errors="coerce")
    return numeric.where(numeric.isin([0, 1])).astype(float)


def load_cohort(label: str, *, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = admixture_inputs.stage_cohort_sources(label, force=force)
    covariates = pd.read_csv(sources["covariates"], sep="\t")
    covariates = covariates.rename(
        columns={_id_column(covariates.columns): "person_id"}
    )
    covariates["person_id"] = covariates["person_id"].astype(str)
    covariates = sample_set.exclude_related(covariates, sample_col="person_id")
    if "age_at_last_ehr_event_squared" not in covariates.columns:
        covariates["age_at_last_ehr_event_squared"] = (
            pd.to_numeric(covariates["age_at_last_ehr_event"], errors="coerce") ** 2
        )
    pca_compression = ("gzip" if str(sources["pca"]).endswith((".gz", ".bgz"))
                       else "infer")
    pca = harmonize_pca(pd.read_csv(
        sources["pca"], sep="\t", compression=pca_compression
    ))
    covariates = covariates.merge(pca, on="person_id", how="inner")
    covariates["sex_at_birth"] = _normalize_sex(covariates["sex_at_birth"])
    excluded_sex = int(covariates["sex_at_birth"].isna().sum())
    if excluded_sex:
        print(
            f"[admixture_reference] {label}: excluding {excluded_sex:,} covariate rows "
            "with missing or non-binary sex_at_birth"
        )
    for column in COVARIATE_COLUMNS:
        covariates[column] = pd.to_numeric(covariates[column], errors="coerce")
    covariates = (covariates.dropna(subset=COVARIATE_COLUMNS)
                  .drop_duplicates("person_id").set_index("person_id"))
    counts = pd.read_csv(
        sources["phenotypes"], sep="\t",
        usecols=lambda column: column in {"person_id", "phecode", "count"},
        dtype={"person_id": str, "phecode": str},
    )
    required = {"person_id", "phecode", "count"}
    if not required.issubset(counts.columns):
        raise ValueError(f"{label} phenotype counts missing {required.difference(counts.columns)}")
    counts = sample_set.exclude_related(counts, sample_col="person_id")
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0)
    return covariates, counts


def phecode_sex_lookup() -> dict[str, str]:
    frame = pd.read_csv(config.ASSETS_DIR / "phecodeX_sex.csv", dtype=str)
    return dict(zip(frame["phecode"].str.strip(), frame["sex"].str.strip()))


def phenotype_vector(group: pd.DataFrame, sample_ids: np.ndarray,
                     sex: np.ndarray, scope: str) -> np.ndarray:
    aggregated = group.groupby("person_id", as_index=False)["count"].sum()
    any_ids = aggregated["person_id"].astype(str).to_numpy()
    case_ids = aggregated.loc[
        aggregated["count"] >= MIN_PHECODE_COUNT, "person_id"
    ].astype(str).to_numpy()
    in_any = np.isin(sample_ids, any_ids)
    in_case = np.isin(sample_ids, case_ids)
    y = np.zeros(len(sample_ids), dtype=float)
    y[in_any] = np.nan
    y[in_case] = 1.0
    if scope == "Female":
        y[sex != 0] = np.nan
    elif scope == "Male":
        y[sex != 1] = np.nan
    elif scope != "Both":
        raise ValueError(f"Unknown phecode sex scope: {scope}")
    return y


def _fit_null(y: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("admixture reference PheWAS requires statsmodels") from exc
    varying = np.nanstd(covariates, axis=0) > 1e-12
    design = np.column_stack([np.ones(len(y)), covariates[:, varying]])
    try:
        fit = sm.Logit(y, design).fit(disp=False, maxiter=300, method="newton")
        probability = np.asarray(fit.predict(design), dtype=float)
    except Exception:
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=300)
        probability = np.asarray(fit.predict(design), dtype=float)
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    return probability, design


def score_test_batch(X: np.ndarray, y: np.ndarray,
                     covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariate-adjusted logistic score tests for every region column."""
    from scipy import stats
    probability, design = _fit_null(y, covariates)
    weight = probability * (1 - probability)
    residual = y - probability
    weighted_design = design * weight[:, None]
    inverse = np.linalg.pinv(design.T @ weighted_design)
    xtwx = np.square(X).T @ weight
    xtwz = X.T @ weighted_design
    adjustment = np.einsum("pk,kl,pl->p", xtwz, inverse, xtwz)
    information = xtwx - adjustment
    score = X.T @ residual
    valid = information > 1e-12
    beta = np.where(valid, score / information, np.nan)
    se = np.where(valid, 1.0 / np.sqrt(information), np.nan)
    chi_square = np.where(valid, np.square(score) / information, np.nan)
    p_value = stats.chi2.sf(chi_square, df=1)
    return beta, se, p_value


def _prediction_masks(wide: pd.DataFrame) -> list[tuple[np.ndarray, list[str]]]:
    groups: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    for column in wide.columns:
        mask = wide[column].notna().to_numpy()
        key = np.packbits(mask).tobytes()
        if key not in groups:
            groups[key] = (mask, [])
        groups[key][1].append(str(column))
    return list(groups.values())


def _safe_region_name(region_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(region_id))


def _phetk_runtime_dir() -> Path:
    override = os.environ.get("AOU_FIG4_PHETK_ENV")
    if override:
        return Path(override).expanduser()
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return (admixture_inputs.work_dir() / ".phetk_runtime" /
            f"phetk-{PHETK_VERSION}-{python_tag}")


def _runtime_python(runtime_dir: Path) -> Path:
    if os.name == "nt":
        return runtime_dir / "Scripts" / "python.exe"
    return runtime_dir / "bin" / "python"


def _phetk_runtime_works(python: Path) -> bool:
    if not python.exists():
        return False
    try:
        check = subprocess.run(
            [str(python), "-c",
             "import importlib.metadata as m, numpy; "
             "print(m.version('phetk') + '|' + numpy.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    expected = f"{PHETK_VERSION}|{PHETK_NUMPY_VERSION}"
    return check.returncode == 0 and check.stdout.strip() == expected


def _ensure_phetk_runtime() -> Path:
    """Create a PheTK environment without changing the notebook's NumPy."""
    runtime_dir = _phetk_runtime_dir()
    python = _runtime_python(runtime_dir)
    if _phetk_runtime_works(python):
        return python

    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    _progress(
        f"creating isolated PheTK {PHETK_VERSION} runtime at {runtime_dir}"
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(runtime_dir)],
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install",
             "--disable-pip-version-check",
             f"numpy=={PHETK_NUMPY_VERSION}", f"phetk=={PHETK_VERSION}"],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not create the isolated admixture reference PheTK runtime. "
            f"Remove or override {runtime_dir} and rerun Stage 3."
        ) from exc
    if not _phetk_runtime_works(python):
        raise RuntimeError(
            f"PheTK {PHETK_VERSION} was not importable from {python}"
        )
    _progress(f"isolated PheTK runtime ready: {python}")
    return python


def _prepare_phetk_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Return numeric, conditioned covariates for PheTK's file interface."""
    frame = covariates.reset_index().copy()
    frame["person_id"] = pd.to_numeric(frame["person_id"], errors="coerce")
    if frame["person_id"].isna().any():
        raise ValueError("PheTK requires numeric person IDs; inspect invalid rows in Workbench.")
    frame["person_id"] = frame["person_id"].astype("int64")

    age = pd.to_numeric(frame["age_at_last_ehr_event"], errors="coerce")
    centered_age = age - age.mean()
    frame["age_at_last_ehr_event"] = centered_age
    frame["age_at_last_ehr_event_squared"] = centered_age.pow(2)
    continuous = [column for column in COVARIATE_COLUMNS
                  if column != "sex_at_birth"]
    for column in continuous:
        values = pd.to_numeric(frame[column], errors="coerce")
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"PheTK covariate {column!r} has no usable variance")
        frame[column] = (values - values.mean()) / scale
    frame["sex_at_birth"] = pd.to_numeric(
        frame["sex_at_birth"], errors="coerce"
    )
    frame = frame.dropna(subset=["person_id", *COVARIATE_COLUMNS])
    return frame[["person_id", *COVARIATE_COLUMNS]]


def _phetk_run_dir(label: str) -> Path:
    path = analysis_dir() / "phetk_runs" / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _phetk_region_paths(label: str, region_id: str) -> tuple[Path, Path, Path]:
    stem = _safe_region_name(region_id)
    root = _phetk_run_dir(label)
    return (
        root / f"{stem}.cohort.tsv",
        root / f"{stem}.phewas.tsv",
        root / f"{stem}.complete.json",
    )


def _region_checkpoint_matches(path: Path, *, region_id: str,
                               target_contract: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload == {
        "analysis_contract": ANALYSIS_CONTRACT,
        "target_contract": target_contract,
        "region_id": str(region_id),
        "phetk_version": PHETK_VERSION,
        "method": PHETK_METHOD,
        "min_cases": MIN_CASES,
        "min_phecode_count": MIN_PHECODE_COUNT,
    }


def _write_region_checkpoint(path: Path, *, region_id: str,
                             target_contract: str) -> None:
    payload = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "target_contract": target_contract,
        "region_id": str(region_id),
        "phetk_version": PHETK_VERSION,
        "method": PHETK_METHOD,
        "min_cases": MIN_CASES,
        "min_phecode_count": MIN_PHECODE_COUNT,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_phetk_result(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype={"phecode": str})


def _run_phetk_region(cohort: pd.DataFrame, phecode_path: Path,
                      output_path: Path, *, phecodes: list[str] | None,
                      runtime_python: Path) -> pd.DataFrame:
    """Run one continuous EUR-dosage PheWAS through PheTK 0.3.3."""
    cohort_path = output_path.with_name(
        output_path.name.replace(".phewas.tsv", ".cohort.tsv")
    )
    cohort.to_csv(cohort_path, sep="\t", index=False)
    worker_config = output_path.with_suffix(output_path.suffix + ".json")
    payload = {
        "phecode_version": "X",
        "phecode_count_file_path": str(phecode_path.resolve()),
        "cohort_file_path": str(cohort_path.resolve()),
        "sex_at_birth_col": "sex_at_birth",
        "male_as_one": True,
        "covariate_cols": COVARIATE_COLUMNS,
        "independent_variable_of_interest": "eur_dosage",
        "phecode_to_process": phecodes,
        "min_cases": MIN_CASES,
        "min_phecode_count": MIN_PHECODE_COUNT,
        "method": PHETK_METHOD,
        "batch_size": 1,
        "fall_back_to_serial": True,
        "suppress_warnings": True,
        "verbose": False,
        "output_file_path": str(output_path.resolve()),
        "parallelization": "multithreading",
        "n_workers": PHETK_WORKERS,
    }
    temporary = worker_config.with_suffix(worker_config.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(worker_config)
    worker = Path(__file__).with_name("admixture_phetk_worker.py")
    try:
        subprocess.run(
            [str(runtime_python), "-u", str(worker), str(worker_config)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"PheTK failed for {output_path.stem}; see the worker output above"
        ) from exc
    return _read_phetk_result(output_path)


def _phetk_converged(frame: pd.DataFrame) -> pd.Series:
    if "converged" not in frame.columns:
        return pd.Series(True, index=frame.index)
    values = frame["converged"].astype(str).str.strip().str.lower()
    return values.isin({"converged", "true", "1", "yes"})


def _canonicalize_phetk(frame: pd.DataFrame, *, label: str, region_id: str,
                        target_contract: str,
                        requested_phecodes: set[str] | None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=ASSOCIATION_COLUMNS)
    normalized = frame.rename(columns={
        "p_value": "p", "standard_error": "se",
        "beta_ind": "beta", "se_ind": "se",
    }).copy()
    required = {"phecode", "cases", "controls", "p", "beta", "se"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"PheTK output missing columns: {sorted(missing)}")
    normalized["phecode"] = normalized["phecode"].astype(str)
    if requested_phecodes is not None:
        normalized = normalized[
            normalized["phecode"].isin(requested_phecodes)
        ].copy()
    for column in ("cases", "controls", "p", "beta", "se"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    valid = (
        _phetk_converged(normalized)
        & np.isfinite(normalized[["p", "beta", "se"]]).all(axis=1)
        & (normalized["se"] > 0)
        & (normalized["cases"] >= MIN_CASES)
        & (normalized["controls"] > MIN_CASES)
    )
    skipped = int((~valid).sum())
    if skipped:
        _progress(
            f"{label} {region_id}: excluded {skipped:,} non-converged/invalid "
            f"or sub-threshold PheTK results"
        )
    normalized = normalized[valid].copy()
    scopes = normalized.get(
        "phecode_sex_restriction",
        pd.Series("Both", index=normalized.index),
    ).astype(str)
    result = pd.DataFrame({
        "cohort": label,
        "region_id": str(region_id),
        "phecode": normalized["phecode"],
        "sex_scope": scopes,
        "beta": normalized["beta"].astype(float),
        "se": normalized["se"].astype(float),
        "p": normalized["p"].astype(float),
        "n": (normalized["cases"] + normalized["controls"]).astype(int),
        "n_cases": normalized["cases"].astype(int),
        "n_controls": normalized["controls"].astype(int),
        "n_pcs": N_PCS,
        "analysis_contract": ANALYSIS_CONTRACT,
        "target_contract": target_contract,
    })
    return result[ASSOCIATION_COLUMNS].reset_index(drop=True)


def _target_contract(targets: pd.DataFrame | None) -> str:
    if targets is None:
        return "all-region-phecode-tests"
    pairs = sorted({
        (str(row.region_id), str(row.phecode))
        for row in targets[["region_id", "phecode"]].itertuples(index=False)
    })
    encoded = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return "selected-" + hashlib.sha256(encoded).hexdigest()[:16]


def _association_manifest(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def _cached_associations(output: Path, target_contract: str) -> pd.DataFrame | None:
    manifest_path = _association_manifest(output)
    if not output.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != {
            "analysis_contract": ANALYSIS_CONTRACT,
            "target_contract": target_contract,
        }:
            return None
        cached = pd.read_csv(output, sep="\t", dtype={"phecode": str})
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if not set(ASSOCIATION_COLUMNS).issubset(cached.columns):
        return None
    if cached.empty:
        return cached
    if not cached["analysis_contract"].astype(str).eq(ANALYSIS_CONTRACT).all():
        return None
    if not cached["target_contract"].astype(str).eq(target_contract).all():
        return None
    return cached


def _write_associations(result: pd.DataFrame, output: Path,
                        target_contract: str) -> None:
    result.to_csv(output, sep="\t", index=False, compression="gzip")
    manifest_path = _association_manifest(output)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "analysis_contract": ANALYSIS_CONTRACT,
        "target_contract": target_contract,
    }, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def run_associations(label: str, prediction_path: Path, *,
                     targets: pd.DataFrame | None = None,
                     force: bool = False) -> pd.DataFrame:
    """Run discovery or targeted replication through PheTK logistic regression."""
    output = analysis_dir() / f"phewas.{label}.tsv.gz"
    target_contract = _target_contract(targets)
    if not force:
        cached = _cached_associations(output, target_contract)
        if cached is not None:
            _progress(f"{label}: using cached PheTK associations from {output}")
            return cached
    runtime_python = _ensure_phetk_runtime()
    _progress(
        f"{label}: PheTK {PHETK_VERSION} {PHETK_METHOD}, "
        f"min_cases={MIN_CASES} (controls > {MIN_CASES}), "
        f"workers={PHETK_WORKERS}, runtime={runtime_python}"
    )
    predictions = pd.read_csv(
        prediction_path, sep="\t",
        dtype={"person_id": str, "region_id": str},
    )
    required_predictions = {"person_id", "region_id", "eur_dosage"}
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"{label} predictions missing {sorted(missing_predictions)}"
        )
    predictions = sample_set.exclude_related(predictions, sample_col="person_id")
    predictions["eur_dosage"] = pd.to_numeric(
        predictions["eur_dosage"], errors="coerce"
    )
    predictions = predictions[
        predictions["eur_dosage"].between(0, 2, inclusive="both")
    ].drop_duplicates(["person_id", "region_id"])
    if predictions.empty:
        empty = pd.DataFrame(columns=ASSOCIATION_COLUMNS)
        _write_associations(empty, output, target_contract)
        return empty

    covariates, counts = load_cohort(label, force=force)
    phetk_covariates = _prepare_phetk_covariates(covariates)
    predictions["person_id"] = pd.to_numeric(
        predictions["person_id"], errors="coerce"
    )
    if predictions["person_id"].isna().any():
        raise ValueError(f"{label} predictions contain non-numeric person IDs")
    predictions["person_id"] = predictions["person_id"].astype("int64")
    sources = admixture_inputs.stage_cohort_sources(label, force=False)
    phecode_path = Path(sources["phenotypes"])
    scopes = phecode_sex_lookup()
    available_phecodes = sorted(
        set(counts["phecode"].astype(str)).intersection(scopes)
    )
    target_lookup: dict[str, set[str]] | None = None
    if targets is not None:
        normalized_targets = targets[["region_id", "phecode"]].astype(str)
        target_lookup = (
            normalized_targets.groupby("region_id")["phecode"]
            .apply(lambda values: set(values).intersection(available_phecodes))
            .to_dict()
        )
    regions = sorted(predictions["region_id"].dropna().astype(str).unique())
    if target_lookup is not None:
        regions = [region for region in regions if target_lookup.get(region)]
    _progress(
        f"{label}: {predictions['person_id'].nunique():,} prediction samples, "
        f"{len(regions):,} regions, {len(available_phecodes):,} available phecodes"
    )

    frames = []
    for region_number, region_id in enumerate(regions, start=1):
        requested = target_lookup.get(region_id) if target_lookup is not None else None
        phecodes = sorted(requested) if requested is not None else available_phecodes
        if not phecodes:
            continue
        _, region_output, checkpoint = _phetk_region_paths(label, region_id)
        if force:
            region_output.unlink(missing_ok=True)
            checkpoint.unlink(missing_ok=True)
        prediction = predictions.loc[
            predictions["region_id"] == region_id,
            ["person_id", "eur_dosage"],
        ]
        cohort = phetk_covariates.merge(prediction, on="person_id", how="inner")
        cohort = cohort.dropna(
            subset=["eur_dosage", *COVARIATE_COLUMNS]
        ).drop_duplicates("person_id")
        dosage_sd = float(cohort["eur_dosage"].std(ddof=0))
        if len(cohort) < 2 * MIN_CASES or not np.isfinite(dosage_sd) or dosage_sd <= 1e-8:
            _progress(
                f"{label} region {region_number}/{len(regions)} {region_id}: "
                f"skipped (n={len(cohort):,}, dosage_sd={dosage_sd:.4g})"
            )
            continue

        completed_now = False
        if _region_checkpoint_matches(
            checkpoint, region_id=region_id, target_contract=target_contract
        ):
            _progress(
                f"{label} region {region_number}/{len(regions)} {region_id}: "
                "resuming from checkpoint"
            )
            raw = _read_phetk_result(region_output)
        else:
            _progress(
                f"{label} region {region_number}/{len(regions)} {region_id}: "
                f"starting n={len(cohort):,}, dosage_sd={dosage_sd:.4f}, "
                f"phecodes={len(phecodes):,}"
            )
            # A mismatched/missing checkpoint means any prior result is stale.
            # Remove it so a no-result rerun cannot be mistaken for old output.
            region_output.unlink(missing_ok=True)
            started = time.perf_counter()
            raw = _run_phetk_region(
                cohort, phecode_path, region_output, phecodes=phecodes,
                runtime_python=runtime_python,
            )
            completed_now = True
            _progress(
                f"{label} region {region_number}/{len(regions)} {region_id}: "
                f"PheTK completed {len(raw):,} rows in "
                f"{time.perf_counter() - started:,.1f}s"
            )
        canonical = _canonicalize_phetk(
            raw, label=label, region_id=region_id,
            target_contract=target_contract,
            requested_phecodes=requested,
        )
        if completed_now:
            _write_region_checkpoint(
                checkpoint, region_id=region_id, target_contract=target_contract
            )
        if not canonical.empty:
            frames.append(canonical)

    result = (pd.concat(frames, ignore_index=True)
              if frames else pd.DataFrame(columns=ASSOCIATION_COLUMNS))
    _write_associations(result, output, target_contract)
    _progress(f"{label} PheTK PheWAS completed {len(result):,} valid tests")
    return result


def _ivw(beta1, se1, beta2, se2) -> tuple[float, float, float]:
    from scipy import stats
    values = np.asarray([beta1, se1, beta2, se2], dtype=float)
    if not np.isfinite(values).all() or se1 <= 0 or se2 <= 0:
        return np.nan, np.nan, np.nan
    weight1, weight2 = 1.0 / se1**2, 1.0 / se2**2
    beta = (beta1 * weight1 + beta2 * weight2) / (weight1 + weight2)
    se = math.sqrt(1.0 / (weight1 + weight2))
    p = float(2 * stats.norm.sf(abs(beta / se)))
    return float(beta), float(se), p


def meta_analyze(discovery: pd.DataFrame, replication: pd.DataFrame,
                 *, force: bool = False) -> pd.DataFrame:
    output = analysis_dir() / "meta_analysis_results.tsv"
    if discovery.empty:
        pd.DataFrame().to_csv(output, sep="\t", index=False)
        return pd.DataFrame()
    labeled = label_discovery_significance(discovery)
    hits = labeled[labeled["discovery_significant"].astype(bool)].copy()
    if hits.empty:
        pd.DataFrame().to_csv(output, sep="\t", index=False)
        return pd.DataFrame()
    replication_columns = ["region_id", "phecode", "beta", "se", "p", "n",
                           "n_cases", "n_controls"]
    if replication.empty:
        replication = pd.DataFrame(columns=replication_columns)
    merged = hits.merge(
        replication[replication_columns].rename(columns={
                         "beta": "beta_replication", "se": "se_replication",
                         "p": "p_replication", "n": "n_replication",
                         "n_cases": "n_cases_replication",
                         "n_controls": "n_controls_replication",
                     }),
        on=["region_id", "phecode"], how="left",
    ).rename(columns={
        "beta": "beta_discovery", "se": "se_discovery",
        "p": "p_discovery", "n": "n_discovery",
        "n_cases": "n_cases_discovery", "n_controls": "n_controls_discovery",
    })
    estimates = merged.apply(
        lambda row: _ivw(row["beta_discovery"], row["se_discovery"],
                         row["beta_replication"], row["se_replication"]),
        axis=1, result_type="expand",
    )
    estimates.columns = ["beta_meta", "se_meta", "p_meta"]
    merged = pd.concat([merged, estimates], axis=1)
    merged["same_direction"] = (
        np.sign(merged["beta_discovery"]) == np.sign(merged["beta_replication"])
    ) & merged["beta_replication"].notna()
    merged["replicated"] = merged["same_direction"] & (merged["p_replication"] < 0.05)
    alpha_meta = 0.05 / max(len(merged), 1)
    merged["meta_alpha"] = alpha_meta
    merged["meta_significant"] = merged["p_meta"] < alpha_meta
    merged["replicated_meta_significant"] = (
        merged["replicated"] & merged["meta_significant"]
    )
    regions = pd.read_csv(admixture_lai.analysis_dir() / "outlier_regions.tsv", sep="\t")
    merged = merged.merge(regions, on="region_id", how="left", suffixes=("", "_region"))
    merged["analysis_contract"] = ANALYSIS_CONTRACT
    merged = merged.sort_values("p_meta", na_position="last").reset_index(drop=True)
    merged.to_csv(output, sep="\t", index=False)
    return merged


def label_discovery_significance(discovery: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed discovery p-value threshold without rerunning Stage 3."""
    labeled = discovery.copy()
    if labeled.empty:
        return labeled
    labeled["p"] = pd.to_numeric(labeled["p"], errors="coerce")
    labeled["discovery_alpha"] = DISCOVERY_P_THRESHOLD
    labeled["discovery_threshold"] = "fixed_p_lt_1e-3"
    labeled["discovery_significant"] = (
        labeled["p"] < DISCOVERY_P_THRESHOLD
    )
    return labeled


def discovery_hits(discovery: pd.DataFrame) -> pd.DataFrame:
    """Return the exact region-phecode pairs that advance to replication."""
    if discovery.empty or "p" not in discovery.columns:
        return pd.DataFrame(columns=["region_id", "phecode"])
    labeled = label_discovery_significance(discovery)
    return (labeled.loc[
        labeled["discovery_significant"].astype(bool), ["region_id", "phecode"]
    ].astype(str).drop_duplicates().reset_index(drop=True))


def load_cached_discovery() -> pd.DataFrame | None:
    """Load and relabel completed Stage 3 results for direct Stage 4 resume."""
    output = analysis_dir() / "phewas.discovery.tsv.gz"
    cached = _cached_associations(output, _target_contract(None))
    if cached is None:
        return None
    labeled = label_discovery_significance(cached)
    labeled.to_csv(output, sep="\t", index=False, compression="gzip")
    return labeled


def write_empty_replication_outputs() -> dict[str, Path]:
    """Write stable empty Stage 5 products when discovery has no hits."""
    replication_output = analysis_dir() / "phewas.replication.tsv.gz"
    admixture_inputs.remove_if_exists(_association_manifest(replication_output))
    pd.DataFrame(columns=ASSOCIATION_COLUMNS).to_csv(
        replication_output, sep="\t", index=False, compression="gzip"
    )
    meta_output = analysis_dir() / "meta_analysis_results.tsv"
    pd.DataFrame().to_csv(meta_output, sep="\t", index=False)
    return {"replication": replication_output, "meta": meta_output}


def run_discovery(*, force: bool = False) -> pd.DataFrame:
    """Run Stage 3 PheWAS across all discovery region-phecode tests."""
    if not force:
        cached = load_cached_discovery()
        if cached is not None:
            _progress(
                "discovery: Stage 3 cache complete; applied fixed p < 1e-3 "
                "threshold without rerunning PheWAS"
            )
            return cached
    discovery_predictions = admixture_plrs.analysis_dir() / "eur_dosage_predictions.discovery.tsv.gz"
    discovery = run_associations(
        "discovery", discovery_predictions, force=force
    )
    discovery_output = analysis_dir() / "phewas.discovery.tsv.gz"
    if discovery.empty:
        return discovery
    discovery = label_discovery_significance(discovery)
    discovery.to_csv(discovery_output, sep="\t", index=False, compression="gzip")
    return discovery


def run_replication(discovery: pd.DataFrame, replication_predictions: Path, *,
                    force: bool = False) -> dict[str, Path]:
    """Run Stage 5 only for Stage 3 region-phecode discoveries."""
    hits = discovery_hits(discovery)
    if hits.empty:
        return write_empty_replication_outputs()
    replication = run_associations(
        "replication", replication_predictions, targets=hits, force=force
    )
    meta = meta_analyze(discovery, replication, force=force)
    print(f"[admixture_reference] meta-analysis completed {len(meta):,} discovery hits")
    return {
        "replication": analysis_dir() / "phewas.replication.tsv.gz",
        "meta": analysis_dir() / "meta_analysis_results.tsv",
    }


def run(*, force: bool = False) -> dict[str, Path]:
    """Compatibility wrapper for already-generated discovery/replication dosage."""
    discovery = run_discovery(force=force)
    outputs = {"discovery": analysis_dir() / "phewas.discovery.tsv.gz"}
    hits = discovery_hits(discovery)
    if hits.empty:
        print(
            f"[admixture_reference] no discovery hits at p < {DISCOVERY_P_THRESHOLD:.0e}"
        )
        outputs.update(write_empty_replication_outputs())
        return outputs
    replication_predictions = (
        admixture_plrs.analysis_dir() / "eur_dosage_predictions.replication.tsv.gz"
    )
    outputs.update(run_replication(
        discovery, replication_predictions, force=force
    ))
    return outputs
