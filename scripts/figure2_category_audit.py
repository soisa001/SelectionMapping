"""Read-only audit of missing Figure 2 PhecodeX categories in local VM caches.

No staging or regressions. Only aggregate audit outputs are written. Case/control
reconstruction follows PheTK v0.3.3 _prepare_case_control_data and _run_phecode:
https://github.com/nhgritctran/PheTK/blob/v0.3.3/src/phetk/phewas.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
import pandas as pd

from . import admixture_phewas, figure2_1025 as corrected
from .common import config

# Category membership is encoded by the official PhecodeX prefix. This avoids
# depending on annotation tables that may themselves omit untested categories.
CATEGORIES = {
    "BI": "Blood/Immune", "CA": "Neoplasms", "CM": "Congenital",
    "CV": "Cardiovascular", "DE": "Dermatological", "EM": "Endocrine/Metab",
    "GE": "Genetic", "GI": "Gastrointestinal", "GU": "Genitourinary",
    "ID": "Infections", "MB": "Mental", "MS": "Musculoskeletal",
    "NB": "Neonatal", "NS": "Neurological", "PP": "Pregnancy",
    "RE": "Respiratory", "SO": "Sense organs", "SS": "Symptoms",
}
FOCUS = ("CM", "NB")


def category(codes):
    result = codes.astype(str).str.split("_").str[0].map(CATEGORIES)
    if result.isna().any():
        raise ValueError("Unknown PhecodeX prefix in results or definitions")
    return result


def read_counts(path, min_count):
    """Stream only the two categories of interest; never sum duplicate rows.

    PheTK tests each count row against min_phecode_count, then takes unique IDs.
    A participant with any occurrence below the case threshold is not a control.
    """
    any_ids, case_ids = {}, {}
    for chunk in pd.read_csv(path, sep="\t", dtype={"person_id": str, "phecode": str},
                             usecols=["person_id", "phecode", "count"], chunksize=500_000):
        chunk = chunk.loc[chunk.phecode.str.startswith(tuple(prefix + "_" for prefix in FOCUS))]
        values = pd.to_numeric(chunk["count"], errors="raise")
        if chunk[["person_id", "phecode"]].isna().any().any() or not np.isfinite(values).all():
            raise ValueError("Invalid phenotype count records")
        chunk = chunk.assign(_case=values >= min_count)
        for code, group in chunk.groupby("phecode", sort=False):
            any_ids.setdefault(code, set()).update(group.person_id)
            case_ids.setdefault(code, set()).update(group.loc[group._case, "person_id"])
    return any_ids, case_ids


def eligibility(cohort, definitions, any_ids, case_ids, requested, minimum):
    if cohort.person_id.isna().any() or cohort.person_id.duplicated().any():
        raise ValueError("Missing or duplicate IDs in saved PheTK cohort")
    if not cohort.sex_at_birth.isin([0, 1]).all():
        raise ValueError("Invalid sex coding in saved PheTK cohort")
    pools = {"Both": set(cohort.person_id),
             "Female": set(cohort.loc[cohort.sex_at_birth.eq(0), "person_id"]),
             "Male": set(cohort.loc[cohort.sex_at_birth.eq(1), "person_id"])}
    rows = []
    for definition in definitions.itertuples(index=False):
        pool = pools[definition.sex]
        code = definition.phecode
        n_cases = len(pool.intersection(case_ids.get(code, set())))
        n_controls = len(pool.difference(any_ids.get(code, set())))
        rows.append({"phecode": code, "category": definition.category,
                     "requested": code in requested,
                     "n_cases": n_cases, "n_controls": n_controls,
                     "passes_counts": n_cases >= minimum and n_controls > minimum})
    return pd.DataFrame(rows)


def classify(row):
    if row["retained"]:
        return "retained_result"
    if not row["requested"]:
        return "not_requested"
    if not row["passes_counts"]:
        return "below_case_control_threshold"
    if row["raw_result"]:
        return "eligible_raw_result_not_retained"
    return "eligible_no_saved_result"


def run(*, threads=4):
    root = corrected.analysis_root()
    source = root / "phewas/phewas.discovery.tsv.gz"
    stage_path = corrected._stage_manifest_path("discovery_phewas")
    if not source.is_file() or not stage_path.is_file():
        raise FileNotFoundError(f"Corrected discovery results/manifest are missing under {root}; run this audit on the VM holding the completed analysis")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if not corrected._stage_current("discovery_phewas", stage["contract"], [source]):
        raise ValueError("Discovery results fail the recorded checksum")
    discovery = pd.read_csv(source, sep="\t", dtype={"region_id": str, "phecode": str})
    if discovery.empty or not discovery.cohort.eq("discovery").all():
        raise ValueError("Expected nonempty discovery associations")
    if discovery.duplicated(["region_id", "phecode"]).any():
        raise ValueError("Duplicate discovery associations")
    discovery["category"] = category(discovery.phecode)
    definition_path = config.ASSETS_DIR / "phecodeX_sex.csv"
    definitions = pd.read_csv(definition_path, dtype=str)
    definitions["category"] = category(definitions.phecode)
    focus = definitions.loc[definitions.phecode.str.startswith(tuple(p + "_" for p in FOCUS))]
    regions = sorted(discovery.region_id.unique())
    nominal = set(discovery.loc[discovery.p < admixture_phewas.DISCOVERY_P_THRESHOLD, "region_id"])
    grid = pd.MultiIndex.from_product([regions, CATEGORIES.values()], names=["region_id", "category"])
    coverage = discovery.groupby(["region_id", "category"]).agg(
        retained_tests=("phecode", "nunique"), nominal_hits=("p", lambda p: int((p < .001).sum())))
    coverage = coverage.reindex(grid, fill_value=0).reset_index()
    coverage["region_has_nominal_hit"] = coverage.region_id.isin(nominal)
    output = root / "category_audit"
    output.mkdir(parents=True, exist_ok=True)
    corrected._atomic_tsv(coverage, output / "category_coverage.tsv")
    sources = {path: corrected._output_fingerprint(path) for path in (source, stage_path, definition_path)}
    jobs, issues, count_cache = [], [], {}
    # Resolve only paths recorded by the completed run; never contact a bucket.
    for region in regions:
        stem = admixture_phewas._safe_region_name(region)
        raw_path = root / "phewas/phetk_runs/discovery" / f"{stem}.phewas.tsv"
        worker_path = raw_path.with_suffix(raw_path.suffix + ".json")
        checkpoint = raw_path.with_name(f"{stem}.complete.json")
        try:
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (saved["analysis_contract"] != stage["contract"]["analysis_contract"]
                    or saved["target_contract"] != "all-region-phecode-tests"
                    or saved["region_id"] != region or saved["phetk_version"] != "0.3.3"
                    or worker["method"] != "logit" or worker["phecode_version"] != "X"
                    or not worker["male_as_one"] or worker.get("use_exclusion", False)
                    or worker["min_cases"] != saved["min_cases"]
                    or worker["min_phecode_count"] != saved["min_phecode_count"]):
                raise ValueError("Worker/checkpoint analysis parameters disagree or are unsupported")
            cohort_path = Path(worker["cohort_file_path"])
            counts_path = Path(worker["phecode_count_file_path"])
            for path in (worker_path, checkpoint, cohort_path, counts_path, raw_path):
                if path not in sources:
                    sources[path] = corrected._output_fingerprint(path)
            key = (counts_path, worker["min_phecode_count"])
            if key not in count_cache:
                count_cache[key] = read_counts(*key)
            jobs.append((region, worker, cohort_path, raw_path, key))
        except (OSError, ValueError, KeyError) as exc:
            issues.append({"region_id": region, "reason": str(exc)})

    def audit_region(job):
        region, worker, cohort_path, raw_path, key = job
        cohort = pd.read_csv(cohort_path, sep="\t", dtype={"person_id": str})
        required = [worker["independent_variable_of_interest"], *worker["covariate_cols"]]
        if cohort[required].isna().any().any():
            raise ValueError("Missing analysis covariates in saved cohort")
        requested = worker["phecode_to_process"]
        if not isinstance(requested, list):
            raise ValueError("Expected explicit requested Phecode list in worker config")
        table = eligibility(cohort, focus, *count_cache[key], set(requested), worker["min_cases"])
        raw = pd.read_csv(raw_path, sep="\t", dtype={"phecode": str})
        retained = discovery.loc[discovery.region_id.eq(region)]
        if not set(retained.phecode).issubset(set(raw.phecode)):
            raise ValueError("Saved raw results do not contain the retained discovery Phecodes")
        # Check reconstructed counts against actual model outputs whenever present.
        compared = table.merge(raw[["phecode", "cases", "controls"]], on="phecode")
        if not (compared.n_cases.eq(compared.cases) & compared.n_controls.eq(compared.controls)).all():
            raise ValueError("Reconstructed case/control counts disagree with saved PheTK results")
        table["region_id"] = region
        table["retained"] = table.phecode.isin(retained.phecode)
        table["raw_result"] = table.phecode.isin(raw.phecode)
        table["status"] = table.apply(classify, axis=1)
        table["min_cases"] = worker["min_cases"]
        table["min_phecode_count"] = worker["min_phecode_count"]
        return table

    frames = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [(job[0], pool.submit(audit_region, job)) for job in jobs]
        for region, future in futures:
            try:
                frames.append(future.result())
            except (OSError, ValueError, KeyError) as exc:
                issues.append({"region_id": region, "reason": str(exc)})
    if frames:
        details = pd.concat(frames, ignore_index=True)
        corrected._atomic_tsv(details, output / "congenital_neonatal_eligibility.tsv")
        summary = details.groupby(["region_id", "category", "status"]).agg(
            phecodes=("phecode", "size"), passing_count_threshold=("passes_counts", "sum")
        ).reset_index()
        corrected._atomic_tsv(summary, output / "eligibility_summary.tsv")
        print(summary.to_string(index=False))
    else:
        # Do not leave a previous successful audit visible as the current result.
        corrected._atomic_tsv(pd.DataFrame(columns=["region_id", "phecode", "status"]), output / "congenital_neonatal_eligibility.tsv")
        corrected._atomic_tsv(pd.DataFrame(columns=["region_id", "category", "status", "phecodes", "passing_count_threshold"]), output / "eligibility_summary.tsv")
    corrected._atomic_json(output / "run_manifest.json", {
        "audit_contract": "figure5_phecodex_counts_v1", "completed_utc": corrected._utc_now(),
        "status": "incomplete" if issues else "complete", "threads": threads,
        "sources": list(sources.values()), "issues": issues,
        "regions_in_results": len(regions), "regions_with_eligibility_audit": len(frames),
        "method_source": "https://github.com/nhgritctran/PheTK/blob/v0.3.3/src/phetk/phewas.py",
        "eligibility": "cases >= saved min_cases; controls > saved min_cases; sex-specific cohort; any occurrence excludes controls",
    })
    print("\nRetained results across all regions:")
    print(discovery.groupby("category").phecode.nunique().reindex(CATEGORIES.values(), fill_value=0).rename("unique_phecodes").to_string())
    print(f"\nAudit saved to {output}; eligibility audited for {len(frames)}/{len(regions)} regions.")
    for issue in issues:
        print(f"UNRESOLVED {issue['region_id']}: {issue['reason']}")
    return {"directory": output, "issues": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    run(threads=args.threads)
