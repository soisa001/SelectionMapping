from pathlib import Path

import pandas as pd

from scripts import figure2_category_audit as audit


def test_exact_thresholds_low_count_exclusion_and_sex_scope():
    cohort = pd.DataFrame({"person_id": [str(i) for i in range(402)],
                           "sex_at_birth": [0] * 201 + [1] * 201})
    definitions = pd.DataFrame({"phecode": ["CM_1", "CM_2", "NB_1"],
                                "category": ["Congenital", "Congenital", "Neonatal"],
                                "sex": ["Both", "Both", "Female"]})
    cases = {code: set(map(str, range(200))) for code in definitions.phecode}
    any_ids = {"CM_1": set(map(str, range(201))), "CM_2": set(map(str, range(202))),
               "NB_1": cases["NB_1"]}
    results = audit.eligibility(cohort, definitions, any_ids, cases, set(cases), 200).set_index("phecode")
    assert results.loc["CM_1", ["n_cases", "n_controls"]].tolist() == [200, 201]
    assert bool(results.loc["CM_1", "passes_counts"])
    assert not results.loc["CM_2", "passes_counts"]  # exactly 200 controls fails
    assert results.loc["NB_1", "n_controls"] == 1  # male participants are not controls
    assert not results.loc["NB_1", "passes_counts"]


def test_counts_follow_phetk_row_threshold_without_summing_duplicates(tmp_path):
    path = tmp_path / "counts.tsv"
    pd.DataFrame({"phecode": ["CM_1"] * 3 + ["BI_1"],
                  "person_id": ["1", "1", "2", "3"], "count": [1, 1, 2, 3]}).to_csv(path, sep="\t", index=False)
    any_ids, cases = audit.read_counts(path, 2)
    assert any_ids == {"CM_1": {"1", "2"}}
    assert cases == {"CM_1": {"2"}}


def test_complete_audit_distinguishes_below_threshold_from_eligible_missing(tmp_path, monkeypatch):
    root, assets = tmp_path / "analysis", tmp_path / "assets"
    root.mkdir()
    assets.mkdir()
    monkeypatch.setattr(audit.corrected, "analysis_root", lambda: root)
    monkeypatch.setattr(audit.config, "ASSETS_DIR", assets)
    definitions = pd.DataFrame({"phecode": ["BI_1", "CM_1", "CM_2", "NB_1"], "sex": ["Both"] * 4})
    definitions.to_csv(assets / "phecodeX_sex.csv", index=False)
    source = root / "phewas/phewas.discovery.tsv.gz"
    source.parent.mkdir()
    pd.DataFrame({"region_id": ["r1"], "phecode": ["BI_1"], "p": [.0001],
                  "cohort": ["discovery"]}).to_csv(source, sep="\t", index=False)
    audit.corrected._write_stage_manifest("discovery_phewas", {"analysis_contract": "fixture"}, [source])
    worker_dir = root / "phewas/phetk_runs/discovery"
    worker_dir.mkdir(parents=True)
    cohort_path = worker_dir / "r1.cohort.tsv"
    pd.DataFrame({"person_id": list(range(402)), "sex_at_birth": [0] * 402,
                  "eur_dosage": [1.] * 402}).to_csv(cohort_path, sep="\t", index=False)
    counts = pd.DataFrame([{"person_id": i, "phecode": code, "count": 2}
                           for code, n in [("CM_1", 199), ("CM_2", 200), ("NB_1", 200)]
                           for i in range(n)])
    count_path = tmp_path / "counts.tsv"
    counts.to_csv(count_path, sep="\t", index=False)
    raw_path = worker_dir / "r1.phewas.tsv"
    pd.DataFrame({"phecode": ["BI_1", "NB_1"], "cases": [200, 200], "controls": [202, 202]}).to_csv(raw_path, sep="\t", index=False)
    audit.corrected._atomic_json(raw_path.with_suffix(".tsv.json"), {
        "cohort_file_path": str(cohort_path), "phecode_count_file_path": str(count_path),
        "phecode_version": "X", "method": "logit", "male_as_one": True,
        "min_cases": 200, "min_phecode_count": 2, "phecode_to_process": definitions.phecode.tolist(),
        "independent_variable_of_interest": "eur_dosage", "covariate_cols": ["sex_at_birth"]})
    checkpoint = worker_dir / "r1.complete.json"
    audit.corrected._atomic_json(checkpoint, {"analysis_contract": "fixture", "region_id": "r1",
        "target_contract": "all-region-phecode-tests", "phetk_version": "0.3.3",
        "min_cases": 200, "min_phecode_count": 2})
    result = audit.run()
    assert result["issues"] == []
    output = result["directory"]
    frame = pd.read_csv(output / "congenital_neonatal_eligibility.tsv", sep="\t").set_index("phecode")
    assert frame.status.to_dict() == {"CM_1": "below_case_control_threshold",
                                     "CM_2": "eligible_no_saved_result",
                                     "NB_1": "eligible_raw_result_not_retained"}
    coverage = pd.read_csv(output / "category_coverage.tsv", sep="\t")
    assert len(coverage) == 18
    assert coverage.retained_tests.sum() == 1
    assert "person_id" not in frame.columns
    # Missing provenance must be unresolved, not interpreted as zero eligible codes.
    checkpoint.unlink()
    result = audit.run()
    assert len(result["issues"]) == 1
    assert pd.read_csv(output / "congenital_neonatal_eligibility.tsv", sep="\t").empty
