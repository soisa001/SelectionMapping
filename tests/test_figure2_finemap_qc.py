"""Exercise fine-map QC on synthetic calls, before imputation and in the fit cohort."""
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scripts import admixture_finemap as fm, admixture_phewas as phewas

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")


def prepare(monkeypatch, columns):
    # Only the first 100 of 120 VCF samples belong to this female phenotype.
    samples = np.array([f"synthetic_{i}" for i in range(120)])
    covariates = pd.DataFrame(0.0, index=samples, columns=phewas.COVARIATE_COLUMNS)
    covariates.loc[samples[100:], "sex_at_birth"] = 1.0
    counts = pd.DataFrame({"person_id": samples[:25], "phecode": "TEST", "count": 2})
    genotype = {
        "samples": samples, "X": np.column_stack(list(columns.values())),
        "variant_id": list(columns), "pos": np.arange(len(columns)) + 100_001,
        "sv_type": np.array([""] * len(columns)),
        "sv_length_bp": np.zeros(len(columns), dtype=int),
    }
    monkeypatch.setattr(fm, "_extract_genotypes", lambda *_: genotype)
    monkeypatch.setattr(phewas, "MIN_CASES", 2)
    hit = pd.Series({"chrom": 1, "start": 100_000, "end": 100_100,
                     "region_id": "test", "phecode": "TEST"})
    return fm._prepare_analysis(hit, Path("synthetic.vcf"), covariates, counts,
                                {"TEST": "Female"})


def test_observed_mac_strictly_over_20_and_20_carriers_before_imputation(monkeypatch):
    columns = {}
    for mac in (5, 20, 21):
        x = np.zeros(120)
        x[:mac] = 1
        columns[f"mac{mac}"] = x
    x = np.zeros(120)
    x[:15] = 2  # MAC 30, but only 15 ALT carriers.
    columns["15_homozygotes"] = x
    x = np.zeros(120)
    x[:20] = 1
    x[0] = 2  # Exactly 20 carriers and MAC 21 passes.
    columns["20_carriers_mac21"] = x
    x = columns["mac20"].copy()
    x[90:95] = np.nan  # Imputation would falsely inflate observed MAC 20 to >21.
    columns["imputed_mac_inflation"] = x
    x = np.full(120, 2.0)
    x[:21] = 1  # REF is the minor allele; MAC must not equal ALT count.
    columns["ref_minor_mac21"] = x
    result = prepare(monkeypatch, columns)
    assert result["variant_id"].tolist() == ["mac21", "20_carriers_mac21", "ref_minor_mac21"]
    np.testing.assert_array_equal(result["mac"], [21, 21, 21])


def test_missingness_and_allele_counts_use_exact_phenotype_subset(monkeypatch):
    columns = {}
    for missing_count in (5, 6):
        x = np.zeros(120)
        x[:21] = 1
        x[90:90 + missing_count] = np.nan
        columns[f"missing{missing_count}"] = x
    x = np.zeros(120)
    x[:4] = x[100:] = 1  # 24 carriers in VCF, only four in phenotype cohort.
    columns["outside_scope"] = x
    columns["all_missing"] = np.full(120, np.nan)
    result = prepare(monkeypatch, columns)
    assert result["variant_id"].tolist() == ["missing5"]
    assert len(result["y"]) == 100
    assert result["mac"].tolist() == [21]
    assert result["n_called"].tolist() == [95]
    assert result["n_carriers"].tolist() == [21]
    assert result["n_case_carriers"].tolist() == [21]
    assert result["n_control_carriers"].tolist() == [0]
    assert result["allele_number"].tolist() == [190]
    assert result["n_called_cases"].tolist() == [25]
    assert result["n_called_controls"].tolist() == [70]
    np.testing.assert_allclose(result["X"][90:95], 21 / 95)
    assert np.isfinite(result["X"]).all()


def test_fine_map_records_qc_diagnostics_and_min_p_fallback(monkeypatch):
    rng = np.random.default_rng(20260715)
    columns = {str(i): rng.binomial(2, 0.3, 120).astype(float) for i in range(3)}
    prepared = prepare(monkeypatch, columns)
    monkeypatch.setattr(fm, "_prepare_analysis", lambda *_: prepared)

    def fit(X, y, *, L, diagnostics):
        diagnostics.update(susie_converged=True, susie_niter=5,
                           susie_version="synthetic", susie_n_active_effects=0,
                           susie_prior_variances="0,0", susie_warnings="")
        return np.zeros(3), {}

    monkeypatch.setattr(fm, "run_susie", fit)
    monkeypatch.setattr(phewas, "score_test_batch", Mock(return_value=(
        np.zeros(3), np.ones(3), np.array([0.5, 0.05, 0.01]))))
    hit = pd.Series({"region_id": "test", "phecode": "TEST"})
    frame = fm.fine_map_hit(hit, None, None, None, None)
    assert frame["lead_variant"].tolist() == [False, False, True]
    assert frame["lead_selection"].eq("minimum_p_no_credible_set").all()
    assert frame["qc_contract"].eq(fm.QC_CONTRACT).all()
    assert frame["mac"].gt(20).all()
    assert frame["susie_converged"].all()
    assert frame.iloc[2]["r2_to_lead"] == pytest.approx(1)
    summary = fm.summarize_fit(frame)
    assert summary["top_pip_variant"] == ""
    assert summary["lead_variant"] == "2"

    def failed_fit(X, y, *, L, diagnostics):
        diagnostics.update(susie_converged=False, susie_niter=100)
        return np.zeros(3), {}

    monkeypatch.setattr(fm, "run_susie", failed_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        fm.fine_map_hit(hit, None, None, None, None)
