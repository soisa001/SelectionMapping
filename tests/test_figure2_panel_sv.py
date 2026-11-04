"""Synthetic checks for the separate panel-SNV + SV analysis."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scripts import admixture_finemap as fm, admixture_inputs
from scripts import figure2_loci as loci, figure2_locus_figures as pairs
from scripts import figure2_1025 as corrected
from test_figure2_loci import staged
from test_figure2_marker_panel import panel_staged, record

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")


def test_panel_svs_use_mac_without_carrier_cutoff_and_exclude_nonpanel_small_variants(monkeypatch):
    samples = np.array([f"synthetic_{i}" for i in range(100)])
    covariates = pd.DataFrame(0.0, index=samples, columns=fm.admixture_phewas.COVARIATE_COLUMNS)
    counts = pd.DataFrame({"person_id": samples[:25], "phecode": "TEST", "count": 2})
    # panel SNV, off-panel SNV, small indel, MAC20 DEL, MAC21 INS,
    # MAC21/11-carrier DEL, high-missingness INS, MAC21 DEL.
    records = [record(100005), record(100006), record(100007, alts=["AT"]),
               record(100008, ref="A" * 51, alts=["A"]),
               record(100009, alts=["A" * 51]),
               record(100010, ref="A" * 51, alts=["A"]),
               record(100011, alts=["A" * 51]),
               record(100012, ref="A" * 51, alts=["A"])]
    calls = np.zeros((100, len(records)))
    calls[:21, :] = 1
    calls[20, 3] = 0
    calls[:, 5] = 0
    calls[:10, 5] = 2
    calls[10, 5] = 1  # MAC21 passes despite only 11 carriers.
    calls[-6:, 6] = np.nan
    reader = Mock(samples=samples, seqnames=["chr1"])
    reader.side_effect = lambda _: iter(records)
    monkeypatch.setitem(sys.modules, "cyvcf2", SimpleNamespace(VCF=lambda *_: reader))
    monkeypatch.setattr(admixture_inputs, "genotype_dosage", lambda item: calls[:, item.POS - 100005])
    monkeypatch.setattr(fm.admixture_phewas, "MIN_CASES", 2)
    hit = dict(chrom=1, start=100000, end=100100, region_id="test", phecode="TEST")
    args = (hit, Path("synthetic.vcf"), covariates, counts, {"TEST": "Both"})
    allowed = frozenset(["chr1:100005:A:G"])
    panel = fm._prepare_analysis(*args, allowed_variants=allowed)
    combined = fm._prepare_analysis(*args, allowed_variants=allowed, include_svs=True)
    full = fm._prepare_analysis(*args)
    assert panel["pos"].tolist() == [100005]
    assert combined["pos"].tolist() == [100005, 100009, 100010, 100012]
    assert combined["sv_type"].tolist() == ["", "INS", "DEL", "DEL"]
    assert combined["mac"].tolist() == [21, 21, 21, 21]
    assert combined["qc_counts"]["n_panel_svs_before_qc"] == 5
    assert combined["qc_counts"]["n_panel_svs_after_qc"] == 3
    assert combined["qc_counts"]["n_fail_panel"] == 2
    assert combined["qc_counts"]["n_fail_mac"] == 1
    assert combined["qc_counts"]["n_fail_carriers"] == 0
    assert combined["qc_counts"]["n_fail_missingness"] == 1
    assert full["pos"][full["sv_type"] != ""].tolist() == [100009, 100012]


def test_separate_panel_sv_outputs_resume_and_repair(panel_staged, monkeypatch):
    from scripts import figure2_sv_linkage
    linkage = Mock(side_effect=lambda summary, output, **kwargs: output)
    monkeypatch.setattr(figure2_sv_linkage, "run_locus_linkage", linkage)
    (ids, fit, *_), _ = panel_staged
    loci.run(ids=ids)
    loci.run(ids=ids, panel_only=True)
    linkage.assert_not_called()
    roots = (corrected.analysis_root(), loci.output_dir().parent)
    original = {path: corrected._sha256_file(path) for root in roots
                for path in root.rglob("*") if path.is_file()}
    old_fit = fit.side_effect

    def fit_with_svs(hit, *args, allowed_variants=None, include_svs=False):
        frame = old_fit(hit, *args, allowed_variants=allowed_variants)
        assert include_svs
        for index, kind, ref, alt in [(1, "DEL", "A" * 51, "A"), (3, "INS", "A", "A" * 51)]:
            frame.loc[index, "variant_id"] = admixture_inputs.variant_id(hit["chrom"], frame.loc[index, "pos"], ref, alt)
            frame.loc[index, ["sv_type", "sv_length_bp"]] = [kind, 50]
            frame.loc[index, ["mac", "n_carriers"]] = [21, 11]
        return frame

    fit.side_effect = fit_with_svs
    result = loci.run(ids=ids, panel_only=True, include_svs=True)
    assert result["directory"].name == "loci_panel_sv"
    linkage.assert_called_once_with(result["summary"], result["directory"] / "sv_linkage", threads=4)
    assert result["linked_sv_table"] == result["directory"] / "sv_linkage" / "linked_svs.csv"
    figures = result["two_panel_figures"]
    assert figures["directory"].parent == result["directory"]
    summary = pd.read_csv(figures["results_table"])
    assert summary["variant_set"].eq("flare_panel_snvs_svs").all()
    assert summary["fine_map_table"].map(lambda path: Path(path).parent.name == "finemap_panel_sv").all()
    assert summary["min_p_is_sv"].all()
    assert summary["sv_min_mac"].eq(20).all() and summary["sv_min_carriers"].eq(0).all()
    assert pd.read_csv(figures["credible_set_table"])["is_sv"].all()
    assert json.loads((figures["directory"] / "run_manifest.json").read_text())["variant_set"] == "flare_panel_snvs_svs"
    assert len(summary) == 13 and fit.call_count == 39
    assert {path: corrected._sha256_file(path) for path in original} == original
    loci.run(ids=ids, panel_only=True, include_svs=True)
    assert fit.call_count == 39
    Path(summary.iloc[0]["fine_map_table"]).write_bytes(b"corrupt fit")
    loci.run(ids=ids, panel_only=True, include_svs=True)
    assert fit.call_count == 40
    assert {path: corrected._sha256_file(path) for path in original} == original
    assert fm.ANALYSIS_SUBDIR.get() == "finemap" and loci.output_dir().name == "loci"


def test_include_svs_requires_panel_mode():
    with pytest.raises(ValueError, match="requires panel_only"):
        loci.run(include_svs=True)


def test_sv_ancestry_uses_shared_loci_interpolation():
    frame = pd.DataFrame({"pos": [50, 100, 125, 200, 250], "sv_type": "DEL"})
    markers = pd.DataFrame({"pos": [100, 200], "eur_frac": [0.2, 0.6]})
    svs = pairs._sv_positions(frame, markers)
    np.testing.assert_allclose(svs["overall_eur_frac"], [0.2, 0.2, 0.3, 0.6, 0.6])
    assert svs["ancestry_method"].tolist() == ["nearest_boundary", "observed_marker",
                                             "linear_interpolation", "observed_marker", "nearest_boundary"]
