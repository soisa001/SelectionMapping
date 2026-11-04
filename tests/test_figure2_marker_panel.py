"""Synthetic FLARE identities, allele matching, QC and independent panel caches."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scripts import admixture_finemap as fm, admixture_inputs
from scripts import figure2_1025 as corrected, figure2_loci as loci
from scripts import figure2_marker_panel as panel_module
from scripts import figure2_locus_figures as pairs
from scripts.common import config
from test_figure2_loci import frame_for, sources, staged, phewas_kwargs  # Reuse isolated synthetic inputs.

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")


def record(pos, ref="A", alts=None, *, chrom="chr1", ancestry=True):
    return SimpleNamespace(POS=int(pos), REF=ref, ALT=["G"] if alts is None else alts,
                           CHROM=chrom, INFO={},
                           format=lambda name: np.array([[0], [1]]) if ancestry else None)


def test_catalog_keeps_only_observed_biallelic_snvs_and_checks_full_lai_positions():
    records = [record(99), record(100), record(101, alts=["AT"]), record(102, alts=["G", "T"]),
               record(103, ref="N"), record(104, alts=["A"]), record(105, ref="C", alts=["T"]),
               record(106, ancestry=False)]
    reader = Mock(side_effect=lambda _: iter(records))
    hit = dict(chrom=1, start=100100, end=100100, region_id="test")
    markers = pd.DataFrame({"pos": range(100, 106)})
    frame, counts = panel_module._read_window(reader, "chr1", hit, markers, 2)
    assert frame["variant_id"].tolist() == ["chr1:100:A:G", "chr1:105:C:T"]
    assert counts == {"n_flare_records": 7, "n_without_ancestry": 1, "n_non_snv": 4,
                      "n_lai_marker_positions": 6}
    reader.assert_called_with("chr1:100-200100")
    with pytest.raises(ValueError, match="positions disagree"):
        panel_module._read_window(reader, "chr1", hit, markers.iloc[:-1], 2)
    with pytest.raises(ValueError, match="ancestry dimensions"):
        panel_module._read_window(reader, "chr1", hit, markers, 3)
    records.append(record(100))
    with pytest.raises(ValueError, match="Duplicate FLARE SNV"):
        panel_module._read_window(reader, "chr1", hit, markers, 2)


def test_discovery_extraction_requires_exact_alleles_and_applies_hardmask_first(monkeypatch, synthetic_finemap_mask):
    synthetic_finemap_mask.write_text("chr1\t109\t110\n")
    records = [record(100), record(100, alts=["T"]), record(100, ref="G", alts=["A"]),
               record(101), record(102, ref="A" * 70, alts=["A"]), record(110), record(111)]
    reader = Mock(samples=["synthetic_1", "synthetic_2"], seqnames=["chr1"])
    reader.side_effect = lambda _: iter(records)
    monkeypatch.setitem(sys.modules, "cyvcf2", SimpleNamespace(VCF=lambda *_: reader))
    decoded = []
    def dosage(item):
        decoded.append(item.POS)
        return np.array([0, 1], dtype=float)
    monkeypatch.setattr(admixture_inputs, "genotype_dosage", dosage)
    allowed = frozenset(["chr1:100:A:G", "chr1:110:A:G", "chr1:111:A:G"])
    extracted = fm._extract_genotypes(Path("synthetic.vcf"), 1, 100, 120, allowed_variants=allowed)
    assert extracted["variant_id"] == ["chr1:100:A:G", "chr1:111:A:G"]
    assert decoded == [100, 111]
    assert extracted["sv_type"].tolist() == ["", ""]
    assert extracted["mask_counts"] == dict(n_variants_before_hardmask=7, n_fail_hardmask=1,
        n_variants_before_panel=6, n_fail_panel=4, n_variants_after_panel=2)
    with pytest.raises(ValueError, match="No eligible variants"):
        fm._extract_genotypes(Path("synthetic.vcf"), 1, 100, 120, allowed_variants=frozenset())


def test_panel_extraction_restores_svs_without_admitting_off_panel_snvs(monkeypatch):
    records = [record(100), record(101), record(102, ref='A' * 51, alts=['A'])]
    reader = Mock(samples=['s0', 's1'], seqnames=['chr1'])
    reader.side_effect = lambda _: iter(records)
    monkeypatch.setitem(sys.modules, 'cyvcf2', SimpleNamespace(VCF=lambda *_: reader))
    monkeypatch.setattr(admixture_inputs, 'genotype_dosage', lambda _: np.array([0., 1.]))
    extracted = fm._extract_genotypes(Path('synthetic.vcf'), 1, 100, 120,
                                     allowed_variants=frozenset(['chr1:100:A:G']), include_svs=True)
    assert extracted['pos'].tolist() == [100, 102]
    assert extracted['sv_type'].tolist() == ['', 'DEL']


@pytest.fixture
def panel_staged(staged, monkeypatch, tmp_path):
    ids, fit, *_ = staged
    directory = tmp_path / "lai_masked"
    directory.mkdir()
    monkeypatch.setattr(config, "LAI_MASKED_DIR", str(directory))
    records = {}
    for chrom in loci.select_hits(*sources())["chrom"].unique():
        path = directory / f"chr{chrom}.anc.vcf.gz"
        path.write_bytes(b"synthetic mocked FLARE file")
        Path(str(path) + ".tbi").write_bytes(b"synthetic mocked index")
        with np.load(corrected.lai_dir() / "chromosome_scans" / f"chr{chrom}.eur_fraction.npz") as scan:
            records[int(chrom)] = [record(pos, chrom=f"chr{chrom}") for pos in scan["pos"]]

    def open_vcf(path, *, samples):
        chrom = int(Path(path).name.split(".")[0].removeprefix("chr"))
        reader = Mock(samples=list(samples), seqnames=[f"chr{chrom}"])
        def query(interval):
            start, end = map(int, interval.split(":")[1].split("-"))
            return (item for item in records[chrom] if start <= item.POS <= end)
        reader.side_effect = query
        return reader
    monkeypatch.setattr(panel_module.admixture_lai, "_cyvcf2", lambda: open_vcf)
    def fit_panel(hit, *_args, allowed_variants=None):
        frame = frame_for(hit)
        if allowed_variants is not None:
            variants = sorted(allowed_variants, key=lambda value: (int(value.split(":")[1]), value))[:6]
            frame["variant_id"] = variants
            frame["pos"] = [int(value.split(":")[1]) for value in variants]
            frame["sv_type"], frame["sv_length_bp"] = "", 0
        return frame
    fit.side_effect = fit_panel
    return staged, records


def test_panel_run_is_separate_resumes_and_repairs_without_touching_full_analysis(panel_staged):
    (ids, fit, render, *_), records = panel_staged
    full = loci.run(ids=ids)
    # Include existing fits, stage manifests and every original figure/table.
    original_files = [path for root in (loci.output_dir(), corrected.analysis_root() / "finemap", corrected.manifest_dir())
                      for path in root.rglob("*") if path.is_file()]
    original_hashes = {path: corrected._sha256_file(path) for path in original_files}
    panel = loci.run(ids=ids, panel_only=True)
    assert panel["directory"].name == "loci_panel"
    assert panel["two_panel_figures"]["directory"].parent.name == "loci_panel"
    assert fit.call_count == 26
    summary = pd.read_csv(panel["two_panel_figures"]["results_table"])
    assert len(summary) == 13 and summary["status"].eq("ready").all()
    assert summary["variant_set"].eq("flare_panel_snvs").all()
    assert summary["fine_map_table"].map(lambda path: Path(path).parent.name == "finemap_panel").all()
    assert summary["n_panel_markers_in_window"].ge(71).all()
    assert summary["analysis_contract"].str.endswith("_flare_panel_snvs_v1").all()
    members = pd.read_csv(panel["two_panel_figures"]["credible_set_table"])
    assert len(members) == 13 and not members["is_sv"].any()
    assert {path: corrected._sha256_file(path) for path in original_files} == original_hashes
    assert loci.output_dir() == full["directory"]
    assert fm.ANALYSIS_SUBDIR.get() == "finemap"
    manifest = json.loads(panel["panel_manifest"].read_text())
    assert manifest["status"] == "ready" and manifest["n_regions"] == 9
    assert len(list(panel["directory"].glob("*.pdf"))) == 13
    assert len(list(panel["two_panel_figures"]["directory"].glob("*.pdf"))) == 13
    loci.run(ids=ids, panel_only=True)
    assert fit.call_count == 26 and pairs.render.call_count == 26
    catalog = Path(summary.iloc[0]["panel_catalog_path"])
    catalog.write_bytes(b"corrupted membership list")
    loci.run(ids=ids, panel_only=True)
    assert len(pd.read_csv(catalog, sep="\t")) >= 71
    assert fit.call_count == 26  # Same source membership restores the catalog without refitting.
    Path(summary.iloc[0]["pdf"]).unlink()
    loci.run(ids=ids, panel_only=True)
    assert pairs.render.call_count == 27 and fit.call_count == 26
    assert {path: corrected._sha256_file(path) for path in original_files} == original_hashes
    loci.run(ids=ids)
    assert fit.call_count == 26 and render.call_count == 26


def test_panel_restriction_still_applies_observed_discovery_mac_and_carrier_qc(monkeypatch):
    phewas = fm.admixture_phewas
    samples = np.array([f"synthetic_{index}" for index in range(100)])
    covariates = pd.DataFrame(0.0, index=samples, columns=phewas.COVARIATE_COLUMNS)
    counts = pd.DataFrame({"person_id": samples[:25], "phecode": "TEST", "count": 2})
    records = [record(pos) for pos in range(100005, 100009)]
    calls = np.zeros((100, 4))
    calls[:20, 0] = 1  # MAC = 20 fails despite panel membership.
    calls[:21, 1] = 1  # MAC = 21, 21 carriers passes.
    calls[:40, 2] = 1  # Passes QC but is not in the panel.
    calls[:15, 3] = 2  # MAC = 30 but only 15 carriers fails.
    reader = Mock(samples=samples, seqnames=["chr1"])
    reader.side_effect = lambda _: iter(records)
    monkeypatch.setitem(sys.modules, "cyvcf2", SimpleNamespace(VCF=lambda *_: reader))
    monkeypatch.setattr(admixture_inputs, "genotype_dosage", lambda item: calls[:, item.POS - 100005])
    monkeypatch.setattr(phewas, "MIN_CASES", 2)
    allowed = frozenset(f"chr1:{pos}:A:G" for pos in (100005, 100006, 100008))
    result = fm._prepare_analysis(dict(chrom=1, start=100000, end=100100, region_id="test", phecode="TEST"),
        Path("synthetic.vcf"), covariates, counts, {"TEST": "Both"}, allowed_variants=allowed)
    assert result["variant_id"].tolist() == ["chr1:100006:A:G"]
    assert result["mac"].tolist() == [21] and result["n_carriers"].tolist() == [21]
    assert result["qc_counts"]["n_fail_panel"] == 1
    assert result["qc_counts"]["n_fail_mac"] == result["qc_counts"]["n_fail_carriers"] == 1


def test_panel_membership_content_change_refits_only_affected_association(panel_staged):
    (ids, fit, *_), records = panel_staged
    result = loci.run(ids=ids, panel_only=True)
    assert fit.call_count == 13
    # Same source file size/mtime and same positions, but a different ALT allele.
    # Chromosome 1 has only one requested association.
    records[1][0].ALT = ["T"]
    loci.run(ids=ids, panel_only=True)
    assert fit.call_count == 14
    summary = pd.read_csv(result["two_panel_figures"]["results_table"])
    row = summary.loc[summary["chrom"].astype(str).isin(["1", "chr1"])].iloc[0]
    assert pd.read_csv(row["fine_map_table"], sep="\t")["variant_id"].str.endswith(":T").any()
    loci.run(ids=ids, panel_only=True)
    assert fit.call_count == 14


def test_panel_source_disagreement_fails_and_restores_default_scope(panel_staged):
    (ids, fit, *_), records = panel_staged
    records[1].pop()
    with pytest.raises(ValueError, match="positions disagree"):
        loci.run(ids=ids, panel_only=True)
    assert fit.call_count == 0
    assert loci.output_dir().name == "loci" and fm.ANALYSIS_SUBDIR.get() == "finemap"
    manifest = corrected.analysis_root() / "finemap_panel" / "panel_markers" / "run_manifest.json"
    assert json.loads(manifest.read_text())["status"] == "failed"


@pytest.mark.parametrize("problem", ["missing_index", "missing_sample"])
def test_missing_panel_inputs_fail_without_staging_or_fitting(panel_staged, monkeypatch, problem):
    (ids, fit, _, _, discovery_staging), _ = panel_staged
    if problem == "missing_index":
        (Path(config.LAI_MASKED_DIR) / "chr1.anc.vcf.gz.tbi").unlink()
        expected = "Local masked FLARE VCF and index required"
    else:
        monkeypatch.setattr(panel_module.admixture_lai, "_cyvcf2",
                            lambda: lambda *args, **kwargs: Mock(samples=["wrong_sample"]))
        expected = "required LAI sample manifest"
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        loci.run(ids=ids, panel_only=True)
    fit.assert_not_called()
    discovery_staging.assert_not_called()


def test_snv_only_figure_saves_vector_pdf_and_png_without_sv_legend(tmp_path):
    hit = loci.select_hits(*sources()).iloc[0]
    frame = frame_for(hit)
    frame["sv_type"], frame["sv_length_bp"] = "", 0
    start, end = fm.expected_window(hit)
    markers = pd.DataFrame({"pos": np.linspace(start, end, 71).astype(int),
                            "eur_frac": np.full(71, 0.19)})
    markers["in_region"] = markers["pos"].between(int(hit["start"]), int(hit["end"]))
    genes = pd.DataFrame(columns=["chrom", "start", "end", "gene", "exons"])
    figure = pairs.make_figure(frame, hit, markers, {"mean_eur": 0.22, "sd_eur": 0.01, "z_threshold": 4},
        genes=genes, lai_sample_n=1025, **phewas_kwargs(hit))
    try:
        assert [text.get_text() for text in figure.axes[1].get_legend().get_texts()] == [
            "Genome-wide mean", "Mean ± 4.00 SD"]
        paths = [tmp_path / "panel.png", tmp_path / "panel.pdf"]
        for path in paths:
            figure.savefig(path)
        loci._validate_plots(paths)
    finally:
        pairs.plt.close(figure)
