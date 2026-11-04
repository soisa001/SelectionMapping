"""Synthetic BED, VCF and upstream cache tests; never read controlled genotypes."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scripts import admixture_finemap as fm, admixture_mask as masks
from scripts import admixture_inputs, admixture_lai, admixture_plrs
from scripts import figure2_1025 as corrected, figure2_mask_audit as audit
from scripts.common import config

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")


def test_bed_endpoints_unsorted_nested_and_adjacent_intervals(synthetic_finemap_mask):
    synthetic_finemap_mask.write_text("# test\nchr1\t20\t30\n1\t10\t25\nchr1\t12\t13\n1\t30\t31\n")
    mask = masks.Hardmask()
    np.testing.assert_array_equal(mask.contains(1, [10, 11, 20, 30, 31, 32]),
                                  [False, True, True, True, True, False])
    assert mask.contains("chr1", 31)
    assert not mask.contains(22, 15)


@pytest.mark.parametrize("content", ["", "# empty\n", "chr1\t-1\t10", "chr1\t20\t20", "chrom start end"])
def test_missing_or_corrupt_bed_fails_closed(synthetic_finemap_mask, content):
    synthetic_finemap_mask.write_text(content)
    with pytest.raises((ValueError, FileNotFoundError)):
        masks.Hardmask()
    synthetic_finemap_mask.unlink()
    with pytest.raises(FileNotFoundError):
        masks.Hardmask()


def test_extraction_masks_all_types_before_genotype_decoding(monkeypatch, synthetic_finemap_mask):
    synthetic_finemap_mask.write_text("chr1\t100\t110\n")
    records = [SimpleNamespace(CHROM="chr1", POS=pos, REF=ref, ALT=[alt], INFO={})
               for pos, ref, alt in [(99, "A" * 60, "A"), (100, "A", "G"),
                                    (101, "A", "G"), (105, "A", "AT"),
                                    (110, "A" * 60, "A"), (111, "A", "A" * 60)]]
    reader = Mock(samples=["synthetic_a", "synthetic_b"], seqnames=["chr1"])
    reader.side_effect = lambda _: iter(records)
    monkeypatch.setitem(sys.modules, "cyvcf2", SimpleNamespace(VCF=lambda *_: reader))
    decoded = []
    def dosage(record):
        decoded.append(record.POS)
        return np.array([0.0, 1.0])
    monkeypatch.setattr(admixture_inputs, "genotype_dosage", dosage)
    result = fm._extract_genotypes("synthetic.vcf", 1, 100, 120)
    assert decoded == [100, 111]
    assert result["pos"].tolist() == [100, 111]
    assert result["sv_type"].tolist() == ["", "INS"]
    assert result["mask_counts"] == {"n_variants_before_hardmask": 5, "n_fail_hardmask": 3}
    reader.close.assert_called_once()


def test_stages_only_required_bed_with_billing_then_reuses(monkeypatch, tmp_path):
    monkeypatch.delenv("AOU_FIG4_HARDMASK_BED")
    monkeypatch.setenv("GOOGLE_PROJECT", "synthetic-billing")
    monkeypatch.setattr(config, "ADMIXTURE_WORK_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(config, "ADMIXTURE_HARDMASK_GS", "gs://synthetic/masks/hardmask.hg38.v4.over99.bed")
    def stage(command, *, check):
        assert check and command[:4] == ["gcloud", "storage", "rsync", "gs://synthetic/masks"]
        assert command[-2:] == ["--billing-project", "synthetic-billing"]
        pattern = command[command.index("--exclude") + 1]
        assert re.search(pattern, "unrelated.vcf.gz")
        assert not re.search(pattern, "hardmask.hg38.v4.over99.bed")
        (Path(command[4]) / "hardmask.hg38.v4.over99.bed").write_text("chr1\t1\t2\n")
    staging = Mock(side_effect=stage)
    monkeypatch.setattr(masks.subprocess, "run", staging)
    masks.Hardmask()
    masks.Hardmask()
    staging.assert_called_once()


def test_upstream_audit_checks_masked_positions_and_rejects_corruption(monkeypatch, tmp_path, synthetic_finemap_mask):
    monkeypatch.setattr(config, "ANALYSIS_DIR", tmp_path / "analysis")
    synthetic_finemap_mask.write_text("chr1\t3000000\t3000010\n")
    with corrected.analysis_context(("synthetic_a", "synthetic_b")):
        scans = []
        for chrom in range(1, 23):
            path = corrected.lai_dir() / "chromosome_scans" / f"chr{chrom}.eur_fraction.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, pos=np.array([3000000, 3000001, 3000011]),
                                eur_frac=np.array([0.2, 0.1, 0.2]), n_samples=np.array([2]))
            scans.append(path)
        corrected._write_stage_manifest("lai", {"sample_n": 2}, scans)
        model = admixture_plrs.model_path("synthetic_region")
        admixture_inputs.atomic_pickle_dump(dict(region={"region_id": "synthetic_region", "chrom": 1},
             feature_ids=["synthetic_1", "synthetic_2"], feature_positions=np.array([3000001, 3000011])), model)
        summary = admixture_plrs.analysis_dir() / "plrs_models_summary.tsv"
        pd.DataFrame({"region_id": ["synthetic_region"]}).to_csv(summary, sep="\t", index=False)
        corrected._write_stage_manifest("plrs_models", {"samples": "synthetic"}, [summary, model])
        result = audit.run()
        assert result["status"] == "overlap"
        frame = pd.read_csv(result["table"], sep="\t")
        assert len(frame) == 45
        assert frame.groupby("stage")["n_in_hardmask"].sum().to_dict() == {
            "lai_markers": 1, "outlier_calling_markers": 1, "plrs_features": 1}
        assert "span gaps" in json.loads(Path(result["manifest"]).read_text())["plrs_target_weighting"]
        synthetic_finemap_mask.write_text("chr1\t9000000\t9000010\n")
        assert audit.run()["status"] == "clear"
        scans[0].write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="corrupt"):
            audit.run()
