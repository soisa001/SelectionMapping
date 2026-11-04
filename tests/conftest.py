"""Explicit synthetic reference fixture; no controlled input staging in tests."""
import pytest


@pytest.fixture(autouse=True)
def controlled_exclusion_manifest(tmp_path, monkeypatch):
    path = tmp_path / "excluded_related_sample_ids.txt"
    path.write_text("excluded-test-1\nexcluded-test-2\n", encoding="utf-8")
    monkeypatch.setenv("AOU_EXCLUDED_SAMPLE_IDS_FILE", str(path))
    return path


@pytest.fixture
def synthetic_finemap_mask(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_hardmask.bed"
    path.write_text("chr1\t300000001\t300000010\n", encoding="utf-8")
    monkeypatch.setenv("AOU_FIG4_HARDMASK_BED", str(path))
    return path


@pytest.fixture
def panel_gene_annotations(tmp_path, monkeypatch):
    from scripts import figure4_finemap
    from scripts.common import config

    lookup = tmp_path / "phecodeX_sex.csv"
    lookup.write_text("phecode,sex\nMB_280.1,Both\nMB_280.11,Both\nGU_626.1,Female\nGU_592,Both\nEM_247.5,Both\n", encoding="utf-8")
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path)

    def genes(locus, start, end):
        names = ["ADH1B", "CFAP69", "ZMYM1"] + [f"LONG_TEST_GENE{i}" for i in range(9)]
        return [{"name": name, "tx_start": start + 10_000 + i * 60_000,
                 "tx_end": start + 30_000 + i * 60_000, "strand": "+",
                 "exons": [(start + 10_000 + i * 60_000, start + 30_000 + i * 60_000)]}
                for i, name in enumerate(names)]

    monkeypatch.setattr(figure4_finemap, "_genes_for_track", genes)
