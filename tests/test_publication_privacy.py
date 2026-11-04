import json
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from panel_fixtures import admixture_frame
from scripts import admixture_plots, figure1_data, figure1_pca, figure2_1025
from scripts.common import samples


def test_exclusions_are_read_from_controlled_manifest(controlled_exclusion_manifest):
    frame = pd.DataFrame({"sample_id": ["kept", "excluded-test-1", "excluded-test-2"]})
    assert samples.exclude_related(frame)["sample_id"].tolist() == ["kept"]
    controlled_exclusion_manifest.write_text("kept\n", encoding="utf-8")
    assert samples.exclude_related(frame)["sample_id"].tolist() == ["excluded-test-1", "excluded-test-2"]


def test_missing_exclusions_fail_without_revealing_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("AOU_EXCLUDED_SAMPLE_IDS_FILE", str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match="Controlled exclusion manifest"):
        samples.load_excluded_sample_ids()


@pytest.mark.parametrize("contents", ["", "sample_id\ntest-only\n", "test-only\ntest-only\n", "test-only\tvalue\n"])
def test_invalid_exclusions_do_not_echo_file_contents(contents, controlled_exclusion_manifest):
    controlled_exclusion_manifest.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        samples.load_excluded_sample_ids()
    assert "test-only" not in str(error.value)


def test_pca_count_error_does_not_disclose_excluded_ids():
    with pytest.raises(ValueError) as error:
        figure1_pca.canonical_aou_ids({"kept", "excluded-test-1"})
    assert "excluded-test" not in str(error.value)


def test_vcf_mismatch_does_not_log_missing_ids(monkeypatch):
    header = Mock(samples=["kept"])
    monkeypatch.setattr(figure2_1025.admixture_lai, "_cyvcf2", lambda: Mock(return_value=header))
    with pytest.raises(ValueError) as error:
        figure2_1025._selected_samples("test.vcf", ("kept", "private-test-identifier"))
    assert "private-test-identifier" not in str(error.value)


def test_finemap_applies_controlled_exclusions(monkeypatch):
    from scripts import figure4_finemap
    monkeypatch.setattr(samples, "N_AOU", 2)
    indices, kept = figure4_finemap.unrelated_panel_indices(
        ["kept-a", "excluded-test-1", "kept-b"],
        allowed_ids={"kept-a", "excluded-test-1", "kept-b"},
    )
    assert indices.tolist() == [0, 2]
    assert kept == ["kept-a", "kept-b"]


def test_tract_plot_has_no_participant_identifier():
    data = admixture_frame()
    data["tracts"]["sample_id"] = "private-test-identifier"
    fig, ax = plt.subplots()
    try:
        admixture_plots.panel_A(ax, data)
        labels = [text.get_text() for text in fig.findobj(plt.Text)]
        assert not any("private-test-identifier" in label for label in labels)
    finally:
        plt.close(fig)


def test_equivalent_exclusion_manifest_preserves_valid_lai_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(figure2_1025, "manifest_dir", lambda: tmp_path)
    output = tmp_path / "output.txt"
    output.write_text("verified output", encoding="utf-8")
    exclusions = samples.load_excluded_sample_ids()
    contract = {"sample_n": 1025, "excluded_ids": sorted(exclusions)}
    figure2_1025._write_stage_manifest("lai", contract, [output])
    current = {"sample_n": 1025, "exclusion_manifest_sha256": figure2_1025._sample_sha(exclusions)}
    assert figure2_1025._stage_current("lai", current, [output])
    manifest = figure2_1025._stage_manifest_path("lai").read_text(encoding="utf-8")
    assert "excluded-test" not in manifest
    assert json.loads(manifest)["contract"] == current
    changed = {**current, "exclusion_manifest_sha256": "changed"}
    assert not figure2_1025._stage_current("lai", changed, [output])
    output.write_text("corrupted output", encoding="utf-8")
    assert not figure2_1025._stage_current("lai", current, [output])


def test_state_display_does_not_emit_exact_positive_counts_below_twenty():
    frame = pd.DataFrame({"state": ["CA", "NY", "TX"], "n": [1, 19, 25]})
    displayed = figure1_data.display_state_counts(frame)
    assert displayed["n"].tolist() == [20, 20, 25]
    assert frame["n"].tolist() == [1, 19, 25]
