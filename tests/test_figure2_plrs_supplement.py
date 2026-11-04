"""Synthetic-only selection, holdout correctness, and output/cache checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import pytest

from scripts import figure2_plrs_supplement as supplement


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "analysis"
    figures = tmp_path / "figures"
    monkeypatch.setattr(supplement.corrected, "analysis_root", lambda: root)
    monkeypatch.setattr(supplement.config, "figure_dir", lambda _: figures)
    ids = [str(i) for i in range(1025)]
    monkeypatch.setattr(supplement.samples, "load_aou_sample_list", lambda **_: ids)
    monkeypatch.setattr(supplement.samples, "load_excluded_sample_ids", lambda: frozenset())
    models, predictions, hits = [], [], []
    for index in range(3):
        truth = np.array([0., .1, .2, .4, .8, 1.2, 1.5, 2.])
        predicted = truth * (0.95 if index < 2 else 0.2)
        r2 = 1 - np.sum((truth - predicted) ** 2) / np.sum((truth - truth.mean()) ** 2)
        region = dict(region_id=f"r{index}", chrom=index + 1,
                      start=100 + index, end=200 + index)
        models.append(dict(**region, n_snvs=15 + index * 12, n_test=8, r2_holdout=r2))
        predictions.append(pd.DataFrame(dict(region_id=f"r{index}", sample_id=ids[:8],
                                             true_eur_dosage=truth, predicted_eur_dosage=predicted)))
        for phecode in ("001", "002"):
            hits.append(dict(**region, phecode=phecode, p_disc=.0001 if index < 2 else .02,
                             replication_status="Not replicable"))
    models, predictions, publication = pd.DataFrame(models), pd.concat(predictions), pd.DataFrame(hits)
    meta = publication.rename(columns={"p_disc": "p_discovery"})
    paths = [root / "plrs/plrs_models_summary.tsv", root / "plrs/plrs_holdout_predictions.tsv.gz"]
    for path, frame in zip(paths, (models, predictions)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    (root / "phewas").mkdir()
    meta.to_csv(root / "phewas/meta_analysis_results.tsv", sep="\t", index=False)
    figures.mkdir()
    publication.drop(columns="region_id").to_csv(figures / "Figure_2_1025_phewas_results.csv", index=False)
    contract = dict(pipeline=supplement.corrected.PIPELINE_CONTRACT,
                    lai_samples=supplement.corrected._sample_sha(ids), r2_min=.5,
                    test_fraction=.2, random_state=42)
    supplement.corrected._write_stage_manifest("plrs_models", contract, paths)
    return models, predictions, publication.loc[publication.p_disc < .001], root, figures


def test_selection_deduplicates_nominal_regions_and_keeps_nonreplicable(inputs):
    models, predictions, regions, provenance = supplement.load_inputs()
    assert len(models) == 3  # B includes the nonnominal model, including negative R2.
    assert models.r2_holdout.min() < 0
    assert regions.region_id.tolist() == ["r0", "r1"]
    assert regions.n_nominal_associations.tolist() == [2, 2]
    assert provenance["n_associations"] == 4
    assert regions.phecodes.tolist() == ["001;002", "001;002"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_region", "wrong_dosage"])
def test_holdout_must_match_each_regions_saved_test_accuracy(inputs, damage):
    models, holdout, hits, *_ = inputs
    holdout = holdout.reset_index(drop=True)
    if damage == "missing":
        holdout = holdout.iloc[1:]
    elif damage == "duplicate":
        holdout.loc[0, "sample_id"] = holdout.loc[1, "sample_id"]
    elif damage == "wrong_region":
        holdout.loc[holdout.region_id == "r0", "region_id"] = "old_chr9"
    else:
        holdout.loc[0, "predicted_eur_dosage"] = 1.9
    with pytest.raises(ValueError, match="holdout|Holdout"):
        supplement.prepare_regions(models, holdout, hits)


def test_corrupt_source_manifest_rejected(inputs):
    *_, root, _ = inputs
    path = root / "plrs/plrs_models_summary.tsv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        supplement.load_inputs()


def test_saves_vectors_and_reuses_only_valid_outputs(inputs, monkeypatch):
    result = supplement.run()
    directory = Path(result["directory"])
    assert len(list(directory.glob("*.png"))) == 3
    pdfs = sorted(directory.glob("*.pdf"))
    assert len(pdfs) == 3
    for path in pdfs:
        with pymupdf.open(path) as doc:
            assert not doc[0].get_images(full=True)
            assert "EUR-like" in doc[0].get_text()
    before = {path: path.stat().st_mtime_ns for path in pdfs}
    supplement.run()
    assert before == {path: path.stat().st_mtime_ns for path in pdfs}
    assert "2 unique regions" in (directory / "caption.txt").read_text(encoding="utf-8")
    pdfs[0].write_bytes(b"corrupt")
    supplement.run()
    assert pdfs[0].read_bytes().startswith(b"%PDF-")
    assert all(path.stat().st_size > 1000 for path in pdfs)
