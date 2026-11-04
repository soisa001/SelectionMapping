from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from panel_fixtures import idw_frames
from scripts import figure1_data, figure1_idw, figure1_plots
from scripts.common import config


@pytest.fixture(autouse=True)
def continent_map(tmp_path, monkeypatch):
    features = []
    for continent, bounds in [("Europe", (-10, 35, 40, 70)), ("Africa", (-20, -35, 40, 30))]:
        west, south, east, north = bounds
        features.append({
            "type": "Feature", "properties": {"CONTINENT": continent},
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, south], [east, south], [east, north], [west, north], [west, south],
            ]]},
        })
    path = tmp_path / "continents.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    monkeypatch.setattr(config, "WORLD_COUNTRIES_GEOJSON", path)


def _write_reference_fixture(tmp_path):
    rows = [
        ("HG00001", "YRI", "1000 Genomes", "AFR", 7.4, 3.9),
        ("HG00002", "GWD", "1000 Genomes", "AFR", 13.4, -15.5),
        ("HG00003", "TSI", "1000 Genomes", "EUR", 43.1, 12.0),
        ("HG00004", "GBR", "1000 Genomes", "EUR", 54.0, -2.0),
        ("HG00005", "MXL", "1000 Genomes", "AMR", 20.7, -103.3),
        ("HG00006", "CHB", "1000 Genomes", "EAS", 39.9, 116.4),
        ("DRU1", "Druze", "HGDP", "MID", 32.0, 35.0),
        ("PAP1", "Papuan", "HGDP", "OCE", -6.1, 145.4),
        # Explicitly excluded relocation locations despite AFR/EUR labels.
        ("NA00001", "CEU", "1000 Genomes", "EUR", 40.3, -111.1),
        ("HG00007", "ASW", "1000 Genomes", "AFR", 35.5, -97.5),
        ("HG00008", "PUR", "1000 Genomes", "AMR", 18.5, -66.1),
    ]
    training = pd.DataFrame(
        {
            "s": [row[0] for row in rows],
            "scores": [
                json.dumps([index + 0.1, index + 0.2, index + 0.3])
                for index in range(len(rows))
            ],
        }
    )
    metadata = pd.DataFrame(
        rows,
        columns=[
            "s",
            "hgdp_tgp_meta.Population",
            "hgdp_tgp_meta.Project",
            "hgdp_tgp_meta.Genetic.region",
            "hgdp_tgp_meta.Latitude",
            "hgdp_tgp_meta.Longitude",
        ],
    )
    training_path = tmp_path / "training.tsv"
    metadata_path = tmp_path / "metadata.tsv"
    training.to_csv(training_path, sep="\t", index=False)
    metadata.to_csv(metadata_path, sep="\t", index=False)
    return training_path, metadata_path


def test_reference_panel_is_only_1kg_afr_eur(tmp_path):
    training_path, metadata_path = _write_reference_fixture(tmp_path)
    reference = figure1_idw._reference_panel(str(training_path), str(metadata_path))

    assert set(reference["genetic_region"]) == {"AFR", "EUR"}
    assert set(reference["pop_code"]) == {"YRI", "GWD", "TSI", "GBR"}
    assert reference["source_project"].eq("1000 Genomes").all()
    assert {"MXL", "CHB", "Druze", "Papuan", "CEU", "ASW", "PUR"}.isdisjoint(
        set(reference["pop_code"])
    )
    audit = reference.attrs["source_audit"]
    assert audit["excluded_us_canada_or_caribbean_rows"] == 2
    assert audit["retained_region_counts"] == {"AFR": 2, "EUR": 2}




def test_europe_count_uses_continent_polygons():
    samples_frame = pd.DataFrame(
        {
            "sample_id": ["london", "paris", "lagos", "atlantic"],
            "pred_longitude": [-0.1276, 2.3522, 3.3792, -30.0],
            "pred_latitude": [51.5072, 48.8566, 6.5244, 30.0],
            "region": "AoU",
        }
    )

    summary = figure1_idw.european_continent_summary(
        samples_frame, write_cache=False
    )

    assert summary["n_projected"] == 4
    assert summary["n_europe"] == 2
    assert summary["n_not_europe"] == 2
    assert summary["percent_europe"] == pytest.approx(50.0)
    assert summary["continent_field"] == "CONTINENT"
    assert summary["continent_value"] == "Europe"
    assert summary["boundary_rule"].startswith("covers")


def test_panel_idw_uses_hollow_sources_and_top_layer_larger_aou_diamonds():
    samples_frame, sources = idw_frames()
    figure, axis = plt.subplots()

    figure1_plots.panel_idw(
        axis, {"idw_samples": samples_frame, "idw_sources": sources}
    )

    assert np.allclose(
        axis.collections[0].get_facecolors()[0], to_rgba("white")
    )
    collections = {
        collection.get_label(): collection for collection in axis.collections
    }
    aou = collections["AoU IDW predictions"]
    one_kg = collections["Source pops (1KG)"]
    assert len(one_kg.get_offsets()) == len(sources)
    assert one_kg.get_sizes().tolist() == pytest.approx([100.0])
    assert one_kg.get_facecolors().size == 0
    assert one_kg.get_edgecolors().tolist() == [[0.0, 0.0, 0.0, 1.0]]
    assert one_kg.get_linewidths().tolist() == pytest.approx([1.1])
    assert aou.get_sizes().tolist() == pytest.approx([22.0])
    assert aou.get_linewidths().tolist() == pytest.approx([0.65])
    assert aou.get_alpha() == pytest.approx(0.82)
    assert aou.get_zorder() > one_kg.get_zorder()
    assert {text.get_text() for text in axis.texts}.issuperset(
        set(sources["pop_code"])
    )
    assert not any(label.startswith("HGDP") for label in collections)
    plt.close(figure)


def test_figure1d_requires_verified_frames(monkeypatch):
    samples_frame, sources = idw_frames()
    data = {"idw_samples": samples_frame, "idw_sources": sources}
    monkeypatch.setattr(figure1_idw, "cache_status", lambda: (False, "checksum mismatch"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        figure1_plots._idw_render_provenance(data)
    monkeypatch.setattr(figure1_idw, "cache_status", lambda: (True, "ok"))
    monkeypatch.setattr(figure1_idw, "european_continent_summary", lambda *a, **k: {"n_europe": 1, "percent_europe": 50.0})
    provenance = figure1_plots._idw_render_provenance(data)
    assert provenance["source"] == "checksum- and sample-manifest-validated IDW cache"
    assert provenance["sample_count"] == len(samples_frame)


def test_build_idw_writes_validated_phase1_manifest(tmp_path, monkeypatch):
    training_path, metadata_path = _write_reference_fixture(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    training_path.replace(raw / "training_pca.tsv")
    metadata_path.replace(raw / "gnomad_meta.tsv")
    pd.DataFrame(
        {
            "research_id": ["a1", "a2"],
            "PC1": [1.1, 3.1],
            "PC2": [1.2, 3.2],
            "PC3": [1.3, 3.3],
        }
    ).to_csv(raw / "ancestry_preds.tsv", sep="\t", index=False)
    (raw / "eigenvalues.txt").write_text("3\n2\n1\n", encoding="utf-8")

    analysis = tmp_path / "analysis"
    monkeypatch.setattr(figure1_idw, "_RAW_DIR", raw)
    monkeypatch.setattr(figure1_idw, "IDW_SAMPLES_CSV", analysis / "samples.csv")
    monkeypatch.setattr(figure1_idw, "IDW_SOURCES_CSV", analysis / "sources.csv")
    monkeypatch.setattr(figure1_idw, "IDW_MANIFEST_JSON", analysis / "manifest.json")
    monkeypatch.setattr(
        figure1_idw,
        "IDW_GEOGRAPHY_SUMMARY_JSON",
        analysis / "geography_summary.json",
    )
    monkeypatch.setattr(config, "IDW_N_PCS", 3)
    monkeypatch.setattr(config, "IDW_K", 3)
    monkeypatch.setattr(
        figure1_idw.samples,
        "load_aou_sample_list",
        lambda download=True: {"a1", "a2"},
    )
    monkeypatch.setattr(figure1_idw, "on_workbench", lambda: True)

    samples_frame, source_frame = figure1_idw.build_idw()

    assert samples_frame["sample_id"].tolist() == ["a1", "a2"]
    assert set(source_frame["region"]) == {"AFR", "EUR"}
    assert source_frame["source_project"].eq("1000 Genomes").all()
    valid, reason = figure1_idw.cache_status()
    assert valid, reason
    manifest = json.loads(figure1_idw.IDW_MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["parameters"]["source_regions"] == ["AFR", "EUR"]
    assert manifest["parameters"]["reference_panel"] == "1000 Genomes only"
    assert manifest["sample_manifest"]["n_projected"] == 2
    assert manifest["geography_audit"]["n_projected"] == 2
    assert figure1_idw.IDW_GEOGRAPHY_SUMMARY_JSON.is_file()
    geography = json.loads(
        figure1_idw.IDW_GEOGRAPHY_SUMMARY_JSON.read_text(encoding="utf-8")
    )
    assert geography["prediction_output_sha256"] == figure1_idw._sha256(
        figure1_idw.IDW_SAMPLES_CSV
    )
    assert geography["world_polygons_sha256"] == figure1_idw._sha256(
        config.WORLD_COUNTRIES_GEOJSON
    )

    monkeypatch.setattr(
        figure1_idw.samples,
        "load_aou_sample_list",
        lambda download=False: None,
    )
    valid, reason = figure1_idw.cache_status()
    assert not valid
    assert "sample manifest is unavailable" in reason

    monkeypatch.setattr(
        figure1_idw.samples,
        "load_aou_sample_list",
        lambda download=False: {"a1", "a2"},
    )
    manifest["sample_manifest"]["n_projected"] = 3
    figure1_idw.IDW_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    valid, reason = figure1_idw.cache_status()
    assert not valid
    assert "row count does not match" in reason
