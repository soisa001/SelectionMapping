from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import figure2_discovery_phewas as plots


def _write_inputs(root: Path) -> tuple[Path, Path]:
    discovery = root / "phewas.discovery.tsv.gz"
    pd.DataFrame(
        [
            {"region_id": "chr1:100-200:high", "phecode": "P1", "p": 1e-5, "beta": 0.4},
            {"region_id": "chr1:100-200:high", "phecode": "P2", "p": 2e-4, "beta": -0.6},
            {"region_id": "chr1:100-200:high", "phecode": "P3", "p": 0.2, "beta": 0.1},
            {"region_id": "chr2:300-400:low", "phecode": "P1", "p": 0.02, "beta": 0.2},
            {"region_id": "chr2:300-400:low", "phecode": "P2", "p": 0.4, "beta": -0.1},
            # The discovery rule is strict, so an exact threshold is not a hit.
            {"region_id": "chr3:500-600:high", "phecode": "P1", "p": 1e-3, "beta": 0.3},
        ]
    ).to_csv(discovery, sep="\t", index=False, compression="gzip")
    annotations = root / "phecodes.tsv"
    pd.DataFrame(
        [
            {"phecode": "P1", "phecode_string": "Phenotype one", "phecode_category": "Circulatory"},
            {"phecode": "P2", "phecode_string": "Phenotype two", "phecode_category": "Mental"},
            {"phecode": "P3", "phecode_string": "Phenotype three", "phecode_category": "Circulatory"},
        ]
    ).to_csv(annotations, sep="\t", index=False)
    return discovery, annotations


def test_prepare_plot_data_keeps_all_tests_only_for_regions_with_strict_hits(
    tmp_path: Path,
) -> None:
    discovery_path, annotation_path = _write_inputs(tmp_path)
    discovery = plots._read_table(
        discovery_path, required={"region_id", "phecode", "p", "beta"}
    )
    annotations, _ = plots.load_annotations([annotation_path])
    frame, categories = plots.prepare_plot_data(discovery, annotations)
    assert frame["region_id"].unique().tolist() == ["chr1:100-200:high"]
    assert len(frame) == 3
    assert frame["discovery_significant"].sum() == 2
    assert categories == ["Circulatory", "Mental"]


def test_render_all_writes_one_vector_panel_per_significant_region_and_reuses_cache(
    tmp_path: Path,
) -> None:
    discovery_path, annotation_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    first = plots.render_all(
        discovery_path, [annotation_path], output_dir, dpi=120
    )
    assert first["status"] == "rendered"
    assert first["regions"] == ["chr1:100-200:high"]
    assert len(first["plots"]) == 2
    pdf = next(path for path in first["plots"] if path.suffix == ".pdf")
    assert pdf.read_bytes().startswith(b"%PDF-")
    # The standalone panel is drawn as vector artists; it embeds no raster image.
    assert b"/Subtype /Image" not in pdf.read_bytes()
    region_manifest = pd.read_csv(first["region_manifest"], sep="\t")
    assert region_manifest.loc[0, "n_discovery_significant"] == 2
    assert region_manifest.loc[0, "significant_phecodes"] == "P1;P2"
    payload = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert payload["summary"]["regions"] == 1
    assert payload["contract"]["parameters"]["rasterized"] is False

    second = plots.render_all(
        discovery_path, [annotation_path], output_dir, dpi=120
    )
    assert second["status"] == "cached"


def test_missing_category_fails_before_plotting(tmp_path: Path) -> None:
    discovery_path, annotation_path = _write_inputs(tmp_path)
    annotations = pd.read_csv(annotation_path, sep="\t")
    annotations.loc[annotations["phecode"].eq("P3"), "phecode_category"] = ""
    annotations.to_csv(annotation_path, sep="\t", index=False)
    discovery = plots._read_table(
        discovery_path, required={"region_id", "phecode", "p", "beta"}
    )
    loaded, _ = plots.load_annotations([annotation_path])
    with pytest.raises(ValueError, match="Missing Phecode categories"):
        plots.prepare_plot_data(discovery, loaded)


def test_cli_recovers_outer_analysis_cache_from_accidental_nested_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer_repo = tmp_path / "AllofUsPhase1Companion"
    nested_repo = outer_repo / "AllofUsPhase1Companion"
    nested_repo.mkdir(parents=True)
    discovery = (
        outer_repo / "analysis" / "figure2" / "phewas"
        / "phewas.discovery.tsv.gz"
    )
    discovery.parent.mkdir(parents=True)
    discovery.write_bytes(b"completed discovery cache")
    monkeypatch.chdir(nested_repo)

    resolved = plots.resolve_discovery_path(
        Path("analysis/figure2/phewas/phewas.discovery.tsv.gz")
    )
    assert resolved == discovery.resolve()
    assert plots.resolve_output_dir(plots.DEFAULT_OUTPUT_DIR, resolved) == (
        outer_repo / "figures" / "figure2"
        / "discovery_phewas"
    ).resolve()
