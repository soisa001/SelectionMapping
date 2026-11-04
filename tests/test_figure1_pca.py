from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import figure1_pca as s10


def _frame(pc1: list[float], pc2: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [str(index) for index in range(len(pc1))],
            "PC1": pc1,
            "PC2": pc2,
            "PC3": [0.0] * len(pc1),
        }
    )


def test_cohort_centroid_is_unweighted_coordinate_mean() -> None:
    stats = s10._cohort_stats(_frame([1.0, 3.0, 8.0], [-2.0, 4.0, 7.0]), "PC1", "PC2")

    assert stats["n"] == 3
    assert stats["x_centroid"] == pytest.approx(4.0)
    assert stats["y_centroid"] == pytest.approx(3.0)


def test_render_figure_s10_writes_paired_outputs_and_panel_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = _frame([-0.2, 0.0, 0.2], [-0.1, 0.0, 0.1])
    replication = _frame([-0.15, -0.05, 0.1], [-0.3, -0.1, 0.0])
    real_close = s10.plt.close
    monkeypatch.setattr(s10.plt, "close", lambda figure: None)

    paths, summary_path, stats = s10.render_figure_s10(
        discovery,
        replication,
        output_stem=tmp_path / "Figure_S10_PCA_10k_vs_30k",
        summary_path=tmp_path / "centroids.tsv",
    )
    figure = s10.plt.gcf()
    try:
        panel_labels = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert panel_labels == ["A", "B"]
        assert [path.suffix for path in paths] == [".png", ".pdf"]
        assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert paths[1].read_bytes().startswith(b"%PDF-")
        assert summary_path.is_file()
        summary = pd.read_csv(summary_path, sep="\t")
        assert summary["panel"].tolist() == ["A", "B"]
        assert summary["cohort"].tolist() == ["discovery_10k", "replication_30k"]
        assert summary["centroid_definition"].nunique() == 1
        assert stats["discovery_10k"]["x_centroid"] == pytest.approx(0.0)
        assert stats["replication_30k"]["y_centroid"] == pytest.approx(-0.1333333333)
    finally:
        real_close(figure)


def test_manifest_cache_rejects_output_corruption(tmp_path: Path) -> None:
    paths, summary_path, _ = s10.render_figure_s10(
        _frame([-0.1, 0.1], [-0.2, 0.0]),
        _frame([-0.2, 0.2], [-0.3, 0.0]),
        output_stem=tmp_path / "Figure_S10",
        summary_path=tmp_path / "centroids.tsv",
    )
    outputs = [*paths, summary_path]
    cache_key = {"contract": s10.S10_CONTRACT}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "outputs": {
                    str(path.resolve()): s10._sha256(path) for path in outputs
                },
            }
        ),
        encoding="utf-8",
    )

    assert s10._manifest_reusable(
        manifest_path,
        cache_key=cache_key,
        expected_outputs=outputs,
    )
    with paths[0].open("ab") as handle:
        handle.write(b"corruption")
    assert not s10._manifest_reusable(
        manifest_path,
        cache_key=cache_key,
        expected_outputs=outputs,
    )
