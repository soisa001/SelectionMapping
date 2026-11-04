from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

from scripts.common import config, panel_io


def test_individual_panels_are_paired_vector_outputs_and_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "figure_dir", lambda _: tmp_path)
    drawn = []

    def draw(axis, data):
        drawn.append(axis)
        axis.plot([0, 1], [0, 1])
        axis.set_xlabel("Reference axis")

    paths = panel_io.render_panels("figure1", {"A": draw, "H": draw}, {}, clean_run_plots=True)
    assert {p.name for p in paths} == {"Figure_1A.png", "Figure_1A.pdf", "Figure_1H.png", "Figure_1H.pdf"}
    assert not (tmp_path / "Figure_1.pdf").exists()
    assert len(drawn) == 2
    assert not plt.get_fignums()
    for path in paths:
        assert path.is_file()
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        else:
            with pymupdf.open(path) as doc:
                assert doc.page_count == 1
                assert not doc[0].get_images()
                assert "Reference axis" in doc[0].get_text()
    previous = {path: path.stat().st_mtime_ns for path in paths}
    assert panel_io.render_panels("figure1", {"A": draw, "H": draw}, {}, clean_run_plots=False) == paths
    assert len(drawn) == 2
    assert previous == {path: path.stat().st_mtime_ns for path in paths}


def test_corrected_admixture_panel_keeps_the_training_cache_context(monkeypatch):
    from contextlib import contextmanager
    from scripts import figure2_1025_plots as plots

    active = []
    ids = ("sample_a", "sample_b")

    @contextmanager
    def context(selected):
        assert selected == ids
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def render(*args, **kwargs):
        assert active == [True]
        return [Path("Figure_2F.pdf")]

    monkeypatch.setattr(plots.figure2_1025, "analysis_context", context)
    monkeypatch.setattr(plots.panel_io, "render_panel", render)
    assert plots.render_panel("F", {"_figure2_training_ids": ids}) == [Path("Figure_2F.pdf")]
    assert active == []
