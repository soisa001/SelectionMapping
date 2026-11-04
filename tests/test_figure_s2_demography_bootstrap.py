from __future__ import annotations

from scripts import figure1_demography, figure1_plots


def test_s2_uses_all_committed_bootstrap_fit_vectors() -> None:
    bootstrap = figure1_demography.load_bootstrap_fits()
    assert bootstrap is not None
    assert len(bootstrap) == 100
    assert bootstrap[
        [column for column, _ in figure1_plots._CORRELOGRAM_PARAMS]
    ].notna().all().all()


def test_s2_paths_are_publication_named_and_paired() -> None:
    png = figure1_plots.s2_image_paths("png")[0]
    pdf = figure1_plots.s2_image_paths("pdf")[0]
    assert png.with_suffix("") == pdf.with_suffix("")
    assert png.name == "Figure_S2_Bootstrap_Correlogram.png"
    assert pdf.name == "Figure_S2_Bootstrap_Correlogram.pdf"
