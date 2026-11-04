from panel_fixtures import finemap_frame
import pytest

pytestmark = pytest.mark.usefixtures("panel_gene_annotations")
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import figure4_finemap


def test_exactly_50bp_is_classified_as_an_sv() -> None:
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "susie_finemap_SNVs.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "".join(notebook["cells"][0]["source"])
    assert "if diff >= SV_LEN_DIFF:" in source
    assert "elif -diff >= SV_LEN_DIFF:" in source
    assert "if diff > SV_LEN_DIFF:" not in source
    assert "elif -diff > SV_LEN_DIFF:" not in source
    assert figure4_finemap.ANALYSIS_CONTRACT.endswith("carriers20_v4")


def test_real_finemap_svs_use_ins_and_del_triangles() -> None:
    locus = figure4_finemap.FINEMAP_LOCI["B"]
    frame = finemap_frame(locus).iloc[:3].copy()
    frame["is_sv"] = ["False", "True", "True"]
    frame["sv_type"] = ["", "INS", "DEL"]
    frame["neg_log_p"] = [2.0, 5.0, 4.0]
    figure = figure4_finemap.plot_finemap(
        frame,
        locus,
        show_title=False,
        show_r2_legend=False,
    )
    try:
        collections = {
            collection.get_gid(): collection
            for collection in figure.axes[0].collections
        }
        assert set(collections) == {
            "finemap-snv",
            "finemap-ins",
            "finemap-del",
        }
        assert len(collections["finemap-snv"].get_offsets()) == 1
        assert len(collections["finemap-ins"].get_offsets()) == 1
        assert len(collections["finemap-del"].get_offsets()) == 1
        assert (
            collections["finemap-ins"].get_sizes()[0]
            > collections["finemap-snv"].get_sizes()[0]
        )
        assert (
            collections["finemap-del"].get_sizes()[0]
            > collections["finemap-snv"].get_sizes()[0]
        )
    finally:
        plt.close(figure)
