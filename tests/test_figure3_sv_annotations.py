from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.text import Text

from scripts import figure3_data, figure3_plots


def test_manhattan_background_downsampling_is_deterministic_and_keeps_extremes() -> None:
    frame = pd.DataFrame(
        {
            "chrom_num": np.repeat([1, 2], 100),
            "position": np.tile(np.arange(100), 2),
            "max_emp_ihs": np.r_[np.linspace(-10, 10, 100), np.linspace(-8, 8, 100)],
            "cumulative_pos": np.arange(200),
            "candidate": False,
        }
    )
    thinned, stats = figure3_plots._downsample_manhattan_background(
        frame,
        max_points=20,
        seed=2,
        extremes_per_chromosome=2,
    )
    repeated, repeated_stats = figure3_plots._downsample_manhattan_background(
        frame,
        max_points=20,
        seed=2,
        extremes_per_chromosome=2,
    )

    assert len(thinned) == 20
    assert thinned.groupby("chrom_num").size().to_dict() == {1: 10, 2: 10}
    for chromosome in (1, 2):
        expected_extremes = set(
            frame.loc[frame["chrom_num"].eq(chromosome), "max_emp_ihs"]
            .abs()
            .nlargest(2)
            .index
        )
        assert expected_extremes <= set(thinned.index)
    pd.testing.assert_frame_equal(thinned, repeated)
    assert repeated_stats == stats == {
        "input_points": 200,
        "output_points": 20,
        "background_cap": 20,
        "seed": 2,
        "extremes_per_chromosome": 2,
    }


def test_figure3a_pdf_is_downsampled_native_vector(tmp_path) -> None:
    data = figure3_data.build(clean_run_data=True)
    figure3_plots.style.apply_style()
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    figure3_plots.panel_A(axis, data)
    assert len(axis.collections[0].get_offsets()) <= (
        figure3_plots.MAX_NONSIGNIFICANT_POINTS
    )
    assert len(axis.collections[1].get_offsets()) == int(
        data["scan"]["candidate"].sum()
    )
    output = tmp_path / "Figure_3A_iHS_manhattan.pdf"
    figure.savefig(output)
    plt.close(figure)

    with pymupdf.open(output) as document:
        assert sum(len(page.get_images(full=True)) for page in document) == 0
        font_types = {
            font[2]
            for page in document
            for font in page.get_fonts(full=True)
        }
    assert "Type3" not in font_types


def test_s3_sv_hit_contract_matches_plotted_scan(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(figure3_data, "_adir", lambda: tmp_path)
    monkeypatch.setattr(figure3_data, "_NEAR_GENE_CONSOLE_EMITTED", False)
    scan = figure3_data._build_scan()
    hits = figure3_data._build_sv_hit_annotations(scan)

    assert len(hits) == figure3_data.SV_HIT_EXPECTED_ROWS == 17
    assert hits["sv_allele_key"].nunique() == 14
    assert hits["label_sv"].sum() == 14
    intergenic_labels = hits.loc[
        hits["sv_gene_context"].str.startswith("intergenic"),
        "display_gene_label",
    ]
    assert intergenic_labels.str.startswith("near").all()
    assert hits["sv_relation"].value_counts().to_dict() == {
        "linked": 14,
        "direct": 3,
    }
    near_distances = dict(
        zip(
            hits.loc[
                hits["sv_gene_context"].str.startswith("intergenic"),
                "sv_gene_label",
            ],
            hits.loc[
                hits["sv_gene_context"].str.startswith("intergenic"),
                "sv_near_gene_distances_bp",
            ],
        )
    )
    assert near_distances == {
        "PARK7/ERRFI1": "PARK7=5037;ERRFI1=21185",
        "CWF19L2": "CWF19L2=38462",
        "TCP10L2": "TCP10L2=6118",
        "MUC3A": "MUC3A=683",
    }
    console = capsys.readouterr().out
    assert "gene=PARK7 distance_bp=5037" in console
    assert "gene=MUC3A distance_bp=683" in console
    log = (tmp_path / "figure3_sv_near_gene_distances.log").read_text(
        encoding="utf-8"
    )
    assert "row_unit=unique intergenic SV x named nearby gene" in log
    assert "gene=CWF19L2 distance_bp=38462" in log

    plekhb2 = hits[hits["sv_gene_label"].eq("PLEKHB2")]
    assert len(plekhb2) == 3
    assert plekhb2.loc[plekhb2["label_sv"], "selection_peak_position"].tolist() == [
        131094186
    ]

    ankrd31 = hits[hits["sv_gene_label"].eq("ANKRD31")]
    assert len(ankrd31) == 2
    assert ankrd31.loc[ankrd31["label_sv"], "selection_peak_position"].tolist() == [
        75131912
    ]






def test_panel_a_uses_uniform_solid_rings_and_no_legend() -> None:
    data = figure3_data.build(clean_run_data=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    try:
        figure3_plots.panel_A(ax, data)

        known_snv_ring = ax.collections[2]
        genic_sv_ring = ax.collections[3]
        assert len(known_snv_ring.get_offsets()) == 2
        assert len(genic_sv_ring.get_offsets()) == 10
        for ring in (known_snv_ring, genic_sv_ring):
            assert ring.get_facecolors().size == 0
            assert ring.get_edgecolors().shape[0] == 1
            assert ring.get_linestyles()[0][1] is None
            assert ring.get_linewidths()[0] == 1.45
        assert ax.get_legend() is None

        expected_labels = {
            "LCT",
            "SLC24A5",
            *data["sv_hits"].loc[
                data["sv_hits"]["label_sv"]
                & ~data["sv_hits"]["sv_gene_context"].str.startswith(
                    "intergenic", na=False
                ),
                "display_gene_label",
            ],
        }
        assert expected_labels <= {text.get_text() for text in ax.texts}
    finally:
        plt.close(fig)
