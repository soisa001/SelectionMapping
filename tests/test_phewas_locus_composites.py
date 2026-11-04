import json
import os
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader

from scripts import figure3_ehh, figure3_locus_combined, figure4_finemap
from scripts import figure5_locus_composites as composites


def _write_meta(path: Path) -> None:
    path.write_text(
        "created on Workbench\n"
        "variant_tag\tphecode\tphecode_string\tphecode_category\t"
        "meta_significant\tp_meta\tbeta_meta\tgene_in_window\n"
        "1_100_A_G\tP1\tPhenotype one\tCirculatory\tTRUE\t1e-5\t0.4\tGENE1\n"
        "1_100_A_G\tP2\tPhenotype two\tRespiratory\tTRUE\t2e-5\t-0.3\tGENE1\n"
        "2_200_C_T\tP3\tPhenotype three\tMental\tTRUE\t3e-5\t0.2\tGENE2\n"
        "2_200_C_T\tP4\tNot retained\tMental\tFALSE\t0.2\t0.1\tGENE2\n",
        encoding="utf-8",
    )


def _write_phewas(path: Path) -> None:
    rows = []
    for variant, target_codes in (("1_100_A_G", ("P1", "P2")), ("2_200_C_T", ("P3",))):
        for index in range(1, 13):
            phecode = f"P{index}"
            beta = (-1 if index % 3 == 0 else 1) * (0.05 + index / 20)
            rows.append(
                {
                    "variant_tag": variant,
                    "phecode": phecode,
                    "phecode_string": f"Phenotype {index}",
                    "phecode_category": ("Circulatory" if index <= 6 else "Respiratory"),
                    "p_value": (1e-5 * index if phecode in target_codes else 0.02 + index / 100),
                    "beta": beta,
                    "odds_ratio": np.exp(beta),
                }
            )
    rows.append(
        {
            "variant_tag": "3_300_G_A", "phecode": "PX", "phecode_string": "Unrelated",
            "phecode_category": "Other", "p_value": 0.5, "beta": 0.1,
            "odds_ratio": np.exp(0.1),
        }
    )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)




def _write_image(path: Path, color: str) -> None:
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 1160, 660), outline=color, width=12)
    draw.line((80, 600, 1120, 120), fill=color, width=10)
    image.save(path)
    figure, axis = composites.plt.subplots(figsize=(12, 7))
    axis.plot([0.05, 0.95], [0.1, 0.9], color=color, linewidth=4)
    axis.set_axis_off()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches=figure.bbox_inches, pad_inches=0)
    composites.plt.close(figure)


def _write_titled_finemap_image(path: Path) -> None:
    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 18, 1050, 40), fill="black")
    draw.rectangle((240, 53, 960, 75), fill="black")
    draw.rectangle((340, 88, 860, 110), fill="black")
    draw.rectangle((320, 130, 880, 150), fill="black")
    draw.rectangle((20, 154, 60, 178), fill="black")
    draw.rectangle((1140, 154, 1180, 178), fill="black")
    draw.line((600, 111, 600, 179), fill="black", width=1)
    draw.rectangle((70, 180, 1130, 850), outline="#117864", width=12)
    draw.line((110, 800, 1090, 240), fill="#117864", width=10)
    image.save(path)
    figure, axis = composites.plt.subplots(figsize=(12, 9))
    axis.plot([0.05, 0.95], [0.1, 0.9], color="#117864", linewidth=4)
    axis.set_axis_off()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches=figure.bbox_inches, pad_inches=0)
    composites.plt.close(figure)


def test_meta_hit_contract_preserves_input_order(tmp_path: Path) -> None:
    path = tmp_path / "meta.tsv"
    _write_meta(path)
    hits = composites.load_meta_hits(path, expected_hits=3, expected_loci=2)
    assert hits["figure_index"].tolist() == [1, 2, 3]
    assert hits["variant_index"].tolist() == [1, 1, 2]
    assert hits[["variant_tag", "phecode"]].values.tolist() == [
        ["1_100_A_G", "P1"], ["1_100_A_G", "P2"], ["2_200_C_T", "P3"]
    ]
    assert hits["gene"].tolist() == ["GENE1", "GENE1", "GENE2"]


def test_marker_areas_are_symmetric_log_scaled_and_capped() -> None:
    values = composites.odds_ratio_marker_areas(
        [1.0, 2.0, 0.5, 4.0, 0.25, 8.0, 0.125, 80.0, 0.001],
        cap=8.0,
        minimum=36.0,
        maximum=220.0,
    )
    assert values[0] == 36.0
    assert values[1] == values[2]
    assert values[3] == values[4]
    assert values[5] == values[6] == values[7] == values[8] == 220.0
    assert values[0] < values[1] < values[3] < values[5]


def test_shared_scales_use_global_or_fold_and_pvalue_maxima() -> None:
    frame = pd.DataFrame(
        {"odds_ratio": [0.1, 1.5, 3.0], "p_value": [1e-4, 1e-9, 0.2]}
    )
    scales = composites.resolve_shared_phewas_scales(frame)
    assert scales["observed_or_fold_max"] == 10.0
    assert scales["odds_ratio_cap"] == 10.0
    assert scales["observed_neglogp_max"] == 9.0
    assert scales["phewas_y_max"] == 10.5


def test_background_tests_descend_by_plotted_significance_within_category() -> None:
    frame = pd.DataFrame(
        {
            "phecode": ["meta", "small", "large_up", "large_down"],
            "phecode_category": ["Circulatory"] * 4,
            "p_value": [1e-5, 0.1, 0.3, 0.2],
            "odds_ratio": [1.2, 1.1, 4.0, 0.25],
            "_meta": [True, False, False, False],
        }
    )
    ordered = composites._category_layout(frame, ["Circulatory"])
    background = ordered.loc[~ordered["_meta"]]
    assert ordered.iloc[0]["phecode"] == "meta"
    assert background["phecode"].tolist() == ["small", "large_down", "large_up"]
    assert np.all(np.diff(background["p_value"].to_numpy()) >= 0)


def test_import_sanitizes_workbench_inline_backend() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import scripts.figure5_locus_composites; "
                "import matplotlib; "
                "print(os.environ['MPLBACKEND']); "
                "print(matplotlib.get_backend())"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [line.casefold() for line in result.stdout.splitlines()] == ["agg", "agg"]


def test_streamed_phewas_loader_retains_only_target_loci(tmp_path: Path) -> None:
    path = tmp_path / "phewas.tsv"
    _write_phewas(path)
    frame, details = composites.load_phewas_for_variants(
        path, {"1_100_A_G", "2_200_C_T"}, chunk_rows=7
    )
    assert len(frame) == 24
    assert set(frame["variant_tag"]) == {"1_100_A_G", "2_200_C_T"}
    assert details["rows_scanned"] == 25
    assert details["rows_retained"] == 24
    assert len(details["relevant_subset_sha256"]) == 64


def test_expected_finemap_name_matches_current_notebook_contract() -> None:
    hit = pd.Series({"variant_tag": "1_100_A_G", "phecode": "P1"})
    assert composites.expected_finemap_filename(hit) == "pval_disc_1_100_A_G__P1.png"


def test_standard_finemap_is_rerendered_on_the_native_shared_gene_canvas(
    tmp_path: Path,
    caplog,
) -> None:
    hit = pd.Series(
        {
            "figure_index": 1,
            "variant_index": 1,
            "variant_tag": "4_99103864_C_T",
            "phecode": "MB_280.1",
            "phecode_name": "Alcohol use disorders",
            "gene": "LOC100507053",
            "p_meta": 1e-5,
            "beta_meta": 0.4,
        }
    )
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    result = result_dir / composites.expected_finemap_result_filename(hit)
    pd.DataFrame(
        {
            "variant_id": ["4_99090000_A_G", "4_99103864_C_T", "4_99120000_G_A"],
            "chrom": ["chr4"] * 3,
            "pos": [99_090_000, 99_103_864, 99_120_000],
            "p_score": [0.02, 1e-6, 0.01],
            "pip": [0.0, 0.8, 0.2],
            "cs": ["", "L1", "L1"],
            "r2_to_minp": [0.2, 1.0, 0.5],
            "is_sv": [False] * 3,
            "sv_type": [""] * 3,
            "_n_pcs": [figure4_finemap.FINEMAP_N_PCS] * 3,
            "_analysis_contract": [figure4_finemap.ANALYSIS_CONTRACT] * 3,
            "_sex_scope": ["Both"] * 3,
            "_window_start": [SNV_START := 99_103_864 - 500_000] * 3,
            "_window_end": [SNV_END := 99_103_864 + 500_000] * 3,
            "_min_carriers": [figure4_finemap.FINEMAP_MIN_CARRIERS] * 3,
        }
    ).to_csv(result, sep="\t", index=False)
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "chr4\trefGene\ttranscript\t99000000\t99100000\t.\t+\t.\t"
        'gene_id "LEFT"; transcript_id "LEFT.1"; gene_name "LEFT";\n'
        "chr4\trefGene\texon\t99000000\t99010000\t.\t+\t.\t"
        'gene_id "LEFT"; transcript_id "LEFT.1"; gene_name "LEFT";\n'
        "chr4\trefGene\ttranscript\t99102000\t99130000\t.\t-\t.\t"
        'gene_id "LOC100507053"; transcript_id "LOC.1"; '
        'gene_name "LOC100507053";\n'
        "chr4\trefGene\texon\t99102000\t99112000\t.\t-\t.\t"
        'gene_id "LOC100507053"; transcript_id "LOC.1"; '
        'gene_name "LOC100507053";\n',
        encoding="utf-8",
    )

    with caplog.at_level("INFO", logger="supp_locus_composites"):
        panels, provenance = composites.render_standard_finemap_panels(
            hit.to_frame().T,
            tmp_path / "rendered",
            [],
            local_dirs=[result_dir],
            gene_annotation=gtf,
            dpi=80,
            force=True,
        )

    panel = panels[1]
    pdf_text = PdfReader(panel.with_suffix(".pdf")).pages[0].extract_text()
    assert "−log₁₀(p)" in pdf_text
    assert "r² to min-p variant" in pdf_text
    with Image.open(panel) as image:
        assert image.width / image.height == pytest.approx(
            figure4_finemap.SUPPLEMENTARY_PANEL_ASPECT,
            rel=0.01,
        )
    contract = provenance["1"]["source_contract"]
    assert contract["window_start"] == SNV_START
    assert contract["window_end"] == SNV_END
    assert contract["gene_track_renderer"] == "figure3_ehh.draw_gene_track"
    assert contract["gene_names"] == ["LEFT", "LOC100507053"]
    summary = contract["finemap_result_summary"]
    assert summary["top_pip_variant"]["pip"] == 0.8
    assert summary["top_pip_variant"]["p_value"] == 1e-6
    assert summary["minimum_p_variant"] == summary["top_pip_variant"]
    assert summary["causal_sets"][0]["causal_set"] == "L1"
    assert {
        json.loads(record.message.split(" | ", 1)[1])["record_type"]
        for record in caplog.records
        if record.message.startswith("PHEWAS_FINEMAP | ")
    } == {
        "summary",
        "top_pip_variant",
        "minimum_p_variant",
        "causal_set_member",
    }


def test_all_phase_recomputes_invalid_standard_finemaps_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = pd.DataFrame(
        {
            "variant_tag": ["4_99103864_C_T"],
            "phecode": ["MB_280.1"],
            "phecode_name": ["Alcohol use disorders"],
            "gene": ["LOC100507053"],
            "p_meta": [1e-5],
        }
    )
    rendered_panel = tmp_path / "panel.png"
    attempts: list[bool] = []

    def fake_render(*args, **kwargs):
        attempts.append(bool(kwargs["force"]))
        if len(attempts) == 1:
            raise ValueError("missing ['_min_carriers']")
        return {1: rendered_panel}, {"1": {"cache": "rendered"}}

    recompute_calls: list[Path] = []
    monkeypatch.setattr(composites, "render_standard_finemap_panels", fake_render)
    monkeypatch.setattr(
        "scripts.recompute_figure4_susie.ensure_figure4_outputs",
        lambda *, task_table: recompute_calls.append(task_table),
    )

    result = composites.render_standard_finemap_panels_with_recompute(
        hits,
        tmp_path / "finemap",
        ["gs://bucket/susie"],
        local_dirs=[],
        gene_annotation=tmp_path / "genes.gtf",
        task_path=tmp_path / "standard_susie_tasks.tsv",
        phase="all",
        dpi=80,
        force=False,
    )

    assert result[0] == {1: rendered_panel}
    assert attempts == [False, False]
    assert recompute_calls == [tmp_path / "standard_susie_tasks.tsv"]
    tasks = pd.read_csv(recompute_calls[0], sep="\t")
    assert tasks.loc[0, "meta_significant"]
    assert tasks.loc[0, "variant_tag"] == "4_99103864_C_T"
    assert tasks.loc[0, "phecode"] == "MB_280.1"


def test_phecode_string_slug_drops_bracketed_text_and_preserves_slash_terms() -> None:
    assert composites._phecode_string_slug(
        "Alcohol use disorder [legacy label] / secondary wording"
    ) == "alcohol_use_disorder_secondary_wording"
    assert composites._phecode_string_slug("[] / suffix", fallback="P1") == "suffix"


def test_output_basename_groups_hits_by_unique_variant(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.tsv"
    _write_meta(meta_path)
    hits = composites.load_meta_hits(meta_path, expected_hits=3, expected_loci=2)
    stems = [composites._phewas_output_stem_name(hit) for _, hit in hits.iterrows()]
    assert stems[0] == "Supplementary_Phewas_01_1_100_A_G__phenotype_one"
    assert stems[1] == "Supplementary_Phewas_01_1_100_A_G__phenotype_two"
    assert stems[2] == "Supplementary_Phewas_02_2_200_C_T__phenotype_three"
    assert stems[0].split("__", 1)[0] == stems[1].split("__", 1)[0]


def test_output_basename_distinguishes_slash_delimited_phecode_names() -> None:
    common = {
        "variant_index": 5,
        "variant_tag": "7_90206732_G_A",
    }
    cycle = pd.Series(
        {
            **common,
            "phecode": "GU_626.13",
            "phecode_name": "Irregular menstrual cycle",
        }
    )
    bleeding = pd.Series(
        {
            **common,
            "phecode": "GU_626.1",
            "phecode_name": "Irregular menstrual cycle/bleeding",
        }
    )

    stems = {
        composites._phewas_output_stem_name(cycle),
        composites._phewas_output_stem_name(bleeding),
    }

    assert stems == {
        "Supplementary_Phewas_05_7_90206732_G_A__irregular_menstrual_cycle",
        "Supplementary_Phewas_05_7_90206732_G_A__irregular_menstrual_cycle_bleeding",
    }


def test_default_meta_matches_current_finemap_notebook_contract() -> None:
    assert composites.DEFAULT_META_OBJECT == "phewas_meta/meta_analysis_results_genic_meff.tsv"


def test_cli_defaults_to_fourfold_triangles_and_matching_500kb_windows() -> None:
    args = composites.build_parser().parse_args([])
    assert args.odds_ratio_cap == 4.0
    assert args.phewas_y_max is None
    assert args.selection_half_window_bp == 500_000
    assert composites.FINEMAP_HALF_WINDOW_BP == 500_000
    assert args.output_dir == composites.DEFAULT_OUTPUT_DIR
    assert args.output_dir.as_posix().endswith("figures/figure5/loci")
    assert composites.SELECTION_CACHE_CONTRACT.endswith("_v14")
    assert composites.ANALYSIS_CONTRACT.endswith("_v23")


def test_selection_gene_track_matches_rendered_finemap_gene_boundary() -> None:
    selection_ratios = figure3_locus_combined.COMBINED_HEIGHT_RATIOS
    expected_finemap_ratios = (sum(selection_ratios[:-1]), selection_ratios[-1])
    np.testing.assert_allclose(
        figure4_finemap.SUPPLEMENTARY_FINEMAP_HEIGHT_RATIOS,
        expected_finemap_ratios,
    )

    notebook = json.loads(
        (composites.REPO_ROOT / "notebooks" / "susie_finemap_SNVs.ipynb").read_text(
            encoding="utf-8"
        )
    )
    primary_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                               if cell["cell_type"] == "code")
    assert "SUPPLEMENTARY_GENE_FONTSIZE = 26" in primary_source
    assert "fontsize=SUPPLEMENTARY_GENE_FONTSIZE" in primary_source
    assert "height_ratios=[7.5, 4.9]" in primary_source
    assert "1.0 + COLORBAR_VERTICAL_HEADROOM" in primary_source
    assert "genes = genes[~genes[\"gene_name\"].str.match(_SMALL_RNA_RE)]" in primary_source


def test_runner_rejects_mismatched_selection_and_finemap_windows() -> None:
    args = composites.build_parser().parse_args(
        ["--selection-half-window-bp", "400000"]
    )
    with pytest.raises(ValueError, match="panels B and C share endpoints"):
        composites.run(args)


def test_meta_labels_are_repelled_on_compact_phewas_panel(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.tsv"
    phewas_path = tmp_path / "phewas.tsv"
    _write_meta(meta_path)
    _write_phewas(phewas_path)
    hits = composites.load_meta_hits(meta_path, expected_hits=3, expected_loci=2)
    phewas, _ = composites.load_phewas_for_variants(
        phewas_path, set(hits["variant_tag"]), chunk_rows=8
    )
    locus = phewas[phewas["variant_tag"] == "1_100_A_G"].copy()
    locus.loc[locus["phecode"] == "P1", "phecode_name"] = "Alcohol use disorders"
    locus.loc[locus["phecode"] == "P2", "phecode_name"] = "Alcohol abuse and dependence"
    locus.loc[locus["phecode"] == "P3", ["phecode_name", "p_value"]] = [
        "Nominal nearby phecode",
        5e-4,
    ]
    scales = composites.resolve_shared_phewas_scales(phewas)
    parameters = composites.PlotParameters(
        odds_ratio_cap=composites.DEFAULT_OR_CAP,
        phewas_y_max=scales["phewas_y_max"],
        dpi=80,
    )
    figure, axis = composites.plt.subplots(figsize=(5.2, 1.8), dpi=100)
    composites.draw_phewas_panel(
        axis,
        locus,
        hits[hits["variant_tag"] == "1_100_A_G"],
        hits.iloc[0],
        list(dict.fromkeys(phewas["phecode_category"])),
        parameters,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = [text for text in axis.texts if text.get_text().startswith("Alcohol")]
    assert len(labels) == 2
    assert {text.get_color() for text in labels} == {"black"}
    assert len({text.get_fontsize() for text in labels}) == 1
    assert {text.get_fontweight() for text in labels} == {"bold"}
    assert not labels[0].get_window_extent(renderer).overlaps(
        labels[1].get_window_extent(renderer)
    )
    assert abs(labels[0].get_position()[0] - labels[1].get_position()[0]) > 0.5
    nominal_label = next(
        text for text in axis.texts if text.get_text() == "Nominal nearby phecode"
    )
    assert nominal_label.get_fontweight() == "bold"
    shape_legend = axis.artists[0]
    assert [text.get_text() for text in shape_legend.get_texts()] == [
        "OR > 1",
        "OR < 1",
        "Meta-significant",
        "Nominal significance",
    ]
    size_legend = axis.get_legend()
    assert [text.get_text() for text in size_legend.get_texts()] == ["1×", "2×", "3×", "4×"]
    assert {line.get_marker() for line in size_legend.get_lines()} == {"^"}
    composites.plt.close(figure)


def test_category_labels_use_requested_rotation_and_palette() -> None:
    categories = [
        "Blood/Immune",
        "Neoplasms",
        "Cardiovascular",
        "Dermatological",
        "Endocrine/Metab",
        "Congenital anomalies",
        "Gastrointestinal",
        "Genitourinary",
        "Infections",
        "Mental",
        "Musculoskeletal",
        "Neurological",
        "Pregnancy complications",
        "Respiratory",
        "Sense organs",
        "Symptoms",
    ]
    frame = pd.DataFrame(
        {
            "phecode": [f"P{index}" for index in range(len(categories))],
            "phecode_name": [f"Phenotype {index}" for index in range(len(categories))],
            "phecode_category": categories,
            "p_value": [0.1] * len(categories),
            "beta": [0.1] * len(categories),
            "odds_ratio": [np.exp(0.1)] * len(categories),
        }
    )
    figure, axis = composites.plt.subplots(figsize=(5.85, 2.6), dpi=100)
    composites.draw_phewas_panel(
        axis,
        frame,
        pd.DataFrame(columns=["phecode"]),
        pd.Series({"phecode": "not-present"}),
        categories,
        composites.PlotParameters(phewas_y_max=3.5),
    )
    figure.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.42)
    figure.canvas.draw()
    assert {label.get_rotation() for label in axis.get_xticklabels()} == {40.0}
    assert composites.PHEWAS_CATEGORY_BACKGROUND_COLORS == ("#6baed6", "#bdbdbd")
    expected_rgb = np.asarray(
        [
            composites.matplotlib.colors.to_rgb(
                composites.PHEWAS_CATEGORY_BACKGROUND_COLORS[index % 2]
            )
            for index in range(len(categories))
        ]
    )
    np.testing.assert_allclose(axis.collections[0].get_facecolors()[:, :3], expected_rgb)
    assert composites._short_category("Neurological") == "Neuro"
    composites.plt.close(figure)


def test_image_panel_can_omit_subtitle_and_strip_embedded_title(tmp_path: Path) -> None:
    source = tmp_path / "titled.png"
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    # Figure 4's long "10k discovery" subtitle can begin left of the plot
    # body after whitespace cropping; the full centered title band must go.
    draw.rectangle((55, 15, 500, 42), fill="black")
    draw.rectangle((40, 88, 130, 100), fill="black")
    # Model a large ``1.0`` colorbar tick whose complete glyph extends roughly
    # one 30-point font height above the chromatic plot row.
    draw.rectangle((570, 70, 590, 119), fill="black")
    draw.rectangle((150, 92, 450, 108), fill="black")
    draw.rectangle((10, 105, 30, 119), fill="black")
    draw.line((300, 43, 300, 119), fill="black", width=1)
    draw.rectangle((40, 120, 560, 360), outline="#117864", width=8)
    image.save(source)

    cropped = composites._crop_white(source)
    stripped = composites._strip_embedded_top_title(cropped)
    assert stripped.shape[0] < cropped.shape[0]
    assert np.all(stripped[:8] == 255)
    chromatic = (np.ptp(stripped, axis=2) >= 12) & np.any(stripped < 245, axis=2)
    first_plot_row = int(np.flatnonzero(chromatic.sum(axis=1) >= 3)[0])
    # Preserve the complete outer y-tick/colorbar-maximum glyphs above the
    # chromatic plot row. Their absolute row depends on the retained font-height
    # band, so assert relative to the plot boundary instead of a fixed offset.
    complete_y_tick = np.all(
        stripped[:first_plot_row, :60] < 20,
        axis=2,
    ).sum()
    complete_colorbar_tick = np.all(
        stripped[:first_plot_row, -60:] < 20,
        axis=2,
    ).sum()
    assert complete_y_tick >= 250
    assert complete_colorbar_tick >= 900
    # The close crop excludes long title fragments while preserving only the
    # immediate outer y-tick/colorbar-label band.
    assert np.all(stripped[:first_plot_row, 50:500] > 248)
    assert np.ptp(stripped[first_plot_row:first_plot_row + 70], axis=2).max() >= 12

    figure, axis = composites.plt.subplots()
    composites._show_image_panel(
        axis,
        source,
        "D",
        None,
        strip_embedded_title=True,
        anchor="N",
    )
    assert [text.get_text() for text in axis.texts] == ["D"]
    assert axis.get_anchor() == "N"
    displayed = np.asarray(axis.images[0].get_array())
    assert np.all(displayed[:composites.FINEMAP_POSTCROP_TOP_PADDING_PX] == 255)
    composites.plt.close(figure)


def test_selection_sources_share_one_top_level_panel(tmp_path: Path) -> None:
    selection = composites._selection_paths(tmp_path, "1_100_A_G")
    _write_image(selection["combined_png"], "#2471a3")
    figure = composites.plt.figure(figsize=(8.5, 3.0))
    grid = figure.add_gridspec(1, 1)

    group_axis, selection_axis = composites._show_grouped_selection_panel(
        figure, grid[0, 0], selection, include_title=True,
    )

    assert [text.get_text() for text in group_axis.texts] == [
        "B",
        "EHH, iHS, nucleotide diversity, and genes (±500 kb)",
    ]
    assert list(selection_axis.texts) == []
    assert len(selection_axis.images) == 1
    composites.plt.close(figure)


def test_grouped_selection_panel_preserves_its_native_canvas(tmp_path: Path) -> None:
    selection = composites._selection_paths(tmp_path, "1_100_A_G")
    image = Image.new("RGB", (1200, 895), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 1100, 795), outline="#2471a3", width=12)
    image.save(selection["combined_png"])
    figure = composites.plt.figure(figsize=(8.5, 4.0))
    grid = figure.add_gridspec(1, 1)

    _, selection_axis = composites._show_grouped_selection_panel(
        figure,
        grid[0, 0],
        selection,
        include_title=False,
    )

    displayed = np.asarray(selection_axis.images[0].get_array())
    assert displayed.shape == (895, 1200, 3)
    composites.plt.close(figure)


def test_selection_and_finemap_images_use_equal_cells_without_stretching(
    tmp_path: Path,
) -> None:
    selection = composites._selection_paths(tmp_path, "1_100_A_G")
    _write_image(selection["combined_png"], "#2471a3")
    fine = tmp_path / "fine.png"
    _write_titled_finemap_image(fine)
    figure = composites.plt.figure(figsize=(8.5, 4.0))
    grid = figure.add_gridspec(1, 2, wspace=0.3)
    _, selection_axis = composites._show_grouped_selection_panel(
        figure,
        grid[0, 0],
        selection,
        include_title=False,
    )
    fine_axis = figure.add_subplot(grid[0, 1])
    composites._show_image_panel(
        fine_axis,
        fine,
        "C",
        None,
        strip_embedded_title=True,
        anchor="N",
        fill_cell=True,
    )
    figure.canvas.draw()
    selection_box = selection_axis.get_subplotspec().get_position(figure)
    fine_box = fine_axis.get_subplotspec().get_position(figure)
    assert abs(selection_box.width - fine_box.width) < 1e-6
    assert abs(selection_box.height - fine_box.height) < 1e-6
    assert selection_axis.get_aspect() == 1.0
    composites.plt.close(figure)


def test_image_panel_recrops_after_removing_a_wide_title(tmp_path: Path) -> None:
    source = tmp_path / "wide_title.png"
    image = Image.new("RGB", (1400, 700), "white")
    draw = ImageDraw.Draw(image)
    # Edge fragments force the initial whitespace crop to retain the full
    # 1,400-pixel title width, while the actual plot is only 700 pixels wide.
    draw.rectangle((5, 30, 180, 55), fill="black")
    draw.rectangle((1180, 30, 1395, 55), fill="black")
    draw.rectangle((350, 150, 1050, 650), outline="#117864", width=10)
    draw.line((390, 600, 1010, 210), fill="#117864", width=8)
    image.save(source)

    figure, axis = composites.plt.subplots()
    composites._show_image_panel(
        axis,
        source,
        "C",
        None,
        strip_embedded_title=True,
        anchor="N",
    )
    displayed = np.asarray(axis.images[0].get_array())
    # The source title must not determine the displayed aspect ratio.  B and C
    # then occupy comparable same-size grid cells without distorting either.
    assert displayed.shape[1] < 800
    assert displayed.shape[0] / displayed.shape[1] > 0.60
    composites.plt.close(figure)


def test_supplementary_selection_gene_track_uses_more_vertical_lanes() -> None:
    genes = {
        f"GENE{i}": {
            "tx_start": i * 100_000,
            "tx_end": i * 100_000 + 10_000,
            "strand": "+",
            "exons": [(i * 100_000, i * 100_000 + 10_000)],
        }
        for i in range(8)
    }
    figure, axis = composites.plt.subplots()
    figure3_ehh.draw_gene_track(axis, genes, 0, 800_000, gene_fontsize=10)
    label_rows = {round(text.get_position()[1], 3) for text in axis.texts}
    assert len(label_rows) >= 3
    assert figure3_ehh.GENE_TRACK_MIN_GAP_FRACTION == 0.24
    assert figure3_locus_combined._FIGSIZE_COMBINED[1] == 12.5
    composites.plt.close(figure)


def test_sv_marker_sits_above_all_gene_lanes_below_pi_track() -> None:
    genes = {
        f"GENE{i}": {
            "tx_start": i * 100_000,
            "tx_end": i * 100_000 + 10_000,
            "strand": "+",
            "exons": [(i * 100_000, i * 100_000 + 10_000)],
        }
        for i in range(8)
    }
    figure, axis = composites.plt.subplots()
    figure3_ehh.draw_gene_track(
        axis,
        genes,
        0,
        800_000,
        highlight_gene="GENE0",
        sv_pos=350_000,
        sv_type="INS",
        gene_fontsize=10,
    )
    marker = axis.collections[-1]
    marker_y = float(marker.get_offsets()[0, 1])
    gene_label_y = max(float(text.get_position()[1]) for text in axis.texts)
    assert marker_y > gene_label_y + 1.0
    assert marker_y > axis.get_ylim()[1] * 0.80
    assert float(marker.get_sizes()[0]) == figure3_ehh.SV_GENE_TRACK_MARKER_SIZE
    assert figure3_ehh.SV_GENE_TRACK_MARKER_SIZE >= 720
    composites.plt.close(figure)


def test_sv_marker_has_pixel_clearance_above_the_top_gene_label() -> None:
    genes = {
        (f"GENE{i}" if i < 7 else "SH3D21"): {
            "tx_start": 350_000 + i * 1_000,
            "tx_end": 371_000 + i * 1_000,
            "strand": "+",
            "exons": [(350_000 + i * 1_000, 355_000 + i * 1_000)],
        }
        for i in range(8)
    }
    figure, axis = composites.plt.subplots(figsize=(6.0, 3.3), dpi=100)
    figure3_ehh.draw_gene_track(
        axis,
        genes,
        0,
        1_000_000,
        sv_pos=357_000,
        sv_type="DEL",
        gene_fontsize=12,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    top_label = max(axis.texts, key=lambda text: text.get_position()[1])
    label_box = top_label.get_window_extent(renderer)
    marker = axis.collections[-1]
    marker_center = axis.transData.transform(marker.get_offsets()[0])
    marker_radius_px = (
        float(marker.get_sizes()[0]) ** 0.5 * figure.dpi / 72.0 / 2.0
    )
    clearance_px = marker_center[1] - marker_radius_px - label_box.y1

    assert top_label.get_text() == "SH3D21"
    assert clearance_px >= 4.0
    composites.plt.close(figure)


def test_sv_finemap_reserves_the_same_gene_track_height_without_duplicate_marker() -> None:
    genes = {
        "SH3D21": {
            "tx_start": 36_250_000,
            "tx_end": 36_300_000,
            "strand": "+",
            "exons": [(36_250_000, 36_260_000)],
        }
    }
    marker_figure, marker_axis = composites.plt.subplots()
    reserved_figure, reserved_axis = composites.plt.subplots()
    figure3_ehh.draw_gene_track(
        marker_axis,
        genes,
        35_780_000,
        36_780_000,
        sv_pos=36_283_616,
        sv_type="DEL",
    )
    figure3_ehh.draw_gene_track(
        reserved_axis,
        genes,
        35_780_000,
        36_780_000,
        reserve_sv_marker_band=True,
    )

    assert marker_axis.get_ylim() == reserved_axis.get_ylim()
    assert len(marker_axis.collections) == 1
    assert len(reserved_axis.collections) == 0
    composites.plt.close(marker_figure)
    composites.plt.close(reserved_figure)


def test_meta_locus_selection_stages_inputs_from_bucket_root(
    tmp_path: Path, monkeypatch,
) -> None:
    meta_path = tmp_path / "meta.tsv"
    meta_path.write_text("fixture\n", encoding="utf-8")
    loci = pd.DataFrame(
        {
            "chrom": ["chr4"],
            "core_pos": [99_100_000],
            "ref": ["A"],
            "alt": ["G"],
        }
    )
    monkeypatch.setattr(figure3_ehh, "select_loci", lambda: loci.copy())
    staged_dir = tmp_path / "selection_inputs"
    staged_sources = {
        "vcf_pattern": str(staged_dir / "SV_biallelic" / "{chrom}.biallelic.vcf.gz"),
        "ihs_pattern": str(staged_dir / "ihs_output_v4" / "{chrom}.ihs.out.100bins.norm.ann"),
        "hardmask_path": str(staged_dir / composites.DEFAULT_HARDMASK_OBJECT),
    }
    monkeypatch.setattr(
        composites,
        "stage_selection_inputs",
        lambda *args, **kwargs: (
            staged_sources,
            {"mode": "workspace-bucket root stage", "chromosomes": ["chr4"]},
        ),
    )

    def fake_stage(remote: str, local: Path, *, force: bool = False):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("fixture\n", encoding="utf-8")
        return {"remote": remote, "local": str(local), "cache": "downloaded"}

    monkeypatch.setattr(composites, "_stage_gcs_file", fake_stage)
    observed: dict[str, object] = {}

    def fake_render(loci_frame, hits, selection_dir, cache_root, **kwargs):
        observed["source_identity"] = kwargs["source_identity"]
        observed["vcf_pattern"] = figure3_ehh.VCF_PATTERN
        observed["ihs_pattern"] = figure3_ehh.IHS_PATTERN
        observed["gtf_path"] = figure3_ehh.GTF_PATH
        observed["hardmask_path"] = figure3_ehh.HARDMASK_PATH
        return {}, {"selection_loci": len(loci_frame)}

    monkeypatch.setattr(composites, "render_selection_loci", fake_render)
    _, provenance = composites.render_selection_panels(
        meta_path,
        pd.DataFrame({"variant_tag": ["4_99100000_A_G"]}),
        tmp_path / "selection",
        tmp_path / "cache",
        bucket="gs://rw-migration-aou-rw-a3e31f4d",
        selection_input_dir=staged_dir,
    )

    assert observed["source_identity"] == {
        "meta_sha256": composites.sha256_file(meta_path),
        "workspace_bucket": "gs://rw-migration-aou-rw-a3e31f4d",
    }
    assert observed["vcf_pattern"] == staged_sources["vcf_pattern"]
    assert observed["ihs_pattern"] == staged_sources["ihs_pattern"]
    assert observed["hardmask_path"] == staged_sources["hardmask_path"]
    assert Path(str(observed["gtf_path"])).name == composites.DEFAULT_GENE_ANNOTATION_OBJECT
    assert provenance["input_stage"]["chromosomes"] == ["chr4"]
    assert provenance["input_stage"]["gene_annotation"]["remote"].endswith(
        "/hg38.refGene.gtf"
    )


def test_letter_composite_is_paired_provenanced_and_cacheable(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.tsv"
    phewas_path = tmp_path / "phewas.tsv"
    _write_meta(meta_path)
    _write_phewas(phewas_path)
    hits = composites.load_meta_hits(meta_path, expected_hits=3, expected_loci=2)
    phewas, _ = composites.load_phewas_for_variants(
        phewas_path, set(hits["variant_tag"]), chunk_rows=8
    )
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection = composites._selection_paths(selection_dir, "1_100_A_G")
    _write_image(selection["combined_png"], "#2471a3")
    fine = tmp_path / "pval_disc_1_100_A_G__P1.png"
    _write_titled_finemap_image(fine)
    output_dir = tmp_path / "outputs"
    parameters = composites.PlotParameters(dpi=80)
    hit = hits.iloc[0]
    category_order = list(dict.fromkeys(phewas["phecode_category"]))
    paths = composites.compose_hit(
        hit,
        hits,
        phewas,
        selection,
        fine,
        output_dir,
        category_order,
        parameters,
        sample_manifest_sha256="a" * 64,
        force=True,
    )
    for key in ("png", "pdf", "no_title_png", "no_title_pdf"):
        assert paths[key].is_file()
    expected_name = "Supplementary_Phewas_01_1_100_A_G__phenotype_one.png"
    assert paths["png"].name == expected_name
    assert paths["pdf"].name == expected_name.replace(".png", ".pdf")
    assert paths["no_title_png"].name == expected_name
    assert paths["no_title_pdf"].name == expected_name.replace(".png", ".pdf")
    assert paths["png"].parent == output_dir / "title"
    assert paths["pdf"].parent == output_dir / "title"
    assert paths["no_title_png"].parent == output_dir / "png"
    assert paths["no_title_pdf"].parent == output_dir / "pdf"
    for key in ("png", "no_title_png"):
        with Image.open(paths[key]) as image:
            assert image.size == (680, 880)
    with Image.open(paths["no_title_png"]).convert("RGB") as image:
        pixels = np.asarray(image)
    lower = pixels[pixels.shape[0] // 3 :, :, :]
    blue = np.all(lower == np.array([36, 113, 163]), axis=2)
    green = np.all(lower == np.array([17, 120, 100]), axis=2)
    blue_rows = np.flatnonzero(blue.any(axis=1))
    blue_cols = np.flatnonzero(blue.any(axis=0))
    rendered_blue_aspect = (
        int(blue_cols[-1] - blue_cols[0] + 1)
        / int(blue_rows[-1] - blue_rows[0] + 1)
    )
    assert rendered_blue_aspect == pytest.approx(1120 / 620, rel=0.08)
    assert green.any()
    assert sha256(paths["png"].read_bytes()).digest() != sha256(
        paths["no_title_png"].read_bytes()
    ).digest()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["analysis_contract"] == composites.ANALYSIS_CONTRACT
    assert manifest["source_contract"]["plot_parameters"]["odds_ratio_cap"] == 4.0
    assert "series_rendering" not in manifest["source_contract"]
    assert manifest["source_contract"]["visual_title_variants"] == {
        "titled": "overall title; B left and C right on one row; C title removed",
        "no_title": "panel letters only; B left and C right; C title removed",
    }
    assert manifest["source_contract"]["panel_layout"] == (
        "A full width; B lower left; C lower right"
    )
    assert manifest["source_contract"]["output_layout"] == {
        "title_free_png_directory": str(output_dir / "png"),
        "title_free_pdf_directory": str(output_dir / "pdf"),
        "titled_directory": str(output_dir / "title"),
        "source_panel_directory": str(
            output_dir / "source_panels" / paths["no_title_png"].stem
        ),
        "source_zip": str(paths["source_zip"]),
        "shared_filename": expected_name,
    }
    assert set(manifest["outputs"]) == {
        "png",
        "pdf",
        "no_title_png",
        "no_title_pdf",
        "source_panels",
        "source_zip",
    }
    source_dir = output_dir / "source_panels" / paths["no_title_png"].stem
    for panel in ("A", "B", "C"):
        assert (source_dir / f"fig_01_{panel}.png").is_file()
        assert (source_dir / f"fig_01_{panel}.pdf").is_file()
    assert paths["source_zip"].is_file()
    with zipfile.ZipFile(paths["source_zip"]) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            "fig_01_A_phewas.pdf",
            "fig_01_B_ehh_ihs_pi_genes.pdf",
            "fig_01_C_phecode_fine_map.pdf",
        ]
    assert len(PdfReader(str(paths["no_title_pdf"])).pages[0].images) == 0

    mtimes = {key: path.stat().st_mtime_ns for key, path in paths.items()}
    cached = composites.compose_hit(
        hit,
        hits,
        phewas,
        selection,
        fine,
        output_dir,
        category_order,
        parameters,
        sample_manifest_sha256="a" * 64,
        force=False,
    )
    assert cached == paths
    assert {key: path.stat().st_mtime_ns for key, path in cached.items()} == mtimes
