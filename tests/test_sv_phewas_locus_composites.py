import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from scripts import (
    figure3_ehh,
    figure3_locus_combined,
    figure4_finemap,
    recompute_figure4_susie,
)
from scripts import figure5_sv_locus_composites as sv_composites


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_path = tmp_path / "figure_s9_source_data.tsv"
    source = pd.DataFrame(
        {
            "variant_tag": [
                "2_131129733_A_G",
                "2_131129733_A_G",
                "22_20391674_C_T",
                "22_20391674_C_T",
                "9_999_A_G",
            ],
            "phecode": ["EM_247.5", "GU_592.1", "GU_592", "GU_592.11", "P5"],
            "phecode_string": ["Calcium", "Second", "Cystitis", "Fourth", "Other"],
            "p": ["1e-6", "2e-5", "3e-6", "4e-5", "0.01"],
            "beta": ["0.2", "-0.3", "0.4", "-0.2", "0.1"],
            "is_sv_linked": ["TRUE", "TRUE", "TRUE", "TRUE", "FALSE"],
            "sv_annotation": [
                "chr2:131129733 (111 bp INS)",
                "chr2:131129733 (111 bp INS)",
                "chr22:20401764 (90 bp INS)",
                "chr22:20401764 (90 bp INS)",
                "",
            ],
            "is_meta_significant": ["TRUE", "FALSE", "FALSE", "FALSE", "FALSE"],
        }
    )
    source.to_csv(source_path, sep="\t", index=False)

    selection_path = tmp_path / "selection_sv_hits.s3.tsv"
    selection = pd.DataFrame(
        {
            "selection_chrom": ["chr2", "chr2", "chr2", "chr22"],
            "region_start": [131_080_000, 131_120_000, 131_160_000, 20_380_000],
            "region_stop": [131_100_000, 131_140_000, 131_220_000, 20_420_000],
            "selection_peak_position": [131_094_186, 131_129_733, 131_208_208, 20_391_674],
            "selection_peak_ihs": [4.245, 3.593, 3.871, 3.991],
            "selection_window_gene": ["FAM168B", "PLEKHB2", "POTEE", "ZNF74"],
            "sv_position": [131_129_733, 131_129_733, 131_129_733, 20_401_764],
            "sv_type": ["INS", "INS", "INS", "INS"],
            "sv_length_bp": [111, 111, 111, 90],
            "sv_ld_r2": [0.70, None, 0.52, 0.62],
            "sv_relation": ["linked", "direct", "linked", "linked"],
            "sv_gene_label": ["PLEKHB2", "PLEKHB2", "PLEKHB2", "ZNF74"],
        }
    )
    selection.to_csv(selection_path, sep="\t", index=False)

    ihs_path = tmp_path / "ihs_significant_regions.linked.tsv.gz"
    ihs = pd.DataFrame(
        {
            "chrom": ["chr2", "chr2", "chr2", "chr22"],
            "max_IHS_position": [131_094_186, 131_129_733, 131_208_208, 20_391_674],
            "max_emp_ihs": [4.245, 3.593, 3.871, 3.991],
            "IHS_REF": ["A", "G", "C", "T"],
            "IHS_ALT": ["G", "A", "T", "C"],
            "IHS_rs": ["rs1", "rs2", "rs3", "rs4"],
        }
    )
    ihs.to_csv(ihs_path, sep="\t", index=False, compression="gzip")
    return source_path, selection_path, ihs_path


def _write_panel_image(path: Path, color: str, *, finemap: bool = False) -> None:
    image = Image.new("RGB", (1200, 900 if finemap else 700), "white")
    draw = ImageDraw.Draw(image)
    if finemap:
        draw.rectangle((150, 18, 1050, 40), fill="black")
        draw.rectangle((250, 55, 950, 78), fill="black")
        draw.rectangle((70, 180, 1130, 850), outline=color, width=12)
        draw.line((110, 800, 1090, 240), fill=color, width=10)
    else:
        draw.rectangle((40, 40, 1160, 660), outline=color, width=12)
        draw.line((80, 600, 1120, 120), fill=color, width=10)
    image.save(path)
    figure, axis = sv_composites.composites.plt.subplots(
        figsize=(12, 9 if finemap else 7)
    )
    axis.plot([0.05, 0.95], [0.1, 0.9], color=color, linewidth=4)
    axis.set_axis_off()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches=figure.bbox_inches, pad_inches=0)
    sv_composites.composites.plt.close(figure)


def _write_igv_asset(path: Path, color: str = "#8e44ad") -> dict[str, object]:
    _write_panel_image(path, color)
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": path,
        "display_path": str(path),
        "chrom": "chr2",
        "sv_position": 131_129_733,
        "sv_length_bp": 111,
        "sv_type": "INS",
        "sha256": sv_composites.composites.sha256_file(path),
        "width_px": width,
        "height_px": height,
        "source_figure": "fixture",
        "source_page": 1,
        "source_image": "fixture.png",
        "source_pdf": "fixture.pdf",
        "source_pdf_sha256": "f" * 64,
    }


def test_sv_series_resolves_association_leads_to_selection_cores(tmp_path: Path) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, loci, provenance = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )

    assert hits[["variant_tag", "phecode"]].values.tolist() == [
        ["2_131129733_A_G", "EM_247.5"],
        ["2_131129733_A_G", "GU_592.1"],
        ["22_20391674_C_T", "GU_592"],
        ["22_20391674_C_T", "GU_592.11"],
    ]
    assert hits["sv_index"].tolist() == [1, 1, 2, 2]
    chr2 = hits[hits["variant_tag"] == "2_131129733_A_G"]
    assert set(chr2["selection_relation_label"]) == {"SV lead"}
    assert set(chr2["selection_core_pos"]) == {131_129_733}
    assert set(chr2["selection_match_method"]) == {"association_at_direct_sv_peak"}
    chr22 = hits[hits["variant_tag"] == "22_20391674_C_T"]
    assert set(chr22["selection_relation_label"]) == {"SV-linked"}
    assert set(chr22["selection_core_pos"]) == {20_391_674}
    assert len(loci) == 4
    assert loci["sv_index"].tolist() == [1, 1, 1, 2]
    assert loci["sv_series"].tolist() == ["selection", "phewas", "selection", "phewas"]
    assert loci["sv_series_index"].tolist() == [1, 1, 1, 2]
    assert loci["association_count"].tolist() == [0, 2, 0, 2]
    assert loci["phewas_panel_status"].tolist() == [
        "skipped_no_exact_s9_sv_linked_association",
        "included",
        "skipped_no_exact_s9_sv_linked_association",
        "included",
    ]
    assert loci.loc[loci["core_pos"].eq(20_391_674), "sv_pos"].item() == 20_401_764
    assert provenance["association_rows"] == 4
    assert provenance["selection_pages"] == 4
    assert provenance["selection_pages_with_association"] == 2
    assert provenance["selection_pages_without_association"] == 2
    assert provenance["selection_series_sv_count"] == 1
    assert provenance["phewas_series_sv_count"] == 2
    assert provenance["selection_reference"]["distinct_sv_alleles"] == 2


def test_committed_s3_contract_builds_all_17_selection_pages(tmp_path: Path) -> None:
    source_path = tmp_path / "figure_s9_source_data.tsv"
    pd.DataFrame(
        {
            "variant_tag": [
                "1_8003347_A_G", "1_8003347_A_G",
                "2_131129733_A_G", "2_131129733_A_G",
            ],
            "phecode": ["P1", "P2", "P3", "P4"],
            "phecode_string": ["One", "Two", "Three", "Four"],
            "p": ["1e-6", "2e-6", "3e-6", "4e-6"],
            "beta": ["0.2", "-0.2", "0.3", "-0.3"],
            "is_sv_linked": ["TRUE"] * 4,
            "sv_annotation": [
                "chr1:7990542 (63 bp INS)",
                "chr1:7990542 (63 bp INS)",
                "chr2:131129733 (111 bp INS)",
                "chr2:131129733 (111 bp INS)",
            ],
            "is_meta_significant": ["FALSE"] * 4,
        }
    ).to_csv(source_path, sep="\t", index=False)

    hits, pages, provenance = sv_composites.load_sv_linked_hits(source_path)

    assert len(hits) == 4
    assert len(pages) == 17
    assert pages["sv_index"].nunique() == 14
    assert pages["variant_tag"].nunique() == 17
    assert pages[["chrom", "sv_pos", "sv_length_bp", "sv_type"]].drop_duplicates().shape[0] == 14
    assert pages["selection_panel_status"].eq("included").all()
    assert pages["association_count"].sum() == 4
    stems = []
    for _, page in pages.iterrows():
        page_hits = hits[
            hits["selection_page_key"].eq(str(page["variant_tag"]))
        ].sort_values("figure_index", kind="mergesort")
        if page_hits.empty:
            stems.append(sv_composites._sv_page_output_stem_name(page, page_hits))
        else:
            for association_index, (_, hit) in enumerate(page_hits.iterrows(), start=1):
                stems.append(
                    sv_composites._sv_page_output_stem_name(
                        page,
                        hit.to_frame().T,
                        association_index=association_index,
                    )
                )
    assert len(stems) == 19
    assert len(set(stems)) == 19
    assert any(stem.startswith("Supplementary_SV_01_") for stem in stems)
    assert any(stem.startswith("Supplementary_SV_Phewas_01_1_") for stem in stems)
    assert any(stem.startswith("Supplementary_SV_Phewas_01_2_") for stem in stems)
    assert provenance["selection_pages"] == 17
    assert provenance["selection_reference"]["distinct_sv_alleles"] == 14


def test_sv_series_writes_exact_susie_tasks_without_fabricating_meta_p(tmp_path: Path) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, _, _ = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )
    task_path = sv_composites.write_susie_tasks(hits, tmp_path / "susie_tasks.tsv")
    tasks = pd.read_csv(task_path, sep="\t", dtype=str)

    assert len(tasks) == 4
    assert set(tasks["meta_significant"]) == {"TRUE"}
    assert tasks["p_meta"].isna().all()
    assert tasks["p_d"].notna().all()
    loci = recompute_figure4_susie._task_table_loci(task_path)
    assert [(locus["variant_tag"], locus["phecode"]) for locus in loci] == [
        ("2_131129733_A_G", "EM_247.5"),
        ("2_131129733_A_G", "GU_592.1"),
        ("22_20391674_C_T", "GU_592"),
        ("22_20391674_C_T", "GU_592.11"),
    ]
    assert all(pd.isna(locus["meta_p"]) for locus in loci)


def test_sv_series_renders_missing_finemap_panel_from_verified_figure4_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, _, _ = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )
    hit = hits.iloc[0]
    locus = sv_composites._figure4_locus(hit)
    assert locus["chrom"] == "chr2"
    assert locus["sv_pos"] == 131_129_733
    assert locus["sv_type"] == "INS"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    result_path = result_dir / sv_composites.expected_finemap_result_filename(hit)
    center = int(hit["pos"])
    result = pd.DataFrame(
        {
            "variant_id": [
                f"chr2:{center - 100}:A:G",
                f"chr2:{center}:INS:111",
            ],
            "chrom": ["chr2", "chr2"],
            "pos": [center - 100, center],
            "p_score": [0.02, 1e-6],
            "pip": [0.25, 0.75],
            "cs": ["L1", "L1,L2"],
            "is_sv": [False, True],
            "sv_type": ["", "INS"],
            "r2_to_minp": [0.2, 1.0],
            "_n_pcs": [20, 20],
            "_sex_scope": [locus["sex_scope"], locus["sex_scope"]],
            "_analysis_contract": [figure4_finemap.ANALYSIS_CONTRACT] * 2,
            "_window_start": [center - figure4_finemap.HALF_WINDOW] * 2,
            "_window_end": [center + figure4_finemap.HALF_WINDOW] * 2,
            "_n_samples": [1_025, 1_025],
            "_n_cases": [25, 25],
            "_min_carriers": [figure4_finemap.FINEMAP_MIN_CARRIERS] * 2,
        }
    )
    result.to_csv(result_path, sep="\t", index=False)
    shared_genes = {
        "PLEKHB2": {
            "tx_start": center - 20_000,
            "tx_end": center + 20_000,
            "strand": "+",
            "exons": [(center - 20_000, center + 20_000)],
        }
    }
    monkeypatch.setattr(
        figure3_ehh, "parse_gtf_region", lambda *_: shared_genes
    )
    gene_annotation = tmp_path / "hg38.refGene.gtf"
    gene_annotation.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        figure4_finemap, "_gene_annotation_path", lambda: gene_annotation
    )

    caplog.set_level(20, logger="sv_linked_locus_composites")
    paths, provenance = sv_composites.render_figure4_finemap_panels(
        hits.iloc[[0]],
        tmp_path / "rendered",
        [],
        local_dirs=[result_dir],
        dpi=80,
    )

    panel = paths[int(hit["figure_index"])]
    assert panel.name == sv_composites.composites.expected_finemap_filename(hit)
    assert panel.is_file()
    assert panel.with_suffix(".pdf").is_file()
    pdf_text = sv_composites.composites.PdfReader(panel.with_suffix(".pdf")).pages[0].extract_text()
    assert "−log₁₀(p)" in pdf_text
    assert "r² to min-p variant" in pdf_text
    assert panel.with_suffix(".json").is_file()
    assert provenance[str(int(hit["figure_index"]))]["cache"] == "rendered"
    panel_manifest = json.loads(panel.with_suffix(".json").read_text(encoding="utf-8"))
    assert panel_manifest["source_contract"]["chrom"] == "chr2"
    assert panel_manifest["source_contract"]["gene_count"] == 1
    assert panel_manifest["source_contract"]["gene_names"] == ["PLEKHB2"]
    assert panel_manifest["source_contract"]["panel_structure"] == (
        "score_test_plus_shared_figure3_gene_track"
    )
    assert panel_manifest["source_contract"]["window_center"] == int(
        hit["selection_core_pos"]
    )
    assert panel_manifest["source_contract"]["gene_track_renderer"] == (
        "figure3_ehh.draw_gene_track"
    )
    result_summary = panel_manifest["source_contract"]["finemap_result_summary"]
    assert result_summary["phewas_association"]["p_value"] == pytest.approx(
        float(hit["discovery_p"])
    )
    assert result_summary["phewas_association"]["beta"] == pytest.approx(
        float(hit["discovery_beta"])
    )
    assert [item["causal_set"] for item in result_summary["causal_sets"]] == [
        "L1", "L2"
    ]
    assert result_summary["causal_sets"][0]["members"] == [
        {
            "variant_id": f"chr2:{center}:INS:111",
            "position": center,
            "pip": 0.75,
            "p_value": 1e-6,
            "is_sv": True,
            "sv_type": "INS",
        },
        {
            "variant_id": f"chr2:{center - 100}:A:G",
            "position": center - 100,
            "pip": 0.25,
            "p_value": 0.02,
            "is_sv": False,
            "sv_type": "",
        },
    ]
    assert result_summary["top_pip_variant"]["pip"] == 0.75
    assert result_summary["top_pip_variant"]["p_value"] == 1e-6
    assert result_summary["minimum_p_variant"]["pip"] == 0.75
    assert result_summary["minimum_p_variant"]["p_value"] == 1e-6
    assert result_summary["sv_matches"] == [result_summary["top_pip_variant"]]
    log_messages = [record.getMessage() for record in caplog.records]
    for record_type in (
        "summary", "top_pip_variant", "minimum_p_variant", "exact_sv",
        "causal_set_member",
    ):
        assert any(
            f'"record_type":"{record_type}"' in message
            for message in log_messages
        )
    assert any('"causal_set":"L1"' in message for message in log_messages)
    assert any('"causal_set":"L2"' in message for message in log_messages)
    assert panel_manifest["source_contract"]["sv_marker"] == {
        "shape": "^",
        "edge_color": figure4_finemap.SV_OUTLINE_COLOR,
        "size": figure4_finemap.SV_MARKER_SIZE,
        "edge_linewidth": figure4_finemap.SV_MARKER_LINEWIDTH,
        "vertical_guide": False,
        "position_match_count": 1,
        "pvalue_match_count": 1,
        "y_mode": "result_p_value",
    }
    with Image.open(panel) as image:
        assert image.width > 500
        assert image.height > 400
        assert image.width / image.height == pytest.approx(
            figure4_finemap.SUPPLEMENTARY_PANEL_ASPECT, rel=0.01
        )

    _, reused = sv_composites.render_figure4_finemap_panels(
        hits.iloc[[0]],
        tmp_path / "rendered",
        [],
        local_dirs=[result_dir],
        dpi=80,
    )
    assert reused[str(int(hit["figure_index"]))]["cache"] == "hit"

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection = sv_composites.composites._selection_paths(
        selection_dir, str(hit["variant_tag"])
    )
    _write_panel_image(selection["combined_png"], "#2471a3")
    phewas = pd.DataFrame(
        {
            "variant_tag": [hit["variant_tag"]] * 3,
            "phecode": ["BI_160", "EM_247.5", "GU_592"],
            "phecode_name": ["Blood", "Calcium", "Cystitis"],
            "phecode_category": ["Blood/Immune", "Endocrine/Metab", "Genitourinary"],
            "p_value": [0.2, 1e-6, 0.04],
            "beta": [0.1, 0.2, -0.1],
            "odds_ratio": [1.105, 1.221, 0.905],
        }
    )
    outputs = sv_composites.composites.compose_hit(
        hit,
        hits,
        phewas,
        selection,
        panel,
        tmp_path / "composites",
        ["Blood/Immune", "Endocrine/Metab", "Genitourinary"],
        sv_composites.composites.PlotParameters(dpi=80, phewas_y_max=7.0),
        sample_manifest_sha256="a" * 64,
        force=True,
        analysis_contract=sv_composites.ANALYSIS_CONTRACT,
        output_prefix="Supplementary_SV_Locus_Composite",
        highlight_color=sv_composites.SV_HIGHLIGHT_COLOR,
        highlight_label="SV-linked/lead",
        association_summary=f"{hit['selection_relation_label']}: {hit['sv_annotation']}",
    )
    with Image.open(outputs["png"]) as image:
        assert image.size == (680, 880)


def test_sv_series_rejects_an_unmapped_annotation(tmp_path: Path) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    source = pd.read_csv(source_path, sep="\t", dtype=str)
    source.loc[0, "sv_annotation"] = "chr2:999999 (111 bp INS)"
    source.to_csv(source_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="No S3 selection reference matches"):
        sv_composites.load_sv_linked_hits(
            source_path,
            selection_path,
            ihs_path,
            expected_hits=4,
            expected_selection_regions=4,
            expected_sv_alleles=2,
        )


def test_sv_cli_defaults_to_separate_single_output_directory() -> None:
    args = sv_composites.build_parser().parse_args([])
    assert args.expected_hits == 4
    assert args.expected_selection_pages == 17
    assert args.expected_sv_alleles == 14
    assert args.threads == 4
    assert args.output_dir == sv_composites.DEFAULT_OUTPUT_DIR
    assert args.igv_dir == sv_composites.DEFAULT_IGV_DIR
    assert args.output_dir.as_posix().endswith("figures/figure5/loci")
    assert args.gene_annotation_object == "hg38.refGene.gtf"
    assert sv_composites.ANALYSIS_CONTRACT.endswith("_v18")
    assert sv_composites.FINEMAP_PANEL_CONTRACT.endswith("_v10")


def test_committed_igv_assets_exactly_cover_the_14_unique_s3_sv_alleles() -> None:
    assets, provenance = sv_composites.load_igv_screenshots(
        sv_composites.DEFAULT_IGV_DIR
    )
    selection = pd.read_csv(
        sv_composites.DEFAULT_SELECTION_HITS,
        sep="\t",
        dtype=str,
    )
    expected = {
        sv_composites._sv_allele_key(
            row["selection_chrom"],
            row["sv_position"],
            row["sv_length_bp"],
            row["sv_type"],
        )
        for _, row in selection.iterrows()
    }

    assert set(assets) == expected
    assert len(assets) == 14
    assert provenance["asset_count"] == 14
    assert provenance["contract"] == sv_composites.IGV_ASSET_CONTRACT
    assert provenance["source_pdf_sha256"] == (
        "2d899ea9f304df6d69a647c3c923ca31b06585222ec9e95ff2b0f21f38cc9716"
    )


def test_sv_runner_rejects_mismatched_selection_and_finemap_windows() -> None:
    args = sv_composites.build_parser().parse_args(
        ["--selection-half-window-bp", "400000"]
    )
    with pytest.raises(ValueError, match="panels B and C share endpoints"):
        sv_composites.run(args)


def test_selection_pages_adapt_panels_to_exact_s9_associations(tmp_path: Path) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, pages, _ = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )
    parameters = sv_composites.composites.PlotParameters(dpi=80, phewas_y_max=7.0)
    category_order = ["Blood/Immune", "Endocrine/Metab", "Genitourinary"]

    selection_only = pages[pages["core_pos"].eq(131_094_186)].iloc[0]
    selection_only_dir = tmp_path / "selection_only_sources"
    selection_only_dir.mkdir()
    selection_only_paths = sv_composites.composites._selection_paths(
        selection_only_dir, str(selection_only["variant_tag"])
    )
    _write_panel_image(selection_only_paths["combined_png"], "#2471a3")
    selection_only_igv = _write_igv_asset(selection_only_dir / "igv.png")
    selection_only_outputs = sv_composites.compose_selection_page(
        selection_only,
        hits.iloc[0:0],
        None,
        selection_only_paths,
        {},
        selection_only_igv,
        tmp_path / "outputs",
        category_order,
        parameters,
        sample_manifest_sha256="a" * 64,
        force=True,
    )
    with Image.open(selection_only_outputs["png"]) as image:
        assert image.size == (680, 880)
    selection_only_manifest = json.loads(
        selection_only_outputs["manifest"].read_text(encoding="utf-8")
    )
    selection_name = (
        "Supplementary_SV_01_2_131129733_111bp_INS__"
        "selection_131094186.png"
    )
    assert selection_only_outputs["png"].name == selection_name
    assert selection_only_outputs["no_title_png"].name == selection_name
    assert selection_only_outputs["png"].parent == tmp_path / "outputs" / "title"
    assert selection_only_outputs["no_title_png"].parent == tmp_path / "outputs" / "png"
    assert selection_only_outputs["no_title_pdf"].parent == tmp_path / "outputs" / "pdf"
    assert selection_only_manifest["source_contract"]["output_layout"] == {
        "title_free_png_directory": str(tmp_path / "outputs" / "png"),
        "title_free_pdf_directory": str(tmp_path / "outputs" / "pdf"),
        "titled_directory": str(tmp_path / "outputs" / "title"),
        "source_panel_directory": str(
            tmp_path
            / "outputs"
            / "source_panels"
            / selection_only_outputs["no_title_png"].stem
        ),
        "source_zip": str(selection_only_outputs["source_zip"]),
        "shared_filename": selection_name,
    }
    assert selection_only_manifest["source_contract"]["panel_decisions"] == {
        "selection": "included",
        "phewas": "skipped_no_exact_s9_sv_linked_association",
        "finemap": "skipped_no_exact_s9_sv_linked_association",
        "igv": "included",
    }
    assert selection_only_manifest["source_contract"]["panel_layout"] == (
        "A selection; B IGV"
    )
    assert selection_only_manifest["source_contract"]["igv"]["sha256"] == (
        selection_only_igv["sha256"]
    )
    assert "phewas_subset_sha256" not in selection_only_manifest["source_contract"]
    assert "finemap" not in selection_only_manifest["source_contract"]
    with Image.open(selection_only_outputs["no_title_png"]).convert("RGB") as image:
        pixels = np.asarray(image)
    assert np.all(pixels == np.array([142, 68, 173]), axis=2).any()
    selection_only_source_dir = (
        tmp_path
        / "outputs"
        / "source_panels"
        / selection_only_outputs["no_title_png"].stem
    )
    for panel in ("A", "B"):
        assert (selection_only_source_dir / f"fig_{int(selection_only['page_index']):02d}_{panel}.png").is_file()
        assert (selection_only_source_dir / f"fig_{int(selection_only['page_index']):02d}_{panel}.pdf").is_file()
    with zipfile.ZipFile(selection_only_outputs["source_zip"]) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            "fig_01_A_ehh_ihs_pi_genes.pdf",
            "fig_01_B_igv_validation.pdf",
        ]

    associated = pages[pages["core_pos"].eq(131_129_733)].iloc[0]
    page_hits = hits[
        hits["selection_page_key"].eq(str(associated["variant_tag"]))
    ].copy()
    associated_dir = tmp_path / "associated_sources"
    associated_dir.mkdir()
    associated_paths = sv_composites.composites._selection_paths(
        associated_dir, str(associated["variant_tag"])
    )
    _write_panel_image(associated_paths["combined_png"], "#2471a3")
    associated_igv = _write_igv_asset(associated_dir / "igv.png")
    finemap_paths: dict[int, Path] = {}
    for _, hit in page_hits.iterrows():
        panel = associated_dir / sv_composites.composites.expected_finemap_filename(hit)
        _write_panel_image(panel, "#117864", finemap=True)
        finemap_paths[int(hit["figure_index"])] = panel
    phewas = pd.DataFrame(
        {
            "variant_tag": [page_hits.iloc[0]["variant_tag"]] * 4,
            "phecode": ["BI_160", "EM_247.5", "GU_592", "GU_592.1"],
            "phecode_name": ["Blood", "Calcium", "Cystitis", "Second"],
            "phecode_category": [
                "Blood/Immune", "Endocrine/Metab", "Genitourinary", "Genitourinary",
            ],
            "p_value": [0.2, 1e-6, 0.04, 2e-5],
            "beta": [0.1, 0.2, -0.1, -0.3],
            "odds_ratio": [1.105, 1.221, 0.905, 0.741],
        }
    )
    associated_outputs = []
    for association_index, (_, hit) in enumerate(
        page_hits.sort_values("figure_index", kind="mergesort").iterrows(), start=1
    ):
        hit_frame = hit.to_frame().T
        hit_index = int(hit["figure_index"])
        outputs = sv_composites.compose_selection_page(
            associated,
            hit_frame,
            phewas,
            associated_paths,
            {hit_index: finemap_paths[hit_index]},
            associated_igv,
            tmp_path / "outputs",
            category_order,
            parameters,
            sample_manifest_sha256="a" * 64,
            association_index=association_index,
            phewas_highlight_hits=page_hits,
            force=True,
        )
        associated_outputs.append(outputs)
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert manifest["source_contract"]["panel_decisions"] == {
            "selection": "included",
            "phewas": "included",
            "finemap": "included",
            "igv": "included",
        }
        assert manifest["source_contract"]["panel_layout"] == (
            "A full width; B lower left; C lower right; D IGV third row"
        )
        assert manifest["source_contract"]["association_index"] == association_index
        assert manifest["source_contract"]["association_total"] == 2
        assert len(manifest["source_contract"]["associations"]) == 1
        assert {
            row["phecode"]
            for row in manifest["source_contract"]["phewas_highlights"]
        } == {"EM_247.5", "GU_592.1"}
        assert len(manifest["source_contract"]["finemap"]) == 1
        with Image.open(outputs["no_title_png"]).convert("RGB") as image:
            assert image.size == (680, 880)
            pixels = np.asarray(image)
        lower = pixels[pixels.shape[0] // 3 :, :, :]
        blue = np.all(lower == np.array([36, 113, 163]), axis=2)
        green = np.all(lower == np.array([17, 120, 100]), axis=2)
        purple = np.all(lower == np.array([142, 68, 173]), axis=2)
        blue_rows = np.flatnonzero(blue.any(axis=1))
        blue_cols = np.flatnonzero(blue.any(axis=0))
        # The selection source fixture has a 1120:620 colored frame.  Its
        # geometry must survive page composition instead of being stretched
        # to the half-page cell's unrelated aspect ratio.
        rendered_blue_aspect = (
            int(blue_cols[-1] - blue_cols[0] + 1)
            / int(blue_rows[-1] - blue_rows[0] + 1)
        )
        assert rendered_blue_aspect == pytest.approx(1120 / 620, rel=0.08)
        assert green.any()
        assert purple.any()
        source_dir = (
            tmp_path
            / "outputs"
            / "source_panels"
            / outputs["no_title_png"].stem
        )
        for panel in ("A", "B", "C", "D"):
            assert (source_dir / f"fig_{int(associated['page_index']):02d}_{panel}.png").is_file()
            assert (source_dir / f"fig_{int(associated['page_index']):02d}_{panel}.pdf").is_file()
        assert outputs["source_zip"].is_file()

    association_names = [outputs["png"].name for outputs in associated_outputs]
    assert association_names == [
        "Supplementary_SV_Phewas_01_1_2_131129733_111bp_INS__calcium.png",
        "Supplementary_SV_Phewas_01_2_2_131129733_111bp_INS__second.png",
    ]
    assert all(outputs["png"].parent == tmp_path / "outputs" / "title" for outputs in associated_outputs)
    assert all(outputs["no_title_png"].parent == tmp_path / "outputs" / "png" for outputs in associated_outputs)
    assert all(outputs["no_title_pdf"].parent == tmp_path / "outputs" / "pdf" for outputs in associated_outputs)


def test_runtime_phewas_applicability_reclassifies_pages_without_target_rows(
    tmp_path: Path,
) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, pages, _ = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )
    # Only the chr2 association is actually present in the runtime PheWAS
    # table.  The chr22 S9 annotation must therefore fall back to the plain SV
    # series and must never produce an SV_Phewas filename.
    runtime_phewas = pd.DataFrame(
        {
            "variant_tag": ["2_131129733_A_G", "2_131129733_A_G"],
            "phecode": ["EM_247.5", "GU_592.1"],
        }
    )
    applicable_hits, applicable_pages, provenance = (
        sv_composites.resolve_runtime_phewas_applicability(
            hits,
            pages,
            runtime_phewas,
        )
    )

    assert applicable_hits[["variant_tag", "phecode"]].values.tolist() == [
        ["2_131129733_A_G", "EM_247.5"],
        ["2_131129733_A_G", "GU_592.1"],
    ]
    chr22 = applicable_pages[applicable_pages["sv_pos"].eq(20_401_764)]
    assert set(chr22["sv_series"]) == {"selection"}
    assert set(chr22["association_count"]) == {0}
    assert set(chr22["phewas_panel_status"]) == {
        "skipped_no_runtime_phewas_target"
    }
    chr22_page = chr22.iloc[0]
    chr22_hits = applicable_hits[
        applicable_hits["selection_page_key"].eq(str(chr22_page["variant_tag"]))
    ]
    assert sv_composites._sv_page_output_stem_name(
        chr22_page,
        chr22_hits,
    ).startswith("Supplementary_SV_")
    assert "Supplementary_SV_Phewas" not in sv_composites._sv_page_output_stem_name(
        chr22_page,
        chr22_hits,
    )
    assert provenance["runtime_association_rows"] == 2
    assert provenance["runtime_pages_with_phewas"] == 1


def test_prune_stale_sv_outputs_keeps_only_runtime_expected_series(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    expected_stem = "Supplementary_SV_01_22_20401764_90bp_INS__selection_20391674"
    stale_stem = "Supplementary_SV_Phewas_99_22_20401764_90bp_INS__cystitis"
    for directory, suffix in (("title", ".png"), ("title", ".pdf"), ("png", ".png"), ("pdf", ".pdf")):
        target_dir = output_dir / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{expected_stem}{suffix}").write_bytes(b"expected")
        (target_dir / f"{stale_stem}{suffix}").write_bytes(b"stale")
    (output_dir / f"{expected_stem}.json").write_text("{}\n", encoding="utf-8")
    (output_dir / f"{stale_stem}.json").write_text("{}\n", encoding="utf-8")
    for stem in (expected_stem, stale_stem):
        source_dir = output_dir / "source_panels" / stem
        source_dir.mkdir(parents=True)
        (source_dir / "fig_01_A.png").write_bytes(b"source")

    removed = sv_composites.prune_stale_sv_outputs(
        output_dir,
        {expected_stem},
    )

    assert len(removed) == 6
    assert all(stale_stem in str(path) for path in removed)
    assert (output_dir / "title" / f"{expected_stem}.png").is_file()
    assert (output_dir / f"{expected_stem}.json").is_file()
    assert (output_dir / "source_panels" / expected_stem).is_dir()
    assert not (output_dir / "source_panels" / stale_stem).exists()


def test_selection_inputs_stage_from_bucket_without_migration_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AOU_MIGRATION_WORKSPACE", raising=False)
    staged: list[str] = []

    def fake_stage(remote: str, local: Path, *, force: bool = False):
        staged.append(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"fixture")
        return {
            "remote": remote,
            "local": str(local),
            "size": local.stat().st_size,
            "cache": "downloaded",
        }

    monkeypatch.setattr(sv_composites.composites, "_stage_gcs_file", fake_stage)
    loci = pd.DataFrame({"chrom": ["chr2", "chr22", "chr1", "chr2"]})

    paths, provenance = sv_composites.stage_selection_inputs(
        loci,
        "gs://rw-migration-aou-rw-a3e31f4d",
        tmp_path / "selection_inputs",
        threads=4,
    )

    assert paths["vcf_pattern"].replace("\\", "/").endswith(
        "SV_biallelic/{chrom}.biallelic.vcf.gz"
    )
    assert paths["ihs_pattern"].replace("\\", "/").endswith(
        "ihs_output_v4/{chrom}.ihs.out.100bins.norm.ann"
    )
    assert Path(paths["hardmask_path"]).is_file()
    assert provenance["chromosomes"] == ["chr1", "chr2", "chr22"]
    assert set(staged) == {
        "gs://rw-migration-aou-rw-a3e31f4d/hardmask.hg38.v4.over99.bed",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr1.biallelic.vcf.gz",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr1.biallelic.vcf.gz.tbi",
        "gs://rw-migration-aou-rw-a3e31f4d/ihs_output_v4/chr1.ihs.out.100bins.norm.ann",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr2.biallelic.vcf.gz",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr2.biallelic.vcf.gz.tbi",
        "gs://rw-migration-aou-rw-a3e31f4d/ihs_output_v4/chr2.ihs.out.100bins.norm.ann",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr22.biallelic.vcf.gz",
        "gs://rw-migration-aou-rw-a3e31f4d/SV_biallelic/chr22.biallelic.vcf.gz.tbi",
        "gs://rw-migration-aou-rw-a3e31f4d/ihs_output_v4/chr22.ihs.out.100bins.norm.ann",
    }


def test_sv_composite_keeps_actual_lead_and_sv_provenance(tmp_path: Path) -> None:
    source_path, selection_path, ihs_path = _write_sources(tmp_path)
    hits, _, _ = sv_composites.load_sv_linked_hits(
        source_path,
        selection_path,
        ihs_path,
        expected_hits=4,
        expected_selection_regions=4,
        expected_sv_alleles=2,
    )
    hit = hits.iloc[2]
    phewas = pd.DataFrame(
        {
            "variant_tag": [hit["variant_tag"]] * 4,
            "phecode": ["BI_160", "EM_247.5", "GU_592", "GU_592.1"],
            "phecode_name": ["Blood", "Calcium", "Cystitis", "Other urinary"],
            "phecode_category": ["Blood/Immune", "Endocrine/Metab", "Genitourinary", "Genitourinary"],
            "p_value": [0.2, 0.01, 3e-6, 0.04],
            "beta": [0.1, -0.2, 0.4, -0.1],
            "odds_ratio": [1.105, 0.819, 1.492, 0.905],
        }
    )
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection = sv_composites.composites._selection_paths(selection_dir, hit["variant_tag"])
    _write_panel_image(selection["combined_png"], "#2471a3")
    finemap = tmp_path / sv_composites.composites.expected_finemap_filename(hit)
    _write_panel_image(finemap, "#117864", finemap=True)
    parameters = sv_composites.composites.PlotParameters(dpi=80, phewas_y_max=7.0)
    outputs = sv_composites.composites.compose_hit(
        hit,
        hits,
        phewas,
        selection,
        finemap,
        tmp_path / "figures" / "supplementary" / "s11_s24",
        ["Blood/Immune", "Endocrine/Metab", "Genitourinary"],
        parameters,
        sample_manifest_sha256="a" * 64,
        force=True,
        analysis_contract=sv_composites.ANALYSIS_CONTRACT,
        output_prefix="Supplementary_SV_Locus_Composite",
        highlight_color=sv_composites.SV_HIGHLIGHT_COLOR,
        highlight_label="SV-linked/lead",
        association_summary=f"{hit['selection_relation_label']}: {hit['sv_annotation']}",
        extra_hit_contract_columns=("sv_annotation", "selection_core_pos", "selection_relation"),
    )

    assert all(outputs[key].is_file() for key in ("png", "pdf", "no_title_png", "no_title_pdf"))
    assert outputs["png"].name.startswith("Supplementary_SV_Locus_Composite_02_")
    assert outputs["png"].parent.name == "title"
    assert outputs["no_title_png"].parent.name == "png"
    assert outputs["no_title_pdf"].parent.name == "pdf"
    with Image.open(outputs["png"]) as image:
        assert image.size == (680, 880)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["analysis_contract"] == sv_composites.ANALYSIS_CONTRACT
    assert manifest["source_contract"]["hit"]["variant_tag"] == "22_20391674_C_T"
    assert manifest["source_contract"]["hit"]["sv_annotation"] == "chr22:20401764 (90 bp INS)"
    assert manifest["source_contract"]["hit"]["selection_core_pos"] == 20_391_674
    assert manifest["source_contract"]["series_rendering"]["highlight_label"] == "SV-linked/lead"


def test_explicit_susie_tasks_require_the_matching_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    locus = {
        "chrom": "chr22",
        "lead_pos": 20_391_674,
        "phecode": "GU_592",
        "sex_scope": "Both",
    }
    result = tmp_path / "susie.tsv"
    result.write_text("placeholder\n", encoding="utf-8")
    task_table = tmp_path / "tasks.tsv"
    task_table.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(
        recompute_figure4_susie,
        "_requested_loci",
        lambda **_: [locus],
    )
    monkeypatch.setattr(
        recompute_figure4_susie.figure4_finemap,
        "_find_discovery_result",
        lambda _: str(result),
    )
    monkeypatch.setattr(
        recompute_figure4_susie.figure4_finemap,
        "validate_discovery_contract",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        recompute_figure4_susie.figure4_finemap,
        "_find_local_artifact",
        lambda *_: None,
    )

    with pytest.raises(RuntimeError, match="requested hits.*plot"):
        recompute_figure4_susie._verify_figure4_outputs(task_table=task_table)

    plot = tmp_path / "pval_disc.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\nvalid")
    monkeypatch.setattr(
        recompute_figure4_susie.figure4_finemap,
        "_find_local_artifact",
        lambda *_: str(plot),
    )
    assert recompute_figure4_susie._verify_figure4_outputs(task_table=task_table) == [result]


def test_selection_worker_forwards_the_sv_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, int, str, tuple[float, float]]] = []

    def fake_combined(*args, sv_pos=None, sv_type=None, figsize=None, **kwargs):
        observed.append(("combined", sv_pos, sv_type, tuple(figsize)))
        return sv_composites.composites.plt.figure(figsize=figsize)

    monkeypatch.setattr(figure3_locus_combined, "plot_combined_locus", fake_combined)
    paths = sv_composites.composites._selection_paths(tmp_path, "22_20391674_C_T")
    payload = {
        "variant_tag": "22_20391674_C_T",
        "locus": {
            "chrom": "chr22",
            "core_pos": 20_391_674,
            "ref": "C",
            "alt": "T",
            "gene": "ZNF74",
            "ihs_val": 3.991,
            "ihs_rs": "rs-test",
            "min_p_meta": float("nan"),
            "sv_pos": 20_401_764,
            "sv_type": "INS",
        },
        "vcf_path": str(tmp_path / "unused.vcf.gz"),
        "gtf_path": str(tmp_path / "unused.gtf"),
        "hardmask_path": str(tmp_path / "unused.bed"),
        "ihs_pattern": str(tmp_path / "unused.{chrom}.tsv"),
        "selection_half_window_bp": 400_000,
        "cache_dir": str(tmp_path / "cache"),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    sv_composites.composites._selection_worker(payload)

    assert observed == [
        (
            "combined",
            20_401_764,
            "INS",
            sv_composites.composites.SELECTION_COMPACT_FIGSIZE,
        )
    ]
    assert paths["combined_png"].is_file()
    assert paths["combined_pdf"].is_file()
    assert paths["combined_wide_png"].is_file()
    assert paths["combined_wide_pdf"].is_file()
    with Image.open(paths["combined_png"]) as compact_image:
        compact_aspect = compact_image.width / compact_image.height
    with Image.open(paths["combined_wide_png"]) as wide_image:
        wide_aspect = wide_image.width / wide_image.height
    assert compact_aspect == pytest.approx(
        sv_composites.composites.SELECTION_COMPACT_PANEL_ASPECT, rel=0.005
    )
    assert wide_aspect == pytest.approx(
        sv_composites.composites.SELECTION_WIDE_PANEL_ASPECT, rel=0.005
    )
