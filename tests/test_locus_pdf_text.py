"""Exercise PDF text through the real supplementary panel output paths."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from scripts import figure3_ehh, figure4_finemap
from scripts import figure5_locus_composites as composites


def assert_single_text_run(path: Path, label: str) -> None:
    """A label must be one PDF text-show operation, not separate math glyphs."""
    runs = []
    for page in PdfReader(path).pages:
        for operands, operator in page.get_contents().operations:
            if operator not in {b"Tj", b"TJ"}:
                continue
            parts = operands[-1] if operator == b"TJ" else operands[-1:]
            raw = b"".join(
                bytes(part) if isinstance(part, bytes) else part.get_original_bytes()
                for part in parts if isinstance(part, (str, bytes))
            )
            runs.append(raw.decode("utf-16-be", errors="replace").replace("\ufeff", ""))
    assert label in runs, f"Axis label is not one editable text run: {label!r}; got {runs!r}"


def assert_editable_text(path: Path, labels: tuple[str, ...]) -> None:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    for label in labels:
        assert label in text
    fonts = [
        ref.get_object()
        for page in reader.pages
        for ref in page["/Resources"]["/Font"].get_object().values()
    ]
    assert fonts
    for font in fonts:
        assert font["/Subtype"] == "/Type0", f"Non-editable font in {path}: {font['/Subtype']}"
        assert font["/ToUnicode"].get_data()
        descendant = font["/DescendantFonts"][0].get_object()
        assert descendant["/Subtype"] == "/CIDFontType2"
        assert descendant["/FontDescriptor"]["/FontFile2"].get_data()


def test_phewas_source_and_merged_pdf_keep_editable_text(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "variant_tag": ["1_100_A_G"] * 3,
        "phecode": ["P1", "P2", "P3"],
        "phecode_name": ["Trait one", "Trait two", "Trait three"],
        "phecode_category": ["Circulatory", "Circulatory", "Respiratory"],
        "p_value": [1e-6, 0.04, 0.5],
        "beta": [0.4, -0.2, 0.1],
        "odds_ratio": np.exp([0.4, -0.2, 0.1]),
    })
    hit = pd.Series({"variant_tag": "1_100_A_G", "phecode": "P1"})
    # Output must survive a caller's default Type 3 font setting.
    with composites.plt.rc_context({"pdf.fonttype": 3}):
        fig, ax = composites.plt.subplots(figsize=(8.5, 5))
        fig.subplots_adjust(top=0.77, bottom=0.22)
        composites.draw_phewas_panel(
            ax, frame, pd.DataFrame([hit]), hit,
            ["Circulatory", "Respiratory"], composites.PlotParameters(dpi=80),
        )
        paths = composites._source_panel_output_paths(tmp_path, "phewas", 1, ("A",))
        composites._save_source_panels(fig, {"A": (ax,)}, paths, dpi=80, metadata={})
        merged = tmp_path / "merged.pdf"
        base, cell = composites.plt.subplots(figsize=(6, 3))
        cell.set_axis_off()
        base.text(0.1, 0.95, "PheWAS")
        composites._save_vector_composite_pdf(base, merged, [(cell, paths["A"]["pdf"])], metadata={})
        composites.plt.close(fig)
        composites.plt.close(base)
    assert_editable_text(paths["A"]["pdf"], ("Circulatory", "Respiratory"))
    assert_editable_text(merged, ("PheWAS", "Circulatory"))
    assert_single_text_run(paths["A"]["pdf"], "−log₁₀(p)")
    assert_single_text_run(merged, "−log₁₀(p)")
    assert composites._validate_editable_panel_pdf(merged)["editable_text"]


def test_finemap_axis_labels_keep_single_text_runs(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "pos": [900_000, 1_000_000, 1_100_000],
        "neg_log_p": [2.0, 6.0, 3.0],
        "r2_to_lead": [0.2, 1.0, 0.5],
        "is_sv": [False] * 3,
    })
    locus = {"lead_pos": 1_000_000, "chrom": "chr1", "variant_tag": "1_1000000_A_G"}
    with composites.plt.rc_context(composites.EDITABLE_PDF_RC):
        fig = figure4_finemap.plot_finemap(
            frame, locus, show_title=False, shared_gene_track={},
            editable_axis_labels=True,
        )
        path = tmp_path / "fine_map.pdf"
        fig.savefig(path)
        composites.plt.close(fig)
    assert_editable_text(path, ("Position (bp)",))
    assert_single_text_run(path, "−log₁₀(p)")
    assert_single_text_run(path, "r² to min-p variant")


def test_panel_merge_transforms_the_clip_box_with_the_ehh_panel(tmp_path: Path) -> None:
    # A short panel placed high on a letter page lies outside its original
    # source-page coordinates. Its crop box must move with its text/curves.
    source = tmp_path / "ehh_source.pdf"
    with composites.plt.rc_context(composites.EDITABLE_PDF_RC):
        fig = composites.plt.figure(figsize=(6, 4))
        fig.text(0.1, 0.9, "Ref. Hap.")
        fig.savefig(source, bbox_inches=fig.bbox_inches, pad_inches=0)
        composites.plt.close(fig)
    fig = composites.plt.figure(figsize=(8.5, 11))
    axis = fig.add_axes([0.1, 0.65, 0.8, 0.3])
    axis.set_axis_off()
    merged = tmp_path / "ehh_page.pdf"
    composites._save_vector_composite_pdf(fig, merged, [(axis, source)], metadata={})
    composites.plt.close(fig)

    operations = PdfReader(merged).pages[0].get_contents().operations
    clip_index = next(i for i, (_, op) in enumerate(operations) if op == b"W")
    transforms_before_clip = [
        tuple(float(value) for value in args)
        for args, op in operations[:clip_index] if op == b"cm"
    ]
    # Height constrains the scale to 3.3 / 4; center horizontally in the cell.
    scale = 3.3 / 4
    expected = (scale, 0, 0, scale, (0.1 * 8.5 + (0.8 * 8.5 - 6 * scale) / 2) * 72, 0.65 * 11 * 72)
    assert any(matrix == pytest.approx(expected) for matrix in transforms_before_clip)
    assert_single_text_run(merged, "Ref. Hap.")


@pytest.mark.parametrize("sv_type", [None, "INS"])
def test_selection_worker_keeps_editable_text(tmp_path: Path, monkeypatch, sv_type) -> None:
    monkeypatch.setattr(figure3_ehh, "compute_or_load_ehh", lambda *a: {
        "pos_left": np.array([900_000, 800_000]),
        "pos_right": np.array([1_100_000, 1_200_000]),
        "ehh0_left": np.array([0.7, 0.4]), "ehh0_right": np.array([0.6, 0.2]),
        "ehh1_left": np.array([0.9, 0.8]), "ehh1_right": np.array([0.8, 0.6]),
    })
    monkeypatch.setattr(figure3_ehh, "load_norm_ihs", lambda *a: (
        np.array([800_000, 1_000_000, 1_200_000]), np.array([-1.0, 3.0, 1.5]),
    ))
    monkeypatch.setattr(figure3_ehh, "compute_or_load_pi", lambda *a: {
        "pi_mids": np.array([800_000, 1_000_000, 1_200_000]),
        "pi_vals": np.array([0.001, 0.002, 0.0015]),
    })
    monkeypatch.setattr(figure3_ehh, "parse_gtf_region", lambda *a: {
        "TEST": {"tx_start": 950_000, "tx_end": 1_080_000, "strand": "+",
                 "exons": [(950_000, 970_000), (1_030_000, 1_080_000)]},
    })
    paths = composites._selection_paths(tmp_path, "1_1000000_A_G")
    payload = {
        "variant_tag": "1_1000000_A_G",
        "locus": {"chrom": "chr1", "core_pos": 1_000_000, "ref": "A", "alt": "G",
                  "gene": "TEST", "ihs_val": 3.0, "ihs_rs": "rs-test",
                  "sv_pos": 1_010_000 if sv_type else None, "sv_type": sv_type},
        "vcf_path": "unused.vcf.gz", "gtf_path": "unused.gtf",
        "hardmask_path": "unused.bed", "ihs_pattern": "unused.{chrom}.tsv",
        "selection_half_window_bp": 500_000, "cache_dir": str(tmp_path / "cache"),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    with composites.plt.rc_context({"pdf.fonttype": 3}):
        composites._selection_worker(payload)
    for key in ("combined_pdf", "combined_wide_pdf"):
        assert_editable_text(paths[key], ("EHH", "|iHS|", "π", "Ref. Hap.", "TEST", "Mb"))
        for label in (
            "EHH", "|iHS|", "π", "Genes", "Position (bp)",
            "Ref. Hap.", "Alt. Hap.", "TEST", "0.50 Mb", "1.00 Mb", "1.50 Mb",
        ):
            assert_single_text_run(paths[key], label)


def test_editable_validation_rejects_type3_glyph_drawings(tmp_path: Path) -> None:
    path = tmp_path / "outlined.pdf"
    with composites.plt.rc_context({"pdf.fonttype": 3}):
        fig, axis = composites.plt.subplots()
        axis.set_ylabel("EHH")
        fig.savefig(path)
        composites.plt.close(fig)
    with pytest.raises(ValueError, match="not embedded Unicode TrueType"):
        composites._validate_editable_panel_pdf(path)


def test_editable_validation_handles_thousands_of_marker_forms(tmp_path: Path, caplog) -> None:
    path = tmp_path / "dense_selection.pdf"
    with composites.plt.rc_context(composites.EDITABLE_PDF_RC):
        fig, axis = composites.plt.subplots()
        positions = np.arange(5_100)
        axis.scatter(positions, positions % 17, s=2)
        axis.set_ylabel("|iHS|")
        fig.savefig(path)
        composites.plt.close(fig)
    assert composites._validate_editable_panel_pdf(path)["editable_text"]
    assert "form XObject invocations" not in caplog.text


def test_font_change_redraws_selection_but_reuses_calculation_cache(tmp_path: Path, monkeypatch) -> None:
    for name in ("chr1.vcf.gz", "chr1.vcf.gz.tbi", "chr1.ihs", "genes.gtf", "mask.bed"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(figure3_ehh, "VCF_PATTERN", str(tmp_path / "{chrom}.vcf.gz"))
    monkeypatch.setattr(figure3_ehh, "IHS_PATTERN", str(tmp_path / "{chrom}.ihs"))
    monkeypatch.setattr(figure3_ehh, "GTF_PATH", str(tmp_path / "genes.gtf"))
    monkeypatch.setattr(figure3_ehh, "HARDMASK_PATH", str(tmp_path / "mask.bed"))
    monkeypatch.setattr(composites, "_sample_manifest", lambda: (None, "a" * 64))
    monkeypatch.setattr(composites.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)
    rendered: list[str] = []

    def render(payload):
        rendered.append(payload["cache_dir"])
        paths = {key: Path(value) for key, value in payload["paths"].items()}
        with composites.plt.rc_context(composites.EDITABLE_PDF_RC):
            fig, axis = composites.plt.subplots()
            axis.set_ylabel("EHH")
            for key, path in paths.items():
                if key != "manifest":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(path)
            composites.plt.close(fig)
        return {"variant_tag": payload["variant_tag"]}

    monkeypatch.setattr(composites, "_selection_worker", render)
    loci = pd.DataFrame([{
        "variant_tag": "1_100_A_G", "chrom": "chr1", "core_pos": 100,
        "ref": "A", "alt": "G", "gene": "TEST", "ihs_val": 3.0,
        "ihs_rs": "rs-test", "min_p_meta": 1e-6,
    }])

    def run():
        return composites.render_selection_loci(
            loci, loci, tmp_path / "selection", tmp_path / "cache",
            source_identity={"manifest_sha256": "b" * 64}, threads=1,
        )

    _, first = run()
    cache_file = Path(first["selection_cache_dir"]) / "existing_calculation.npz"
    np.savez(cache_file, values=np.array([1.0, 2.0]))
    original_hash = composites.sha256_file(cache_file)
    run()
    assert len(rendered) == 1
    monkeypatch.setitem(composites.EDITABLE_PDF_RC, "pdf.compression", 0)
    _, updated = run()
    assert len(rendered) == 2
    assert rendered[0] == rendered[1] == updated["selection_cache_dir"]
    assert composites.sha256_file(cache_file) == original_hash
