from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_all_figure_notebooks.py"
SPEC = importlib.util.spec_from_file_location("run_all_figure_notebooks", SCRIPT_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_default_notebook_order_is_all_five_main_figures() -> None:
    notebooks = runner.resolve_notebooks(None)
    assert [path.name for path in notebooks] == [f"Figure_{i}.ipynb" for i in range(1, 6)]



def test_aliases_are_deduplicated_and_supplement_is_opt_in() -> None:
    notebooks = runner.resolve_notebooks(
        ["1", "Figure_1.ipynb", "2"],
    )
    assert [path.name for path in notebooks] == ["Figure_1.ipynb", "Figure_2.ipynb"]


def test_parser_accepts_a_admixture_only_kernel_override() -> None:
    args = runner.build_parser().parse_args(
        ["--admixture-kernel", "allofus-figure4"]
    )
    assert args.kernel_name is None
    assert args.admixture_kernel == "allofus-figure4"


def test_figure2_uses_the_standard_uv_interpreter() -> None:
    args = runner.build_parser().parse_args(
        ["--notebook", "5", "--admixture-kernel", "allofus-figure4"]
    )
    assert runner._selected_kernel(runner.MAIN_NOTEBOOKS[4], args) is None


def test_parser_rejects_the_removed_figure2_kernel_override() -> None:
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--figure2_sources-kernel", "allofus-figure4"]
        )


def test_main_artifact_contract_contains_individual_panels():
    expected_counts = {1: 16, 2: 12, 3: 10, 4: 8, 5: 2}
    for number, count in expected_counts.items():
        paths = runner.expected_artifacts(runner.MAIN_NOTEBOOKS[number - 1])
        assert len(paths) == count
        assert not any(path.name in {f"Figure_{number}.png", f"Figure_{number}.pdf"} for path in paths)


def test_artifact_validation_checks_png_and_pdf_structure(tmp_path: Path) -> None:
    png = tmp_path / "plot.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + (100).to_bytes(4, "big")
        + (80).to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
        + b"padding"
    )
    pdf = tmp_path / "plot.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + b"0" * 32 + b"\n%%EOF\n")

    assert runner.validate_artifact(png) == (True, "ok")
    assert runner.validate_artifact(pdf) == (True, "ok")

    png.write_bytes(b"not a png" * 5)
    pdf.write_bytes(b"%PDF-1.4\n" + b"0" * 32)
    assert runner.validate_artifact(png)[0] is False
    assert runner.validate_artifact(pdf)[0] is False


@pytest.mark.parametrize("name", ["figure_summary.tsv", "credible_set_variants.tsv"])
def test_figure2_tsv_artifacts_are_accepted(tmp_path, name, monkeypatch):
    path = tmp_path / name
    path.write_text("chrom\tstart\tend\tregion_id\tphecode\tvariant_id\tcredible_set_ids\n"
                    "1\t100\t200\tr1\tSS_001\tchr1:123:A:G\t1\n", encoding="utf-8")
    assert runner.validate_artifact(path) == (True, "ok")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    records, errors = runner.verify_artifacts([path], before={}, run_started_epoch=0)
    assert errors == []
    assert records[0]["valid"] and records[0]["sha256"]


@pytest.mark.parametrize("extension,delimiter", [("csv", ","), ("tsv", "\t")])
def test_empty_credible_set_table_is_valid_but_empty_summary_is_not(tmp_path, extension, delimiter):
    path = tmp_path / f"credible_set_variants.{extension}"
    path.write_text(delimiter.join(["region_id", "phecode", "variant_id", "credible_set_ids"]) + "\n", encoding="utf-8")
    assert runner.validate_artifact(path) == (True, "ok")
    summary = tmp_path / "figure_summary.tsv"
    summary.write_text("chrom\tstart\tend\tregion_id\tphecode\n", encoding="utf-8")
    assert runner.validate_artifact(summary) == (False, "TSV has no data rows")


@pytest.mark.parametrize("content", [
    "chrom,start,end,region_id,phecode\n1,100,200,r1,SS_001\n",
    "chrom\tstart\tend\tregion_id\tphecode\n1\t100\t200\tr1\n",
    'chrom\tstart\tend\tregion_id\tphecode\n1\t100\t200\tr1\t"unfinished\n',
    "chrom\tstart\tend\tregion_id\tregion_id\n1\t100\t200\tr1\tr1\n",
])
def test_malformed_tsv_artifacts_are_rejected(tmp_path, content):
    path = tmp_path / "figure_summary.tsv"
    path.write_text(content, encoding="utf-8")
    assert runner.validate_artifact(path)[0] is False


def test_admixture_notebook_runs_summary_from_workbench_products():
    notebook = json.loads((REPO_ROOT / "notebooks/Admixture_reference.ipynb").read_text(encoding="utf-8"))
    summary = next("".join(c["source"]) for c in notebook["cells"] if "admixture_summary.run" in "".join(c["source"]))
    assert summary.startswith("summary_results = admixture_summary.run")


def test_figure3_installs_vcf_dependency():
    notebook = json.loads((REPO_ROOT / "notebooks/Figure_3.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert 'deps.ensure([("allel", "scikit-allel")])' in source
    assert 'if figure3_ehh.on_workbench():' not in source


def test_completed_runner_reports_actual_locus_folder(tmp_path, monkeypatch, capsys):
    notebook = tmp_path / "notebooks" / "Figure_2.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "resolve_notebooks", lambda *args, **kwargs: [notebook])
    monkeypatch.setattr(runner, "validate_requested_kernels", lambda *args: None)
    monkeypatch.setattr(runner, "_selected_kernel", lambda *args: None)
    monkeypatch.setattr(runner, "_configure_environment", lambda *args: {})
    monkeypatch.setattr(runner, "_git", lambda *args: "synthetic")
    monkeypatch.chdir(tmp_path)
    artifacts = [
        {"path": f"figures/figure2_sources/loci/locus_{i}.{ext}", "valid": True}
        for i in range(13) for ext in ("png", "pdf")
    ]
    monkeypatch.setattr(runner, "execute_notebook", lambda *args, **kwargs: {
        "status": "completed", "duration_seconds": 1.0, "error": None,
        "log": "analysis/run.log", "artifacts": artifacts,
        "executed_notebook": "analysis/executed/Figure_2.ipynb",
    })
    assert runner.main(["--notebook", "5", "--run-root", str(tmp_path / "runs")]) == 0
    output = capsys.readouterr().out
    assert str(tmp_path / "figures" / "figure2_sources" / "loci") in output
    assert "13 PNG, 13 PDF" in output
    assert "saves images to disk" in output
