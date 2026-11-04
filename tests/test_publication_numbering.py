"""Regression checks for publication numbering and existing cache isolation."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.common import config

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_ids_reuse_the_correct_distinct_cache_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANALYSIS_DIR", tmp_path / "analysis")
    expected = {"figure2": "figure5_v2", "figure3": "figure2", "figure4": "figure3",
                "figure2_sources": "figure5", "admixture_reference": "figure4"}
    for manuscript_id, cache_id in expected.items():
        assert config.analysis_dir(manuscript_id) == tmp_path / "analysis" / cache_id
    assert len({config.analysis_dir(key) for key in expected}) == len(expected)








def test_local_module_imports_resolve_after_renaming():
    """Resolve local imports without importing optional scientific runtimes."""
    failures = []
    for path in (ROOT / "scripts").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = list(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package[:len(package) - node.level + 1]
                if node.module:
                    base += node.module.split(".")
            elif node.module and node.module.startswith("scripts"):
                base = node.module.split(".")
            else:
                continue
            folder = ROOT.joinpath(*base)
            if folder.with_suffix(".py").is_file():
                continue  # importing attributes from an existing module
            if not folder.is_dir():
                failures.append(f"{path.name}:{node.lineno}: {'.'.join(base)}")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = folder / alias.name
                if not candidate.is_dir() and not candidate.with_suffix(".py").is_file():
                    failures.append(f"{path.name}:{node.lineno}: {'.'.join(base)}.{alias.name}")
    assert not failures, "\n".join(failures)


def test_main_notebooks_are_panel_references_without_saved_output():
    for number in range(1, 6):
        nb = json.loads((ROOT / f"notebooks/Figure_{number}.ipynb").read_text(encoding="utf-8"))
        assert not nb.get("metadata", {}).get("publication", {}).get("reference_only")
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "assembly" not in text
        assert "Q–Q" not in text
        assert "composite_image_paths" not in text
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert not cell["outputs"]
                assert cell["execution_count"] is None
    nb = json.loads((ROOT / "notebooks/Figure_3.ipynb").read_text(encoding="utf-8"))
    headings = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert headings[1].startswith("## Panel A — Genome-wide iHS selection scan")
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "build_figure3" not in code
    for letter in "BCD":
        assert f'panel_letters=("{letter}",)' in code


def test_required_support_modules_live_with_their_main_figures():
    for name in ("figure1_model_fits", "figure2_discovery_phewas", "figure4_discovery"):
        assert (ROOT / "scripts" / f"{name}.py").is_file()
    assert not list((ROOT / "scripts/supplementary").glob("*.py"))
    assert not (ROOT / "supp_figures/supp_figures_new.ipynb").exists()
