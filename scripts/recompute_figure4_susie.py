"""Run the primary 10k-discovery SuSiE cell on Researcher Workbench.

The repository's fine-mapping notebook also contains downstream exploratory
cells and an older single-cohort workflow. This runner executes only its first,
current discovery-only analysis cell for either the default meta-significant
set or an explicit ``--task-table``, writes products to
``config.SUSIE_FINEMAP_DIR`` on the Workbench VM, and verifies the complete
result set.

Run from the repository root on Workbench::

    python -m scripts.recompute_figure4_susie
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd

from . import figure4_finemap
from .common import config, gcs


def _primary_cell_source(notebook: Path) -> str:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.get("cell_type") == "code" and "susie-discovery" in tags:
            return source
    raise RuntimeError(f"primary 10k-discovery SuSiE cell not found in {notebook}")


def _task_table_loci(task_table: Path) -> list[dict]:
    """Resolve an explicit notebook task table into the verification schema."""
    task_table = task_table.expanduser().resolve()
    if not task_table.is_file() or task_table.stat().st_size == 0:
        raise FileNotFoundError(f"SuSiE task table is missing or empty: {task_table}")
    frame = pd.read_csv(task_table, sep="\t", dtype=str)
    required = {"variant_tag", "phecode", "meta_significant"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SuSiE task table is missing required columns: {missing}")
    selected = frame["meta_significant"].astype(str).str.strip().str.upper().isin(
        {"TRUE", "T", "1", "YES", "Y"}
    )
    frame = frame[selected].drop_duplicates(["variant_tag", "phecode"], keep="first")
    if frame.empty:
        raise ValueError("SuSiE task table contains no selected variant-phecode rows")

    sex_lookup = figure4_finemap._sex_scope_lookup()
    loci: list[dict] = []
    for _, row in frame.iterrows():
        parsed = figure4_finemap._parse_variant_tag(row["variant_tag"])
        if parsed is None:
            raise ValueError(f"Invalid SuSiE task variant_tag: {row['variant_tag']!r}")
        chrom, pos, ref, alt = parsed
        phecode = str(row["phecode"]).strip()
        sex_scope = sex_lookup.get(phecode)
        if sex_scope not in {"Both", "Female", "Male"}:
            raise ValueError(f"No valid sex scope for SuSiE task phecode {phecode}")
        name = str(row.get("phecode_string", phecode)).strip() or phecode
        gene = figure4_finemap._first_gene(
            row.get("gene_in_window"), f"chr{chrom.removeprefix('chr')}:{pos}"
        )
        p_value = pd.to_numeric(
            row.get("p_meta", row.get("p_d", row.get("p", float("nan")))),
            errors="coerce",
        )
        loci.append(
            {
                "chrom": chrom,
                "lead_pos": pos,
                "lead_ref": ref,
                "lead_alt": alt,
                "phecode": phecode,
                "phecode_name": name,
                "lead_gene": gene,
                "finemap_gene": gene,
                "bold_gene": None,
                "sex_scope": sex_scope,
                "meta_p": float(p_value) if pd.notna(p_value) else float("nan"),
                "variant_tag": str(row["variant_tag"]).strip(),
                "source_kind": f"explicit SuSiE task table {task_table}",
            }
        )
    return loci


def _requested_loci(
    *,
    include_v5: bool = False,
    task_table: Path | None = None,
) -> list[dict]:
    if include_v5 and task_table is not None:
        raise ValueError("include_v5 and task_table are mutually exclusive")
    loci = (
        _task_table_loci(task_table)
        if task_table is not None
        else figure4_finemap.meta_significant_loci()
    )
    if include_v5:
        loci.extend(figure4_finemap.linked_sv_loci().values())
    unique = {}
    for locus in loci:
        key = (locus["variant_tag"], locus["phecode"])
        unique[key] = locus
    return list(unique.values())


def _verify_figure4_outputs(
    *,
    include_v5: bool = False,
    task_table: Path | None = None,
) -> list[Path]:
    verified: list[Path] = []
    missing = []
    loci = _requested_loci(include_v5=include_v5, task_table=task_table)
    if not loci:
        raise RuntimeError("requested SuSiE task set contains no loci")
    for locus in loci:
        discovery_result = figure4_finemap._find_discovery_result(locus)
        if discovery_result is None:
            missing.append(
                f"{locus['chrom']}:{locus['lead_pos']} x {locus['phecode']}"
            )
            continue
        figure4_finemap.validate_discovery_contract(
            discovery_result,
            expected_scope=locus["sex_scope"],
            expected_locus=locus,
        )
        if task_table is not None:
            plot_path = figure4_finemap._find_local_artifact(locus, "plot")
            if plot_path is None:
                missing.append(
                    f"{locus['chrom']}:{locus['lead_pos']} x {locus['phecode']} plot"
                )
                continue
            plot = Path(plot_path)
            if plot.stat().st_size == 0 or plot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Invalid SuSiE task plot: {plot}")
        verified.append(Path(discovery_result))
    if missing:
        raise RuntimeError(
            "SuSiE did not create discovery results for all requested hits: "
            + ", ".join(missing)
        )
    return verified


def _stage_figure4_outputs(
    *,
    include_v5: bool = False,
    task_table: Path | None = None,
) -> None:
    """Stage any existing bucket products before deciding to recompute."""
    # A no-match warning is expected when recomputation is required; replace
    # the four low-level copy warnings with the single clear status message in
    # ``ensure_figure4_outputs`` below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"\[figure4\] could not stage .* artifacts.*"
        )
        for locus in _requested_loci(include_v5=include_v5, task_table=task_table):
            figure4_finemap._stage_remote_artifacts(locus, "result", force=False)
            if task_table is not None:
                figure4_finemap._stage_remote_artifacts(locus, "plot", force=False)


def ensure_figure4_outputs(
    *,
    include_v5: bool = False,
    task_table: Path | None = None,
) -> list[Path]:
    """Return verified outputs, recomputing once only when they are absent.

    This is the notebook entry point. A repeated Figure 4 notebook run is
    cheap: a complete set of valid local products is returned immediately;
    otherwise existing bucket products are staged and checked. The expensive
    discovery SuSiE cell runs only when a requested hit remains absent.
    """
    try:
        return _verify_figure4_outputs(include_v5=include_v5, task_table=task_table)
    except (RuntimeError, ValueError):
        pass

    _stage_figure4_outputs(include_v5=include_v5, task_table=task_table)
    try:
        verified = _verify_figure4_outputs(include_v5=include_v5, task_table=task_table)
    except (RuntimeError, ValueError):
        print("Figure 4 SuSiE products are missing or invalid; recomputing them now.")
        return main(include_v5=include_v5, task_table=task_table)

    print("Using verified Figure 4 SuSiE products staged from the workspace bucket.")
    return verified


def _write_task_table(loci: list[dict]) -> Path:
    """Write the exact requested locus set in the notebook's input schema."""
    rows = []
    for locus in loci:
        rows.append(
            {
                "variant_tag": locus["variant_tag"],
                "phecode": locus["phecode"],
                "phecode_string": locus["phecode_name"],
                "meta_significant": "TRUE",
                "gene_in_window": locus.get("lead_gene", ""),
                "p_meta": locus.get("meta_p"),
            }
        )
    path = config.analysis_dir("figure4") / "figure4_v5_susie_tasks.tsv"
    pd.DataFrame(rows).drop_duplicates(
        ["variant_tag", "phecode"], keep="first"
    ).to_csv(path, sep="\t", index=False)
    print(f"Figure 4 SuSiE task table: {path} ({len(rows)} requested loci)")
    return path


def main(
    *,
    include_v5: bool = False,
    task_table: Path | None = None,
) -> list[Path]:
    if not gcs.available():
        raise RuntimeError(
            "SuSiE recomputation requires Workbench bucket access via gcloud or gsutil"
        )
    notebook = config.NOTEBOOKS_DIR / "susie_finemap_SNVs.ipynb"
    source = _primary_cell_source(notebook)
    config.figure_dir("figure4")
    workdir = Path(config.SUSIE_FINEMAP_DIR)
    workdir.mkdir(parents=True, exist_ok=True)

    # The panel covariate object is intentionally no longer required. Rebuild
    # the discovery covariate merge from the latest bucket-staged age/sex and
    # 20-PC inputs whenever a missing/invalid result triggers this runner.
    refresh = [
        workdir / "disc_covariates.tsv",
        workdir / "workspace_inputs" / "phewas_sv_v5" / "covariates_base.tsv",
        workdir / "workspace_inputs" / "phewas_sv_v5" / "pca_scores_afr.tsv.bgz",
    ]
    for path in refresh:
        path.unlink(missing_ok=True)

    previous_cwd = Path.cwd()
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook),
    }
    previous_tasks = os.environ.get("AOU_FIGURE3_TASKS_TSV")
    task_path: Path | None = None
    if task_table is not None:
        task_path = task_table.expanduser().resolve()
        _requested_loci(task_table=task_path)
    elif include_v5:
        task_path = _write_task_table(_requested_loci(include_v5=True))
    if task_path is not None:
        os.environ["AOU_FIGURE3_TASKS_TSV"] = str(task_path)
    try:
        os.chdir(config.REPO_ROOT)
        exec(compile(source, str(notebook), "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(previous_cwd)
        if task_path is not None:
            if previous_tasks is None:
                os.environ.pop("AOU_FIGURE3_TASKS_TSV", None)
            else:
                os.environ["AOU_FIGURE3_TASKS_TSV"] = previous_tasks

    verified = _verify_figure4_outputs(include_v5=include_v5, task_table=task_table)
    print("Verified Figure 4 SuSiE outputs:")
    for path in verified:
        print(f"  {path}")
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="verify local/bucket products first and run SuSiE only if needed",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--include-v5",
        action="store_true",
        help="also compute the chr22/chr2 S9-linked loci used by Figure 4 v5",
    )
    task_group.add_argument(
        "--task-table",
        type=Path,
        default=None,
        help="run exactly the selected rows in an explicit notebook-format task TSV",
    )
    args = parser.parse_args()
    if args.reuse_existing:
        ensure_figure4_outputs(include_v5=args.include_v5, task_table=args.task_table)
        _verify_figure4_outputs(include_v5=args.include_v5, task_table=args.task_table)
    else:
        main(include_v5=args.include_v5, task_table=args.task_table)
