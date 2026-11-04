"""Execute the canonical figure notebooks and verify their plot artifacts.

The main notebooks remain the source of truth.  This helper calls them in order,
saves executed copies and text logs under ``analysis/figure_notebook_runs/``, and
checks the individual panel outputs and locus run manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Direct ``python scripts/run_all_figure_notebooks.py`` execution otherwise
    # exposes only scripts/ on sys.path, which breaks imports of scripts.common.
    sys.path.insert(0, str(REPO_ROOT))
MAIN_NOTEBOOKS = tuple(
    REPO_ROOT / "notebooks" / f"Figure_{number}.ipynb"
    for number in range(1, 6)
)
FIGURE2_1025_NOTEBOOK = MAIN_NOTEBOOKS[1]
DEFAULT_SEED = 20260715
DEFAULT_THREADS = 4
DEFAULT_CELL_TIMEOUT_SECONDS = 21_600
FIGURE_FORMATS = ("png", "pdf")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _normalise_notebook_name(value: str) -> Path:
    aliases: dict[str, Path] = {}
    for number, notebook in enumerate(MAIN_NOTEBOOKS, start=1):
        aliases[str(number)] = notebook
        aliases[f"figure_{number}"] = notebook
        aliases[f"figure_{number}.ipynb"] = notebook
    for alias in ("2_1025", "figure_2_1025", "figure_2_1025.ipynb"):
        aliases[alias] = FIGURE2_1025_NOTEBOOK

    key = value.strip().lower()
    if key in aliases:
        return aliases[key]

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"notebook must be inside {REPO_ROOT}: {candidate}") from exc
    return candidate


def resolve_notebooks(
    requested: Iterable[str] | None,
) -> list[Path]:
    """Resolve CLI notebook names, preserving order and removing duplicates."""
    notebooks = (
        [_normalise_notebook_name(value) for value in requested]
        if requested
        else list(MAIN_NOTEBOOKS)
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for notebook in notebooks:
        resolved = notebook.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _figure_number(notebook: Path) -> int | str | None:
    try:
        relative = notebook.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    for number, canonical in enumerate(MAIN_NOTEBOOKS, start=1):
        if relative == canonical.relative_to(REPO_ROOT):
            return number
    if relative == FIGURE2_1025_NOTEBOOK.relative_to(REPO_ROOT):
        return 2
    return None


def expected_artifacts(notebook: Path) -> list[Path]:
    """Return individual panel outputs; locus workflows also verify their own manifests."""
    number = _figure_number(notebook)
    if number is None:
        return []
    from scripts.common import config

    directory = REPO_ROOT / "figures" / f"figure{number}"
    if number in (1, 2):
        stems = [directory / "panels" / f"Figure_{number}{letter}"
                 for letter in config.FIGURES[f"figure{number}"]["panels"]]
    elif number == 3:
        stems = [directory / "Figure_3A_iHS_manhattan",
                 *(directory / f"Figure_3{letter}_{gene}_combined"
                   for letter, gene in (("B", "LCT"), ("C", "SLC24A5"), ("D", "ANKRD31"))),
                 directory / "panels" / "Figure_3E_enrichment"]
    elif number == 4:
        stems = [directory / name for name in (
            "Figure_4A_volcano", "Figure_4B_MAF_Effect",
            "Figure_4C_Discovery_Meta", "Figure_4D_forest")]
    else:
        return [directory / "loci" / name for name in ("run_manifest.json", "sv_selection_run_manifest.json")]
    return [stem.with_suffix(f".{ext}") for stem in stems for ext in FIGURE_FORMATS]


def validate_artifact(path: Path) -> tuple[bool, str]:
    """Perform cheap structural checks without adding image/PDF dependencies."""
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size < 32:
        return False, f"too small ({size} bytes)"

    with path.open("rb") as handle:
        prefix = handle.read(32)
        if path.suffix.lower() == ".png":
            if not prefix.startswith(b"\x89PNG\r\n\x1a\n"):
                return False, "invalid PNG signature"
            width = int.from_bytes(prefix[16:20], "big")
            height = int.from_bytes(prefix[20:24], "big")
            if width <= 0 or height <= 0:
                return False, f"invalid PNG dimensions ({width}x{height})"
        elif path.suffix.lower() == ".pdf":
            if not prefix.startswith(b"%PDF-"):
                return False, "invalid PDF signature"
            handle.seek(max(0, size - 2048))
            if b"%%EOF" not in handle.read():
                return False, "missing PDF EOF marker"
        elif path.suffix.lower() in {".csv", ".tsv"}:
            table_format = path.suffix[1:].upper()
            handle.seek(0)
            required_by_name = {
                "Figure_2_1025_all_outlier_regions.csv": {
                    "chrom",
                    "start",
                    "end",
                    "direction",
                    "peak_z",
                    "genes_in_region",
                    "region_id",
                },
                "Figure_2_1025_plrs_regions.csv": {
                    "chrom",
                    "start",
                    "end",
                    "region_id",
                    "n_snvs",
                    "r2_holdout",
                    "retained_for_phewas",
                },
                "Figure_2_1025_phewas_results.csv": {
                    "chrom",
                    "start",
                    "end",
                    "genes_in_region",
                    "phecode",
                    "phecode_string",
                    "beta_disc",
                    "p_disc",
                    "beta_rep",
                    "p_rep",
                    "beta_meta",
                    "p_meta",
                    "meta_significant",
                },
                "figure_summary.tsv": {"chrom", "start", "end", "region_id", "phecode"},
                "fine_mapping_results.csv": {"chrom", "start", "end", "region_id", "phecode"},
                "credible_set_variants.csv": {"region_id", "phecode", "variant_id", "credible_set_ids"},
                "credible_set_variants.tsv": {"region_id", "phecode", "variant_id", "credible_set_ids"},
            }
            try:
                with io.TextIOWrapper(handle, encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream, delimiter="\t" if table_format == "TSV" else ",", strict=True)
                    header = next(reader)
                    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                        return False, f"invalid {table_format} header"
                    missing = required_by_name.get(path.name, set()).difference(header)
                    if missing:
                        return False, f"{table_format} missing columns {sorted(missing)}"
                    n_rows = 0
                    for row in reader:
                        if not row:  # Ignore blank lines, not malformed records.
                            continue
                        if len(row) != len(header):
                            return False, f"{table_format} row {reader.line_num} has {len(row)} fields; expected {len(header)}"
                        n_rows += 1
                    # An empty credible-set membership output is valid when
                    # no variants belong to any credible set; its schema is required.
                    allow_empty = path.name in {"credible_set_variants.csv", "credible_set_variants.tsv"}
                    if not n_rows and not allow_empty:
                        return False, f"{table_format} has no data rows"
            except (StopIteration, UnicodeDecodeError, csv.Error):
                return False, f"invalid {table_format} content"
        else:
            return False, f"unsupported artifact extension {path.suffix}"
    return True, "ok"


def _artifact_snapshot(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_file():
            snapshot[str(path.resolve())] = {
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": _sha256(path),
            }
    return snapshot


def verify_artifacts(
    paths: Iterable[Path],
    *,
    before: dict[str, dict[str, Any]],
    run_started_epoch: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate expected files and record whether this run refreshed each one."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        valid, detail = validate_artifact(path)
        key = str(path.resolve())
        record: dict[str, Any] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "valid": valid,
            "validation": detail,
        }
        if valid:
            stat = path.stat()
            checksum = _sha256(path)
            previous = before.get(key)
            record.update(
                {
                    "size_bytes": stat.st_size,
                    "sha256": checksum,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(timespec="seconds"),
                    "refreshed_this_run": stat.st_mtime >= run_started_epoch - 2.0,
                    "content_changed": previous is None
                    or previous.get("sha256") != checksum,
                }
            )
        else:
            errors.append(f"{record['path']}: {detail}")
        records.append(record)
    return records, errors


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def write_notebook_log(notebook: Any, path: Path, header: Iterable[str]) -> None:
    """Write useful textual cell output while omitting embedded binary images."""
    lines = [*header, ""]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        lines.append(
            f"--- code cell {index} (execution_count={cell.get('execution_count')}) ---"
        )
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                lines.append(_as_text(output.get("text", "")).rstrip())
            elif output_type == "error":
                trace = output.get("traceback") or [
                    f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
                ]
                lines.append("\n".join(_as_text(item) for item in trace))
            elif output_type in {"display_data", "execute_result"}:
                text_value = output.get("data", {}).get("text/plain")
                if text_value is not None:
                    lines.append(_as_text(text_value).rstrip())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _save_notebook(notebook: Any, path: Path) -> None:
    import nbformat

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)


def _kernel_manager(kernel_name: str | None) -> Any:
    from jupyter_client import KernelManager

    if kernel_name:
        return KernelManager(kernel_name=kernel_name)

    class CurrentInterpreterKernelManager(KernelManager):
        """Launch the native Python kernel with the uv runner's interpreter."""

        def format_kernel_cmd(
            self, extra_arguments: list[str] | None = None
        ) -> list[str]:
            command = super().format_kernel_cmd(extra_arguments)
            command[0] = sys.executable
            return command

    return CurrentInterpreterKernelManager(kernel_name="python3")


def _selected_kernel(notebook: Path, args: Any) -> str | None:
    """Resolve the global or per-figure kernel override for one notebook."""
    figure_number = _figure_number(notebook)
    if notebook.name == "Admixture_reference.ipynb" and args.admixture_kernel:
        return args.admixture_kernel
    return args.kernel_name


def validate_requested_kernels(
    notebooks: Iterable[Path],
    args: Any,
    *,
    installed: set[str] | None = None,
) -> None:
    """Fail before execution when an explicit Jupyter kernel is unavailable."""
    requested: dict[str, list[Path]] = {}
    for notebook in notebooks:
        kernel_name = _selected_kernel(notebook, args)
        if kernel_name:
            requested.setdefault(kernel_name, []).append(notebook)
    if not requested:
        return
    if installed is None:
        from jupyter_client.kernelspec import KernelSpecManager

        installed = set(KernelSpecManager().find_kernel_specs())
    missing = {
        name: paths for name, paths in requested.items() if name not in installed
    }
    if not missing:
        return
    details = "; ".join(
        f"{name!r} requested for "
        + ", ".join(str(path.relative_to(REPO_ROOT)) for path in paths)
        for name, paths in sorted(missing.items())
    )
    guidance = "List installed kernels with: jupyter kernelspec list"
    if "allofus-figure4" in missing:
        guidance = (
            "Install the required runtime first with: "
            "uv run --with-requirements requirements.txt python "
            "scripts/setup_admixture_runtime.py"
        )
    raise SystemExit(f"Requested Jupyter kernel is not installed: {details}. {guidance}")


def execute_notebook(
    notebook_path: Path,
    *,
    run_dir: Path,
    timeout_seconds: int,
    kernel_name: str | None,
) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient
    from nbformat.validator import MissingIDFieldWarning

    relative = notebook_path.relative_to(REPO_ROOT)
    slug = "__".join(relative.with_suffix("").parts)
    executed_path = run_dir / "executed" / f"{slug}.executed.ipynb"
    log_path = run_dir / "logs" / f"{slug}.log"
    expected = expected_artifacts(notebook_path)
    before = _artifact_snapshot(expected)
    started_epoch = time.time()
    started_utc = _utc_now()
    # Older tracked notebooks predate cell IDs. nbformat currently normalises
    # them in memory; keep the source notebooks untouched and the runner output
    # quiet while ensuring every executed copy receives valid IDs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingIDFieldWarning)
        notebook = nbformat.read(notebook_path, as_version=4)
    status = "completed"
    error: str | None = None

    try:
        client = NotebookClient(
            notebook,
            km=_kernel_manager(kernel_name),
            timeout=None if timeout_seconds == 0 else timeout_seconds,
            allow_errors=False,
            allow_stdin=False,
            record_timing=True,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
    except BaseException as exc:  # noqa: BLE001 - persist logs for Ctrl-C too
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = "".join(traceback.format_exception(exc)).rstrip()
    finally:
        _save_notebook(notebook, executed_path)
        write_notebook_log(
            notebook,
            log_path,
            (
                f"source_notebook={relative}",
                f"started_utc={started_utc}",
                f"finished_utc={_utc_now()}",
                f"status={status}",
                f"kernel={kernel_name or ('current interpreter: ' + sys.executable)}",
                f"cell_timeout_seconds={timeout_seconds}",
                *(f"runner_error={error}" if error else "",),
            ),
        )

    artifacts: list[dict[str, Any]] = []
    artifact_errors: list[str] = []
    if status == "completed":
        # Corrected Figure 2 can create its selection CSV during this run.
        expected = expected_artifacts(notebook_path)
    if status == "completed" and expected:
        artifacts, artifact_errors = verify_artifacts(
            expected,
            before=before,
            run_started_epoch=started_epoch,
        )
        if artifact_errors:
            status = "invalid_artifacts"
            error = "; ".join(artifact_errors)

    finished_epoch = time.time()
    return {
        "source": str(relative),
        "source_sha256": _sha256(notebook_path),
        "kernel": kernel_name or "current-interpreter",
        "status": status,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "duration_seconds": round(finished_epoch - started_epoch, 3),
        "executed_notebook": str(executed_path.relative_to(REPO_ROOT)),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "expected_artifact_count": len(expected),
        "artifacts": artifacts,
        "error": error,
    }


def _configure_environment(threads: int) -> dict[str, str]:
    values = {
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": str(DEFAULT_SEED),
        "AOU_FIGURE_SEED": str(DEFAULT_SEED),
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "POLARS_MAX_THREADS": str(threads),
    }
    os.environ.update(values)
    return values


def _provenance_environment() -> dict[str, str]:
    prefixes = ("AOU_", "SUPP_")
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(prefixes)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        action="append",
        metavar="NAME_OR_PATH",
        help=(
            "Run only this notebook (repeatable). Accepts 1, Figure_1, or a "
            "repository-relative .ipynb path. Default: Figure_1 through Figure_5."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Threads exposed to numerical libraries (default: {DEFAULT_THREADS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CELL_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Maximum time for each notebook cell; 0 disables the timeout "
            f"(default: {DEFAULT_CELL_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--kernel-name",
        help=(
            "Installed Jupyter kernel to use. By default the current uv Python "
            "interpreter is forced for reproducibility."
        ),
    )
    parser.add_argument(
        "--admixture-kernel",
        help=(
            "Kernel override for Admixture_reference.ipynb only (for example, the "
            "Workbench allofus-figure4 kernel). Other notebooks keep --kernel-name."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "analysis" / "figure_notebook_runs",
        help="Parent directory for executed notebooks, logs, and run manifests.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique run directory name (default: UTC timestamp).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first notebook failure instead of running independent figures.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the resolved notebook order and exit without executing anything.",
    )
    return parser


def report_plot_locations(result: dict[str, Any]) -> None:
    """Expose verified plot folders in the outer shell cell, outside kernel logs."""
    folders: dict[Path, dict[str, int]] = {}
    for artifact in result.get("artifacts", []):
        path = REPO_ROOT / artifact["path"]
        extension = path.suffix.lower().removeprefix(".")
        if artifact.get("valid") and extension in FIGURE_FORMATS:
            counts = folders.setdefault(path.parent, {"png": 0, "pdf": 0})
            counts[extension] += 1
    for folder, counts in sorted(folders.items()):
        print(f"  Saved plots: {folder} ({counts['png']} PNG, {counts['pdf']} PDF)")
    if result.get("executed_notebook"):
        print(f"  Executed notebook: {REPO_ROOT / result['executed_notebook']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.timeout < 0:
        raise SystemExit("--timeout must be 0 or greater")

    try:
        notebooks = resolve_notebooks(
            args.notebook,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    missing = [path for path in notebooks if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing notebook(s): " + ", ".join(str(path) for path in missing)
        )

    print("Notebook execution order:")
    for path in notebooks:
        print(f"  - {path.relative_to(REPO_ROOT)}")
    if args.list:
        return 0

    validate_requested_kernels(notebooks, args)

    os.chdir(REPO_ROOT)
    configured_environment = _configure_environment(args.threads)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.run_root.expanduser()
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_dir = run_root.resolve() / run_id
    if run_dir.exists():
        raise SystemExit(
            f"Run directory already exists: {run_dir}. Choose a unique --run-id."
        )
    run_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "repo_root": str(REPO_ROOT),
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_porcelain": _git(["status", "--porcelain=v1"]),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "parameters": {
            "notebooks": [str(path.relative_to(REPO_ROOT)) for path in notebooks],
            "threads": args.threads,
            "cell_timeout_seconds": args.timeout,
            "kernel_name": args.kernel_name or "current-interpreter",
            "admixture_kernel": args.admixture_kernel,
            "fail_fast": args.fail_fast,
            "run_dir": str(run_dir),
            "seed": DEFAULT_SEED,
        },
        "configured_environment": configured_environment,
        "aou_and_supp_environment": _provenance_environment(),
        "notebook_results": [],
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    for index, notebook in enumerate(notebooks, start=1):
        print(f"\n[{index}/{len(notebooks)}] Executing {notebook.relative_to(REPO_ROOT)}")
        selected_kernel = _selected_kernel(notebook, args)
        result = execute_notebook(
            notebook,
            run_dir=run_dir,
            timeout_seconds=args.timeout,
            kernel_name=selected_kernel,
        )
        manifest["notebook_results"].append(result)
        _write_json_atomic(manifest_path, manifest)
        print(
            f"  {result['status']} in {result['duration_seconds']:.1f}s; "
            f"log={result['log']}"
        )
        report_plot_locations(result)
        if result["error"]:
            print(f"  error: {result['error']}", file=sys.stderr)
        if result["status"] == "interrupted" or (
            result["status"] != "completed" and args.fail_fast
        ):
            break

    statuses = [result["status"] for result in manifest["notebook_results"]]
    manifest["status"] = (
        "completed" if len(statuses) == len(notebooks) and set(statuses) == {"completed"}
        else "failed"
    )
    manifest["finished_utc"] = _utc_now()
    _write_json_atomic(manifest_path, manifest)

    print(f"\nRun status: {manifest['status']}")
    print(f"Manifest: {manifest_path}")
    print(f"Plots: {REPO_ROOT / 'figures'}")
    if any(path.name in {"Figure_2.ipynb", "Admixture_loci.ipynb"} for path in notebooks):
        print("Figure 2 saves images to disk; notebook output contains paths and counts only.")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
