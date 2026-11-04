"""Run the complete admixture reference local-ancestry analysis on Researcher Workbench."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from . import admixture_finemap, admixture_lai, admixture_phewas, admixture_plrs


PYTHON_DEPENDENCIES = {
    "cyvcf2": "cyvcf2>=0.31",
    "sklearn": "scikit-learn>=1.3",
    "statsmodels": "statsmodels>=0.14",
    "scipy": "scipy>=1.10",
}


def ensure_dependencies() -> None:
    missing = [(module, package) for module, package in PYTHON_DEPENDENCIES.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return
    print("Installing missing admixture reference dependencies:",
          " ".join(package for _, package in missing))
    for module, package in missing:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Could not install required admixture reference dependency {package!r} "
                f"into {sys.executable}. Install it in the active Workbench "
                "kernel and rerun the cell."
            ) from exc
    importlib.invalidate_caches()
    unresolved = [module for module, _ in missing
                  if importlib.util.find_spec(module) is None]
    if unresolved:
        raise ModuleNotFoundError(
            "admixture reference dependencies remained unavailable after installation: "
            + ", ".join(unresolved)
        )


def _has_prediction(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import pandas as pd
        return not pd.read_csv(path, sep="\t", nrows=1).empty
    except Exception:
        return False


def run_all(*, force: bool = False, run_phewas: bool = True,
            run_finemap: bool = True) -> dict:
    ensure_dependencies()
    discovery_predictions = admixture_plrs.analysis_dir() / \
        "eur_dosage_predictions.discovery.tsv.gz"
    cached_discovery = None
    if run_phewas and not force and _has_prediction(discovery_predictions):
        cached_discovery = admixture_phewas.load_cached_discovery()

    if cached_discovery is not None:
        print("\n" + "=" * 72)
        print("FIGURE 4 RESUME - Stage 3 complete; starting from Stage 4")
        print("=" * 72)
        print(
            "[admixture_reference] reused cached Stage 3 associations and applied "
            f"p < {admixture_phewas.DISCOVERY_P_THRESHOLD:.0e}"
        )
        lai_outputs = {"status": "cached-through-stage3"}
        plrs_outputs = {
            "summary": admixture_plrs.analysis_dir() / "plrs_models_summary.tsv",
            "discovery_predictions": discovery_predictions,
        }
    else:
        print("\n" + "=" * 72)
        print("FIGURE 4 STAGE 1 - marker-level LAI scan and outlier regions")
        print("=" * 72)
        lai_outputs = admixture_lai.run(force=force)

        print("\n" + "=" * 72)
        print("FIGURE 4 STAGE 2 - generate EUR dosage for all discovery regions")
        print("=" * 72)
        plrs_outputs = admixture_plrs.run_discovery(force=force)

    phewas_outputs = {}
    if run_phewas and _has_prediction(discovery_predictions):
        if cached_discovery is None:
            print("\n" + "=" * 72)
            print("FIGURE 4 STAGE 3 - PheWAS of all discovery regions by EUR dosage")
            print("=" * 72)
            discovery = admixture_phewas.run_discovery(force=force)
        else:
            discovery = cached_discovery
        discovery_output = admixture_phewas.analysis_dir() / "phewas.discovery.tsv.gz"
        phewas_outputs = {"discovery": discovery_output}
        hits = admixture_phewas.discovery_hits(discovery)
        if hits.empty:
            print(
                "[admixture_reference] no discovery hits at "
                f"p < {admixture_phewas.DISCOVERY_P_THRESHOLD:.0e}"
            )
            phewas_outputs.update(admixture_phewas.write_empty_replication_outputs())
        else:
            replication_regions = hits["region_id"].drop_duplicates().tolist()
            print("\n" + "=" * 72)
            print("FIGURE 4 STAGE 4 - generate discovery-significant replication regions")
            print("=" * 72)
            replication_predictions = admixture_plrs.predict_replication_regions(
                replication_regions, force=force
            )
            plrs_outputs["replication_predictions"] = replication_predictions

            print("\n" + "=" * 72)
            print("FIGURE 4 STAGE 5 - targeted replication PheWAS and IVW meta-analysis")
            print("=" * 72)
            phewas_outputs.update(admixture_phewas.run_replication(
                discovery, replication_predictions, force=force
            ))
    elif run_phewas:
        print("[admixture_reference] PheWAS skipped because discovery has no retained PLRS predictions")

    finemap_outputs = {}
    if run_finemap and "meta" in phewas_outputs:
        print("\n" + "=" * 72)
        print("FIGURE 4 STAGE 6 - regional SuSiE fine-mapping")
        print("=" * 72)
        finemap_outputs = admixture_finemap.run(force=force)

    return {
        "lai": lai_outputs,
        "plrs": plrs_outputs,
        "phewas": phewas_outputs,
        "finemap": finemap_outputs,
    }


def ensure_admixture_outputs(*, force: bool = False) -> dict:
    """Run or resume the Workbench analysis using each stage's completion checks."""
    return run_all(force=force)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute cached chromosome/model/results products")
    parser.add_argument("--skip-phewas", action="store_true")
    parser.add_argument("--skip-finemap", action="store_true")
    args = parser.parse_args(argv)
    return run_all(
        force=args.force,
        run_phewas=not args.skip_phewas,
        run_finemap=not args.skip_finemap,
    )


if __name__ == "__main__":
    main()
