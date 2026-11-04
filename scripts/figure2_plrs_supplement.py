"""Cache-only Figure 2 supplements: regional model size and nominal-hit calibration.

Run with ``python -m scripts.figure2_plrs_supplement`` after the corrected
Figure_2_1025 analysis. No staging, model fitting, or inline display occurs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import admixture_phewas, figure2_1025 as corrected
from .common import config, samples, style

PLOT_CONTRACT = "figure5_plrs_supplement_v1"
FORMATS = ("png", "pdf")


def output_dir() -> Path:
    return config.figure_dir("figure2_sources") / "supplementary" / "plrs"


def prepare_regions(models, holdout, hits):
    """Require each nominal region's own complete, numerically matching holdout."""
    required = {"region_id", "chrom", "start", "end", "n_snvs", "n_test", "r2_holdout"}
    if not required.issubset(models) or models.empty:
        raise ValueError("Missing regional PLRS model summary columns or rows")
    if models.region_id.isna().any() or models.region_id.duplicated().any():
        raise ValueError("Missing or duplicated PLRS region IDs")
    for column in ("n_snvs", "n_test", "r2_holdout"):
        values = pd.to_numeric(models[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite PLRS {column}")
        if column != "r2_holdout" and ((values < 1).any() or (values % 1 != 0).any()):
            raise ValueError(f"Invalid PLRS {column}")
    if (models.r2_holdout > 1).any():
        raise ValueError("PLRS test R2 exceeds 1")
    columns = {"region_id", "sample_id", "true_eur_dosage", "predicted_eur_dosage"}
    if not columns.issubset(holdout):
        raise ValueError("Missing PLRS holdout columns")
    selected = hits[["region_id", "chrom", "start", "end"]].drop_duplicates()
    selected = selected.merge(models, on=["region_id", "chrom", "start", "end"],
                              how="left", validate="one_to_one", indicator=True)
    if not selected["_merge"].eq("both").all():
        raise ValueError("Nominal regions do not match corrected PLRS models")
    selected = selected.drop(columns="_merge").sort_values(["chrom", "start", "end"])
    rows = []
    for region in selected.to_dict("records"):
        frame = holdout.loc[holdout.region_id.eq(region["region_id"])]
        if len(frame) != region["n_test"] or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Incomplete or duplicate holdout for {region['region_id']}")
        values = frame[["true_eur_dosage", "predicted_eur_dosage"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 2).any():
            raise ValueError(f"Invalid holdout dosage for {region['region_id']}")
        truth, predicted = values.T
        total = float(np.sum((truth - truth.mean()) ** 2))
        error = float(np.sum((truth - predicted) ** 2))
        # Match sklearn r2_score(force_finite=True), including constant targets.
        r2 = 1 - error / total if total > 0 else float(error == 0)
        if not np.isclose(r2, region["r2_holdout"], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Holdout R2 disagrees with model summary for {region['region_id']}")
        associations = hits.loc[hits.region_id.eq(region["region_id"])]
        rows.append({**region, "n_nominal_associations": len(associations),
                     "phecodes": ";".join(associations.phecode.astype(str)),
                     "min_p_discovery": float(associations.p_disc.min()),
                     "mean_true_dosage": float(truth.mean())})
    return pd.DataFrame(rows)


def load_inputs():
    """Verify corrected sample/source manifests before reading plotting data."""
    from .figure2_loci import select_hits, source_table

    root = corrected.analysis_root()
    summary = root / "plrs" / "plrs_models_summary.tsv"
    holdout = root / "plrs" / "plrs_holdout_predictions.tsv.gz"
    meta = root / "phewas" / "meta_analysis_results.tsv"
    publication = source_table()
    stage_path = corrected._stage_manifest_path("plrs_models")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    contract = stage["contract"]
    ids = samples.load_aou_sample_list(download=False)
    if ids is None:
        raise FileNotFoundError("Missing local canonical LAI sample manifest; run Figure_2_1025 first")
    ids = sorted(set(map(str, ids)).difference(samples.load_excluded_sample_ids()))
    if (len(ids) != 1025 or contract.get("lai_samples") != corrected._sample_sha(ids)
            or contract.get("pipeline") != corrected.PIPELINE_CONTRACT):
        raise ValueError("PLRS manifest does not match the corrected 1,025-person cohort")
    if not corrected._stage_current("plrs_models", contract, [summary, holdout]):
        raise ValueError("Missing or corrupt corrected PLRS caches; repair Figure_2_1025 PLRS stage")
    models = pd.read_csv(summary, sep="\t", dtype={"region_id": str})
    predictions = pd.read_csv(holdout, sep="\t", dtype={"region_id": str, "sample_id": str})
    if not set(predictions.sample_id).issubset(ids):
        raise ValueError("Holdout samples are outside the corrected LAI cohort")
    hits = select_hits(pd.read_csv(publication, dtype={"phecode": str}),
                       pd.read_csv(meta, sep="\t", dtype={"phecode": str, "region_id": str}))
    regions = prepare_regions(models, predictions, hits)
    provenance = {"inputs": [corrected._output_fingerprint(path)
                             for path in (summary, holdout, meta, publication)],
                  "model_contract": contract, "plot_contract": PLOT_CONTRACT,
                  "selection": f"p_disc < {admixture_phewas.DISCOVERY_P_THRESHOLD}; no replication/meta filter",
                  "n_models": len(models), "n_regions": len(regions), "n_associations": len(hits),
                  "parameters": {"dpi": 300, "figsize": [8, 7], "font_size": 18,
                                 "formats": list(FORMATS), "dosage_limits": [0, 2],
                                 "random_sampling": False}}
    return models, predictions, regions, provenance


def draw_model_size(axis, models, threshold):
    retained = models.r2_holdout > threshold
    for mask, color, label in ((~retained, "0.65", f"Test r² ≤ {threshold:g}"),
                               (retained, style.PALETTE["blue"], f"Test r² > {threshold:g}")):
        axis.scatter(models.loc[mask, "n_snvs"], models.loc[mask, "r2_holdout"],
                     s=45, color=color, alpha=0.8, edgecolors="none", label=label,
                     rasterized=False)
    axis.axhline(threshold, color=style.PALETTE["vermillion"], ls="--")
    correlation = (np.corrcoef(models.n_snvs, models.r2_holdout)[0, 1]
                   if models.n_snvs.nunique() > 1 and models.r2_holdout.nunique() > 1 else np.nan)
    label = f"Pearson r = {correlation:.3f}" if np.isfinite(correlation) else "Pearson r undefined"
    axis.set_title(f"{label}; {len(models)} regions", pad=58)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=15)
    axis.set_xlabel("SNVs in admixture-mapping\noutlier-region model")
    axis.set_ylabel("Test r² for predicted EUR-like\nancestry dosage")
    return float(correlation) if np.isfinite(correlation) else None


def draw_calibration(axis, frame, region):
    axis.scatter(frame.true_eur_dosage, frame.predicted_eur_dosage,
                 color=style.PALETTE["orange"], s=38, alpha=0.7, edgecolors="none", rasterized=False)
    axis.plot([0, 2], [0, 2], color="0.4", ls="--", label="Predicted = true")
    axis.set(xlim=(-0.03, 2.03), ylim=(-0.03, 2.03),
             xlabel="True EUR-like ancestry dosage", ylabel="Predicted EUR-like ancestry dosage")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"chr{int(region['chrom'])}:{int(region['start']):,}–{int(region['end']):,}\n"
                   f"Test r² = {region['r2_holdout']:.3f}; n = {int(region['n_test'])}", pad=48)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=15)


def run(*, force=False):
    models, predictions, regions, provenance = load_inputs()
    directory = output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stems = ["Figure_2_supp_B_snvs_vs_test_r2"] + [
        f"Figure_2_supp_C_chr{int(row.chrom)}_{int(row.start)}_{int(row.end)}"
        for row in regions.itertuples()]
    figures = [directory / f"{stem}.{extension}" for stem in stems for extension in FORMATS]
    table = directory / "region_summary.tsv"
    caption = directory / "caption.txt"
    outputs = [*figures, table, caption]
    stage_name = "plrs_supplement"
    print(f"[figure2_sources][plrs supplement] {json.dumps(provenance, sort_keys=True)}; output={directory}; "
          f"threads={os.environ['OMP_NUM_THREADS']}", flush=True)
    if not force and corrected._stage_current(stage_name, provenance, outputs):
        print("[figure2_sources][plrs supplement] verified cached PNG/PDF outputs", flush=True)
        return {"status": "ready", "directory": str(directory), "n_regions": len(regions)}
    style.apply_style()
    with plt.rc_context({"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18,
                         "xtick.labelsize": 16, "ytick.labelsize": 16}):
        for index, stem in enumerate(stems):
            figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
            try:
                if index == 0:
                    draw_model_size(axis, models, provenance["model_contract"]["r2_min"])
                else:
                    region = regions.iloc[index - 1]
                    draw_calibration(axis, predictions.loc[predictions.region_id.eq(region.region_id)], region)
                for extension in FORMATS:
                    figure.savefig(directory / f"{stem}.{extension}", dpi=300)
            finally:
                plt.close(figure)
    regions["png"] = [str(directory / f"{stem}.png") for stem in stems[1:]]
    regions["pdf"] = [str(directory / f"{stem}.pdf") for stem in stems[1:]]
    corrected._atomic_tsv(regions, table)
    contract = provenance["model_contract"]
    caption.write_text(
        "Figure 2 supplementary PLRS validation. "
        f"B: Number of eligible SNVs in each outlier-region PLRS model versus test r² across {len(models)} fitted regions. "
        f"The dashed line indicates the strict r² > {contract['r2_min']:g} PheWAS retention criterion; Pearson correlation uses all fitted regions. "
        f"C: True versus predicted EUR-like ancestry dosage (0–2) in the held-out samples, shown separately for all {len(regions)} unique regions "
        f"with at least one discovery PheWAS p < {admixture_phewas.DISCOVERY_P_THRESHOLD:g} association. "
        "No replication or meta-analysis significance filter is applied. Each point is one held-out sample from that region's own model; "
        "the dashed diagonal indicates perfect prediction. "
        f"The models used {100 * (1 - contract['test_fraction']):g}% training and {100 * contract['test_fraction']:g}% test samples "
        f"from the corrected 1,025-person LAI panel (split seed {contract['random_state']}). "
        "Reported test r² is the coefficient of determination, 1 − SSE/SST, not squared Pearson correlation.\n",
        encoding="utf-8")
    from .figure2_loci import _validate_plots
    for index in range(0, len(figures), 2):
        _validate_plots(figures[index:index + 2])
    corrected._write_stage_manifest(stage_name, provenance, outputs)
    print(f"[figure2_sources][plrs supplement] saved and verified {len(figures)} plots; summary={table}", flush=True)
    return {"status": "ready", "directory": str(directory), "n_regions": len(regions)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redraw plots from verified caches")
    run(force=parser.parse_args().force)
