"""Load and cache Figure 1 cohort, ancestry, geographic, and demographic inputs on Workbench.

The unrelated AoU analysis set contains 1,025 samples. Controlled inputs and
analysis products retain their established workspace paths."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common import config, io, samples

FIGURE_ID = "figure1"

# Fig 1A input: ancestry predictions with PCA features (the AoU ancestry_preds
# table). Columns: an ID column (research_id / person_id), ``ancestry_pred``,
# and ``pca_features`` (list of 16 PCs). On the Workbench it is copied from
# ``config.ANCESTRY_PREDS_GS`` (v7) if absent. See data/README.md. Git-ignored.
ANCESTRY_TSV_INPUT = config.RAW_DATA_DIR / "ancestry.tsv"
_PCA_ID_CANDIDATES = ("research_id", "person_id", "s", "sample_id", "sample")


def _ensure_ancestry_tsv() -> None:
    """Stage the ancestry predictions from the controlled bucket when absent."""
    if ANCESTRY_TSV_INPUT.is_file():
        return
    from .common import gcs
    ANCESTRY_TSV_INPUT.parent.mkdir(parents=True, exist_ok=True)
    result = gcs.gcs_cp(config.ANCESTRY_PREDS_GS, ANCESTRY_TSV_INPUT)
    if result.returncode != 0:
        raise RuntimeError(f"Could not stage {config.ANCESTRY_PREDS_GS}: {result.stderr.strip()}")
    if not ANCESTRY_TSV_INPUT.is_file() or not ANCESTRY_TSV_INPUT.stat().st_size:
        raise FileNotFoundError(f"Ancestry predictions missing or empty: {ANCESTRY_TSV_INPUT}")

# All of Us data-disclosure rule: no state cell of 1-19 may be shown in a plot;
# such counts are rendered as this floor value. Applied at PLOT time only -- raw
# counts are kept on disk (files are git-ignored; see data/README.md).
AOU_MIN_COUNT = 20

# Fig 1C (git-ignored, AoU-derived). The sample-level (person_id, state) table is
# the SOURCE pull and lives under data/raw/; the per-state summary is DERIVED and
# lives under analysis/figure1/ (regenerable; wiped on clean_run_data).
STATE_SAMPLES_INPUT = config.RAW_DATA_DIR / "person_state.csv"          # person_id, state (source)


def _adir():
    return config.analysis_dir(FIGURE_ID)


def _parse_pca_features(value) -> list[float]:
    """Parse a ``pca_features`` cell (bracketed/space/comma list) to floats."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    s = str(value).strip().strip("[]")
    if not s:
        return []
    return [float(x) for x in re.split(r"[,\s]+", s) if x]


def _build_pca() -> pd.DataFrame:
    """PC1/PC2 per sample for Fig 1A, labelled AOU vs. ancestry_pred.

    Reads ``ancestry.tsv`` (columns: an ID column, ``ancestry_pred``,
    ``pca_features`` [16 PCs]) if present. AoU samples (numeric IDs, restricted
    to the 1,025 list when available) are labelled "AOU"; every other sample
    takes its ``ancestry_pred`` label (e.g. "afr"). Output columns: sample_id, PC1, PC2, label,
    is_aou.
    """
    _ensure_ancestry_tsv()  # cp from gs (v7) on the Workbench if missing
    if ANCESTRY_TSV_INPUT.exists():
        df = pd.read_csv(ANCESTRY_TSV_INPUT, sep="\t")
        idcol = next((c for c in _PCA_ID_CANDIDATES if c in df.columns), df.columns[0])
        ids = df[idcol].astype(str).str.strip()
        feats = df["pca_features"].map(_parse_pca_features)
        pc1 = feats.map(lambda v: v[0] if len(v) > 0 else np.nan).astype(float)
        pc2 = feats.map(lambda v: v[1] if len(v) > 1 else np.nan).astype(float)
        # AoU samples carry numeric IDs (1000G reference are HG*/NA*).
        is_aou = ids.str.fullmatch(r"\d+").fillna(False)
        aou_ids = samples.load_aou_sample_list(download=False)
        if aou_ids is not None:
            is_aou = is_aou & ids.isin(aou_ids)
        pred = (
            df["ancestry_pred"].astype(str).str.strip().str.upper()
            if "ancestry_pred" in df.columns else pd.Series("OTH", index=df.index)
        )
        out = pd.DataFrame({
            "sample_id": ids,
            "PC1": pc1,
            "PC2": pc2,
            "is_aou": is_aou.to_numpy(),
            "label": np.where(is_aou.to_numpy(), "AOU", pred.to_numpy()),
        })
        # Drop the always-excluded related IDs and any unparsable rows.
        out = out[~out["sample_id"].isin(set(samples.load_excluded_sample_ids()))]
        return out.dropna(subset=["PC1", "PC2"]).reset_index(drop=True)

    raise FileNotFoundError(f"Ancestry predictions missing: {ANCESTRY_TSV_INPUT}")


def _annotated_admixture() -> "tuple[pd.DataFrame, str]":
    """Return (annotated merged ADMIXTURE frame, AFR component name).

    Uses the merged ADMIXTURE .Q from figure1_admixture. Annotates each sample: cohort ('AoU'/'1kg') and
    group ('AOU', 'AFR', 'EUR', or 'other'). The AFR component is identified
    from the 1kg AFR reference samples (their higher-mean component), so colours
    are anchored, not guessed.
    """
    from . import figure1_admixture
    frame = figure1_admixture.load_admixture_frame()
    if frame is None:
        raise FileNotFoundError("Merged ADMIXTURE products are missing; run figure1_admixture first.")

    ref = samples.load_1kg_reference()
    frame = frame.copy()
    # Normalise IDs and annotate strictly from the CSV: a non-AoU sample is a
    # 1kg reference ONLY if its ID is in assets/1kg_ref_populations.csv; every
    # other 1000G sample in the fileset is 'other' and dropped from the ref plot.
    sid = frame["sample_id"].astype(str).str.strip()
    frame["sample_id"] = sid
    is_aou = sid.str.fullmatch(r"\d+").fillna(False)
    superpop = sid.map(ref)  # AFR/EUR for CSV samples, NaN otherwise
    frame["cohort"] = np.where(is_aou.to_numpy(), "AoU", "1kg")
    frame["group"] = np.where(is_aou.to_numpy(), "AOU", superpop.fillna("other").to_numpy())

    afr_rows = frame[frame["group"] == "AFR"]
    ref_rows = afr_rows if len(afr_rows) else frame
    afr_comp = "comp1" if ref_rows["comp1"].mean() >= ref_rows["comp2"].mean() else "comp2"
    return frame, afr_comp


def _admixture_frames() -> "tuple[pd.DataFrame, pd.DataFrame, str]":
    """Split the merged ADMIXTURE frame into the two plot frames.

    Returns (aou, ref, afr_comp):
      - ``aou``: AoU-only, filtered to the 1,025-sample analysis set.
      - ``ref``: AoU + 1kg AFR/EUR, with the 1kg side restricted to samples in
        assets/1kg_ref_populations.csv; the 2 related AoU dropped.
      - ``afr_comp``: name of the AFR-anchored component for colouring.
    """
    frame, afr_comp = _annotated_admixture()
    aou = samples.filter_to_unrelated(
        frame[frame["cohort"] == "AoU"], label="Fig1 ADMIXTURE (AoU)",
    ).reset_index(drop=True)
    csv_ids = set(samples.load_1kg_reference())
    keep_ref = (frame["cohort"] == "AoU") | (
        frame["sample_id"].isin(csv_ids) & frame["group"].isin(["AFR", "EUR"])
    )
    ref = samples.exclude_related(frame[keep_ref]).reset_index(drop=True)
    return aou, ref, afr_comp


def admixture_summary() -> pd.DataFrame:
    """Mean/median ADMIXTURE fractions per population (AoU 1025, 1kg AFR, EUR).

    Reports both k=2 components (AFR- and EUR-anchored) for each group, using the
    same frames as the plot. Also cached to
    ``analysis/figure1/admixture_summary.csv``.
    """
    aou, ref, afr_comp = _admixture_frames()
    eur_comp = "comp2" if afr_comp == "comp1" else "comp1"
    groups = [
        (f"AoU (n={len(aou)})", aou),
        ("1kg AFR", ref[ref["group"] == "AFR"]),
        ("1kg EUR", ref[ref["group"] == "EUR"]),
    ]
    rows = [{
        "population": label,
        "n": len(sub),
        "AFR_mean": round(float(sub[afr_comp].mean()), 4) if len(sub) else np.nan,
        "AFR_median": round(float(sub[afr_comp].median()), 4) if len(sub) else np.nan,
        "EUR_mean": round(float(sub[eur_comp].mean()), 4) if len(sub) else np.nan,
        "EUR_median": round(float(sub[eur_comp].median()), 4) if len(sub) else np.nan,
    } for label, sub in groups]
    summary = pd.DataFrame(rows)
    summary.to_csv(_adir() / "admixture_summary.csv", index=False)
    return summary


def _build_idw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load validated IDW frames or build them from Workbench inputs."""
    from . import figure1_idw
    frames = figure1_idw.load_idw_frames()
    return frames if frames is not None else figure1_idw.build_idw()


def display_state_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the AoU plot-disclosure rule: floor any 1-19 count to AOU_MIN_COUNT.

    Call this on the RAW counts immediately before plotting so no small cell
    (1-19) is ever rendered. Does not mutate the stored raw data.
    """
    out = df.copy()
    out["n"] = out["n"].clip(lower=AOU_MIN_COUNT).astype(int)
    return out


def _counts_from_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a sample-level (person_id, state) table to raw per-state counts."""
    counts = samples_df.groupby("state", as_index=False).size().rename(columns={"size": "n"})
    return counts.sort_values("n", ascending=False).reset_index(drop=True)


def _build_state_counts() -> pd.DataFrame:
    """Aggregate the sample-level residence output, querying Workbench when absent."""
    if STATE_SAMPLES_INPUT.is_file():
        frame = pd.read_csv(STATE_SAMPLES_INPUT)
    else:
        from . import figure1_query
        frame = figure1_query.fetch_state_samples()
        STATE_SAMPLES_INPUT.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(STATE_SAMPLES_INPUT, index=False)
    return _counts_from_samples(frame)


def _build_model_params() -> dict:
    """Best-fit demographic model (physical units) from the committed dadi fit.

    Sourced from ``figure1_demography`` (the ``demes`` best-fit output + reported
    migration/CIs); see that module for the model spec and how the fit was
    chosen. Fig 1E renders the labelled schematic that these numbers back.
    """
    from . import figure1_demography
    return figure1_demography.best_fit_physical()


def _build_sfs_aou_fit() -> pd.DataFrame:
    """Load the AoU SFS fit or recompute it from the observed spectrum and fitted model."""
    from . import figure1_demography
    fit = figure1_demography.load_aou_fit()
    return fit if fit is not None else figure1_demography.compute_aou_fit()


# Physical-unit columns the Fig 1F/1G bootstrap panels + correlogram consume.
_BOOTSTRAP_COLUMNS = (
    "N_AFR_postsplit", "N_EUR_initial", "N_EUR_final", "f_AFR", "f_EUR",
    "T_split_years", "T_ancestral_years", "m_EUR_to_AFR", "m_AFR_to_EUR",
)


def _build_bootstrap() -> pd.DataFrame:
    """Load the demographic bootstrap parameter fits in physical units."""
    from . import figure1_demography
    boot = figure1_demography.load_bootstrap_fits()
    if boot is None:
        raise FileNotFoundError(f"Demographic bootstrap fits missing: {config.DEMOGRAPHY_BOOTSTRAP}")
    return boot[list(_BOOTSTRAP_COLUMNS)].reset_index(drop=True)


def build(clean_run_data: bool = False) -> dict:
    """Return all Figure 1 intermediates, (re)building caches as needed.

    ADMIXTURE runs on the merged AoU + 1000G fileset; here we drop the two
    related AoU samples and split into two frames: ``admixture`` (AoU only,
    1,025) and ``admixture_ref`` (AoU + 1kg AFR/EUR). ``admixture_afr_comp``
    names the AFR-anchored component for consistent colouring.

    PCA and ADMIXTURE are NOT CSV-cached: they are cheap to (re)build and this
    ensures the current ancestry.tsv / .Q is used. The freshly
    built PCA is still written to pca.csv for inspection.

    ``clean_run_data=True`` first wipes every regenerable intermediate in
    ``analysis/figure1/`` (incl. the IDW predictions / source-pop CSVs) so a
    stale cache can never survive a clean run; source data under ``data/`` and
    downloaded GCS inputs are untouched.
    """
    d = _adir()
    if clean_run_data:
        import shutil
        for p in d.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    admixture, admixture_ref, afr_comp = _admixture_frames()
    pca = _build_pca()
    pca.to_csv(d / "pca.csv", index=False)  # for inspection; not used as a cache
    idw_samples, idw_sources = _build_idw()
    state_counts = _build_state_counts()
    bootstrap = _build_bootstrap()
    model_params = _build_model_params()
    state_counts.to_csv(d / "state_counts.csv", index=False)
    bootstrap.to_csv(d / "bootstrap.csv", index=False)
    return {
        "pca": pca,
        "idw_samples": idw_samples,
        "idw_sources": idw_sources,
        "state_counts": state_counts,
        "admixture": admixture,
        "admixture_ref": admixture_ref,
        "admixture_afr_comp": afr_comp,
        "model_params": model_params,
        "sfs_aou_fit": io.cached_frame(
            d / "sfs_aou_fit.csv", clean_run_data, _build_sfs_aou_fit,
            expected_columns=(
                "allele_count", "data", "model", "poisson_resid", "model_id",
                "model_parameter_sha256", "model_grid_points", "model_scaling",
                "model_theta",
            ),
        ),
        "bootstrap": bootstrap,
    }
