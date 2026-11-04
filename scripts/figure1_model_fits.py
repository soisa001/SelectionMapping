"""Evaluate saved dadi fits and generate Supplementary Figure S1 source data.

The demographic fits are evaluated against the full AFR/EUR/AoU joint SFS.
For the plotted comparison only, each joint expectation is marginalized to AoU
and independently rescaled to the observed AoU one-dimensional SFS.  No
parameter optimization is performed.

B1 and B2 are reduced versions of B3 and were optimized against a direct
85-point full joint SFS. B2 fixes EUR size as constant; B1 additionally fixes
AFR size as constant while estimating the admixture fraction. The original
35/45/55 extrapolation is retained as a compatibility diagnostic, because it
contains small negative cells for these models and is not stable enough to rank
the newly optimized points.

The S1A demographic panels are native ``demesdraw.tubes`` plots. AFR is one
continuous root deme; historical African size changes are epochs of AFR.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import config

# Keep labels editable in vector outputs.  Type 42 embeds TrueType outlines;
# the default Type 3 PDF fonts are vector shapes but are awkward to edit in
# Illustrator/Inkscape and can render poorly in downstream manuscript tools.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

MODEL_CONTRACT = "s1_reduced_fits_joint_grid85_b1_free_f_v7_physical_units"
EXTRAPOLATION = "dadi.Numerics.make_extrap_func"
GRID_POINTS = (35, 45, 55)
REPORT_GRID_POINT = 85
GENERATION_TIME_YEARS = 25.0
SEQUENCE_LENGTH_BP = 481_020_725
MUTATION_RATE_PER_BP_PER_GENERATION = 1.5e-8
FIXED_EARLY_MODEL_AFR_FRACTION = 0.8
DEMESDRAW_SEED = 20260813
DEMESDRAW_MAX_TIME_YEARS = 1_000_000.0
# The fitted dadi models create AoU by instantaneous admixture (T2 = 0), while
# Demes requires a positive-duration child deme. This plotting-only floor is in
# literal years and is not a fitted time.
AOU_PLOT_START_TIME_YEARS = 1.2

REPORTED_GODAMBE_ADJUSTED_D = 61.11
REPORTED_GODAMBE_P_BOUND = "<1e-8"

FULL_PARAMETER_NAMES = (
    "nu_growth",
    "nu2B",
    "nu2F",
    "nu3",
    "Tgrowth",
    "T1",
    "T2",
    "f",
    "m12",
    "m21",
)

HARMONIZED_PARAMETER_METADATA = OrderedDict(
    (
        (
            "Na",
            {
                "physical_column": "Na",
                "source_parameter": "theta / (4 * mu * L)",
                "units": "individuals",
                "explanation": "Initial effective size of the ancestral/AFR root deme.",
            },
        ),
        (
            "N_AFR",
            {
                "physical_column": "N_AFR",
                "source_parameter": "nu_growth * Na",
                "units": "individuals",
                "explanation": "AFR effective size after the historical AFR size-change epoch.",
            },
        ),
        (
            "N_EUR_INITIAL",
            {
                "physical_column": "N_EUR_initial",
                "source_parameter": "nu2B * Na",
                "units": "individuals",
                "explanation": "EUR effective size immediately after the AFR-EUR split.",
            },
        ),
        (
            "N_EUR_FINAL",
            {
                "physical_column": "N_EUR_final",
                "source_parameter": "nu2F * Na",
                "units": "individuals",
                "explanation": "Present-day EUR effective size.",
            },
        ),
        (
            "T_AFR_CHANGE",
            {
                "physical_column": "Tgrowth_duration_years",
                "source_parameter": "Tgrowth * 2 * Na * generation_time",
                "units": "years",
                "explanation": (
                    "Duration from the historical AFR size change to the AFR-EUR split."
                ),
            },
        ),
        (
            "T_SPLIT",
            {
                "physical_column": "T_AFR_EUR_split_years_before_present",
                "source_parameter": "(T1 + T2) * 2 * Na * generation_time",
                "units": "years before present",
                "explanation": (
                    "AFR-EUR split time before present; equals dadi T1 because T2=0."
                ),
            },
        ),
        (
            "f",
            {
                "physical_column": "f_AFR",
                "source_parameter": "f",
                "units": "fraction AFR in AoU",
                "explanation": "AFR ancestry proportion in the instantaneously admixed AoU population.",
            },
        ),
        (
            "p_misid",
            {
                "physical_column": "p_misid",
                "source_parameter": "p_misid",
                "units": "fraction misidentification",
                "explanation": "Proportion of ancestral-state misidentification.",
            },
        ),
        (
            "theta",
            {
                "physical_column": "theta_full_3D",
                "source_parameter": "optimal theta for the full 3D SFS",
                "units": "population mutation rate",
                "explanation": "Population mutation rate, theta = 4 * Na * mu * L.",
            },
        ),
        (
            "m_EUR_AFR",
            {
                "physical_column": "m12_EUR_to_AFR_per_generation",
                "source_parameter": "m12 / (2 * Na)",
                "units": "per generation",
                "explanation": "Forward migration fraction from EUR into AFR per generation.",
            },
        ),
        (
            "m_AFR_EUR",
            {
                "physical_column": "m21_AFR_to_EUR_per_generation",
                "source_parameter": "m21 / (2 * Na)",
                "units": "per generation",
                "explanation": "Forward migration fraction from AFR into EUR per generation.",
            },
        ),
    )
)

# C2 point estimates and percentile intervals are transcribed from the
# user-supplied full-model workbook. They are deliberately not recomputed from
# the legacy saved vector, so the publication table preserves the reported fit.
C2_REPORTED_PARAMETER_INTERVALS = {
    "Na": (11727.0, 11121.0, 12438.0),
    "N_AFR": (18277.0, 17944.0, 18696.0),
    "N_EUR_INITIAL": (921.0, 753.0, 1126.0),
    "N_EUR_FINAL": (8529.0, 7464.0, 10245.0),
    "T_AFR_CHANGE": (202131.0, 159072.0, 264888.0),
    "T_SPLIT": (44129.0, 39638.0, 50338.0),
    "f": (0.802, 0.801, 0.803),
    "p_misid": (0.040, 0.037, 0.042),
    "theta": (338448.8, 320979.2, 358983.2),
    "m_EUR_AFR": (0.000074, 0.000065, 0.000084),
    "m_AFR_EUR": (0.000033, 0.000027, 0.000038),
}

COLORS = {
    "B1": "#666666",
    "B2": "#377eb8",
    "B3": "#e68613",
    "C1": "#2a9d8f",
    "C2": "#8e5ab5",
}

POPULATION_COLORS = {
    "AFR": "#4C78A8",
    "EUR": "#F28E2B",
    "AoU": "#59A14F",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    figure_panel: str
    label: str
    description: str
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    expanded_parameters: tuple[float, ...]
    constraints: str
    misid: float
    source: str
    mapping_status: str = "confirmed"
    saved_theta: float | None = None


# B1/B2 were refit to the full joint SFS as reduced versions of B3. B3/C1/C2
# are the latest supplied parameter points.
MODEL_SPECS: OrderedDict[str, ModelSpec] = OrderedDict(
    (
        (
            "B1",
            ModelSpec(
                "B1",
                "B1",
                "Constant AFR and EUR sizes",
                "Constant AFR and EUR sizes; instantaneous AoU admixture; no migration.",
                ("nu2", "T1", "f"),
                (0.2031224169385347, 0.05202325831926894, 0.7778222000865582),
                (
                    1.0,
                    0.2031224169385347,
                    0.2031224169385347,
                    1.0,
                    0.0,
                    0.05202325831926894,
                    0.0,
                    0.7778222000865582,
                    0.0,
                    0.0,
                ),
                "nu_growth=1; Tgrowth=T2=0; nu2F=nu2B; m12=m21=0",
                0.03356716427065076,
                "full-3D reduced-model optimization (2026-08-13)",
                "confirmed; two starts converged at direct pts=85 with f estimated",
            ),
        ),
        (
            "B2",
            ModelSpec(
                "B2",
                "B2",
                "AFR size change; constant EUR",
                "Ancestral AFR size change; constant EUR; instantaneous AoU admixture; no migration.",
                ("nu_growth", "nu2", "Tgrowth", "T1", "f"),
                (1.58726652511399, 0.24533575793421822, 0.20315571257484996,
                 0.0628953441302355, 0.7824984540208955),
                (
                    1.58726652511399,
                    0.24533575793421822,
                    0.24533575793421822,
                    1.0,
                    0.20315571257484996,
                    0.0628953441302355,
                    0.0,
                    0.7824984540208955,
                    0.0,
                    0.0,
                ),
                "nu2F=nu2B; T2=0; m12=m21=0",
                0.04016672697822522,
                "full-3D reduced-model optimization (2026-08-13)",
                "confirmed; two starts converged at direct pts=85",
            ),
        ),
        (
            "B3",
            ModelSpec(
                "B3",
                "B3",
                "AFR size change + EUR growth",
                "Ancestral AFR size change and EUR bottleneck/recovery; no migration.",
                ("nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f"),
                (1.585655606, 0.044900732, 1.535172265, 0.343290685, 0.045840737, 0.804246326),
                (
                    1.585655606,
                    0.044900732,
                    1.535172265,
                    1.0,
                    0.343290685,
                    0.045840737,
                    0.0,
                    0.804246326,
                    0.0,
                    0.0,
                ),
                "T2=0 (nu3 inactive); m12=m21=0",
                0.040357568,
                "user-supplied candidate vector (2026-08-13)",
            ),
        ),
        (
            "C1",
            ModelSpec(
                "C1",
                "C1",
                "One shared migration rate",
                "B3 with one shared bidirectional migration rate.",
                ("nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f", "m"),
                (
                    1.6011265012522724,
                    0.0641435969935697,
                    0.9120731314346658,
                    0.46472098181027055,
                    0.06729711019710978,
                    0.8017056364025,
                    1.2133481947792528,
                ),
                (
                    1.6011265012522724,
                    0.0641435969935697,
                    0.9120731314346658,
                    1.0,
                    0.46472098181027055,
                    0.06729711019710978,
                    0.0,
                    0.8017056364025,
                    1.2133481947792528,
                    1.2133481947792528,
                ),
                "T2=0 (nu3 inactive); m12=m21=m",
                0.04074401941207889,
                "user-supplied candidate vector (2026-08-13)",
            ),
        ),
        (
            "C2",
            ModelSpec(
                "C2",
                "C2",
                "Two directional migration rates",
                "C1 with separate EUR-to-AFR and AFR-to-EUR migration rates.",
                ("nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f", "m12", "m21"),
                (
                    1.558584501,
                    0.078546083,
                    0.727276039,
                    0.344734944,
                    0.075261347,
                    0.801915874,
                    1.741892163,
                    1.219863597,
                ),
                (
                    1.558584501,
                    0.078546083,
                    0.727276039,
                    1.0,
                    0.344734944,
                    0.075261347,
                    0.0,
                    0.801915874,
                    1.741892163,
                    1.219863597,
                ),
                "T2=0 (nu3 inactive); m12 and m21 free",
                0.040393927,
                "user-supplied/published best-fit vector",
                saved_theta=338448.7526,
            ),
        ),
    )
)


# Previously supplied points are evaluated only as a sanity-check comparison.
# They do not replace the vectors above or appear as extra figure curves.
REFERENCE_SPECS: OrderedDict[str, ModelSpec] = OrderedDict(
    (
        (
            "B1_previous",
            ModelSpec(
                "B1_previous", "B1", "Previous B1 point", "Previous expanded B1 point.",
                ("nu1", "nu2", "nu3", "T1", "T2", "f"),
                (1.44797, 0.154883, 152.701, 0.396869, 0.0528349, 0.776814),
                (1.44797, 0.154883, 0.154883, 152.701, 0.0, 0.396869, 0.0528349, 0.776814, 0.0, 0.0),
                "Tgrowth=0; nu2F=nu2B; m12=m21=0", 0.0383904,
                "previously supplied point",
            ),
        ),
        (
            "B2_previous",
            ModelSpec(
                "B2_previous", "B2", "Previous B2 point", "Previous expanded B2 point.",
                ("nu_growth", "nu2B", "nu3", "Tgrowth", "T1", "T2", "f"),
                (1.87503, 0.139483, 9.37521, 0.369277, 0.0289119, 0.0150655, 0.575085),
                (1.87503, 0.139483, 0.139483, 9.37521, 0.369277, 0.0289119, 0.0150655, 0.575085, 0.0, 0.0),
                "nu2F=nu2B; m12=m21=0", 0.0336129,
                "previously supplied point",
            ),
        ),
        (
            "B3_previous",
            ModelSpec(
                "B3_previous", "B3", "Previous B3 point", "Previous no-migration B3 point.",
                ("nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f"),
                (1.567252618, 0.072435327, 0.721614683, 0.325135383, 0.071395801, 0.801629075),
                (1.567252618, 0.072435327, 0.721614683, 1.0, 0.325135383, 0.071395801, 0.0, 0.801629075, 0.0, 0.0),
                "T2=0; m12=m21=0", 0.040266162,
                "previously supplied point",
            ),
        ),
        (
            "C1_previous",
            ModelSpec(
                "C1_previous", "C1", "Previous C1 point", "Previous shared-migration C1 point.",
                ("nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f", "m"),
                (1.567252618, 0.072435327, 0.721614683, 0.325135383, 0.071395801, 0.801629075, 1.449212918),
                (1.567252618, 0.072435327, 0.721614683, 1.0, 0.325135383, 0.071395801, 0.0, 0.801629075, 1.449212918, 1.449212918),
                "T2=0; m12=m21=m", 0.040266162,
                "previously supplied point",
            ),
        ),
    )
)


def _dadi():
    try:
        import dadi
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Recomputing S1 requires dadi. Install with "
            "`uv pip install -r requirements-demography.txt`."
        ) from exc
    return dadi


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance_source_path(path: Path) -> str:
    """Use a repository-relative path when possible, else an absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(config.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def nested_two_migration_model(params, ns, pts):
    """Superset model used for every saved parameter point."""
    dadi = _dadi()
    (
        nu_growth, nu2b, nu2f, nu3, tgrowth, t1, t2, f, m12, m21,
    ) = params
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    if tgrowth > 0:
        phi = dadi.Integration.one_pop(phi, xx, tgrowth, nu=nu_growth)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    if math.isclose(nu2b, nu2f):
        nu2_during_split = nu2b
    else:
        nu2_during_split = lambda t: nu2b * (nu2f / nu2b) ** (t / t1)
    phi = dadi.Integration.two_pops(
        phi, xx, t1, nu1=nu_growth, nu2=nu2_during_split, m12=m12, m21=m21
    )
    phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
    if t2 > 0:
        phi = dadi.Integration.three_pops(
            phi, xx, t2, nu1=nu_growth, nu2=nu2f, nu3=nu3,
            m12=m12, m21=m21,
        )
    return dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx))


def c1_c2_equality_lrt_model(params, ns, pts):
    """C2 reparameterized so C1 is the interior constraint ``delta_m=0``."""
    (
        nu_growth, nu2b, nu2f, tgrowth, t1, f, m12, delta_m,
    ) = params
    return nested_two_migration_model(
        (nu_growth, nu2b, nu2f, 1.0, tgrowth, t1, 0.0, f, m12, m12 + delta_m),
        ns,
        pts,
    )


def _spec_payload(spec: ModelSpec) -> dict:
    return {
        "parameter_names": list(spec.parameter_names),
        "parameters": list(spec.parameters),
        "expanded_parameters": list(spec.expanded_parameters),
        "constraints": spec.constraints,
        "misid": spec.misid,
        "source": spec.source,
        "mapping_status": spec.mapping_status,
        "saved_theta": spec.saved_theta,
    }


def _contract_payload() -> dict:
    return {
        "model_contract": MODEL_CONTRACT,
        "observed_sfs_sha256": _sha256(config.DEMOGRAPHY_SFS),
        "likelihood_spectra": ["full AFR/EUR/AoU 3D", "AoU 1D marginal"],
        "primary_report_grid_point": REPORT_GRID_POINT,
        "compatibility_extrapolation_grid_points": list(GRID_POINTS),
        "extrapolation": EXTRAPOLATION,
        "generation_time_years": GENERATION_TIME_YEARS,
        "sequence_length_bp": SEQUENCE_LENGTH_BP,
        "mutation_rate_per_bp_per_generation": MUTATION_RATE_PER_BP_PER_GENERATION,
        "theta_to_Na": "theta / (4 * mutation_rate * sequence_length)",
        "full_parameter_names": list(FULL_PARAMETER_NAMES),
        "models": {key: _spec_payload(value) for key, value in MODEL_SPECS.items()},
        "reference_models": {
            key: _spec_payload(value) for key, value in REFERENCE_SPECS.items()
        },
    }


def _contract_hash() -> str:
    raw = json.dumps(_contract_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _harmonized_contract_payload() -> dict:
    return {
        "parameters": HARMONIZED_PARAMETER_METADATA,
        "c2_reported_parameter_intervals": C2_REPORTED_PARAMETER_INTERVALS,
        "source_parameter_mapping_version": 2,
        "unused_display_value": "-",
    }


def _harmonized_contract_hash() -> str:
    raw = json.dumps(
        _harmonized_contract_payload(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _core_source_paths() -> tuple[Path, ...]:
    return (
        config.DEMOGRAPHY_S1_MODEL_SUMMARY,
        config.DEMOGRAPHY_S1_PARAMETERS,
        config.DEMOGRAPHY_S1_WIDE_PARAMETERS,
        config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS,
        config.DEMOGRAPHY_S1_COMPARISONS,
        config.DEMOGRAPHY_S1_CANDIDATE_CHECKS,
        config.DEMOGRAPHY_S1_MARGINAL_SFS,
    )


def _source_paths() -> tuple[Path, ...]:
    return (
        *_core_source_paths(),
        config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS,
        config.DEMOGRAPHY_S1_PROVENANCE,
    )


def _base_source_tables_current() -> bool:
    paths = (*_core_source_paths(), config.DEMOGRAPHY_S1_PROVENANCE)
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        provenance = json.loads(config.DEMOGRAPHY_S1_PROVENANCE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return provenance.get("contract_hash") == _contract_hash()


def source_tables_current() -> bool:
    if not _base_source_tables_current():
        return False
    if not (
        config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS.exists()
        and config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS.stat().st_size > 0
    ):
        return False
    provenance = json.loads(
        config.DEMOGRAPHY_S1_PROVENANCE.read_text(encoding="utf-8")
    )
    return provenance.get("harmonized_contract_hash") == _harmonized_contract_hash()


def _filled(spectrum) -> np.ndarray:
    return np.asarray(np.ma.filled(spectrum, np.nan), dtype=float)


def _evaluate_spec(spec: ModelSpec, wrapped, extrapolated, joint_data, aou_data, ns) -> dict:
    dadi = _dadi()
    all_parameters = (*spec.expanded_parameters, spec.misid)
    joint_unscaled = wrapped(all_parameters, ns, REPORT_GRID_POINT)
    theta_joint = float(dadi.Inference.optimal_sfs_scaling(joint_unscaled, joint_data))
    ll_joint = float(dadi.Inference.ll(theta_joint * joint_unscaled, joint_data))
    aou_unscaled = joint_unscaled.marginalize([0, 1])
    theta_aou = float(dadi.Inference.optimal_sfs_scaling(aou_unscaled, aou_data))
    aou_scaled = theta_aou * aou_unscaled
    ll_aou = float(dadi.Inference.ll(aou_scaled, aou_data))

    compatibility_unscaled = extrapolated(
        all_parameters, ns, list(GRID_POINTS)
    )
    theta_compatibility = float(
        dadi.Inference.optimal_sfs_scaling(compatibility_unscaled, joint_data)
    )
    ll_compatibility = float(
        dadi.Inference.ll(theta_compatibility * compatibility_unscaled, joint_data)
    )
    unmasked = ~np.ma.getmaskarray(joint_data)
    negative = unmasked & (np.asarray(compatibility_unscaled) < 0)
    return {
        "spec": spec,
        "theta_joint": theta_joint,
        "theta_aou": theta_aou,
        "Na_joint": theta_joint
        / (4.0 * MUTATION_RATE_PER_BP_PER_GENERATION * SEQUENCE_LENGTH_BP),
        "ll_joint": ll_joint,
        "ll_aou": ll_aou,
        "aou_spectrum": aou_scaled,
        "theta_compatibility": theta_compatibility,
        "ll_compatibility": ll_compatibility,
        "negative_compatibility_cells": int(np.sum(negative)),
        "data_count_in_negative_compatibility_cells": float(
            np.sum(np.asarray(joint_data)[negative])
        ),
        "compatibility_spectrum": compatibility_unscaled,
    }


def _evaluate_models():
    dadi = _dadi()
    joint_data = dadi.Spectrum.from_file(str(config.DEMOGRAPHY_SFS))
    aou_data = joint_data.marginalize([0, 1])
    ns = tuple(int(value) for value in joint_data.sample_sizes)
    wrapped = dadi.Numerics.make_anc_state_misid_func(nested_two_migration_model)
    extrapolated = dadi.Numerics.make_extrap_func(wrapped)
    evaluated = OrderedDict(
        (key, _evaluate_spec(spec, wrapped, extrapolated, joint_data, aou_data, ns))
        for key, spec in MODEL_SPECS.items()
    )
    references = OrderedDict(
        (key, _evaluate_spec(spec, wrapped, extrapolated, joint_data, aou_data, ns))
        for key, spec in REFERENCE_SPECS.items()
    )

    cached = pd.read_csv(config.DEMOGRAPHY_AOU_FIT)
    if not np.allclose(np.asarray(aou_data), cached["data"], rtol=1e-10, atol=1e-8):
        raise RuntimeError("Observed AoU marginal no longer matches aou_sfs_fit.csv")
    regenerated = np.asarray(
        evaluated["C2"]["compatibility_spectrum"].marginalize([0, 1])
    )
    cached_model = cached["model"].to_numpy(dtype=float)
    if not np.allclose(
        regenerated / np.nansum(regenerated),
        cached_model / np.nansum(cached_model),
        rtol=2e-5,
        atol=1e-10,
    ):
        raise RuntimeError("C2 AoU SFS shape no longer matches aou_sfs_fit.csv")
    return joint_data, aou_data, evaluated, references


def _chi2_df1_sf(statistic: float) -> float:
    return math.erfc(math.sqrt(statistic / 2.0)) if statistic >= 0 else math.nan


def _summary_table(joint_data, aou_data, evaluated) -> pd.DataFrame:
    rows = []
    for model_id, result in evaluated.items():
        spec = result["spec"]
        rows.append(
            {
                "model_id": model_id,
                "figure_panel": spec.figure_panel,
                "model_label": spec.label,
                "description": spec.description,
                "model_function": "nested_two_migration_model",
                "nested_constraints": spec.constraints,
                "parameter_source": spec.source,
                "parameter_mapping_status": spec.mapping_status,
                "n_demographic_parameters_saved": len(spec.parameters),
                "n_nuisance_parameters": 1,
                "ancestral_misid_fraction": spec.misid,
                "saved_theta_if_provided": spec.saved_theta,
                "optimal_theta_full_3D": result["theta_joint"],
                "log_likelihood_full_3D": result["ll_joint"],
                "full_3D_report_grid_point": REPORT_GRID_POINT,
                "optimal_theta_full_3D_extrap_35_45_55": result[
                    "theta_compatibility"
                ],
                "log_likelihood_full_3D_extrap_35_45_55": result[
                    "ll_compatibility"
                ],
                "negative_extrapolated_cells": result[
                    "negative_compatibility_cells"
                ],
                "data_count_in_negative_extrapolated_cells": result[
                    "data_count_in_negative_compatibility_cells"
                ],
                "optimal_theta_AoU_1D": result["theta_aou"],
                "log_likelihood_AoU_1D": result["ll_aou"],
                "Na_from_full_3D_theta": result["Na_joint"],
                "joint_sample_sizes": ",".join(map(str, joint_data.sample_sizes)),
                "AoU_sample_size": int(aou_data.sample_sizes[0]),
                "compatibility_extrapolation_grid_points": ",".join(
                    map(str, GRID_POINTS)
                ),
                "extrapolation": EXTRAPOLATION,
                "model_contract": MODEL_CONTRACT,
                "observed_sfs": config.DEMOGRAPHY_SFS.name,
                "observed_sfs_sha256": _sha256(config.DEMOGRAPHY_SFS),
            }
        )
    return pd.DataFrame(rows)


def _parameter_table(evaluated) -> pd.DataFrame:
    rows: list[dict] = []

    def add(spec, order, source_parameter, source_value, source_unit,
            physical_parameter, physical_value, physical_unit, conversion,
            notes="Direct parameter conversion."):
        rows.append(
            {
                "model_id": spec.model_id,
                "figure_panel": spec.figure_panel,
                "model_label": spec.label,
                "parameter_order": order,
                "source_parameter": source_parameter,
                "source_value": source_value,
                "source_unit": source_unit,
                "physical_parameter": physical_parameter,
                "physical_value": physical_value,
                "physical_unit": physical_unit,
                "conversion": conversion,
                "notes": notes,
            }
        )

    for result in evaluated.values():
        spec = result["spec"]
        raw = dict(zip(spec.parameter_names, spec.parameters))
        full = dict(zip(FULL_PARAMETER_NAMES, spec.expanded_parameters))
        na = float(result["Na_joint"])
        years_per_scaled_time = 2.0 * na * GENERATION_TIME_YEARS
        add(
            spec,
            0,
            "optimal_theta_full_3D",
            result["theta_joint"],
            "SFS scale",
            "Na",
            na,
            "diploid individuals",
            "theta / (4 * 1.5e-8 * 481020725)",
            "Primary physical conversion uses the full joint-SFS scale.",
        )
        add(spec, "plot-scale", "optimal_theta_AoU_1D", result["theta_aou"], "SFS scale",
            "AoU_plot_scale", result["theta_aou"], "SFS scale", "optimal_sfs_scaling",
            "Used only for the plotted AoU marginal.")

        for order, name in enumerate(spec.parameter_names, start=1):
            value = raw[name]
            canonical = "nu3" if name == "nu3_inactive" else name
            size_map = {
                "nu1": "N_AFR", "nu2": "N_EUR", "nu3": "N_AoU",
                "nu_growth": "N_AFR_postchange", "nu2B": "N_EUR_initial",
                "nu2F": "N_EUR_final",
            }
            if canonical in size_map:
                inactive = name == "nu3_inactive"
                add(spec, order, name, value, "N/Na", size_map[canonical],
                    math.nan if inactive else value * na,
                    "diploid individuals", f"{name} * Na",
                    "Inactive because T2=0; retained to preserve the supplied vector."
                    if inactive else "Direct parameter conversion.")
            elif name in {"Tgrowth", "T1", "T2"}:
                physical_name = {
                    "Tgrowth": "ancestral_epoch_duration",
                    "T1": "split_to_admixture_duration",
                    "T2": "time_since_admixture",
                }[name]
                add(spec, order, name, value, "2*Na generations", physical_name,
                    value * years_per_scaled_time, "years",
                    f"{name} * 2 * Na * {GENERATION_TIME_YEARS:g}")
            elif name == "f":
                add(spec, order, name, value, "fraction", "admixture_fraction_AFR",
                    value, "fraction", "identity")
            elif name in {"m", "m12", "m21"}:
                physical_name = {
                    "m": "symmetric_migration", "m12": "migration_EUR_to_AFR",
                    "m21": "migration_AFR_to_EUR",
                }[name]
                add(spec, order, name, value, "2*Na*m", physical_name,
                    value / (2.0 * na), "per generation", f"{name} / (2 * Na)")

        if "f" not in raw:
            add(spec, "fixed-f", "f", full["f"], "fraction",
                "admixture_fraction_AFR", full["f"], "fraction", "fixed",
                "Fixed by the reduced B1 model contract.")
        add(spec, "misid", "p_misid", spec.misid, "fraction",
            "ancestral_state_misidentification", spec.misid, "fraction", "identity")
        add(spec, "derived-f", "1-f", 1.0 - full["f"], "fraction",
            "admixture_fraction_EUR", 1.0 - full["f"], "fraction", "1 - f")
        split_scaled = full["T1"] + full["T2"]
        add(spec, "derived-split", "T1+T2", split_scaled, "2*Na generations",
            "T_AFR_EUR_split", split_scaled * years_per_scaled_time,
            "years before present", "(T1 + T2) * 2 * Na * generation_time")
        if full["Tgrowth"] > 0:
            ancestral_scaled = full["Tgrowth"] + split_scaled
            add(spec, "derived-change", "Tgrowth+T1+T2", ancestral_scaled,
                "2*Na generations", "T_ancestral_size_change",
                ancestral_scaled * years_per_scaled_time, "years before present",
                "(Tgrowth + T1 + T2) * 2 * Na * generation_time")
    return pd.DataFrame(rows)


def _candidate_checks(evaluated, references) -> pd.DataFrame:
    rows = []
    for model_id in ("B1", "B2", "B3", "C1"):
        current = evaluated[model_id]
        previous = references[f"{model_id}_previous"]
        delta_joint = current["ll_joint"] - previous["ll_joint"]
        delta_aou = current["ll_aou"] - previous["ll_aou"]
        rows.append(
            {
                "figure_panel": model_id,
                "supplied_candidate_source": current["spec"].source,
                "mapping_status": current["spec"].mapping_status,
                "candidate_log_likelihood_full_3D": current["ll_joint"],
                "previous_log_likelihood_full_3D": previous["ll_joint"],
                "candidate_minus_previous_log_likelihood_full_3D": delta_joint,
                "candidate_log_likelihood_AoU_1D": current["ll_aou"],
                "previous_log_likelihood_AoU_1D": previous["ll_aou"],
                "candidate_minus_previous_log_likelihood_AoU_1D": delta_aou,
                "candidate_better_full_3D": delta_joint > 0,
                "candidate_better_AoU_1D": delta_aou > 0,
                "interpretation": (
                    "Direct pts=85 evaluation through the same superset model; B1/B2 are newly optimized reduced fits."
                ),
            }
        )
    return pd.DataFrame(rows)


def _wide_parameter_table(evaluated) -> pd.DataFrame:
    """One row per nested model with inactive superset parameters blank."""
    rows = []
    for model_id, result in evaluated.items():
        spec = result["spec"]
        full = dict(zip(FULL_PARAMETER_NAMES, spec.expanded_parameters))
        rows.append(
            {
                "model_id": model_id,
                "model_label": spec.label,
                "nu_growth": full["nu_growth"],
                "nu2B": full["nu2B"],
                "nu2F": full["nu2F"],
                "nu3": np.nan if full["T2"] == 0 else full["nu3"],
                "Tgrowth": np.nan if full["Tgrowth"] == 0 else full["Tgrowth"],
                "T1": full["T1"],
                "T2": np.nan if full["T2"] == 0 else full["T2"],
                "f": full["f"],
                "m12": np.nan if full["m12"] == 0 else full["m12"],
                "m21": np.nan if full["m21"] == 0 else full["m21"],
                "p_misid": spec.misid,
                "constraint_notes": spec.constraints,
            }
        )
    return pd.DataFrame(rows)


def _wide_physical_parameter_table(evaluated) -> pd.DataFrame:
    """One row per model after conversion with theta = 4 * Na * mu * L."""
    rows = []
    for model_id, result in evaluated.items():
        spec = result["spec"]
        full = dict(zip(FULL_PARAMETER_NAMES, spec.expanded_parameters))
        na = float(result["Na_joint"])
        years_per_scaled_time = 2.0 * na * GENERATION_TIME_YEARS
        rows.append(
            {
                "model_id": model_id,
                "model_label": spec.label,
                "theta_full_3D": result["theta_joint"],
                "Na": na,
                "N_AFR": full["nu_growth"] * na,
                "N_EUR_initial": full["nu2B"] * na,
                "N_EUR_final": full["nu2F"] * na,
                "N_AoU": np.nan if full["T2"] == 0 else full["nu3"] * na,
                "Tgrowth_duration_years": (
                    np.nan
                    if full["Tgrowth"] == 0
                    else full["Tgrowth"] * years_per_scaled_time
                ),
                "T1_split_to_admixture_years": full["T1"] * years_per_scaled_time,
                "T2_since_admixture_years": (
                    np.nan
                    if full["T2"] == 0
                    else full["T2"] * years_per_scaled_time
                ),
                "f_AFR": full["f"],
                "m12_EUR_to_AFR_per_generation": (
                    np.nan if full["m12"] == 0 else full["m12"] / (2.0 * na)
                ),
                "m21_AFR_to_EUR_per_generation": (
                    np.nan if full["m21"] == 0 else full["m21"] / (2.0 * na)
                ),
                "p_misid": spec.misid,
                "T_AFR_EUR_split_years_before_present": (
                    full["T1"] + full["T2"]
                )
                * years_per_scaled_time,
                "T_ancestral_size_change_years_before_present": (
                    np.nan
                    if full["Tgrowth"] == 0
                    else (full["Tgrowth"] + full["T1"] + full["T2"])
                    * years_per_scaled_time
                ),
                "constraint_notes": spec.constraints,
            }
        )
    return pd.DataFrame(rows)


def _harmonized_parameter_status(model_id: str, parameter: str) -> tuple[str, str]:
    if parameter == "T_AFR_CHANGE" and model_id == "B1":
        return "unused", "No historical AFR size-change epoch in B1."
    if parameter in {"m_EUR_AFR", "m_AFR_EUR"} and model_id in {"B1", "B2", "B3"}:
        return "unused", "Migration is not included in this model."
    if parameter == "N_AFR" and model_id == "B1":
        return "constrained", "N_AFR = Na."
    if parameter == "N_EUR_FINAL" and model_id in {"B1", "B2"}:
        return "constrained", "N_EUR_FINAL = N_EUR_INITIAL."
    if parameter in {"m_EUR_AFR", "m_AFR_EUR"} and model_id == "C1":
        return "constrained", "m_EUR_AFR = m_AFR_EUR."
    if parameter == "Na":
        return "derived", "Derived from theta = 4 * Na * mu * L."
    if parameter == "theta":
        return "derived", "Optimal multinomial scaling of the full 3D model SFS."
    return "fitted", ""


def _harmonized_source_parameter(
    model_id: str, parameter: str, default: str
) -> str:
    if parameter == "N_AFR" and model_id == "B1":
        return "nu_growth=1; N_AFR = Na"
    if parameter in {"N_EUR_INITIAL", "N_EUR_FINAL"} and model_id in {"B1", "B2"}:
        return "nu2 * Na"
    if parameter in {"m_EUR_AFR", "m_AFR_EUR"} and model_id == "C1":
        return "m / (2 * Na)"
    return default


def _harmonized_parameter_table(physical: pd.DataFrame) -> pd.DataFrame:
    """Long publication table with identical physical-parameter rows for B1-C2."""
    physical_by_model = physical.set_index("model_id", drop=False)
    rows: list[dict] = []
    for model_id, spec in MODEL_SPECS.items():
        physical_row = physical_by_model.loc[model_id]
        for order, (parameter, metadata) in enumerate(
            HARMONIZED_PARAMETER_METADATA.items(), start=1
        ):
            status, constraint_detail = _harmonized_parameter_status(
                model_id, parameter
            )
            best = (
                math.nan
                if status == "unused"
                else float(physical_row[metadata["physical_column"]])
            )
            ci_low = math.nan
            ci_high = math.nan
            estimate_source = (
                "Saved best point evaluated at direct pts=85 with model-specific "
                "optimal full-3D theta."
            )
            ci_source = "Not available in the saved B1-C1 results."
            if model_id == "C2":
                best, ci_low, ci_high = C2_REPORTED_PARAMETER_INTERVALS[parameter]
                estimate_source = "User-supplied C2 full-model workbook."
                ci_source = "User-supplied 2.5% and 97.5% percentile intervals."
                if parameter == "m_AFR_EUR":
                    ci_source += (
                        " Retained as reported; not recalculated from the legacy m21 vector."
                    )
            if status == "unused":
                ci_source = "Not applicable because the parameter is unused."
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": spec.label,
                    "parameter_order": order,
                    "parameter": parameter,
                    "best_ll_estimate": best,
                    "ci_2_5": ci_low,
                    "ci_97_5": ci_high,
                    "units": metadata["units"],
                    "explanation": metadata["explanation"],
                    "source_parameter": _harmonized_source_parameter(
                        model_id, parameter, metadata["source_parameter"]
                    ),
                    "parameter_status": status,
                    "constraint_detail": constraint_detail,
                    "estimate_source": estimate_source,
                    "ci_source": ci_source,
                }
            )
    return pd.DataFrame(rows)


def _comparison_table(evaluated, *, godambe_adjustment: float | None = None) -> pd.DataFrame:
    rows = []
    for null_id, alt_id, change in (
        ("B1", "B2", "add ancestral AFR size-change epoch"),
        ("B2", "B3", "add EUR bottleneck/recovery"),
        ("B3", "C1", "add one shared bidirectional migration rate"),
        ("C1", "C2", "replace shared migration with two directional rates"),
    ):
        null = evaluated[null_id]
        alt = evaluated[alt_id]
        raw_d = 2.0 * (alt["ll_joint"] - null["ll_joint"])
        compatibility_d = 2.0 * (
            alt["ll_compatibility"] - null["ll_compatibility"]
        )
        row = {
            "null_model_id": null_id,
            "alternative_model_id": alt_id,
            "null_log_likelihood_full_3D": null["ll_joint"],
            "alternative_log_likelihood_full_3D": alt["ll_joint"],
            "2_delta_log_likelihood_full_3D": raw_d,
            "full_3D_report_grid_point": REPORT_GRID_POINT,
            "null_log_likelihood_full_3D_extrap_35_45_55": null[
                "ll_compatibility"
            ],
            "alternative_log_likelihood_full_3D_extrap_35_45_55": alt[
                "ll_compatibility"
            ],
            "2_delta_log_likelihood_full_3D_extrap_35_45_55": compatibility_d,
            "null_log_likelihood_AoU_1D": null["ll_aou"],
            "alternative_log_likelihood_AoU_1D": alt["ll_aou"],
            "2_delta_log_likelihood_AoU_1D": 2.0 * (alt["ll_aou"] - null["ll_aou"]),
            "nested_change": change,
            "lrt_df": math.nan,
            "unadjusted_lrt_p_value": math.nan,
            "unadjusted_lrt_p_value_extrap_35_45_55": math.nan,
            "godambe_adjustment_recomputed": math.nan,
            "godambe_adjusted_D_recomputed": math.nan,
            "godambe_adjusted_p_value_recomputed": math.nan,
            "godambe_adjusted_D_reported": math.nan,
            "reported_p_value_bound": "",
            "notes": "Sequential saved-point comparison; not an LRT unless stated.",
        }
        if (null_id, alt_id) == ("C1", "C2"):
            row["lrt_df"] = 1
            row["unadjusted_lrt_p_value"] = _chi2_df1_sf(raw_d)
            row["unadjusted_lrt_p_value_extrap_35_45_55"] = _chi2_df1_sf(
                compatibility_d
            )
            row["godambe_adjusted_D_reported"] = REPORTED_GODAMBE_ADJUSTED_D
            row["reported_p_value_bound"] = REPORTED_GODAMBE_P_BOUND
            row["notes"] = (
                "C1 is the interior equality constraint m12=m21 (one df); "
                "the boundary-mixture weights used for a no-migration null do "
                "not apply. The historical reported Godambe D is retained "
                "separately for comparison because it may correspond to the "
                "prior C1 point; recomputed columns use the committed bootstrap "
                "joint SFSs and the current C1/C2 points."
            )
            if godambe_adjustment is not None:
                row["godambe_adjustment_recomputed"] = godambe_adjustment
                adjusted_d = godambe_adjustment * compatibility_d
                row["godambe_adjusted_D_recomputed"] = adjusted_d
                row["godambe_adjusted_p_value_recomputed"] = _chi2_df1_sf(adjusted_d)
        rows.append(row)
    return pd.DataFrame(rows)


def _aou_sfs_table(aou_data, evaluated) -> pd.DataFrame:
    rows = []
    spectra = OrderedDict([("observed", {"aou_spectrum": aou_data, "ll_aou": np.nan})])
    spectra.update(evaluated)
    for spectrum_id, result in spectra.items():
        spectrum = result["aou_spectrum"]
        values = _filled(spectrum)
        mask = np.ma.getmaskarray(spectrum)
        for allele_count, (value, is_masked) in enumerate(zip(values, mask)):
            rows.append(
                {
                    "spectrum_id": spectrum_id,
                    "spectrum_label": (
                        "Observed AoU marginal" if spectrum_id == "observed"
                        else MODEL_SPECS[spectrum_id].label
                    ),
                    "population": "AoU",
                    "derived_allele_count": allele_count,
                    "value": value,
                    "is_masked": bool(is_masked),
                    "log_likelihood_AoU_1D": result["ll_aou"],
                    "sfs_scale": "observed" if spectrum_id == "observed"
                    else "optimal theta to observed AoU 1D SFS",
                    "model_joint_grid_point": (
                        math.nan if spectrum_id == "observed" else REPORT_GRID_POINT
                    ),
                }
            )
    return pd.DataFrame(rows)


@contextmanager
def _bootstrap_sfs_directory(bootstrap_source: Path):
    """Yield a directory of 100 SFSs from a directory or tar archive.

    Archive members are flattened into a private temporary directory after
    validating their basenames, so path traversal and notebook checkpoints
    cannot enter the Godambe input set.
    """
    source = Path(bootstrap_source)
    if source.is_dir():
        yield source
        return
    if not source.is_file():
        raise FileNotFoundError(f"Bootstrap SFS source does not exist: {source}")
    if not tarfile.is_tarfile(source):
        raise ValueError(
            f"Bootstrap SFS source is neither a directory nor a tar archive: {source}"
        )
    with (
        tarfile.open(source, mode="r:*") as archive,
        tempfile.TemporaryDirectory(prefix="aou_s1_bootstrap_sfs_") as tmp,
    ):
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and re.fullmatch(r"(?:.*/)?\d+_AFR_EUR_AOU\.fs", member.name)
        ]
        indices = sorted(
            int(re.match(r"(\d+)_", Path(member.name).name).group(1))
            for member in members
        )
        if indices != list(range(100)):
            raise ValueError(
                f"Expected bootstrap SFS indices 0..99 in {source}; found {indices}"
            )
        target = Path(tmp)
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read archive member {member.name}")
            with stream, (target / Path(member.name).name).open("wb") as handle:
                shutil.copyfileobj(stream, handle)
        yield target


def _bootstrap_spectra(bootstrap_source: Path) -> list:
    dadi = _dadi()
    with _bootstrap_sfs_directory(bootstrap_source) as bootstrap_dir:
        files = list(bootstrap_dir.glob("*_AFR_EUR_AOU.fs"))
        files.sort(key=lambda path: int(re.match(r"(\d+)_", path.name).group(1)))
        if len(files) != 100:
            raise ValueError(
                f"Expected 100 bootstrap SFS files in {bootstrap_source}; "
                f"found {len(files)}"
            )
        return [dadi.Spectrum.from_file(str(path)) for path in files]


def recompute_c1_c2_godambe_adjustment(bootstrap_source: Path) -> float:
    """Recompute the full-joint-SFS Godambe adjustment from 100 bootstraps."""
    dadi = _dadi()
    data = dadi.Spectrum.from_file(str(config.DEMOGRAPHY_SFS))
    bootstraps = _bootstrap_spectra(bootstrap_source)
    c2 = MODEL_SPECS["C2"]
    raw = dict(zip(c2.parameter_names, c2.parameters))
    transformed = np.array(
        [raw["nu_growth"], raw["nu2B"], raw["nu2F"], raw["Tgrowth"],
         raw["T1"], raw["f"], raw["m12"], raw["m21"] - raw["m12"], c2.misid]
    )
    wrapped = dadi.Numerics.make_anc_state_misid_func(c1_c2_equality_lrt_model)
    extrapolated = dadi.Numerics.make_extrap_func(wrapped)
    adjustment = dadi.Godambe.LRT_adjust(
        extrapolated, list(GRID_POINTS), bootstraps, transformed, data,
        nested_indices=[7], multinom=True,
    )
    return float(np.asarray(adjustment).squeeze())


def recompute_source_tables(*, bootstrap_dir: Path | None = None) -> list[Path]:
    """Recompute source tables from saved points; never optimize parameters."""
    joint_data, aou_data, evaluated, references = _evaluate_models()
    godambe = (
        recompute_c1_c2_godambe_adjustment(bootstrap_dir)
        if bootstrap_dir is not None else None
    )
    physical_parameters = _wide_physical_parameter_table(evaluated)
    tables = (
        (_summary_table(joint_data, aou_data, evaluated), config.DEMOGRAPHY_S1_MODEL_SUMMARY, False),
        (_parameter_table(evaluated), config.DEMOGRAPHY_S1_PARAMETERS, False),
        (_wide_parameter_table(evaluated), config.DEMOGRAPHY_S1_WIDE_PARAMETERS, False),
        (
            physical_parameters,
            config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS,
            False,
        ),
        (
            _harmonized_parameter_table(physical_parameters),
            config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS,
            False,
        ),
        (_comparison_table(evaluated, godambe_adjustment=godambe), config.DEMOGRAPHY_S1_COMPARISONS, False),
        (_candidate_checks(evaluated, references), config.DEMOGRAPHY_S1_CANDIDATE_CHECKS, False),
        (_aou_sfs_table(aou_data, evaluated), config.DEMOGRAPHY_S1_MARGINAL_SFS, True),
    )
    config.DEMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path, compress in tables:
        kwargs = {"sep": "\t", "index": False, "float_format": "%.10g"}
        if compress:
            kwargs["compression"] = {"method": "gzip", "compresslevel": 9, "mtime": 0}
        frame.to_csv(path, **kwargs)

    output_hashes = {
        path.name: _sha256(path) for path in _source_paths()
        if path != config.DEMOGRAPHY_S1_PROVENANCE and path.exists()
    }
    provenance = {
        **_contract_payload(),
        "contract_hash": _contract_hash(),
        "harmonized_contract": _harmonized_contract_payload(),
        "harmonized_contract_hash": _harmonized_contract_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation": (
            "saved parameters -> common nested dadi model -> ancestry mis-ID -> "
            "direct pts=85 spectrum -> independent optimal scaling and LL for "
            "full joint SFS and AoU marginal; 35/45/55 extrapolation retained "
            "as a numerical compatibility diagnostic"
        ),
        "optimization_performed": False,
        "godambe_bootstrap_source": (
            _provenance_source_path(bootstrap_dir) if bootstrap_dir else None
        ),
        "godambe_bootstrap_source_sha256": (
            _sha256(bootstrap_dir)
            if bootstrap_dir and bootstrap_dir.is_file()
            else None
        ),
        "outputs_sha256": output_hashes,
    }
    config.DEMOGRAPHY_S1_PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return list(_source_paths())


def refresh_harmonized_parameter_table() -> Path:
    """Refresh the publication parameter table from committed physical estimates."""
    if not _base_source_tables_current():
        raise RuntimeError(
            "Core S1 source tables are stale; run the full dadi regeneration first."
        )
    physical = pd.read_csv(
        config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS, sep="\t"
    )
    harmonized = _harmonized_parameter_table(physical)
    harmonized_tsv = harmonized.to_csv(
        None,
        sep="\t",
        index=False,
        float_format="%.10g",
    )
    harmonized_bytes = harmonized_tsv.encode("utf-8")
    current_tsv = (
        config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS.read_bytes()
        if config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS.exists()
        else None
    )
    table_changed = current_tsv != harmonized_bytes
    if table_changed:
        config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS.write_bytes(harmonized_bytes)
    provenance = json.loads(
        config.DEMOGRAPHY_S1_PROVENANCE.read_text(encoding="utf-8")
    )
    output_hashes = {
        path.name: _sha256(path)
        for path in _source_paths()
        if path != config.DEMOGRAPHY_S1_PROVENANCE and path.exists()
    }
    metadata_changed = (
        provenance.get("harmonized_contract_hash") != _harmonized_contract_hash()
        or provenance.get("outputs_sha256") != output_hashes
    )
    if not table_changed and not metadata_changed:
        return config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS
    provenance["harmonized_contract"] = _harmonized_contract_payload()
    provenance["harmonized_contract_hash"] = _harmonized_contract_hash()
    provenance["harmonized_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["outputs_sha256"] = output_hashes
    config.DEMOGRAPHY_S1_PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS


def ensure_source_tables(*, force: bool = False, bootstrap_dir: Path | None = None) -> list[Path]:
    if not force and bootstrap_dir is None:
        if source_tables_current():
            return list(_source_paths())
        if _base_source_tables_current():
            refresh_harmonized_parameter_table()
            return list(_source_paths())
    return recompute_source_tables(bootstrap_dir=bootstrap_dir)


def _physical_lookup(parameters: pd.DataFrame, model_id: str) -> dict[str, float]:
    subset = parameters[parameters["model_id"] == model_id]
    return dict(zip(subset["physical_parameter"], subset["physical_value"]))


def _annotation(model_id: str, summary: pd.DataFrame, parameters: pd.DataFrame) -> str:
    row = summary.loc[summary["model_id"] == model_id].iloc[0]
    phys = _physical_lookup(parameters, model_id)
    head = (
        f"{model_id}: {row['model_label']}\n"
        f"LL3D = {row['log_likelihood_full_3D']:,.1f}; "
        f"LLAoU = {row['log_likelihood_AoU_1D']:,.1f}"
    )
    if model_id == "B1":
        body = (
            f"Na = {phys['Na']:,.0f}; N_EUR = {phys['N_EUR']:,.0f}\n"
            f"split = {phys['T_AFR_EUR_split']/1000:,.1f} kya\n"
            f"AFR fraction = {phys['admixture_fraction_AFR']:.3f}"
        )
    elif model_id == "B2":
        body = (
            f"Na = {phys['Na']:,.0f}; N_AFR = {phys['N_AFR_postchange']:,.0f}\n"
            f"N_EUR = {phys['N_EUR']:,.0f}\n"
            f"ancestral change = {phys['T_ancestral_size_change']/1000:,.1f} kya\n"
            f"split = {phys['T_AFR_EUR_split']/1000:,.1f} kya; "
            f"fAFR = {phys['admixture_fraction_AFR']:.3f}"
        )
    else:
        body = (
            f"Na = {phys['Na']:,.0f}; N_AFR = {phys['N_AFR_postchange']:,.0f}\n"
            f"N_EUR = {phys['N_EUR_initial']:,.0f} to {phys['N_EUR_final']:,.0f}\n"
            f"ancestral change = {phys['T_ancestral_size_change']/1000:,.1f} kya\n"
            f"split = {phys['T_AFR_EUR_split']/1000:,.1f} kya; "
            f"fAFR = {phys['admixture_fraction_AFR']:.3f}"
        )
    return head + "\n" + body


def demography_figure_paths() -> list[Path]:
    output_dir = config.figure_dir("figure1") / "models"
    return [
        output_dir / "Figure_S1A_Demographic_Models.png",
        output_dir / "Figure_S1A_Demographic_Models.pdf",
    ]


def streamlined_demography_figure_paths() -> list[Path]:
    output_dir = config.figure_dir("figure1") / "models"
    return [
        output_dir / "Figure_S1A_Demographic_Models_Streamlined.png",
        output_dir / "Figure_S1A_Demographic_Models_Streamlined.pdf",
    ]


def sfs_figure_paths() -> list[Path]:
    output_dir = config.figure_dir("figure1") / "models"
    return [
        output_dir / "Figure_S1B_AoU_SFS.png",
        output_dir / "Figure_S1B_AoU_SFS.pdf",
    ]


def figure_paths() -> list[Path]:
    return [
        *demography_figure_paths(),
        *streamlined_demography_figure_paths(),
        *sfs_figure_paths(),
    ]


def build_demes_graph(row: pd.Series):
    """Build one standard Demes graph from a physical-units source-data row.

    AFR is deliberately the root deme and remains extant at time zero. Historical
    African size changes are represented as epochs of AFR rather than as a
    separate ancestral population. AoU's positive duration is only a plotting
    floor because the fitted dadi contract has instantaneous admixture (T2=0).
    """
    import demes

    model_id = str(row["model_id"])
    na = float(row["Na"])
    n_afr = float(row["N_AFR"])
    n_eur_initial = float(row["N_EUR_initial"])
    n_eur_final = float(row["N_EUR_final"])
    split_time = float(row["T_AFR_EUR_split_years_before_present"])
    aou_plot_start_time = AOU_PLOT_START_TIME_YEARS
    change_time_raw = row["T_ancestral_size_change_years_before_present"]
    change_time = None if pd.isna(change_time_raw) else float(change_time_raw)

    builder = demes.Builder(
        description=(
            f"Supplementary Figure S1 {model_id}: {row['model_label']}. "
            "Physical units derived from the fitted dadi theta."
        ),
        time_units="years",
        generation_time=GENERATION_TIME_YEARS,
    )
    if change_time is None:
        afr_epochs = [{"start_size": na}]
    else:
        afr_epochs = [
            {"start_size": na, "end_time": change_time},
            {"start_size": n_afr},
        ]
    builder.add_deme(
        "AFR",
        description="Continuous African lineage, including the ancestral epoch",
        epochs=afr_epochs,
    )
    builder.add_deme(
        "EUR",
        ancestors=["AFR"],
        start_time=split_time,
        epochs=[{"start_size": n_eur_initial, "end_size": n_eur_final}],
    )
    builder.add_deme(
        "AoU",
        description=(
            "Instantaneously admixed AoU population; its positive duration is "
            "a plotting floor and its displayed size is not a fitted parameter"
        ),
        ancestors=["AFR", "EUR"],
        proportions=[float(row["f_AFR"]), 1.0 - float(row["f_AFR"])],
        start_time=aou_plot_start_time,
        epochs=[{"start_size": na}],
    )

    m12 = row["m12_EUR_to_AFR_per_generation"]
    m21 = row["m21_AFR_to_EUR_per_generation"]
    if not pd.isna(m12):
        builder.add_migration(
            source="EUR",
            dest="AFR",
            rate=float(m12),
            start_time=split_time,
            end_time=aou_plot_start_time,
        )
    if not pd.isna(m21):
        builder.add_migration(
            source="AFR",
            dest="EUR",
            rate=float(m21),
            start_time=split_time,
            end_time=aou_plot_start_time,
        )
    return builder.resolve()


def _draw_demes_model(
    ax,
    row: pd.Series,
    *,
    show_y_label: bool,
    show_title: bool = True,
    title_fontsize: float = 13.5,
    axis_label_fontsize: float = 12,
    tick_label_fontsize: float = 11,
    deme_label_fontsize: float = 12,
) -> None:
    import demesdraw

    graph = build_demes_graph(row)
    separation = demesdraw.utils.separation_heuristic(graph)
    positions = {"AFR": 0.0, "AoU": separation, "EUR": 2.0 * separation}
    demesdraw.tubes(
        graph,
        ax=ax,
        colours=POPULATION_COLORS,
        log_time=True,
        title=f"{row['model_id']}  {row['model_label']}",
        inf_ratio=0.12,
        positions=positions,
        num_lines_per_migration=4,
        seed=DEMESDRAW_SEED,
        max_time=DEMESDRAW_MAX_TIME_YEARS,
        labels="xticks-mid",
        fill=True,
        scale_bar=True,
    )
    if show_title:
        ax.set_title(
            ax.get_title(),
            fontsize=title_fontsize,
            fontweight="bold",
            color=COLORS[str(row["model_id"])],
            pad=9,
        )
    else:
        ax.set_title("")
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.set_xlabel("")
    ax.set_ylabel(
        "Time before present (years)" if show_y_label else "",
        fontsize=axis_label_fontsize,
    )
    for label in ax.get_xticklabels():
        label.set_fontsize(deme_label_fontsize)
        label.set_fontweight("bold")


def render_demographic_models(*, force: bool = False) -> list[Path]:
    """Render S1A with native ``demesdraw.tubes`` in physical units."""
    outputs = demography_figure_paths()
    if not force and all(path.exists() and path.stat().st_size > 0 for path in outputs):
        newest_input = max(path.stat().st_mtime for path in _source_paths())
        if min(path.stat().st_mtime for path in outputs) >= newest_input:
            return outputs

    physical = pd.read_csv(config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS, sep="\t")
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.2))
    axes_flat = axes.ravel()
    for index, (_, row) in enumerate(physical.iterrows()):
        _draw_demes_model(axes_flat[index], row, show_y_label=index % 3 == 0)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    legend_ax.text(
        0.02,
        0.94,
        "Nested model progression",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    legend_ax.text(
        0.02,
        0.84,
        "B1  constant AFR + EUR sizes\n"
        "B2  + historical AFR size change\n"
        "B3  + EUR bottleneck/recovery\n"
        "C1  + shared bidirectional migration\n"
        "C2  + two directional migration rates",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        linespacing=1.55,
    )
    legend_ax.text(
        0.02,
        0.49,
        "DemesDraw encoding",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    legend_ax.text(
        0.02,
        0.41,
        "Tube width is proportional to population size\n"
        "Blue: AFR   Orange: EUR   Green: AoU\n"
        "AFR is one continuous root deme; size changes are epochs\n"
        "Dashed branches show AoU ancestry; thin lines show migration\n"
        f"AoU display floor: {AOU_PLOT_START_TIME_YEARS:g} years (fitted T2 = 0)\n"
        "AoU display width is not an inferred population size",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.7,
        linespacing=1.45,
    )
    legend_ax.text(
        0.02,
        0.025,
        "Conversions: L=481,020,725 bp; mu=1.5e-8/bp/gen; 25 years/gen",
        transform=legend_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#4A4A4A",
    )
    fig.suptitle(
        "Supplementary Figure S1A - nested demographic models",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.08,
        top=0.92,
        hspace=0.34,
        wspace=0.18,
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return outputs


def render_streamlined_demographic_models(*, force: bool = False) -> list[Path]:
    """Render a title-free, tightly packed S1A variant with enlarged labels."""
    outputs = streamlined_demography_figure_paths()
    if not force and all(
        path.exists() and path.stat().st_size > 0 for path in outputs
    ):
        newest_input = max(path.stat().st_mtime for path in _source_paths())
        if min(path.stat().st_mtime for path in outputs) >= newest_input:
            return outputs

    physical = pd.read_csv(config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS, sep="\t")
    rows = {str(row["model_id"]): row for _, row in physical.iterrows()}
    fig = plt.figure(figsize=(17.5, 9.5))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.05,
        right=0.99,
        bottom=0.055,
        top=0.98,
        hspace=0.31,
        wspace=0.2,
    )
    panel_slots = {
        "B1": (0, slice(0, 2)),
        "B2": (0, slice(2, 4)),
        "B3": (0, slice(4, 6)),
        "C1": (1, slice(1, 3)),
        "C2": (1, slice(3, 5)),
    }
    for model_id, (row_index, column_slice) in panel_slots.items():
        ax = fig.add_subplot(grid[row_index, column_slice])
        _draw_demes_model(
            ax,
            rows[model_id],
            show_y_label=model_id in {"B1", "C1"},
            show_title=False,
            title_fontsize=19.5,
            axis_label_fontsize=17,
            tick_label_fontsize=15,
            deme_label_fontsize=18,
        )

    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(
            path,
            dpi=300 if path.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)
    return outputs


def render_model_sfs(*, force: bool = False) -> list[Path]:
    outputs = sfs_figure_paths()
    if not force and all(path.exists() and path.stat().st_size > 0 for path in outputs):
        newest_input = max(path.stat().st_mtime for path in _source_paths())
        if min(path.stat().st_mtime for path in outputs) >= newest_input:
            return outputs

    summary = pd.read_csv(config.DEMOGRAPHY_S1_MODEL_SUMMARY, sep="\t")
    parameters = pd.read_csv(config.DEMOGRAPHY_S1_PARAMETERS, sep="\t")
    sfs = pd.read_csv(config.DEMOGRAPHY_S1_MARGINAL_SFS, sep="\t")
    display_models = ("B1", "B2", "B3")
    fig = plt.figure(figsize=(15, 10.5))
    grid = fig.add_gridspec(3, 4, width_ratios=(1, 1, 1, 1.34), hspace=0.38, wspace=0.35)
    ax = fig.add_subplot(grid[:, :3])
    observed = sfs[(sfs["spectrum_id"] == "observed") & ~sfs["is_masked"] & (sfs["value"] > 0)]
    ax.scatter(observed["derived_allele_count"], observed["value"], s=70,
               facecolor="black", edgecolor="white", linewidth=0.8, zorder=5,
               label="Observed AoU SFS")
    for model_id in display_models:
        subset = sfs[(sfs["spectrum_id"] == model_id) & ~sfs["is_masked"] & (sfs["value"] > 0)]
        ll = summary.loc[summary["model_id"] == model_id, "log_likelihood_full_3D"].iloc[0]
        ax.plot(subset["derived_allele_count"], subset["value"], color=COLORS[model_id],
                linewidth=3.2,
                label=f"{model_id} (full 3D LL, pts={REPORT_GRID_POINT}: {ll:,.1f})")
    ax.set_yscale("log")
    ax.set_title("AoU one-dimensional site-frequency spectrum", fontsize=22, fontweight="bold")
    ax.set_xlabel("Derived allele count", fontsize=18)
    ax.set_ylabel("Observed count / optimally scaled expectation", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=13, frameon=False, loc="best")

    for axis_index, model_id in enumerate(display_models):
        text_ax = fig.add_subplot(grid[axis_index, 3])
        text_ax.axis("off")
        text_ax.text(0.02, 0.96, _annotation(model_id, summary, parameters),
                     transform=text_ax.transAxes, ha="left", va="top", fontsize=12.2,
                     linespacing=1.32,
                     bbox={"boxstyle": "round,pad=0.62", "facecolor": COLORS[model_id],
                           "alpha": 0.12, "edgecolor": COLORS[model_id], "linewidth": 2})
    fig.suptitle("Supplementary Figure S1B - nested demographic fits",
                 fontsize=23, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.88)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return outputs


def run(*, clean_run_data: bool = False, clean_run_plots: bool = True,
        bootstrap_dir: Path | None = None) -> list[Path]:
    source_paths = ensure_source_tables(force=clean_run_data, bootstrap_dir=bootstrap_dir)
    plot_paths = [
        *render_demographic_models(force=clean_run_plots),
        *render_streamlined_demographic_models(force=clean_run_plots),
        *render_model_sfs(force=clean_run_plots),
    ]
    return [*source_paths, *plot_paths]


def main(argv: Iterable[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-data", action="store_true")
    parser.add_argument("--reuse-plots", action="store_true")
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        help=(
            "Directory or .tar.gz archive containing "
            "0..99_AFR_EUR_AOU.fs for the Godambe adjustment"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    for path in run(clean_run_data=args.recompute_data,
                    clean_run_plots=not args.reuse_plots,
                    bootstrap_dir=args.bootstrap_dir):
        print(path.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
