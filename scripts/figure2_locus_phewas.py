"""Verified discovery admixture tests and the shared selection-hit PheWAS style."""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.text import Text

from . import admixture_phewas, figure2_1025 as corrected
from . import figure2_discovery_phewas as discovery_plots
from . import figure5_locus_composites as selection_plots

THRESHOLD = admixture_phewas.DISCOVERY_P_THRESHOLD
EFFECT_BASIS = "OR = exp(discovery beta), per +1 predicted EUR ancestry dosage (0-2)"
FONT_SCALE = 1.6
LABEL_WRAP = 26
CATEGORY_FONT_SIZE = 7.5 * FONT_SCALE
CATEGORY_LABEL_GAP_PT = 3.0
CATEGORY_ORDER = (
    "Blood/Immune", "Neoplasms", "Cardiovascular", "Dermatological", "Endocrine/Metab",
    "Genetic", "Gastrointestinal", "Genitourinary", "Infectious", "Mental",
    "Musculoskeletal", "Neurological", "Pregnancy complications", "Respiratory",
    "Sense organs", "Symptoms",
)
CATEGORY_LABELS = {category: category for category in CATEGORY_ORDER} | {
    "Endocrine/Metab": "Endocrine/metabolic", "Infectious": "Infections",
    "Neurological": "Neuro", "Pregnancy complications": "Pregnancy", "Symptoms": "Signs/symptoms",
}
CATEGORY_ALIASES = {category.casefold(): category for category in CATEGORY_ORDER} | {
    "muscloskeletal": "Musculoskeletal", "endocrine/metabolic": "Endocrine/Metab",
    "neuro": "Neurological", "infections": "Infectious", "pregnancy": "Pregnancy complications",
    "signs/symptoms": "Symptoms",
}


def _canonical_category(value):
    name = " ".join(str(value).split())
    return CATEGORY_ALIASES.get(name.casefold(), name)


def _category_order(values):
    present = {_canonical_category(value) for value in values}
    return ([category for category in CATEGORY_ORDER if category in present]
            + sorted(present.difference(CATEGORY_ORDER), key=str.casefold))


def _category_widths(axis, frame, categories, labels):
    """Reserve measured label clearance; share remaining space by test count."""
    counts = frame["phecode_category"].value_counts()
    present = [category for category in categories if category in counts]
    renderer = axis.figure.canvas.get_renderer()
    label_height = max(Text(text=labels[category], fontsize=CATEGORY_FONT_SIZE, figure=axis.figure)
                       .get_window_extent(renderer).height for category in present)
    # The perpendicular separation of parallel rotated labels is dx * sin(angle).
    # Keeping that above the text height prevents even singleton categories from
    # colliding, without the unnecessary gaps implied by axis-aligned text boxes.
    gap = CATEGORY_LABEL_GAP_PT * axis.figure.dpi / 72
    rotation = np.deg2rad(selection_plots.PHEWAS_CATEGORY_LABEL_ROTATION)
    minimum = (label_height + gap) / (np.sin(rotation) * axis.bbox.width)
    remaining = 1 - len(present) * minimum
    if remaining < 0:
        raise ValueError("PheWAS panel is too narrow for readable category labels")
    return {category: minimum + remaining * counts[category] / counts.sum() for category in present}


class DiscoveryPhewas:
    """Load full discovery tests, verifying the cohort and saved file hashes.

    Annotation sources supply names/categories only. All p-values and effect
    estimates come exclusively from the corrected admixture discovery cache.
    This phase performs no staging, phenotype regression, or model fitting.
    """

    def __init__(self, hits):
        path = corrected.analysis_root() / "phewas" / "phewas.discovery.tsv.gz"
        metadata_path = path.with_suffix(path.suffix + ".manifest.json")
        stage_path = corrected._stage_manifest_path("discovery_phewas")
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        contract = stage["contract"]
        if contract.get("cohort") != corrected._cohort_contract("discovery"):
            raise ValueError("Discovery PheWAS manifest does not match the corrected discovery cohort")
        if not corrected._stage_current("discovery_phewas", contract, [path, metadata_path]):
            raise ValueError(f"Missing or corrupt discovery PheWAS cache: {path}; repair the corrected discovery stage")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {"analysis_contract": contract["analysis_contract"],
                    "target_contract": admixture_phewas._target_contract(None)}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Discovery PheWAS metadata is not the full discovery analysis")
        frame = pd.read_csv(path, sep="\t", dtype={"phecode": str, "region_id": str})
        if not set(admixture_phewas.ASSOCIATION_COLUMNS).issubset(frame) or frame.empty:
            raise ValueError("Missing full discovery PheWAS columns or tests")
        if (not frame["cohort"].eq("discovery").all() or not frame["n_pcs"].eq(20).all()
                or any(not frame[key].eq(value).all() for key, value in expected.items())
                or frame.duplicated(["region_id", "phecode"]).any()):
            raise ValueError("Discovery PheWAS rows have inconsistent analysis provenance or duplicate tests")
        frame = frame.loc[frame["region_id"].isin(hits["region_id"]), admixture_phewas.ASSOCIATION_COLUMNS]
        override = os.environ.get("AOU_FIG5_PHECODE_ANNOTATIONS")
        paths = [Path(override)] if override else discovery_plots._default_annotation_paths()
        if not paths:
            raise FileNotFoundError("No cached Phecode name/category table; set AOU_FIG5_PHECODE_ANNOTATIONS to a local annotation TSV/CSV")
        annotations, annotation_source = discovery_plots.load_annotations(paths)
        # Use exactly the same phenotype wording as the published hit table.
        descriptions = hits[["phecode", "phecode_string"]].drop_duplicates("phecode").set_index("phecode")["phecode_string"]
        annotations["phecode_name"] = annotations["phecode"].map(descriptions).fillna(annotations["phecode_name"])
        self.frame, annotation_order = discovery_plots.prepare_plot_data(frame, annotations, threshold=THRESHOLD)
        self.frame["phecode_category"] = self.frame["phecode_category"].map(_canonical_category)
        self.categories = _category_order(self.frame["phecode_category"])
        nominal = self.frame.loc[self.frame["discovery_significant"]].set_index(["region_id", "phecode"])
        wanted = hits.set_index(["region_id", "phecode"])
        if (set(nominal.index) != set(wanted.index) or
                not np.allclose(nominal.loc[wanted.index, "p_value"], wanted["p_disc"], rtol=1e-6, atol=0)):
            raise ValueError("Discovery PheWAS tests disagree with the requested discovery hit table")
        scales = selection_plots.resolve_shared_phewas_scales(self.frame, odds_ratio_cap=4.0)
        self.parameters = selection_plots.PlotParameters(
            phewas_threshold=THRESHOLD, phewas_y_max=scales["phewas_y_max"], dpi=300)
        self.source = {"discovery": corrected._output_fingerprint(path),
                       "discovery_contract": contract, "annotations": annotation_source,
                       "effect_basis": EFFECT_BASIS, "category_order": self.categories,
                       "annotation_category_order": annotation_order,
                       "category_labels": {category: CATEGORY_LABELS.get(category, category) for category in self.categories},
                       "category_layout": {"minimum_label_gap_pt": CATEGORY_LABEL_GAP_PT,
                           "font_size": CATEGORY_FONT_SIZE, "remaining_width": "proportional_to_test_count"},
                       "label_parameters": {"font_scale": FONT_SCALE, "wrap_columns": LABEL_WRAP,
                           "seed": int(os.environ.get("AOU_FIGURE_SEED", "20260715"))},
                       "parameters": asdict(self.parameters), "scales": scales}

    def region(self, hit):
        return self.frame.loc[self.frame["region_id"].eq(hit["region_id"])].copy()


def draw_panel(axis, frame, hit, categories, parameters):
    """Shared odds-ratio sizes, direction triangles, categories and repelled labels."""
    plotted = frame.copy()
    plotted["phecode_category"] = plotted["phecode_category"].map(_canonical_category)
    categories = _category_order([*categories, *plotted["phecode_category"]])
    category_labels = {category: CATEGORY_LABELS.get(category, category) for category in categories}
    category_widths = _category_widths(axis, plotted, categories, category_labels)
    plotted["phecode_name"] = plotted["phecode_name"].map(
        lambda value: textwrap.fill(str(value), width=LABEL_WRAP, break_long_words=False))
    # adjustText uses NumPy's RNG when labels coincide. Preserve caller state.
    state = np.random.get_state()
    np.random.seed(int(os.environ.get("AOU_FIGURE_SEED", "20260715")))
    try:
        selection_plots.draw_phewas_panel(
            axis, plotted, plotted.loc[plotted["p_value"] < THRESHOLD], hit,
            categories, parameters, highlight_label=None, nominal_label=None,
            threshold_inclusive=False, font_scale=FONT_SCALE,
            category_widths=category_widths, category_labels=category_labels)
    finally:
        np.random.set_state(state)
    axis.get_legend().set_title("OR fold (log scale)")
    axis.set_ylabel(selection_plots.style.P_VALUE_AXIS_LABEL, fontsize=20, parse_math=False)
    axis.tick_params(axis="y", labelsize=16)
    axis.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, np.floor(axis.get_ylim()[1]) + 1)))
    axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
