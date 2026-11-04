"""Figure 1E -- demographic inference (dadi 3D-SFS fit of the AFR/EUR/AoU model).

This module documents HOW the best-fit demographic model was chosen and exposes
the chosen parameters; it does not re-run the inference. The fit is expensive
(many optimizations from random starts), so for the companion repo we take the
already-selected best fit and only convert/serve it.

Provenance of the committed inputs (all tracked in git, under
``assets/demography/``):

  * ``3pop_AFR_NFE_AMR_combined.fs`` -- the observed 26x26x26 unfolded joint SFS
    (AFR, EUR, AoU), projected to 25 alleles/pop, that dadi was fit to.
  * ``3pop_2mig.yaml`` -- the best-fit model in ``demes`` format, already
    converted from dadi's scaled units to physical units (years, individuals,
    per-generation rates). This is the source of truth for the panel labels.
  * Figure 1E is drawn directly into the supplied Matplotlib axes by
    ``render_demographic_model_schematic(ax=...)``. It does not read or rewrite
    a tracked intermediate schematic image.
  * ``runs_3d_demo_fits_pulse_admix.txt.gz`` -- 100 bootstrap best-fit parameter
    sets (one dadi fit per line: loglik, the 8 model params, misid, theta). These
    drive the Fig 1F/1G bootstrap scatters and the parameter correlogram;
    ``load_bootstrap_fits`` converts them to physical units.
  * ``aou_sfs_fit.csv`` -- the AoU-marginal observed vs. C2 best-fit-model SFS +
    Poisson residuals, precomputed with dadi (``compute_aou_fit``: plug the data
    ``.fs`` and C2 parameters into ``pulse_admix_mig``, marginalize to AoU, then
    optimally rescale to that 1D SFS). Committed so the Fig 1F fit/residual panel
    needs no dadi at runtime.

The model: pop 1 (AFR) and pop 2 (EUR) split; at present (t=0) they admix
(fractions f, 1-f) into a third population (AoU). Sizes are ratios to the
ancestral size Na; times are in units of 2*Na generations. Two-parameter
migration (m12, m21) between AFR and EUR is included, and was justified over the
no-migration model by a Godambe-adjusted likelihood-ratio test (below).

Model specification (dadi), kept for the record -- NOT executed here (dadi is a
Workbench/analysis dependency; running it re-does the fit we are reusing):

    def pulse_admix_mig(params, ns, pts):
        '''
        Split between pop 1 (AFR) and pop 2 (EUR). At time=0, both admix into a
        third population (AoU).

        nu_growth : ancestral size change (growth); also pop 1 (AFR) size after
                    the split. All sizes are ratios to the ancestral size.
        nu2B      : initial size of pop 2 (EUR) after the split.
        nu2F      : final size of pop 2 (EUR).
        Tgrowth   : scaled time between ancestral growth and the split
                    (units of 2*Na generations).
        T1        : scaled time between the AFR/EUR split and the origin of AoU.
        f         : admixture fraction from pop 1 (AFR) into AoU.
        m12, m21  : migration into pop 1 from pop 2, and the reverse.
        '''
        nu_growth, nu2B, nu2F, Tgrowth, T1, f, m12, m21 = params
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.Integration.one_pop(phi, xx, Tgrowth, nu=nu_growth)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        nu2_func = lambda t: nu2B * (nu2F / nu2B) ** (t / T1)
        phi = dadi.Integration.two_pops(phi, xx, T1, nu1=nu_growth, nu2=nu2_func,
                                        m12=m12, m21=m21)
        phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
        fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx))
        return fs

    # The no-migration model is the same function evaluated with m12 = m21 = 0.

How the fit was chosen (descriptive; see ``lrt_migration_test.__doc__``):
    A migration and a no-migration version of the model were each optimized from
    many random starts. The migration model achieved the higher composite
    log-likelihood, and a Godambe-adjusted LRT on the two nested migration
    parameters (with 100 bootstrap SFS to correct for SFS non-independence)
    rejected the no-migration null. The migration best fit is therefore the model
    reported in Fig 1E.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

# Workbench can set the inline notebook backend, which is unavailable in the
# lightweight plotting environment used by the standalone renderer.
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from .common import config

# Parameter order of ``pulse_admix_mig`` (before the misid nuisance parameter).
PARAM_NAMES: tuple[str, ...] = (
    "nu_growth", "nu2B", "nu2F", "Tgrowth", "T1", "f", "m12", "m21",
)

# Columns of one bootstrap-fit line: composite log-likelihood, the 8 model
# parameters, the misid nuisance parameter, and dadi's optimal theta.
BOOTSTRAP_COLUMNS: tuple[str, ...] = ("loglik", *PARAM_NAMES, "misid", "theta")

# Reference scaling for raw -> physical conversion. dadi sizes/times are relative
# to the ancestral reference Ne = theta / (4*mu*L); since 4*mu*L is fixed by the
# data, Ne scales linearly with a fit's theta. We pin the constant with the
# committed best fit: NA_REF individuals at THETA_REF (both reproduce the demes
# output exactly -- verified against 3pop_2mig.yaml).
NA_REF = 11727.0            # ancestral Ne of the best fit (demes AFR epoch 0)
THETA_REF = 338448.7526     # dadi optimal theta of the best fit

# Best-fit raw (dadi-scaled) parameters, found by multiple optimizations. The
# trailing value in each array is the ancestral-state misidentification rate
# added by ``make_anc_state_misid_func``; m12/m21 are 0 in the no-mig fit.
# Migration best fit, composite log-likelihood -35290.57201:
BEST_FIT_MODEL_ID = "C2"
BEST_FIT_MIG = np.array([
    1.558584501, 0.078546083, 0.727276039, 0.344734944, 0.075261347,
    0.801915874, 1.741892163, 1.219863597, 0.040393927,
])
# No-migration best fit (m12 = m21 = 0):
BEST_FIT_NOMIG = np.array([
    1.558584501, 0.078546083, 0.727276039, 0.344734944, 0.075261347,
    0.801915874, 0.0, 0.0, 0.040393927,
])
BEST_FIT_MIG_LOGLIK = -35290.57201


def best_fit_parameter_sha256() -> str:
    """Stable fingerprint of the C2 raw dadi vector, including ``p_misid``."""
    little_endian = np.asarray(BEST_FIT_MIG, dtype="<f8")
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def lrt_migration_test() -> None:
    r"""Godambe-adjusted LRT of migration vs. no-migration -- descriptive only.

    NOT run in this main-figure plotting module. The 100 spectra are committed
    in ``assets/demography/s1_demography_bootstrap_sfs.tar.gz`` and the active
    implementation is
    ``scripts.figure1_model_fits.recompute_c1_c2_godambe_adjustment``.
    Kept here as a compact description of the model-selection procedure::

        data = dadi.Spectrum.from_file(config.DEMOGRAPHY_SFS)
        all_boot = [dadi.Spectrum.from_file(f"fs/{i}_AFR_EUR_AOU.fs")
                    for i in range(100)]

        demo_model = dadi.Numerics.make_anc_state_misid_func(pulse_admix_mig)
        demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)

        ns = [25, 25, 25]
        pts_l = [max(ns) + 10, max(ns) + 20, max(ns) + 30]

        # No-migration best fit (m12 = m21 = 0).
        model_nomig = demo_model_ex(BEST_FIT_NOMIG, ns, pts_l)
        ll_nomig = dadi.Inference.ll_multinom(model_nomig, data)

        # Complex (migration) model evaluated at its best fit; nested params are
        # the two migration rates at indices 6 and 7.
        model = demo_model_ex(BEST_FIT_MIG, ns, pts_l)
        ll_model = dadi.Inference.ll_multinom(model, data)

        adj = dadi.Godambe.LRT_adjust(demo_model_ex, pts_l, all_boot,
                                      BEST_FIT_MIG, data,
                                      nested_indices=[6, 7], multinom=True)
        D_adj = adj * 2 * (ll_model - ll_nomig)

        # Migration rates are bounded at 0, so the null is a 50:50 mixture of
        # chi^2_0 and chi^2_1; a weighted-sum-of-chi^2 ppf gives the p-value.
        pval = dadi.Godambe.sum_chi2_ppf(D_adj, weights=(0, 0, 1))
    """
    raise NotImplementedError(
        "Descriptive only: the LRT reuses the committed best fit and is not "
        "re-run in the companion repo (needs dadi + bootstrap spectra)."
    )


def load_demes_model() -> dict:
    """Load the fitted demographic model from its demes YAML output."""
    import yaml
    with open(config.DEMOGRAPHY_DEMES_YAML, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _demes_physical() -> dict:
    """Sizes, times, and generation time from the fitted demes YAML."""
    model = load_demes_model()
    demes = {d["name"]: d for d in model["demes"]}
    afr_epochs = demes["AFR"]["epochs"]
    eur_epoch = demes["EUR"]["epochs"][0]
    aou = demes["AOU"]
    return {
        "N_ancestral_start": afr_epochs[0]["start_size"],
        "N_ancestral_end": afr_epochs[1]["start_size"],
        "T_ancestral_years": afr_epochs[0]["end_time"],
        "T_split_years": demes["EUR"]["start_time"],
        "N_EUR_start": eur_epoch["start_size"],
        "N_EUR_end": eur_epoch["end_size"],
        "admix_frac_AFR": aou["proportions"][0],
        "admix_frac_EUR": aou["proportions"][1],
        "generation_time": model["generation_time"],
    }


def best_fit_physical() -> dict:
    """Best-fit demographic parameters in physical units for the Fig 1E panel.

    Effective sizes and epoch times come from the committed ``demes`` model (the
    authoritative physical-unit output); the migration rates, admixture
    fractions and 95% CIs are the
    reported C2 best fit and conservatively propagated marginal bootstrap
    bounds shown on the schematic. The Figure 1 renderer uses these values
    directly when drawing panel E.
    """
    phys = _demes_physical()

    return {
        # AFR: ancestral size then post-growth size (demes epochs).
        "N_ancestral_start": phys["N_ancestral_start"],
        "N_ancestral_end": phys["N_ancestral_end"],
        "T_ancestral_years": phys["T_ancestral_years"],
        # EUR: split time and its exponential-growth endpoints.
        "T_split_years": phys["T_split_years"],
        "N_EUR_start": phys["N_EUR_start"],
        "N_EUR_end": phys["N_EUR_end"],
        # Admixture proportions into AoU (AFR, EUR).
        "admix_frac_AFR": phys["admix_frac_AFR"],
        "admix_frac_EUR": phys["admix_frac_EUR"],
        # Migration best fit (physical, per-generation) and conservatively
        # propagated marginal bootstrap bounds.
        "mig_eur_to_afr": 7.4e-5,
        "mig_eur_to_afr_ci": (5.8e-5, 9.4e-5),
        "mig_afr_to_eur": 5.2e-5,
        "mig_afr_to_eur_ci": (4.0e-5, 6.3e-5),
        # Conservatively propagated marginal bootstrap bounds. The ancestral
        # event age is Tgrowth + T1, so its endpoints are the matching sums.
        "N_ancestral_start_ci": (11121, 12438),
        "N_ancestral_end_ci": (17018, 19830),
        "N_EUR_start_ci": (714, 1195),
        "N_EUR_end_ci": (7078, 10867),
        "T_ancestral_years_ci": (188454, 334352),
        "T_split_years_ci": (37592, 53392),
        "admix_frac_AFR_ci": (0.801, 0.803),
        "admix_frac_EUR_ci": (0.197, 0.199),
        "generation_time_years": phys["generation_time"],
        "mutation_rate_per_bp_gen": 1.5e-8,
    }


def _format_interval(bounds: tuple[int | float, int | float]) -> str:
    """Format an integer-valued publication interval."""
    low, high = bounds
    return f"[{float(low):,.0f}-{float(high):,.0f}]"


def _schematic_paths(
    *, output_dir: Path, basename: str
) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{basename}.png",
        output_dir / f"{basename}.pdf",
        output_dir / f"{basename}.manifest.json",
    )


def best_fit_demes_row() -> pd.Series:
    """C2 physical parameters in the row contract used by Supplementary S1."""
    values = best_fit_physical()
    return pd.Series(
        {
            "model_id": BEST_FIT_MODEL_ID,
            "model_label": "Two directional migration rates",
            "Na": values["N_ancestral_start"],
            "N_AFR": values["N_ancestral_end"],
            "N_EUR_initial": values["N_EUR_start"],
            "N_EUR_final": values["N_EUR_end"],
            "T_AFR_EUR_split_years_before_present": values["T_split_years"],
            "T_ancestral_size_change_years_before_present": values[
                "T_ancestral_years"
            ],
            "f_AFR": values["admix_frac_AFR"],
            "m12_EUR_to_AFR_per_generation": values["mig_eur_to_afr"],
            "m21_AFR_to_EUR_per_generation": values["mig_afr_to_eur"],
        }
    )


def best_fit_demes_graph():
    """Build Fig. 1E with the exact C2 Demes contract used by Figure S1A."""
    from . import figure1_model_fits as s1

    return s1.build_demes_graph(best_fit_demes_row())


def render_demographic_model_schematic(
    *,
    ax=None,
    output_dir: Path | str | None = None,
    basename: str = "demographic_model_schematic",
    dpi: int = 300,
    font_scale: float | None = None,
) -> tuple[Path, Path, Path] | None:
    """Draw Fig. 1E directly, optionally saving a standalone diagnostic.

    The native DemesDraw tubes geometry encodes the single continuous AFR root
    deme, the exponentially changing EUR deme, instantaneous AoU admixture, and
    both directional migration rates. Displayed estimates come from
    best_fit_physical; uncertainty labels are the conservatively propagated
    marginal bootstrap bounds used in the manuscript table.

    When ``ax`` is supplied, drawing is entirely in-memory on that axes and no
    asset files are read or written. This is the Figure 1 production path. The
    no-``ax`` form is retained only as an explicit standalone diagnostic output.
    """
    write_outputs = ax is None
    import matplotlib

    if write_outputs:
        matplotlib.use("Agg", force=True)
    import demesdraw
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text

    from . import figure1_model_fits as s1

    if write_outputs and dpi < 72:
        raise ValueError("dpi must be at least 72")
    if write_outputs:
        target_dir = (
            Path(output_dir) if output_dir is not None else config.DEMOGRAPHY_DIR
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        png_path, pdf_path, manifest_path = _schematic_paths(
            output_dir=target_dir, basename=basename
        )
    values = best_fit_physical()
    row = best_fit_demes_row()
    regular_font_path = (
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    )
    if not regular_font_path.is_file():
        raise FileNotFoundError(
            f"Bundled regular font is missing: {regular_font_path}"
        )
    regular_font = FontProperties(fname=str(regular_font_path), weight=400)
    base_font_sizes = {
        "axis_label": 28,
        "tick_label": 22,
        "deme_label": 28,
        "parameter_label": 24,
        "minimum": 20,
    }
    if font_scale is None:
        if write_outputs:
            font_scale = 1.0
        else:
            axis_width_inches = ax.get_position().width * ax.figure.get_figwidth()
            font_scale = min(1.0, max(0.47, axis_width_inches / 13.4))
    if font_scale <= 0:
        raise ValueError("font_scale must be positive")
    font_sizes = {
        name: size * float(font_scale) for name, size in base_font_sizes.items()
    }

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_sizes["tick_label"],
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        if write_outputs:
            fig, ax = plt.subplots(figsize=(13.4, 7.5), constrained_layout=False)
        else:
            fig = ax.figure
        axes_before_draw = set(fig.axes)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # This is the same graph builder and DemesDraw rendering contract used
        # for the C2 panel in Supplementary Figure S1A.
        s1._draw_demes_model(
            ax,
            row,
            show_y_label=True,
            show_title=False,
            axis_label_fontsize=font_sizes["axis_label"],
            tick_label_fontsize=font_sizes["tick_label"],
            deme_label_fontsize=font_sizes["deme_label"],
        )
        ax.set_ylabel(
            "Time before present (years)",
            fontsize=font_sizes["axis_label"],
            fontproperties=regular_font,
        )
        ax.tick_params(
            axis="both",
            which="major",
            width=1.4,
            labelsize=font_sizes["tick_label"],
        )
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontproperties(regular_font)
            label.set_fontsize(font_sizes["tick_label"])

        # x is in axes coordinates and y remains in demographic time units.
        time_transform = ax.get_yaxis_transform()
        text_kw = {
            "transform": time_transform,
            "fontproperties": regular_font,
            "color": "black",
            "zorder": 20,
        }
        ax.text(
            0.14,
            760_000,
            f"$N_e$ = {values['N_ancestral_start']:,.0f}\n"
            + _format_interval(values["N_ancestral_start_ci"]),
            ha="center",
            va="center",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.14,
            4.0,
            f"$N_e$ = {values['N_ancestral_end']:,.0f}\n"
            + _format_interval(values["N_ancestral_end_ci"]),
            ha="center",
            va="center",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.64,
            285_000,
            f"T = {values['T_ancestral_years']:,.0f} "
            + _format_interval(values["T_ancestral_years_ci"])
            + " years",
            ha="center",
            va="bottom",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.55,
            45_000,
            f"T = {values['T_split_years']:,.0f} "
            + _format_interval(values["T_split_years_ci"])
            + " years",
            ha="center",
            va="bottom",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.56,
            1_500,
            "EUR$\\rightarrow$AFR: $m = 7.4\\ [5.8-9.4] "
            "\\times 10^{-5}$/gen",
            ha="center",
            va="bottom",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.56,
            95,
            "AFR$\\rightarrow$EUR: $m = 5.2\\ [4.0-6.3] "
            "\\times 10^{-5}$/gen",
            ha="center",
            va="top",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.985,
            13_000,
            f"$N_e$ = {values['N_EUR_start']:,.0f}\n"
            + _format_interval(values["N_EUR_start_ci"]),
            ha="right",
            va="center",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.985,
            4.0,
            f"$N_e$ = {values['N_EUR_end']:,.0f}\n"
            + _format_interval(values["N_EUR_end_ci"]),
            ha="right",
            va="center",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.40,
            1.75,
            "f = 80.2%\n[80.1%-80.3%]",
            ha="center",
            va="bottom",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )
        ax.text(
            0.64,
            1.75,
            "f = 19.8%\n[19.7%-19.9%]",
            ha="center",
            va="bottom",
            fontsize=font_sizes["parameter_label"],
            **text_kw,
        )

        # DemesDraw creates a size scale bar on an inset axes. Limit font
        # changes to panel E and the inset axes created by this call so direct
        # composite rendering cannot alter typography in any other panel.
        schematic_axes = [
            ax,
            *(panel_ax for panel_ax in fig.axes if panel_ax not in axes_before_draw),
        ]
        for panel_ax in schematic_axes:
            for text in panel_ax.findobj(match=Text):
                if text.get_text():
                    requested_size = max(
                        text.get_fontsize(), font_sizes["minimum"]
                    )
                    text.set_fontproperties(regular_font)
                    text.set_fontsize(requested_size)

        if write_outputs:
            fig.subplots_adjust(left=0.09, right=0.995, top=0.995, bottom=0.16)
            fig.savefig(
                png_path,
                dpi=dpi,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.04,
                metadata={"Software": "AllofUsPhase1Companion"},
            )
            fig.savefig(
                pdf_path,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.04,
                metadata={"Creator": "AllofUsPhase1Companion"},
            )
            plt.close(fig)

    if not write_outputs:
        return None

    from PIL import Image

    with Image.open(png_path) as rendered:
        pixel_dimensions = list(rendered.size)
        rendered_dpi = [float(value) for value in rendered.info.get("dpi", (dpi, dpi))]
    manifest = {
        "model_id": BEST_FIT_MODEL_ID,
        "model_parameter_sha256": best_fit_parameter_sha256(),
        "renderer": "demesdraw.tubes",
        "demes_source": (
            "scripts.figure1_model_fits.build_demes_graph"
        ),
        "demesdraw_version": demesdraw.__version__,
        "aou_plot_start_time_years": s1.AOU_PLOT_START_TIME_YEARS,
        "uncertainty": "conservatively propagated marginal bootstrap bounds",
        "requested_dpi": dpi,
        "font_file": regular_font_path.name,
        "font_weight": 400,
        "font_scale": font_scale,
        "font_sizes_points": font_sizes,
        "rendered_dpi": rendered_dpi,
        "png_pixel_dimensions": pixel_dimensions,
        "png": png_path.name,
        "pdf": pdf_path.name,
        "parameters": values,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return png_path, pdf_path, manifest_path


def best_fit_point(*, generation_time: float = 25.0) -> dict:
    """Physical-unit best-fit point (for the 'best fit' marker on the panels)."""
    nu_growth, nu2B, nu2F, Tgrowth, T1, f, m12, m21, _misid = BEST_FIT_MIG
    na = NA_REF
    return {
        "N_AFR_postsplit": nu_growth * na,
        "N_EUR_initial": nu2B * na,
        "N_EUR_final": nu2F * na,
        "f_AFR": f,
        "f_EUR": 1.0 - f,
        "T_split_years": T1 * 2.0 * na * generation_time,
        "T_ancestral_years": (Tgrowth + T1) * 2.0 * na * generation_time,
        "m_EUR_to_AFR": m12 / (2.0 * na),
        "m_AFR_to_EUR": m21 / (2.0 * na),
    }


def raw_to_physical(df: pd.DataFrame, *, generation_time: float = 25.0) -> pd.DataFrame:
    """Add physical-unit columns to a frame of raw (dadi-scaled) bootstrap fits.

    ``df`` must have the raw columns in ``BOOTSTRAP_COLUMNS``. Each fit's own
    ``theta`` sets its reference size ``Na = NA_REF * theta / THETA_REF``; sizes
    are ``nu * Na`` individuals, times are ``T * 2 * Na`` generations (converted
    to years), migration rates are ``m / (2 * Na)`` per generation, and ``f`` is
    the AFR admixture fraction into AoU (already dimensionless).
    """
    out = df.copy()
    na = NA_REF * out["theta"] / THETA_REF
    out["Na"] = na
    out["N_AFR_postsplit"] = out["nu_growth"] * na
    out["N_EUR_initial"] = out["nu2B"] * na
    out["N_EUR_final"] = out["nu2F"] * na
    out["f_AFR"] = out["f"]
    out["f_EUR"] = 1.0 - out["f"]
    out["T_split_years"] = out["T1"] * 2.0 * na * generation_time
    out["T_ancestral_years"] = (out["Tgrowth"] + out["T1"]) * 2.0 * na * generation_time
    out["m_EUR_to_AFR"] = out["m12"] / (2.0 * na)
    out["m_AFR_to_EUR"] = out["m21"] / (2.0 * na)
    return out


def pulse_admix_mig(params, ns, pts):
    """dadi model spec (see module docstring). Imported lazily -- dadi is only
    needed to (re)compute the model SFS, not to plot the cached fit."""
    import dadi
    nu_growth, nu2B, nu2F, Tgrowth, T1, f, m12, m21 = params
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, Tgrowth, nu=nu_growth)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    nu2_func = lambda t: nu2B * (nu2F / nu2B) ** (t / T1)
    phi = dadi.Integration.two_pops(phi, xx, T1, nu1=nu_growth, nu2=nu2_func,
                                    m12=m12, m21=m21)
    phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
    return dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx))


def compute_aou_fit() -> pd.DataFrame:
    """Observed vs. C2 best-fit AoU-marginal SFS + residuals (needs dadi).

    Loads the committed 3D data spectrum, evaluates the C2 best-fit model
    (misid-wrapped and extrapolated over grids 35/45/55), marginalizes model and
    data to AoU (axis 2), and optimally rescales the model to that plotted 1D
    spectrum. This is a forward evaluation, not re-inference. The returned
    provenance columns bind the cached curve to the exact C2 parameter vector.
    """
    import dadi
    data = dadi.Spectrum.from_file(str(config.DEMOGRAPHY_SFS))
    ns = data.sample_sizes
    pts_l = [max(ns) + 10, max(ns) + 20, max(ns) + 30]
    model_ex = dadi.Numerics.make_extrap_func(
        dadi.Numerics.make_anc_state_misid_func(pulse_admix_mig))
    model = model_ex(BEST_FIT_MIG, ns, pts_l)
    data_aou = data.marginalize([0, 1])
    model_aou = model.marginalize([0, 1])
    theta_aou = float(dadi.Inference.optimal_sfs_scaling(model_aou, data_aou))
    d = np.asarray(data_aou, dtype=float)
    m = np.asarray(theta_aou * model_aou, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = (m - d) / np.sqrt(m)
    return pd.DataFrame({
        "allele_count": np.arange(len(d)),
        "data": d, "model": m, "poisson_resid": resid,
        "model_id": BEST_FIT_MODEL_ID,
        "model_parameter_sha256": best_fit_parameter_sha256(),
        "model_grid_points": ",".join(str(int(point)) for point in pts_l),
        "model_scaling": "optimal_to_AoU_1D_SFS",
        "model_theta": theta_aou,
    })


def load_aou_fit() -> pd.DataFrame | None:
    """Return the cached C2 AoU-marginal fit, rejecting stale provenance."""
    if config.DEMOGRAPHY_AOU_FIT.exists():
        fit = pd.read_csv(config.DEMOGRAPHY_AOU_FIT)
        required = {
            "allele_count", "data", "model", "poisson_resid", "model_id",
            "model_parameter_sha256", "model_grid_points", "model_scaling",
            "model_theta",
        }
        missing = required.difference(fit.columns)
        if missing:
            raise ValueError(
                f"AoU SFS cache is missing provenance columns: {sorted(missing)}"
            )
        expected = {
            "model_id": BEST_FIT_MODEL_ID,
            "model_parameter_sha256": best_fit_parameter_sha256(),
            "model_grid_points": "35,45,55",
            "model_scaling": "optimal_to_AoU_1D_SFS",
        }
        for column, value in expected.items():
            if set(fit[column].astype(str)) != {value}:
                raise ValueError(
                    f"AoU SFS cache {column} does not identify the C2 best fit"
                )
        return fit
    return None


def load_bootstrap_fits() -> pd.DataFrame | None:
    """Load the 100 bootstrap best-fit parameter sets (raw + physical columns).

    Returns None if the bootstrap file is absent. Reads either the gzipped or
    a plain whitespace-delimited table at ``config.DEMOGRAPHY_BOOTSTRAP``.
    """
    path = config.DEMOGRAPHY_BOOTSTRAP
    plain = path.with_suffix("") if path.suffix == ".gz" else path
    if path.exists():
        opener = gzip.open(path, "rt") if path.suffix == ".gz" else open(path)
        with opener as fh:
            raw = pd.read_csv(fh, sep=r"\s+", header=None, names=list(BOOTSTRAP_COLUMNS))
    elif plain.exists():
        raw = pd.read_csv(plain, sep=r"\s+", header=None, names=list(BOOTSTRAP_COLUMNS))
    else:
        return None
    gen = float(_demes_physical()["generation_time"])
    return raw_to_physical(raw, generation_time=gen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the corrected Fig. 1E schematic."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DEMOGRAPHY_DIR,
        help="Destination for PNG, PDF, and provenance manifest.",
    )
    parser.add_argument("--basename", default="demographic_model_schematic")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    for path in render_demographic_model_schematic(
        output_dir=args.output_dir, basename=args.basename, dpi=args.dpi
    ):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
