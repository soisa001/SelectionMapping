from __future__ import annotations

import gzip
import math
import tarfile
from pathlib import Path

import matplotlib
import pandas as pd

from scripts.common import config
from scripts import figure1_model_fits as s1


def test_saved_models_are_constraints_on_one_superset() -> None:
    assert set(s1.MODEL_SPECS) == {"B1", "B2", "B3", "C1", "C2"}
    for spec in s1.MODEL_SPECS.values():
        assert len(spec.expanded_parameters) == len(s1.FULL_PARAMETER_NAMES)

    expanded = {
        model_id: dict(zip(s1.FULL_PARAMETER_NAMES, spec.expanded_parameters))
        for model_id, spec in s1.MODEL_SPECS.items()
    }
    assert expanded["B1"]["Tgrowth"] == 0
    assert expanded["B1"]["nu2B"] == expanded["B1"]["nu2F"]
    assert expanded["B2"]["nu2B"] == expanded["B2"]["nu2F"]
    assert "f" in s1.MODEL_SPECS["B1"].parameter_names
    assert 0.5 < expanded["B1"]["f"] < 0.99
    assert expanded["B3"]["m12"] == expanded["B3"]["m21"] == 0
    assert expanded["C1"]["m12"] == expanded["C1"]["m21"]
    assert expanded["C2"]["m12"] != expanded["C2"]["m21"]


def test_committed_s1_source_tables_are_current_and_include_joint_ll() -> None:
    assert s1.source_tables_current()
    summary = pd.read_csv(config.DEMOGRAPHY_S1_MODEL_SUMMARY, sep="\t")
    assert summary["model_id"].tolist() == ["B1", "B2", "B3", "C1", "C2"]
    assert summary["log_likelihood_full_3D"].notna().all()
    assert summary["log_likelihood_AoU_1D"].notna().all()
    assert set(summary["full_3D_report_grid_point"]) == {s1.REPORT_GRID_POINT}
    c2 = summary.loc[summary["model_id"] == "C2"].iloc[0]
    assert abs(c2["log_likelihood_full_3D_extrap_35_45_55"] + 35290.5720167) < 1e-3

    wide = pd.read_csv(config.DEMOGRAPHY_S1_WIDE_PARAMETERS, sep="\t")
    assert wide["model_id"].tolist() == ["B1", "B2", "B3", "C1", "C2"]
    assert wide[["nu3", "T2"]].isna().all().all()
    assert wide.loc[wide["model_id"] == "B1", "f"].notna().all()
    assert wide.loc[wide["model_id"] == "C1", "m12"].iloc[0] == wide.loc[
        wide["model_id"] == "C1", "m21"
    ].iloc[0]

    physical = pd.read_csv(config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS, sep="\t")
    assert physical["model_id"].tolist() == ["B1", "B2", "B3", "C1", "C2"]
    assert physical[["N_AoU", "T2_since_admixture_years"]].isna().all().all()
    b1_physical = physical.loc[physical["model_id"] == "B1"].iloc[0]
    expected_na = b1_physical["theta_full_3D"] / (
        4 * s1.MUTATION_RATE_PER_BP_PER_GENERATION * s1.SEQUENCE_LENGTH_BP
    )
    assert math.isclose(b1_physical["Na"], expected_na, rel_tol=1e-8)
    assert math.isclose(b1_physical["N_AFR"], b1_physical["Na"], rel_tol=1e-8)
    assert math.isclose(
        b1_physical["N_EUR_initial"], b1_physical["N_EUR_final"], rel_tol=1e-8
    )
    c1_physical = physical.loc[physical["model_id"] == "C1"].iloc[0]
    assert math.isclose(
        c1_physical["m12_EUR_to_AFR_per_generation"],
        c1_physical["m21_AFR_to_EUR_per_generation"],
        rel_tol=1e-8,
    )

    harmonized = pd.read_csv(config.DEMOGRAPHY_S1_HARMONIZED_PARAMETERS, sep="\t")
    expected_parameters = list(s1.HARMONIZED_PARAMETER_METADATA)
    assert len(harmonized) == len(s1.MODEL_SPECS) * len(expected_parameters)
    for model_id in s1.MODEL_SPECS:
        subset = harmonized.loc[harmonized["model_id"] == model_id]
        assert subset["parameter"].tolist() == expected_parameters
    assert "m_EUR_EUR" not in set(harmonized["parameter"])
    b1_change = harmonized[
        (harmonized["model_id"] == "B1")
        & (harmonized["parameter"] == "T_AFR_CHANGE")
    ].iloc[0]
    assert b1_change["parameter_status"] == "unused"
    assert pd.isna(b1_change["best_ll_estimate"])
    c1_migration = harmonized[
        (harmonized["model_id"] == "C1")
        & harmonized["parameter"].isin(["m_EUR_AFR", "m_AFR_EUR"])
    ]
    assert c1_migration["best_ll_estimate"].nunique() == 1
    assert set(c1_migration["source_parameter"]) == {"m / (2 * Na)"}
    b2_eur = harmonized[
        (harmonized["model_id"] == "B2")
        & harmonized["parameter"].isin(["N_EUR_INITIAL", "N_EUR_FINAL"])
    ]
    assert set(b2_eur["source_parameter"]) == {"nu2 * Na"}
    c2 = harmonized.loc[harmonized["model_id"] == "C2"].set_index("parameter")
    assert c2.loc["Na", "best_ll_estimate"] == 11727
    assert c2.loc["theta", "ci_97_5"] == 358983.2
    assert c2.loc["m_EUR_AFR", "best_ll_estimate"] == 0.000074
    assert c2.loc["m_AFR_EUR", "best_ll_estimate"] == 0.000033

    with gzip.open(config.DEMOGRAPHY_S1_MARGINAL_SFS, "rt", encoding="utf-8") as handle:
        spectra = pd.read_csv(handle, sep="\t")
    assert set(spectra["population"]) == {"AoU"}
    assert set(spectra["spectrum_id"]) == {
        "observed",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
    }


def test_s1_figure_names_are_paired() -> None:
    assert s1.DEMESDRAW_MAX_TIME_YEARS == 1_000_000
    paths = s1.figure_paths()
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    stems = {Path(path).with_suffix("") for path in paths}
    assert len(stems) == 3
    for stem in stems:
        assert {path.suffix for path in paths if path.with_suffix("") == stem} == {
            ".png",
            ".pdf",
        }
    assert {path.name for path in s1.demography_figure_paths()} == {
        "Figure_S1A_Demographic_Models.png",
        "Figure_S1A_Demographic_Models.pdf",
    }
    assert {path.name for path in s1.streamlined_demography_figure_paths()} == {
        "Figure_S1A_Demographic_Models_Streamlined.png",
        "Figure_S1A_Demographic_Models_Streamlined.pdf",
    }


def test_s1_pdf_output_uses_editable_vector_fonts() -> None:
    assert matplotlib.get_backend().lower() == "agg"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def test_demes_graph_keeps_afr_as_one_continuous_root_deme() -> None:
    physical = pd.read_csv(config.DEMOGRAPHY_S1_WIDE_PHYSICAL_PARAMETERS, sep="\t")
    graphs = {
        row["model_id"]: s1.build_demes_graph(row)
        for _, row in physical.iterrows()
    }

    for graph in graphs.values():
        assert [deme.name for deme in graph.demes] == ["AFR", "EUR", "AoU"]
        assert math.isinf(graph["AFR"].start_time)
        assert graph["AFR"].end_time == 0
        assert graph["AFR"].ancestors == []
        assert graph["AoU"].ancestors == ["AFR", "EUR"]
        assert math.isclose(
            graph["AoU"].start_time, s1.AOU_PLOT_START_TIME_YEARS
        )
        assert graph["AoU"].start_time < graph["EUR"].start_time

    assert len(graphs["B1"]["AFR"].epochs) == 1
    for model_id in ("B2", "B3", "C1", "C2"):
        assert len(graphs[model_id]["AFR"].epochs) == 2
    assert len(graphs["B3"].migrations) == 0
    assert len(graphs["C1"].migrations) == 2
    assert len(graphs["C2"].migrations) == 2


def test_c1_c2_lrt_is_one_df_equality_constraint() -> None:
    comparisons = pd.read_csv(config.DEMOGRAPHY_S1_COMPARISONS, sep="\t")
    row = comparisons[
        (comparisons["null_model_id"] == "C1")
        & (comparisons["alternative_model_id"] == "C2")
    ].iloc[0]
    assert row["lrt_df"] == 1
    assert row["2_delta_log_likelihood_full_3D"] > 0
    assert row["unadjusted_lrt_p_value"] < 1e-200


def test_committed_godambe_archive_contains_exactly_100_joint_sfs() -> None:
    archive_path = config.DEMOGRAPHY_BOOTSTRAP_SFS_ARCHIVE
    assert archive_path.is_file()
    with tarfile.open(archive_path, mode="r:*") as archive:
        names = [
            Path(member.name).name
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("_AFR_EUR_AOU.fs")
        ]
    assert len(names) == 100
    assert {
        int(name.split("_", maxsplit=1)[0]) for name in names
    } == set(range(100))

    with s1._bootstrap_sfs_directory(archive_path) as bootstrap_dir:
        assert len(list(bootstrap_dir.glob("*_AFR_EUR_AOU.fs"))) == 100

    provenance = pd.read_json(config.DEMOGRAPHY_S1_PROVENANCE, typ="series")
    assert provenance["godambe_bootstrap_source"] == (
        "assets/demography/s1_demography_bootstrap_sfs.tar.gz"
    )
