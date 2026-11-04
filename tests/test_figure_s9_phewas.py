from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import figure4_discovery as figure_s9_phewas

REPO_ROOT = Path(__file__).resolve().parents[1]


def _s9_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_hits = 446
    discovery = pd.DataFrame(
        {
            "variant_tag": [f"{1 + i % 22}_{10_000 + i}_A_G" for i in range(n_hits)],
            "phecode": [f"P{i:03d}" for i in range(n_hits)],
            "phecode_string": [f"Phenotype {i}" for i in range(n_hits)],
            "p": np.linspace(1e-7, 9.99e-4, n_hits),
            "beta": np.linspace(-1.8, 1.8, n_hits),
            "se": np.full(n_hits, 0.2),
            "converged": np.ones(n_hits, dtype=bool),
            "mac_disc": np.arange(20, 20 + n_hits),
            "n_cases": np.full(n_hits, 1_000),
            "n_controls": np.full(n_hits, 4_000),
            "maf": np.linspace(0.002, 0.45, n_hits),
        }
    )
    # Exercise row-level MAF recovery: an eligible row has no aggregate MAF,
    # although MAC and the tested sample denominator recover it exactly.
    discovery.loc[123, "maf"] = np.nan

    meta = discovery.loc[:16, ["variant_tag", "phecode"]].copy()
    meta["meta_significant"] = True

    sentinels = discovery[["variant_tag"]].copy()
    sentinels["gene_in_window"] = "GENE"
    sentinels["is_sv"] = False
    sentinels["SV_CHR"] = pd.NA
    sentinels["SV_POS"] = pd.NA
    sentinels["SV_TYPE"] = pd.NA
    sentinels["SV_LEN"] = pd.NA
    sentinels.loc[[17, 18, 19, 20], "SV_CHR"] = "chr1"
    sentinels.loc[[17, 18, 19, 20], "SV_POS"] = [20_017, 20_018, 20_019, 20_020]
    sentinels.loc[[17, 18, 19, 20], "SV_TYPE"] = ["Del", "Ins", "Inv", "Del"]
    sentinels.loc[[17, 18, 19, 20], "SV_LEN"] = [72, 84, 1_120, 55]
    return discovery, meta, sentinels


def test_s9_overrides_workbench_inline_backend_for_headless_import() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scripts.figure4_discovery; "
                "import matplotlib; "
                "assert matplotlib.get_backend().lower() == 'agg'"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_s9_preserves_446_hits_and_uses_exclusive_sv_color_category() -> None:
    discovery, meta, sentinels = _s9_inputs()

    prepared, audit = figure_s9_phewas.prepare_s9_data(
        discovery,
        meta,
        sentinels,
    )

    assert len(prepared) == 446
    assert prepared["maf"].notna().all()
    assert prepared.loc[123, "maf"] == 143 / (2 * 5_000)
    assert audit["counts"]["plot_rows"] == 446
    assert audit["counts"]["maf_recovered_from_mac"] == 1
    assert audit["category_counts"] == {
        "discovery hit": 425,
        "SV-linked hit": 4,
        "meta-significant": 17,
    }
    assert sum(audit["category_counts"].values()) == 446


def test_s9_plot_legend_reports_all_exclusive_rows() -> None:
    discovery, meta, sentinels = _s9_inputs()
    prepared, _ = figure_s9_phewas.prepare_s9_data(
        discovery,
        meta,
        sentinels,
        expected_hit_count=446,
    )

    figure = figure_s9_phewas.plot_s9(prepared)
    legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]

    np.testing.assert_allclose(figure.get_size_inches(), [11, 8.5])
    assert legend_labels == [
        "discovery hit (425)",
        "SV-linked hit (4)",
        "meta-significant (17)",
    ]
    assert len(figure.axes[0].texts) == 4
    assert all("Phenotype " in text.get_text() for text in figure.axes[0].texts)
    assert all("Phecode:" not in text.get_text() for text in figure.axes[0].texts)
    assert all(" bp " in text.get_text() for text in figure.axes[0].texts)
    assert all(text.get_fontweight() == "bold" for text in figure.axes[0].texts)
    assert len(figure.axes[0].patches) >= 4
    assert all(patch.get_linewidth() >= 2 for patch in figure.axes[0].patches[-4:])
    figure.canvas.draw()
    point_pixels = figure.axes[0].transData.transform(
        prepared[["maf", "abs_beta"]].to_numpy()
    )
    for text in figure.axes[0].texts:
        box = text.get_window_extent()
        assert not any(box.contains(x, y) for x, y in point_pixels)


def test_s9_rejects_a_445_row_artifact() -> None:
    discovery, meta, sentinels = _s9_inputs()

    with pytest.raises(figure_s9_phewas.S9ContractError, match="expected 446.*found 445"):
        figure_s9_phewas.prepare_s9_data(
            discovery.iloc[:-1],
            meta,
            sentinels,
            expected_hit_count=446,
        )


def test_s9_requires_phenotype_string_names() -> None:
    discovery, meta, sentinels = _s9_inputs()

    with pytest.raises(
        figure_s9_phewas.S9ContractError,
        match="missing: phecode_string",
    ):
        figure_s9_phewas.prepare_s9_data(
            discovery.drop(columns="phecode_string"),
            meta,
            sentinels,
        )


def test_s9_requires_size_aware_sv_annotations() -> None:
    discovery, meta, sentinels = _s9_inputs()

    with pytest.raises(
        figure_s9_phewas.S9ContractError,
        match="need chromosome/position and size/type annotations",
    ):
        figure_s9_phewas.prepare_s9_data(
            discovery,
            meta,
            sentinels.drop(columns="SV_LEN"),
        )


def test_s9_uses_the_source_notebook_max_ihs_schema_for_sv_links() -> None:
    discovery, meta, sentinels = _s9_inputs()
    lead = discovery.iloc[[20]]["variant_tag"].item().split("_")
    interval_hit = discovery.iloc[[42]]["variant_tag"].item().split("_")
    assert lead[0] == interval_hit[0]
    sv_links = pd.DataFrame(
        {
            "variant_id": ["not_the_source_notebook_lead_tag"],
            "SV_ID": ["opaque_sv_id_that_should_not_be_displayed"],
            "chrom": [lead[0]],
            # The archived linked table can also carry a different generic pos;
            # the source notebook explicitly constructs the lead from max_IHS.
            "pos": [999_999_999],
            "max_IHS_position": [int(lead[1])],
            "IHS_REF": [lead[2]],
            "IHS_ALT": [lead[3]],
            "region_start": [int(lead[1]) - 1],
            "region_end": [int(interval_hit[1]) + 1],
            # Exact field names from the archived *.sigp.linked.tsv.
            "lead_is_SV": [True],
            "has_linked_SV": [False],
            "SV_n_linked": [0],
            "SV_TYPE": ["Ins"],
            "SV_POS": [int(lead[1]) + 7],
            "SV_LEN": [72],
        }
    )

    prepared, audit = figure_s9_phewas.prepare_s9_data(
        discovery,
        meta,
        sentinels,
        sv_links=sv_links,
        expected_hit_count=446,
        expected_sv_region_count=1,
    )

    assert prepared.loc[
        prepared["variant_tag"].eq("_".join(lead)), "category"
    ].item() == "SV-linked hit"
    assert prepared.loc[
        prepared["variant_tag"].eq("_".join(interval_hit)), "category"
    ].item() == "SV-linked hit"
    expected_annotation = f"chr{lead[0]}:{int(lead[1]) + 7} (72 bp INS)"
    assert prepared.loc[
        prepared["variant_tag"].eq("_".join(lead)), "sv_annotation"
    ].item() == expected_annotation
    assert prepared.loc[
        prepared["variant_tag"].eq("_".join(interval_hit)), "sv_annotation"
    ].item() == expected_annotation
    assert audit["sv_linkage"] == {
        "reference_rows": 1,
        "linked_reference_regions": 1,
        "exact_lead_association_rows": 1,
        "large_allele_association_rows": 0,
        "interval_association_rows": 2,
        "sv_linked_association_rows": 2,
        "annotated_sv_association_rows": 2,
    }


def test_s9_rejects_zero_hits_from_a_dedicated_sv_reference() -> None:
    discovery, meta, sentinels = _s9_inputs()
    sv_links = pd.DataFrame(
        {
            "chrom": ["22"],
            "max_IHS_position": [999_999_999],
            "IHS_REF": ["A"],
            "IHS_ALT": ["G"],
            "start": [999_999_998],
            "stop": [1_000_000_000],
            "lead_is_SV": [False],
            "has_linked_SV": [True],
            "SV_n_linked": [1],
            "SV_TYPE": ["Del"],
        }
    )

    with pytest.raises(
        figure_s9_phewas.S9ContractError,
        match="matched zero discovery-hit associations",
    ) as error:
        figure_s9_phewas.prepare_s9_data(
            discovery,
            meta,
            sentinels,
            sv_links=sv_links,
            expected_hit_count=446,
            expected_sv_region_count=1,
        )

    assert error.value.audit["sv_linkage"]["linked_reference_regions"] == 1
    assert error.value.audit["sv_linkage"]["sv_linked_association_rows"] == 0


def test_s9_run_recovers_13_missing_mafs_from_supplied_cohorts(tmp_path) -> None:
    discovery, meta, sentinels = _s9_inputs()
    discovery = discovery.iloc[:13].drop(columns=["n_cases", "n_controls"]).copy()
    discovery["maf"] = np.nan
    meta = meta.iloc[:0].copy()
    sentinels = sentinels.loc[
        sentinels["variant_tag"].isin(discovery["variant_tag"])
    ].copy()

    discovery_path = tmp_path / "discovery.tsv"
    meta_path = tmp_path / "meta.tsv"
    sentinels_path = tmp_path / "sentinels.tsv"
    cohort_dir = tmp_path / "cohorts"
    cohort_dir.mkdir()
    discovery.to_csv(discovery_path, sep="\t", index=False)
    meta.to_csv(meta_path, sep="\t", index=False)
    sentinels.to_csv(sentinels_path, sep="\t", index=False)
    for row in discovery.itertuples(index=False):
        mac = int(row.mac_disc)
        pd.DataFrame({"genotype": [1] * mac + [0] * (100 - mac)}).to_csv(
            cohort_dir / f"{row.variant_tag}_cohort.tsv", sep="\t", index=False
        )

    outputs = figure_s9_phewas.run(
        discovery_path,
        meta_path,
        sentinels_path,
        cohort_dir=cohort_dir,
        analysis_dir=tmp_path / "analysis",
        figure_dir=tmp_path / "figures",
        expected_hit_count=13,
        force=True,
    )

    source = pd.read_csv(outputs["source_data"], sep="\t")
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert source["maf"].notna().all()
    assert "sv_annotation" in source.columns
    assert audit["counts"]["plot_rows"] == 13
    assert audit["counts"]["maf_recovered_from_cohort"] == 13

    with pytest.raises(figure_s9_phewas.S9ContractError, match="expected 14.*found 13"):
        figure_s9_phewas.run(
            discovery_path,
            meta_path,
            sentinels_path,
            cohort_dir=cohort_dir,
            analysis_dir=tmp_path / "analysis",
            figure_dir=tmp_path / "figures",
            expected_hit_count=14,
            force=True,
        )
    failure_audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    assert failure_audit["status"] == "failed_contract"
    assert failure_audit["counts"]["plot_rows"] == 13


def test_s9_run_writes_verified_pairs_and_reuses_a_valid_cache(tmp_path) -> None:
    discovery, meta, sentinels = _s9_inputs()
    discovery_path = tmp_path / "discovery.tsv"
    meta_path = tmp_path / "meta.tsv"
    sentinels_path = tmp_path / "sentinels.tsv"
    discovery.to_csv(discovery_path, sep="\t", index=False)
    meta.to_csv(meta_path, sep="\t", index=False)
    sentinels.to_csv(sentinels_path, sep="\t", index=False)

    outputs = figure_s9_phewas.run(
        discovery_path,
        meta_path,
        sentinels_path,
        analysis_dir=tmp_path / "analysis",
        figure_dir=tmp_path / "figures",
        expected_hit_count=446,
        force=True,
    )
    mtimes = {name: path.stat().st_mtime_ns for name, path in outputs.items()}

    assert outputs["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert outputs["pdf"].read_bytes()[:5] == b"%PDF-"
    import matplotlib.pyplot as plt

    assert plt.imread(outputs["png"]).shape[:2] == (2_550, 3_300)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["plot_rows"] == 446
    assert manifest["category_counts"]["SV-linked hit"] == 4
    source = pd.read_csv(outputs["source_data"], sep="\t")
    assert source["phecode_string"].str.startswith("Phenotype ").all()
    assert source.loc[source["is_sv_linked"], "sv_annotation"].notna().all()

    cached = figure_s9_phewas.run(
        discovery_path,
        meta_path,
        sentinels_path,
        analysis_dir=tmp_path / "analysis",
        figure_dir=tmp_path / "figures",
        expected_hit_count=446,
    )
    assert {name: path.stat().st_mtime_ns for name, path in cached.items()} == mtimes


def test_s9_cache_rebuilds_source_data_missing_required_annotation(tmp_path) -> None:
    discovery, meta, sentinels = _s9_inputs()
    discovery_path = tmp_path / "discovery.tsv"
    meta_path = tmp_path / "meta.tsv"
    sentinels_path = tmp_path / "sentinels.tsv"
    discovery.to_csv(discovery_path, sep="\t", index=False)
    meta.to_csv(meta_path, sep="\t", index=False)
    sentinels.to_csv(sentinels_path, sep="\t", index=False)

    outputs = figure_s9_phewas.run(
        discovery_path,
        meta_path,
        sentinels_path,
        analysis_dir=tmp_path / "analysis",
        figure_dir=tmp_path / "figures",
        expected_hit_count=446,
        force=True,
    )

    stale_source = pd.read_csv(outputs["source_data"], sep="\t").drop(
        columns="sv_annotation"
    )
    stale_source.to_csv(outputs["source_data"], sep="\t", index=False)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    source_key = str(outputs["source_data"].resolve())
    manifest["output_sha256"][source_key] = figure_s9_phewas._sha256(
        outputs["source_data"]
    )
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    rebuilt = figure_s9_phewas.run(
        discovery_path,
        meta_path,
        sentinels_path,
        analysis_dir=tmp_path / "analysis",
        figure_dir=tmp_path / "figures",
        expected_hit_count=446,
    )

    repaired_source = pd.read_csv(rebuilt["source_data"], sep="\t")
    assert "sv_annotation" in repaired_source.columns
