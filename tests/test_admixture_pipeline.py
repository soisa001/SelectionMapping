import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.usefixtures("synthetic_finemap_mask")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import (
    admixture_data,
    admixture_finemap,
    admixture_inputs,
    admixture_lai,
    admixture_phewas,
    admixture_plots,
    admixture_plrs,
    admixture_summary,
    recompute_admixture_reference,
    setup_admixture_runtime,
)
from scripts.common import config, gcs, samples


class Figure4LaiTests(unittest.TestCase):
    def test_bucket_defaults_use_migration_prefix(self):
        root = "gs://rw-migration-aou-rw-a3e31f4d"
        self.assertEqual(config.LAI_MASKED_GS_DIR, f"{root}/lai_masked")
        self.assertTrue(config.ADMIXTURE_DISCOVERY_COV_GS.startswith(root + "/"))
        self.assertTrue(config.ADMIXTURE_REPLICATION_COV_GS.startswith(root + "/"))
        self.assertEqual(
            config.ADMIXTURE_GWAS_GS,
            f"{root}/GWAS_catalog/gwas_catalog_associations.tsv",
        )

    def test_ancestry_indices_do_not_assume_header_order(self):
        resolved = admixture_lai.ancestry_indices({"SAS": 0, "AFR": 3, "EUR": 7})
        self.assertEqual(resolved, (7, 3, "EUR", "AFR"))

    def test_region_merging_is_sign_specific_and_valley_aware(self):
        frame = pd.DataFrame({
            "chrom": [1] * 10,
            "pos": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "eur_frac": [0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05],
            "eur_z": [5.0, 5.2, 0, 0, 0, 0, 0, 0, -5.1, -5.3],
        })
        regions = admixture_lai.merge_regions_signed_valley(
            frame, gap_bp=1_000, valley_min_markers=4
        )
        self.assertEqual(len(regions), 2)
        self.assertEqual(set(regions["direction"]), {"EUR-excess", "AFR-excess"})

    def test_centromere_majority_overlap_is_removed(self):
        regions = pd.DataFrame({
            "region_id": ["drop", "keep"], "chrom": [1, 1],
            "start": [100, 0], "end": [200, 80],
        })
        centromeres = pd.DataFrame({"chrom": [1], "start": [100], "end": [200]})
        kept = admixture_lai.filter_centromeric_regions(regions, centromeres)
        self.assertEqual(kept["region_id"].tolist(), ["keep"])


class Figure4AssociationContractTests(unittest.TestCase):
    def test_numpy_two_pickle_error_has_recovery_instructions(self):
        error = ModuleNotFoundError(
            "No module named 'numpy._core.numeric'",
            name="numpy._core.numeric",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            path.write_bytes(b"placeholder")
            with patch("pickle.load", side_effect=error), \
                 self.assertRaisesRegex(RuntimeError, "numpy==2.0.2"):
                admixture_inputs.pickle_load(path)

    def test_plrs_threshold_is_strictly_greater_than_half(self):
        self.assertEqual(admixture_plrs.R2_MIN, 0.5)

    def test_completed_stage2_skips_model_pickle_deserialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "region_id": ["region-a"],
                "retained_for_phewas": [True],
            }).to_csv(root / "plrs_models_summary.tsv", sep="\t", index=False)
            pd.DataFrame({
                "region_id": ["region-a"],
                "plrs_model_contract": [admixture_plrs.MODEL_CONTRACT],
            }).to_csv(
                root / "eur_dosage_predictions.discovery.tsv.gz",
                sep="\t", index=False, compression="gzip",
            )
            with patch.object(admixture_plrs, "analysis_dir", return_value=root), \
                 patch.object(admixture_plrs, "fit_plrs_models") as fit:
                result = admixture_plrs.run_discovery(force=False)

        fit.assert_not_called()
        self.assertEqual(
            result["discovery_predictions"],
            root / "eur_dosage_predictions.discovery.tsv.gz",
        )

    def test_plrs_contract_requires_unrelated_1025_training_panel(self):
        self.assertIn("unrelated1025", admixture_plrs.MODEL_CONTRACT)
        self.assertEqual(samples.N_AOU, 1025)

    def test_prediction_sample_ids_drop_related_individuals(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "covariates.tsv"
            pd.DataFrame({
                "person_id": ["keep", *samples.load_excluded_sample_ids()]
            }).to_csv(path, sep="\t", index=False)
            self.assertEqual(admixture_plrs._cohort_ids(path), ["keep"])

    def test_old_prediction_contract_is_not_reused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv.gz"
            pd.DataFrame({"plrs_model_contract": ["old"]}).to_csv(
                path, sep="\t", index=False, compression="gzip"
            )
            self.assertFalse(admixture_plrs._prediction_cache_current(path))
            pd.DataFrame({
                "plrs_model_contract": [admixture_plrs.MODEL_CONTRACT]
            }).to_csv(path, sep="\t", index=False, compression="gzip")
            self.assertTrue(admixture_plrs._prediction_cache_current(path))

    def test_replication_payloads_are_limited_to_discovery_regions(self):
        summary = pd.DataFrame({
            "region_id": ["region-a", "region-b", "region-c"],
            "retained_for_phewas": [True, True, False],
        })
        with patch.object(admixture_plrs, "model_path",
                          side_effect=lambda region_id: Path(region_id)), \
             patch.object(admixture_plrs.admixture_inputs, "pickle_load",
                          side_effect=lambda path: {
                              "region": {"region_id": path.name}
                          }):
            payloads = admixture_plrs._retained_payloads(
                summary, region_ids=["region-b", "region-c"]
            )
        self.assertEqual(
            [payload["region"]["region_id"] for payload in payloads],
            ["region-b"],
        )

    def test_replication_target_contract_uses_exact_region_phecode_pairs(self):
        targets = pd.DataFrame({
            "region_id": ["region-b", "region-a"],
            "phecode": ["200.1", "100.2"],
        })
        reordered = targets.iloc[::-1].reset_index(drop=True)
        changed = targets.copy()
        changed.loc[0, "phecode"] = "200.2"
        self.assertEqual(
            admixture_phewas._target_contract(targets),
            admixture_phewas._target_contract(reordered),
        )
        self.assertNotEqual(
            admixture_phewas._target_contract(targets),
            admixture_phewas._target_contract(changed),
        )

    def test_replication_phewas_runs_only_discovered_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "replication_predictions.tsv.gz"
            people = [1, 2, 3, 4]
            pd.DataFrame([
                {"person_id": person, "region_id": region,
                 "eur_dosage": float(index % 3)}
                for index, person in enumerate(people)
                for region in ("region-a", "region-b")
            ]).to_csv(
                prediction_path, sep="\t", index=False, compression="gzip"
            )
            covariates = pd.DataFrame(
                0.0, index=people, columns=admixture_phewas.COVARIATE_COLUMNS
            )
            covariates.index.name = "person_id"
            counts = pd.DataFrame({
                "person_id": ["1", "2"],
                "phecode": ["100.1", "200.1"],
                "count": [2, 2],
            })
            targets = pd.DataFrame({
                "region_id": ["region-a", "region-b"],
                "phecode": ["100.1", "200.1"],
            })

            prepared_covariates = covariates.reset_index()

            def fake_phetk(*_, phecodes, **__):
                return pd.DataFrame({
                    "phecode": list(phecodes),
                    "cases": [1] * len(phecodes),
                    "controls": [2] * len(phecodes),
                    "p_value": [0.5] * len(phecodes),
                    "beta": [0.1] * len(phecodes),
                    "standard_error": [0.2] * len(phecodes),
                    "converged": ["Converged"] * len(phecodes),
                    "phecode_sex_restriction": ["Both"] * len(phecodes),
                })

            with patch.object(admixture_phewas, "analysis_dir", return_value=root), \
                 patch.object(admixture_phewas, "load_cohort",
                               return_value=(covariates, counts)), \
                 patch.object(admixture_phewas, "_prepare_phetk_covariates",
                              return_value=prepared_covariates), \
                 patch.object(admixture_phewas, "_ensure_phetk_runtime",
                              return_value=Path("phetk-python")), \
                 patch.object(admixture_phewas.admixture_inputs,
                              "stage_cohort_sources",
                              return_value={"phenotypes": root / "counts.tsv"}), \
                 patch.object(admixture_phewas, "_run_phetk_region",
                              side_effect=fake_phetk), \
                 patch.object(admixture_phewas, "phecode_sex_lookup",
                               return_value={"100.1": "Both", "200.1": "Both"}), \
                 patch.object(admixture_phewas, "MIN_CASES", 1):
                result = admixture_phewas.run_associations(
                    "replication", prediction_path, targets=targets, force=True
                )

        self.assertEqual(
            set(zip(result["region_id"], result["phecode"])),
            {("region-a", "100.1"), ("region-b", "200.1")},
        )

    def test_female_phecode_excludes_males_and_subthreshold_counts(self):
        group = pd.DataFrame({
            "person_id": ["a", "b", "c"], "count": [2, 2, 1]
        })
        y = admixture_phewas.phenotype_vector(
            group, np.asarray(["a", "b", "c", "d"]),
            np.asarray([0, 1, 0, 0]), "Female",
        )
        np.testing.assert_equal(y, np.asarray([1.0, np.nan, np.nan, 0.0]))

    def test_sex_normalization_excludes_unmapped_values_without_aborting(self):
        values = pd.Series([
            "Female", "male", "F", "M", "woman", "man", "0.0", "1.0",
            "Intersex", "Prefer not to answer", None,
        ])
        normalized = admixture_phewas._normalize_sex(values)
        np.testing.assert_equal(
            normalized.to_numpy(),
            np.asarray([0, 1, 0, 1, 0, 1, 0, 1, np.nan, np.nan, np.nan]),
        )

    def test_covariate_contract_uses_age_age_squared_sex_and_twenty_pcs(self):
        expected = [
            "age_at_last_ehr_event", "age_at_last_ehr_event_squared",
            "sex_at_birth", *[f"pc{i}" for i in range(1, 21)],
        ]
        self.assertEqual(admixture_phewas.COVARIATE_COLUMNS, expected)

    def test_stage3_uses_phetk_logit_and_two_hundred_case_threshold(self):
        self.assertEqual(admixture_phewas.PHETK_METHOD, "logit")
        self.assertEqual(admixture_phewas.PHETK_NUMPY_VERSION, "1.26.4")
        self.assertEqual(admixture_phewas.MIN_CASES, 200)
        self.assertNotIn("phetk", recompute_admixture_reference.PYTHON_DEPENDENCIES)

    def test_phetk_runtime_install_uses_isolated_python(self):
        with TemporaryDirectory() as directory:
            runtime = Path(directory) / "phetk-runtime"
            runtime_python = admixture_phewas._runtime_python(runtime)
            with patch.object(admixture_phewas, "_phetk_runtime_dir",
                              return_value=runtime), \
                 patch.object(admixture_phewas, "_phetk_runtime_works",
                              side_effect=[False, True]), \
                 patch.object(admixture_phewas.subprocess, "run") as run:
                result = admixture_phewas._ensure_phetk_runtime()

        self.assertEqual(result, runtime_python)
        self.assertEqual(run.call_args_list[0].args[0][:3], [
            admixture_phewas.sys.executable, "-m", "venv",
        ])
        self.assertEqual(run.call_args_list[1].args[0][:4], [
            str(runtime_python), "-m", "pip", "install",
        ])
        self.assertIn(
            "numpy==1.26.4", run.call_args_list[1].args[0]
        )

    def test_phetk_region_runs_worker_with_isolated_python(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_python = root / "isolated-python"
            output = root / "region.phewas.tsv"
            phecodes = root / "phecodes.tsv"
            cohort = pd.DataFrame({
                "person_id": [1], "eur_dosage": [1.0],
                **{column: [0.0] for column in admixture_phewas.COVARIATE_COLUMNS},
            })
            with patch.object(admixture_phewas.subprocess, "run") as run:
                admixture_phewas._run_phetk_region(
                    cohort, phecodes, output, phecodes=["100.1"],
                    runtime_python=runtime_python,
                )
            command = run.call_args.args[0]
            payload = json.loads(
                output.with_suffix(output.suffix + ".json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(command[0], str(runtime_python))
        self.assertEqual(command[1], "-u")
        self.assertEqual(payload["method"], "logit")
        self.assertEqual(payload["min_cases"], 200)

    def test_phetk_results_require_200_cases_and_more_than_200_controls(self):
        raw = pd.DataFrame({
            "phecode": ["valid", "few-cases", "few-controls"],
            "cases": [200, 199, 200],
            "controls": [201, 201, 200],
            "p_value": [0.1, 0.1, 0.1],
            "beta": [0.2, 0.2, 0.2],
            "standard_error": [0.1, 0.1, 0.1],
            "converged": ["Converged"] * 3,
        })
        result = admixture_phewas._canonicalize_phetk(
            raw,
            label="discovery",
            region_id="region-a",
            target_contract="all-region-phecode-tests",
            requested_phecodes=None,
        )
        self.assertEqual(result["phecode"].tolist(), ["valid"])

    def test_discovery_passes_every_exact_pair_below_one_e_minus_three(self):
        discovery = pd.DataFrame({
            "region_id": ["region-a", "region-a", "region-b", "region-c"],
            "phecode": ["100.1", "100.2", "200.1", "300.1"],
            "p": [1e-6, 9.99e-4, 1e-3, 2e-3],
        })
        labeled = admixture_phewas.label_discovery_significance(discovery)
        hits = admixture_phewas.discovery_hits(discovery)

        self.assertEqual(
            labeled["discovery_significant"].tolist(),
            [True, True, False, False],
        )
        self.assertTrue(
            labeled["discovery_alpha"].eq(1e-3).all()
        )
        self.assertEqual(
            set(zip(hits["region_id"], hits["phecode"])),
            {("region-a", "100.1"), ("region-a", "100.2")},
        )


class Figure4RuntimeTests(unittest.TestCase):
    def test_runtime_pins_numpy_two_and_keeps_phetk_outside_kernel(self):
        requirements = (
            setup_admixture_runtime.REQUIREMENTS.read_text(encoding="utf-8")
            .splitlines()
        )
        packages = [line for line in requirements
                    if line and not line.startswith("#")]
        self.assertIn("numpy==2.0.2", packages)
        self.assertFalse(any(line.startswith("phetk") for line in packages))

    def test_runtime_environment_does_not_inherit_pythonpath(self):
        with patch.dict(os.environ, {"PYTHONPATH": "broken-site-packages"}):
            env = setup_admixture_runtime._repo_environment(Path("phetk-env"))
        self.assertEqual(
            env["PYTHONPATH"], str(setup_admixture_runtime.REPO_ROOT)
        )
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_registered_kernel_uses_isolated_phetk_environment(self):
        with TemporaryDirectory() as directory:
            resource_dir = Path(directory) / "kernel"
            resource_dir.mkdir()
            spec_path = resource_dir / "kernel.json"
            spec_path.write_text(json.dumps({
                "argv": ["python", "-m", "ipykernel_launcher", "-f",
                         "{connection_file}"],
                "display_name": "admixture reference",
                "language": "python",
            }), encoding="utf-8")
            listing = Mock(stdout=str(resource_dir))
            with patch.object(
                setup_admixture_runtime, "_run",
                return_value=listing,
            ):
                result = setup_admixture_runtime._register_kernel(
                    Path("kernel-python"), Path("isolated-phetk"),
                    kernel_name="allofus-admixture_reference",
                    display_name="admixture reference",
                    env={},
                )
            spec = json.loads(spec_path.read_text(encoding="utf-8"))

        self.assertEqual(result, resource_dir)
        self.assertEqual(
            spec["env"]["AOU_FIG4_PHETK_ENV"], "isolated-phetk"
        )
        self.assertEqual(spec["env"]["PYTHONNOUSERSITE"], "1")


class GcsInputIntegrityTests(unittest.TestCase):

    def test_content_match_requires_size_and_md5(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.tsv"
            path.write_bytes(b"same bytes")
            metadata = {
                "size": path.stat().st_size,
                "md5": gcs._local_md5(path),
            }
            with patch.object(gcs, "object_metadata", return_value=metadata):
                self.assertTrue(gcs.gcs_same_file(path, "gs://bucket/result.tsv"))
            with patch.object(
                gcs, "object_metadata", return_value={**metadata, "md5": "different"}
            ):
                self.assertFalse(gcs.gcs_same_file(path, "gs://bucket/result.tsv"))


class Figure4PlotTests(unittest.TestCase):

    def test_finemap_cache_requires_exact_hundred_kb_window(self):
        hit = {
            "region_id": "chr9_100000_200000", "phecode": "401.1",
            "start": 100_000, "end": 200_000,
        }
        current = pd.DataFrame({
            "analysis_contract": [admixture_finemap.FINEMAP_CONTRACT],
            "region_id": [hit["region_id"]], "phecode": [hit["phecode"]],
            "window_start": [1], "window_end": [300_000],
            **admixture_finemap.filter_parameters(),
        })
        stale = current.copy()
        stale["window_start"] = 1
        stale["window_end"] = 450_000
        self.assertTrue(admixture_finemap.cache_matches_hit(current, hit))
        self.assertFalse(admixture_finemap.cache_matches_hit(stale, hit))


class Figure4SummaryTests(unittest.TestCase):
    def test_summary_counts_regions_lengths_and_retained_models(self):
        regions = pd.DataFrame({
            "start": [100, 1_000], "end": [199, 1_499],
        })
        models = pd.DataFrame({
            "r2_holdout": [0.2, 0.5, 0.5001, 0.8], "n_snvs": [2, 3, 4, 5],
        })
        qtl_status = pd.DataFrame({
            "source": ["MAGE eQTL", "MAGE sQTL"],
            "n_linked_associations": [1, 0],
        })
        gwas_status = pd.DataFrame({
            "source": ["GWAS Catalog"],
            "n_linked_associations": [3],
            "n_direct_associations": [1],
        })
        discovery = pd.DataFrame({
            "region_id": ["r1", "r1", "r2"],
            "phecode": ["P1", "P2", "P1"],
            "p": [5e-4, 0.2, 8e-4],
        })
        meta = pd.DataFrame({
            "phecode": ["P1", "P2"],
            "meta_significant": [True, False],
        })
        result = admixture_summary._summary_statistics(
            regions, models, discovery, meta, qtl_status, gwas_status,
            "chr9:150:A:G",
        ).set_index("statistic")["value"]
        self.assertEqual(result["outlier_regions"], 2)
        self.assertEqual(result["outlier_length_min_bp"], 100)
        self.assertEqual(result["outlier_length_max_bp"], 500)
        self.assertEqual(result["plrs_models_r2_gt_0.5"], 2)
        self.assertEqual(result["mage_eqtl_linked_associations"], 1)
        self.assertEqual(result["gwas_catalog_linked_associations"], 3)
        self.assertEqual(result["gwas_catalog_direct_associations"], 1)
        self.assertEqual(result["discovery_unique_phecodes_passing_filters"], 2)
        self.assertEqual(
            result["discovery_region_phecode_tests_passing_filters"], 3
        )
        self.assertEqual(result["discovery_significant_region_phecode_pairs"], 2)
        self.assertEqual(result["meta_significant_associations"], 1)
        self.assertEqual(result["meta_significant_unique_phecodes"], 1)

    def test_meta_summary_table_matches_requested_contract(self):
        region_id = "chr9_63748785_63837789"
        regions = pd.DataFrame({
            "region_id": [region_id], "chrom": [9],
            "start": [63_748_785], "end": [63_837_789],
            "direction": ["AFR-excess"], "peak_z": [-15.30452427],
        })
        genes = pd.DataFrame({
            "chrom": ["chr9", "chr9", "chr9", "chr9"],
            "start": [63_750_000, 63_780_000, 63_820_000, 63_900_000],
            "end": [63_760_000, 63_790_000, 63_830_000, 63_910_000],
            "gene": ["LINC00537", "MIR4477A", "FRG1JP", "OUTSIDE"],
        })
        meta = pd.DataFrame({
            "region_id": [region_id], "phecode": ["SS_809.1"],
            "beta_discovery": [0.653656812], "se_discovery": [0.193],
            "p_discovery": [7.17e-4], "n_discovery": [8771],
            "beta_replication": [0.2221122139], "se_replication": [0.348],
            "p_replication": [0.5228330909], "n_replication": [28552],
            "beta_meta": [0.5521456014], "se_meta": [0.1687809415],
            "p_meta": [0.001070262], "same_direction": [True],
            "meta_significant": [True],
            "replicated_meta_significant": [False],
            "discovery_alpha": [1e-3],
        })
        result = admixture_summary._meta_summary_table(
            meta, regions, genes, {"SS_809.1": "Acute pain"}
        )
        self.assertEqual(result.columns.tolist(), admixture_summary.META_SUMMARY_COLUMNS)
        row = result.iloc[0]
        self.assertEqual(row["variant_tag"], region_id)
        self.assertEqual(row["chrom"], "chr9")
        self.assertEqual(
            row["genes_in_region"], "LINC00537,MIR4477A,FRG1JP"
        )
        self.assertEqual(row["phecode_string"], "Acute pain")
        self.assertTrue(row["same_direction"])
        self.assertFalse(row["rep_nominal"])
        self.assertTrue(row["meta_significant"])
        self.assertFalse(row["replicated_full"])
        self.assertEqual(row["disc_hit_threshold"], 1e-3)
        self.assertEqual(row["disc_hit_kind"], "fallback_1e-3")

    def test_phecode_strings_are_recovered_from_raw_discovery_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "phetk_runs" / "discovery"
            raw_dir.mkdir(parents=True)
            pd.DataFrame({
                "phecode": ["SS_809.1"],
                "phecode_string": ["Acute pain"],
                "p_value": [7.17e-4],
            }).to_csv(raw_dir / "region.phewas.tsv", sep="\t", index=False)
            with patch.object(admixture_phewas, "analysis_dir", return_value=root):
                result = admixture_summary._phecode_string_lookup(pd.DataFrame({
                    "phecode": ["SS_809.1"],
                }))
        self.assertEqual(result, {"SS_809.1": "Acute pain"})

    def test_mage_lookup_is_limited_to_variants_linked_to_minimum_p(self):
        finemap = pd.DataFrame({
            "variant_id": ["chr9:100:A:G", "chr9:110:C:T", "chr9:120:G:A"],
            "minimum_p_variant": [True, False, False],
            "r2_to_min_p": [1.0, 0.7, 0.2],
        })
        minimum_id, linked = admixture_summary._linked_variants(finemap)
        self.assertEqual(minimum_id, "chr9:100:A:G")
        self.assertEqual(
            set(linked), {("chr9", 100, "A", "G"), ("chr9", 110, "C", "T")}
        )

    def test_focused_mage_scan_reports_direct_and_linked_hits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mage.tsv.gz"
            pd.DataFrame({
                "variantChrom": ["9", "chr9", "chr9"],
                "variantPosition": [100, 110, 120],
                "variantRef": ["A", "C", "G"],
                "variantAlt": ["G", "T", "A"],
                "geneSymbol": ["DIRECT", "LINKED", "UNLINKED"],
                "qval": [0.01, 0.02, 0.01],
            }).to_csv(path, sep="\t", index=False, compression="gzip")
            linked = {
                ("chr9", 100, "A", "G"): 1.0,
                ("chr9", 110, "C", "T"): 0.7,
            }
            matches, status = admixture_summary._scan_mage(
                path, "MAGE eQTL", linked, "chr9:100:A:G"
            )
        self.assertEqual(set(matches["gene"]), {"DIRECT", "LINKED"})
        self.assertEqual(status["n_linked_associations"], 2)
        self.assertEqual(status["n_direct_associations"], 1)

    def test_focused_gwas_scan_reports_direct_and_linked_hits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gwas.tsv"
            pd.DataFrame({
                "CHR_ID": ["9", "chr9", "chr9", "chr9"],
                "CHR_POS": [100, 110, 120, 110],
                "P-VALUE": [1e-12, 2e-9, 1e-15, 1e-5],
                "SNPS": ["rsDirect", "rsLinked", "rsUnlinked", "rsWeak"],
                "DISEASE/TRAIT": ["Trait A", "Trait B", "Trait C", "Trait D"],
                "MAPPED_GENE": ["DIRECT", "LINKED", "UNLINKED", "WEAK"],
                "STUDY ACCESSION": ["GCST1", "GCST2", "GCST3", "GCST4"],
                "PUBMEDID": ["1001", "1002", "1003", "1004"],
            }).to_csv(path, sep="\t", index=False)
            linked = {
                ("chr9", 100, "A", "G"): 1.0,
                ("chr9", 110, "C", "T"): 0.7,
            }
            matches, status = admixture_summary._scan_gwas(
                path, linked, "chr9:100:A:G"
            )
        self.assertEqual(set(matches["trait"]), {"Trait A", "Trait B"})
        self.assertEqual(set(matches["linked_variant_id"]), {
            "chr9:100:A:G", "chr9:110:C:T",
        })
        self.assertTrue(matches["gwas_catalog_url"].str.startswith(
            "https://www.ebi.ac.uk/gwas/studies/"
        ).all())
        self.assertEqual(status["match_basis"], "chromosome_position")
        self.assertEqual(status["n_linked_associations"], 2)
        self.assertEqual(status["n_direct_associations"], 1)

    def test_gwas_lookup_stages_once_then_reuses_matching_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "gwas.tsv"
            pd.DataFrame({
                "CHR_ID": ["9"], "CHR_POS": [100], "P-VALUE": [1e-12],
                "DISEASE/TRAIT": ["Trait A"],
            }).to_csv(catalog, sep="\t", index=False)
            finemap = pd.DataFrame({
                "variant_id": ["chr9:100:A:G"],
                "minimum_p_variant": [True],
                "r2_to_min_p": [1.0],
            })
            with patch.object(admixture_summary, "analysis_dir", return_value=root), \
                 patch.object(
                     admixture_inputs, "stage_file", return_value=catalog
                 ) as stage:
                first, status, minimum_id = admixture_summary._gwas_lookup(finemap)
                second, _, _ = admixture_summary._gwas_lookup(finemap)
        stage.assert_called_once_with(
            config.ADMIXTURE_GWAS_GS,
            Path("annotation") / "gwas_catalog_associations.tsv",
            force=False,
            required=False,
        )
        self.assertEqual(minimum_id, "chr9:100:A:G")
        self.assertEqual(len(first), 1)
        self.assertTrue(status["available"].iloc[0])
        pd.testing.assert_frame_equal(first, second, check_dtype=False)


class Figure4WorkflowTests(unittest.TestCase):
    def test_cli_forwards_local_analysis_options(self):
        with patch.object(recompute_admixture_reference, "run_all", return_value={}) as run:
            recompute_admixture_reference.main([])
            self.assertEqual(run.call_args.kwargs, {
                "force": False, "run_phewas": True, "run_finemap": True,
            })
            recompute_admixture_reference.main(["--force", "--skip-phewas", "--skip-finemap"])
            self.assertEqual(run.call_args.kwargs, {
                "force": True, "run_phewas": False, "run_finemap": False,
            })

    def test_replication_generation_receives_only_significant_discovery_regions(self):
        discovery = pd.DataFrame({
            "region_id": ["region-a", "region-a", "region-b", "region-c"],
            "phecode": ["100.1", "100.2", "200.1", "300.1"],
            "p": [1e-6, 2e-5, 9e-4, 2e-3],
        })
        events = []

        with patch.object(recompute_admixture_reference, "ensure_dependencies"), \
             patch.object(recompute_admixture_reference.admixture_lai, "run",
                          side_effect=lambda **_: events.append("stage1") or {}), \
             patch.object(recompute_admixture_reference.admixture_plrs, "run_discovery",
                          side_effect=lambda **_: events.append("stage2") or {
                              "summary": Path("summary.tsv"),
                              "discovery_predictions": Path("discovery.tsv.gz"),
                          }), \
             patch.object(recompute_admixture_reference, "_has_prediction", return_value=True), \
             patch.object(recompute_admixture_reference.admixture_phewas, "run_discovery",
                          side_effect=lambda **_: events.append("stage3") or discovery), \
             patch.object(recompute_admixture_reference.admixture_phewas, "analysis_dir",
                          return_value=Path("phewas")), \
             patch.object(recompute_admixture_reference.admixture_plrs,
                          "predict_replication_regions",
                          side_effect=lambda regions, **_: (
                              events.append(("stage4", list(regions)))
                              or Path("replication.tsv.gz")
                          )), \
             patch.object(recompute_admixture_reference.admixture_phewas, "run_replication",
                          side_effect=lambda *_, **__: events.append("stage5") or {
                              "replication": Path("replication-phewas.tsv.gz"),
                              "meta": Path("meta.tsv"),
                          }):
            outputs = recompute_admixture_reference.run_all(
                run_finemap=False
            )

        self.assertEqual(events, [
            "stage1", "stage2", "stage3",
            ("stage4", ["region-a", "region-b"]), "stage5",
        ])
        self.assertEqual(
            outputs["plrs"]["replication_predictions"],
            Path("replication.tsv.gz"),
        )

    def test_completed_stage3_resumes_directly_at_stage4(self):
        discovery = pd.DataFrame({
            "region_id": ["region-a", "region-b", "region-c"],
            "phecode": ["100.1", "200.1", "300.1"],
            "p": [1e-6, 9e-4, 2e-3],
        })
        events = []

        with patch.object(recompute_admixture_reference, "ensure_dependencies"), \
             patch.object(recompute_admixture_reference, "_has_prediction", return_value=True), \
             patch.object(recompute_admixture_reference.admixture_phewas,
                          "load_cached_discovery", return_value=discovery), \
             patch.object(recompute_admixture_reference.admixture_lai, "run") as stage1, \
             patch.object(recompute_admixture_reference.admixture_plrs,
                          "run_discovery") as stage2, \
             patch.object(recompute_admixture_reference.admixture_phewas,
                          "run_discovery") as stage3, \
             patch.object(recompute_admixture_reference.admixture_phewas, "analysis_dir",
                          return_value=Path("phewas")), \
             patch.object(recompute_admixture_reference.admixture_plrs,
                          "predict_replication_regions",
                          side_effect=lambda regions, **_: (
                              events.append(("stage4", list(regions)))
                              or Path("replication.tsv.gz")
                          )), \
             patch.object(recompute_admixture_reference.admixture_phewas, "run_replication",
                          side_effect=lambda *_, **__: events.append("stage5") or {}):
            recompute_admixture_reference.run_all(
                run_finemap=False, force=False
            )

        stage1.assert_not_called()
        stage2.assert_not_called()
        stage3.assert_not_called()
        self.assertEqual(events, [
            ("stage4", ["region-a", "region-b"]), "stage5",
        ])


if __name__ == "__main__":
    unittest.main()
