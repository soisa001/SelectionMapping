from panel_fixtures import finemap_frame, discovery_frame
import pytest

pytestmark = pytest.mark.usefixtures("panel_gene_annotations")
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import (
    figure3_ehh,
    figure3_locus_combined,
    figure4_finemap,
    figure4_meta,
    figure4_plots,
    figure4_s9,
    recompute_figure4_susie,
)


class Figure3ForestTests(unittest.TestCase):
    def test_forest_needs_only_meta_effect_columns(self):
        frame = pd.DataFrame({
            "variant_tag": ["1_100_A_G"],
            "phecode": ["CV_401"],
            "phecode_string": ["Hypertension"],
            "beta_meta": [-0.25],
            "se_meta": [0.05],
            "p_meta": [1e-8],
            "meta_significant": [True],
        })
        fig = figure4_meta.plot_forest(frame)
        try:
            label_axis, p_axis, effect_axis = fig.axes
            self.assertIsNone(effect_axis.get_legend())
            self.assertEqual(label_axis.get_title(loc="left"), "Phecode (Variant)")
            self.assertEqual(p_axis.get_title(), "$p$")
            self.assertEqual(effect_axis.get_title(), "$\\beta$ and 95% CI")
            self.assertEqual(
                effect_axis.get_xlabel(), "Meta-analysis $\\beta$ (log-odds)"
            )
            self.assertEqual(
                label_axis.texts[0].get_text(), "Hypertension (1:100 A>G)"
            )
            self.assertEqual(p_axis.texts[0].get_text(), "$1.0\\times 10^{-8}$")
            self.assertLessEqual(fig.get_figheight(), 5.2)
        finally:
            plt.close(fig)

    def test_v4_forest_omits_variant_information(self):
        frame = pd.DataFrame({
            "variant_tag": ["4_99103864_C_T"],
            "phecode": ["MB_280.1"],
            "phecode_string": ["Alcohol use disorders"],
            "beta_meta": [-0.55],
            "se_meta": [0.08],
            "p_meta": [3.8e-21],
            "meta_significant": [True],
        })
        fig = figure4_meta.plot_forest(frame, show_variant=False)
        try:
            label_axis = fig.axes[0]
            self.assertEqual(label_axis.get_title(loc="left"), "Phecode")
            self.assertEqual(label_axis.texts[0].get_text(), "Alcohol use disorders")
        finally:
            plt.close(fig)



class Figure3LocusTests(unittest.TestCase):
    def test_gene_track_keeps_complete_b_and_c_sized_windows(self):
        genes = [
            {
                "name": f"GENE{i}",
                "tx_start": 100_000 + i * 45_000,
                "tx_end": 112_000 + i * 45_000,
                "strand": "+",
                "exons": [(100_000 + i * 45_000, 112_000 + i * 45_000)],
            }
            for i in range(16)
        ]
        genes[12]["name"] = "ADH1B"
        fig, ax = plt.subplots()
        try:
            figure4_finemap._draw_gene_track(
                ax,
                genes,
                50_000,
                1_050_000,
                highlight=None,
                required="ADH1B",
            )
            labels = [text.get_text() for text in ax.texts]
            self.assertEqual(len(labels), 16)
            self.assertIn("ADH1B", labels)
        finally:
            plt.close(fig)

    def test_gene_track_adds_a_row_instead_of_overlapping_labels(self):
        genes = [
            {
                "name": f"LONG_GENE_{i}",
                "tx_start": 450_000,
                "tx_end": 550_000,
                "strand": "+",
                "exons": [(450_000, 550_000)],
            }
            for i in range(7)
        ]
        fig, ax = plt.subplots()
        try:
            figure4_finemap._draw_gene_track(ax, genes, 0, 1_000_000)
            label_rows = {text.get_position()[1] for text in ax.texts}
            self.assertEqual(len(label_rows), 7)
        finally:
            plt.close(fig)

    def test_all_finemaps_use_500kb_flanks_and_spacious_gene_tracks(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        frame = finemap_frame(locus)
        fig = figure4_finemap.plot_finemap(frame, locus)
        try:
            expected = np.array([
                (locus["lead_pos"] - 500_000) / 1e6,
                (locus["lead_pos"] + 500_000) / 1e6,
            ])
            np.testing.assert_allclose(fig.axes[0].get_xlim(), expected)
            np.testing.assert_allclose(fig.axes[1].get_xlim(), expected)
            self.assertEqual(figure4_finemap.GENE_TRACK_ROWS, 6)
            self.assertGreater(fig.axes[1].get_ylim()[1], 9.6)
        finally:
            plt.close(fig)

    def test_compact_locus_can_omit_title_and_r2_scale(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        frame = finemap_frame(locus)
        fig = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            show_r2_legend=False,
        )
        try:
            self.assertIsNone(fig._suptitle)
            self.assertEqual(len(fig.axes), 2)
        finally:
            plt.close(fig)

    def test_compact_pair_reserves_equal_width_for_the_shared_r2_scale(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        frame = finemap_frame(locus)
        left = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            show_r2_legend=False,
            reserve_r2_legend_space=True,
        )
        right = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            show_r2_legend=True,
            reserve_r2_legend_space=True,
        )
        try:
            left.canvas.draw()
            right.canvas.draw()
            self.assertAlmostEqual(
                left.axes[0].get_position().width,
                right.axes[0].get_position().width,
                places=6,
            )
            self.assertAlmostEqual(
                left.axes[1].get_position().width,
                right.axes[1].get_position().width,
                places=6,
            )
            np.testing.assert_allclose(
                left.get_size_inches(), figure4_finemap._FIGSIZE_COMPACT
            )
        finally:
            plt.close(left)
            plt.close(right)

    def test_panel_e_compact_gene_track_has_no_bold_gene_names(self):
        locus = figure4_finemap.FINEMAP_LOCI["C"]
        frame = finemap_frame(locus)
        fig = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            show_r2_legend=True,
            reserve_r2_legend_space=True,
            highlight_gene=False,
        )
        try:
            gene_labels = [text for text in fig.axes[1].texts if text.get_text()]
            cfap69 = next(text for text in gene_labels if text.get_text() == "CFAP69")
            self.assertEqual(cfap69.get_fontweight(), "normal")
            self.assertTrue(
                all(text.get_fontweight() == "normal" for text in gene_labels)
            )
        finally:
            plt.close(fig)

    def test_locus_plot_omits_threshold_and_min_p_legend_label(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        frame = finemap_frame(locus)
        fig = figure4_finemap.plot_finemap(frame, locus)
        try:
            pvalue_axis = fig.axes[0]
            horizontal_thresholds = []
            for line in pvalue_axis.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if len(y) > 1 and np.allclose(y, y[0]) and line.get_linestyle() == "--":
                    horizontal_thresholds.append(line)
            self.assertEqual(horizontal_thresholds, [])
            legend = pvalue_axis.get_legend()
            labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
            self.assertNotIn("min-p variant", labels)
            self.assertNotIn("iHS lead SNV", labels)
            self.assertEqual(len(pvalue_axis.collections), 1)
            collection_sizes = np.concatenate([
                collection.get_sizes() for collection in pvalue_axis.collections
                if len(collection.get_sizes())
            ])
            self.assertEqual(
                int(np.sum(np.isclose(
                    collection_sizes,
                    figure4_finemap.FINEMAP_BEST_P_MARKER_SIZE,
                ))),
                1,
            )
            self.assertTrue(np.any(np.isclose(
                collection_sizes, figure4_finemap.FINEMAP_MARKER_SIZE
            )))
            self.assertGreater(
                figure4_finemap.FINEMAP_BEST_P_MARKER_SIZE,
                figure4_finemap.FINEMAP_MARKER_SIZE,
            )
            self.assertGreaterEqual(figure4_finemap.FINEMAP_MARKER_SIZE, 120)
            self.assertFalse(np.any(np.isclose(collection_sizes, 260)))
            self.assertFalse(any(
                line.get_color() == "goldenrod" for line in pvalue_axis.lines
            ))

            gene_axis = fig.axes[1]
            gene_labels = [text for text in gene_axis.texts if text.get_text() != ""]
            self.assertTrue(gene_labels)
            self.assertGreaterEqual(min(text.get_fontsize() for text in gene_labels), 26)
            self.assertTrue(all(text.get_color() == "black" for text in gene_labels))
            adh1b = next(text for text in gene_labels if text.get_text() == "ADH1B")
            self.assertEqual(adh1b.get_fontweight(), "normal")
            self.assertFalse(any(
                line.get_color() == "goldenrod" for line in gene_axis.lines
            ))
        finally:
            plt.close(fig)

    def test_supplementary_finemap_expands_rows_with_matched_gene_fraction(self):
        locus = figure4_finemap.FINEMAP_LOCI["E"]
        frame = finemap_frame(locus)
        standard = figure4_finemap.plot_finemap(
            frame, locus, show_title=False
        )
        expanded = figure4_finemap.plot_finemap(
            frame, locus, show_title=False, expanded_gene_track=True
        )
        try:
            standard_rows = {
                text.get_position()[1] for text in standard.axes[1].texts
            }
            expanded_rows = {
                text.get_position()[1] for text in expanded.axes[1].texts
            }
            self.assertTrue(standard_rows)
            self.assertTrue(expanded_rows)
            self.assertEqual(
                {text.get_text() for text in expanded.axes[1].texts},
                {text.get_text() for text in standard.axes[1].texts},
            )
            self.assertEqual(
                figure4_finemap.SUPPLEMENTARY_FINEMAP_HEIGHT_RATIOS,
                (7.5, 4.9),
            )
            pvalue_height = expanded.axes[0].get_position().height
            gene_height = expanded.axes[1].get_position().height
            self.assertAlmostEqual(
                gene_height / (pvalue_height + gene_height),
                4.9 / 12.4,
                places=3,
            )
            np.testing.assert_allclose(
                expanded.get_size_inches(),
                figure4_finemap.SUPPLEMENTARY_FIGSIZE_LOCUS,
            )
            self.assertAlmostEqual(
                expanded.get_size_inches()[0] / expanded.get_size_inches()[1],
                1.3405649529424606,
                places=6,
            )
            colorbar_axis = expanded.axes[2]
            self.assertEqual(colorbar_axis.get_ylim(), (0.0, 1.0))
        finally:
            plt.close(standard)
            plt.close(expanded)

    def test_supplementary_shared_gene_track_reuses_figure3_geometry(self):
        locus = figure4_finemap.FINEMAP_LOCI["E"]
        center = int(locus["lead_pos"])
        start = center - figure4_finemap.HALF_WINDOW
        end = center + figure4_finemap.HALF_WINDOW
        shared_genes = {
            "LEFT1": {
                "tx_start": center - 360_000,
                "tx_end": center - 250_000,
                "strand": "+",
                "exons": [(center - 360_000, center - 350_000)],
            },
            "MIDDLE2": {
                "tx_start": center - 20_000,
                "tx_end": center + 80_000,
                "strand": "-",
                "exons": [(center - 20_000, center - 10_000)],
            },
        }
        frame = finemap_frame(locus)
        fine = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            expanded_gene_track=True,
            window_center_pos=center,
            shared_gene_track=shared_genes,
        )
        reference, reference_axis = plt.subplots()
        figure3_ehh.draw_gene_track(
            reference_axis,
            shared_genes,
            start,
            end,
            highlight_gene=None,
            gene_fontsize=figure3_locus_combined.GENE_FONTSIZE_COMBINED,
        )
        try:
            fine_axis = fine.axes[1]
            fine_labels = sorted(
                (text.get_text(), text.get_position(), text.get_fontsize())
                for text in fine_axis.texts
            )
            reference_labels = sorted(
                (text.get_text(), text.get_position(), text.get_fontsize())
                for text in reference_axis.texts
            )
            self.assertEqual(fine_labels, reference_labels)
            self.assertEqual(fine_axis.get_ylim(), reference_axis.get_ylim())
            np.testing.assert_allclose(fine_axis.get_xlim(), (start, end))
            np.testing.assert_allclose(
                fine_axis.get_xticks(), (start, center, end)
            )
        finally:
            plt.close(fine)
            plt.close(reference)

    def test_optional_sv_marker_uses_type_shape_and_red_outline(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        frame = finemap_frame(locus)
        sv_pos = int(frame.iloc[len(frame) // 2]["pos"])
        fig = figure4_finemap.plot_finemap(
            frame,
            locus,
            show_title=False,
            sv_pos=sv_pos,
            sv_type="INS",
        )
        try:
            pvalue_axis = fig.axes[0]
            self.assertEqual(figure4_finemap.sv_marker_shape("INS"), "^")
            self.assertEqual(figure4_finemap.sv_marker_shape("DEL"), "v")
            with self.assertRaisesRegex(ValueError, "expected INS or DEL"):
                figure4_finemap.sv_marker_shape("INV")
            self.assertEqual(len(pvalue_axis.collections), 2)
            sv_marker = pvalue_axis.collections[-1]
            self.assertTrue(
                np.allclose(sv_marker.get_sizes(), figure4_finemap.SV_MARKER_SIZE)
            )
            self.assertTrue(np.allclose(
                sv_marker.get_edgecolors()[0],
                to_rgba(figure4_finemap.SV_OUTLINE_COLOR),
            ))
            self.assertGreaterEqual(figure4_finemap.SV_MARKER_SIZE, 1400)
            self.assertEqual(figure4_finemap.SV_OUTLINE_COLOR, "#d95f02")
            self.assertGreaterEqual(figure4_finemap.SV_MARKER_LINEWIDTH, 7.0)
            self.assertFalse(any(
                line.get_color() == figure4_finemap.SV_OUTLINE_COLOR
                for line in pvalue_axis.lines
            ))
            self.assertIn("INS SV", [text.get_text() for text in pvalue_axis.texts])
        finally:
            plt.close(fig)

    def test_showcase_includes_requested_hypertension_and_breast_lump_loci(self):
        hypertension = figure4_finemap.FINEMAP_LOCI["D"]
        breast_lump = figure4_finemap.FINEMAP_LOCI["E"]
        self.assertEqual(
            (hypertension["chrom"], hypertension["lead_pos"],
             hypertension["lead_ref"], hypertension["lead_alt"],
             hypertension["phecode"]),
            ("chr1", 35_126_025, "A", "G", "CV_401"),
        )
        self.assertEqual(
            (breast_lump["chrom"], breast_lump["lead_pos"],
             breast_lump["lead_ref"], breast_lump["lead_alt"],
             breast_lump["phecode"]),
            ("chr2", 85_632_712, "T", "A", "CA_142.1"),
        )

    def test_cache_contract_rejects_narrower_window(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        expected_start = locus["lead_pos"] - figure4_finemap.HALF_WINDOW
        expected_end = locus["lead_pos"] + figure4_finemap.HALF_WINDOW
        row = {
            "_n_pcs": 20,
            "_analysis_contract": figure4_finemap.ANALYSIS_CONTRACT,
            "_min_carriers": figure4_finemap.FINEMAP_MIN_CARRIERS,
            "_sex_scope": locus["sex_scope"],
            "_window_start": expected_start + 250_000,
            "_window_end": expected_end - 250_000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "narrow.tsv"
            pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "expected full window"):
                figure4_finemap.validate_discovery_contract(
                    path, expected_scope=locus["sex_scope"],
                    expected_locus=locus,
                )
            row["_window_start"] = expected_start
            row["_window_end"] = expected_end
            row["_min_carriers"] = 19
            pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "expected _min_carriers=20"):
                figure4_finemap.validate_discovery_contract(
                    path, expected_scope=locus["sex_scope"],
                    expected_locus=locus,
                )
            row["_min_carriers"] = figure4_finemap.FINEMAP_MIN_CARRIERS
            pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
            figure4_finemap.validate_discovery_contract(
                path, expected_scope=locus["sex_scope"],
                expected_locus=locus,
            )

    def test_plot_cache_invalidates_when_renderer_dependency_is_newer(self):
        locus = figure4_finemap.FINEMAP_LOCI["B"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result = directory / "result.tsv"
            dependency = directory / "renderer.py"
            targets = [directory / "panel.png", directory / "panel.pdf"]
            result.write_text("result", encoding="utf-8")
            dependency.write_text("renderer", encoding="utf-8")
            for target in targets:
                target.write_text("plot", encoding="utf-8")

            os.utime(result, (100, 100))
            os.utime(dependency, (200, 200))
            for target in targets:
                os.utime(target, (300, 300))

            with (
                patch.object(
                    figure4_finemap,
                    "_find_discovery_result",
                    return_value=str(result),
                ),
                patch.object(figure4_finemap, "validate_discovery_contract"),
                patch.object(
                    figure4_finemap,
                    "_render_dependency_paths",
                    return_value=[dependency],
                ),
            ):
                self.assertTrue(
                    figure4_finemap._render_targets_current(locus, targets)
                )
                os.utime(dependency, (400, 400))
                self.assertFalse(
                    figure4_finemap._render_targets_current(locus, targets)
                )

    def test_meta_significant_loci_keep_variant_phecode_pairs(self):
        meta = pd.DataFrame({
            "variant_tag": ["4_99103864_C_T", "4_99103864_C_T", "7_90206732_G_A"],
            "phecode": ["MB_280.1", "MB_280.11", "GU_626.1"],
            "phecode_string": ["Alcohol use disorders", "Alcohol dependence", "Cycle disorder"],
            "p_meta": [1e-10, 2e-9, 3e-8],
            "meta_significant": [True, True, True],
        })
        with patch.object(figure4_meta, "load_meta_results", return_value=meta):
            loci = figure4_finemap.meta_significant_loci()
        keys = {(locus["variant_tag"], locus["phecode"]) for locus in loci}
        self.assertEqual(len(keys), 3)
        self.assertIn(("4_99103864_C_T", "MB_280.11"), keys)

    def test_v5_sv_loci_use_s9_association_leads_and_exact_sv_annotations(self):
        frame = pd.DataFrame(
            {
                "variant_tag": [
                    "22_20395000_A_G",
                    "2_131129733_A_AT",
                ],
                "phecode": ["GU_592", "EM_247.5"],
                "phecode_string": [
                    "Cystitis and urethritis",
                    "Disorders of calcium metabolism",
                ],
                "sv_annotation": [
                    "chr22:20401764 (90 bp INS)",
                    "chr2:131129733 (111 bp INS)",
                ],
                "gene_in_window": ["TEST22", "TEST2"],
                "p": [1e-4, 2e-4],
            }
        )
        loci = figure4_finemap.linked_sv_loci(frame)
        self.assertEqual(loci["F"]["variant_tag"], "22_20395000_A_G")
        self.assertEqual(loci["F"]["sv_pos"], 20_401_764)
        self.assertEqual(loci["F"]["phecode"], "GU_592")
        self.assertEqual(loci["G"]["variant_tag"], "2_131129733_A_AT")
        self.assertEqual(loci["G"]["sv_length"], 111)
        self.assertEqual(loci["G"]["sex_scope"], "Both")

    def test_v5_sv_locus_refuses_a_lead_window_that_excludes_the_sv(self):
        frame = pd.DataFrame(
            {
                "variant_tag": ["22_19000000_A_G", "2_131129733_A_AT"],
                "phecode": ["GU_592", "EM_247.5"],
                "phecode_string": [
                    "Cystitis and urethritis",
                    "Disorders of calcium metabolism",
                ],
                "sv_annotation": [
                    "chr22:20401764 (90 bp INS)",
                    "chr2:131129733 (111 bp INS)",
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, r"inside the \+/-500,000 bp window"):
            figure4_finemap.linked_sv_loci(frame)


    def test_controlled_v5_panel_refuses_a_missing_susie_result(self):
        locus = {
            "chrom": "chr22",
            "lead_pos": 20_395_000,
            "phecode": "GU_592",
            "variant_tag": "22_20395000_A_G",
            "lead_gene": "TEST22",
        }
        with (
            patch.object(figure4_finemap, "_stage_remote_artifacts"),
            patch.object(figure4_finemap, "_find_result_tsv", return_value=None),
            self.assertRaisesRegex(FileNotFoundError, "Verified SuSiE result missing"),
        ):
            figure4_finemap.load_finemap_result(locus)


class Figure3RecomputeTests(unittest.TestCase):
    def test_primary_susie_dependency_contract_includes_pyarrow(self):
        notebook = (
            recompute_figure4_susie.config.NOTEBOOKS_DIR
            / "susie_finemap_SNVs.ipynb"
        )
        source = recompute_figure4_susie._primary_cell_source(notebook)
        self.assertIn('"pyarrow": "pyarrow>=14"', source)

        requirements = (
            recompute_figure4_susie.config.REPO_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8")
        active = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(any(line.startswith("pyarrow>=14") for line in active))

    def test_primary_susie_cell_accepts_exact_v5_task_table_override(self):
        notebook = (
            recompute_figure4_susie.config.NOTEBOOKS_DIR
            / "susie_finemap_SNVs.ipynb"
        )
        source = recompute_figure4_susie._primary_cell_source(notebook)
        self.assertIn('os.environ.get("AOU_FIGURE3_TASKS_TSV")', source)

    def test_primary_susie_cell_filters_variants_with_fewer_than_20_carriers(self):
        notebook = (
            recompute_figure4_susie.config.NOTEBOOKS_DIR
            / "susie_finemap_SNVs.ipynb"
        )
        source = recompute_figure4_susie._primary_cell_source(notebook)
        self.assertIn("MIN_CARRIERS      = 20", source)
        self.assertIn("observed_carriers = nz & (dos > 0)", source)
        self.assertIn(
            "carrier_counts = np.sum(carrier_matrix[mask], axis=0).astype(int)",
            source,
        )
        self.assertIn("eligible = carrier_counts >= MIN_CARRIERS", source)
        self.assertIn('df["_min_carriers"] = MIN_CARRIERS', source)
        self.assertIn('"n_carriers": carrier_counts', source)
        self.assertEqual(figure4_finemap.FINEMAP_MIN_CARRIERS, 20)
        self.assertTrue(figure4_finemap.ANALYSIS_CONTRACT.endswith("carriers20_v4"))

    def test_v5_task_table_preserves_variant_phecode_pairs(self):
        loci = [
            {
                "variant_tag": "22_20395000_A_G",
                "phecode": "GU_592",
                "phecode_name": "Cystitis and urethritis",
                "lead_gene": "TEST22",
                "meta_p": 1e-4,
            },
            {
                "variant_tag": "2_131129733_A_AT",
                "phecode": "EM_247.5",
                "phecode_name": "Disorders of calcium metabolism",
                "lead_gene": "TEST2",
                "meta_p": 2e-4,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            recompute_figure4_susie.config,
            "analysis_dir",
            return_value=Path(tmp),
        ):
            path = recompute_figure4_susie._write_task_table(loci)
            tasks = pd.read_csv(path, sep="\t", dtype=str)
        self.assertEqual(
            list(zip(tasks["variant_tag"], tasks["phecode"])),
            [("22_20395000_A_G", "GU_592"), ("2_131129733_A_AT", "EM_247.5")],
        )
        self.assertTrue(tasks["meta_significant"].eq("TRUE").all())


class Figure3VolcanoTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "variant_tag": ["1_100_A_G", "1_100_A_G", "2_200_C_T", "3_300_G_A"],
            "phecode": ["CV_401", "EM_202", "DE_684", "NS_333"],
            "phecode_string": ["Hypertension", "Anemia", "Alopecia", "Sleep"],
            # two nominally significant (p<=1e-3), two not.
            "beta": [0.6, -0.5, 0.4, -0.2],
            "p_value": [1e-5, 1e-4, 1e-2, 5e-1],
        })

    def test_volcano_filters_the_full_phewas_frame_to_genic_sentinels(self):
        from scripts import figure4_volcano

        phewas = figure4_volcano._normalize(pd.DataFrame({
            "variant_tag": ["1_100_A_G", "1_100_A_G", "2_200_C_T", "3_300_G_A"],
            "phecode": ["CV_401", "EM_202", "DE_684", "MS_705"],
            "phecode_string": ["Hypertension", "Anemia", "Alopecia", "Arthritis"],
            "beta": [0.6, -0.5, 0.4, -2.4],
            "p_value": [1e-5, 1e-4, 1e-2, 1e-4],
        }))
        sentinels = pd.DataFrame({
            "variant_tag": ["1_100_A_G", "2_200_C_T", "3_300_G_A"],
            "gene_in_window": ["GENE1", "GENE2", ""],
        })

        filtered = figure4_volcano._filter_to_genic_sentinels(
            phewas,
            sentinels,
            expected_genic_sentinels=2,
            expected_nominal_hits=2,
        )

        self.assertEqual(set(filtered["variant_tag"]), {"1_100_A_G", "2_200_C_T"})
        self.assertEqual(len(filtered), 3)
        self.assertFalse(filtered["beta"].abs().ge(2.0).any())

    def test_nonsignificant_points_are_gray_and_hits_are_coloured(self):
        from scripts import figure4_volcano
        df = figure4_volcano._normalize(self._frame())
        fig = figure4_volcano.plot_volcano(df)
        try:
            ax = fig.axes[0]
            # First two path collections are the background then the hits scatter.
            bg_colors = ax.collections[0].get_facecolors()
            hit_colors = ax.collections[1].get_facecolors()
            self.assertTrue(len(bg_colors) and len(hit_colors))
            gray = matplotlib.colors.to_rgb(figure4_volcano.BG_COLOR)
            self.assertTrue(np.allclose(bg_colors[0][:3], gray, atol=1e-3))
            pos = matplotlib.colors.to_rgb(figure4_volcano.POS_COLOR)
            neg = matplotlib.colors.to_rgb(figure4_volcano.NEG_COLOR)
            for row in hit_colors:
                self.assertTrue(
                    np.allclose(row[:3], pos, atol=1e-3)
                    or np.allclose(row[:3], neg, atol=1e-3))
            # threshold line drawn at -log10(1e-3) = 3.
            hlines = [ln for ln in ax.lines
                      if len(ln.get_ydata()) and np.allclose(ln.get_ydata(),
                                                             ln.get_ydata()[0])]
            self.assertTrue(any(np.isclose(ln.get_ydata()[0], 3.0) for ln in hlines))
        finally:
            plt.close(fig)

    def test_v3_outlines_meta_significant_hits_with_black_circles(self):
        from scripts import figure4_volcano
        df = figure4_volcano._normalize(self._frame())
        meta_keys = {("1_100_A_G", "CV_401")}
        base = figure4_volcano.plot_volcano(df)
        outlined = figure4_volcano.plot_volcano(
            df, circle_meta=True, meta_keys=meta_keys)
        try:
            # v3 adds one more scatter collection (the open black outlines).
            self.assertGreater(len(outlined.axes[0].collections),
                               len(base.axes[0].collections))
            edge = outlined.axes[0].collections[2].get_edgecolors()
            self.assertTrue(np.allclose(edge[0][:3], (0.0, 0.0, 0.0), atol=1e-3))
            legend_labels = [t.get_text()
                             for t in outlined.axes[0].get_legend().get_texts()]
            self.assertIn("meta-significant", legend_labels)
        finally:
            plt.close(base)
            plt.close(outlined)

    def test_marker_area_scales_with_effect_size_capped_like_manhattan(self):
        from scripts import figure4_volcano
        # area == SIZE_FLOOR + SIZE_SCALE * ln(min(|b|, BETA_CAP) + 1)
        self.assertAlmostEqual(
            float(figure4_volcano._size_from_beta(np.array([0.0]))[0]),
            figure4_volcano.SIZE_FLOOR)
        capped = figure4_volcano._size_from_beta(np.array([figure4_volcano.BETA_CAP]))
        beyond = figure4_volcano._size_from_beta(np.array([50.0]))
        self.assertAlmostEqual(float(capped[0]), float(beyond[0]))  # cap saturates

    def test_labels_top_four_per_effect_direction(self):
        from scripts import figure4_volcano
        self.assertEqual(figure4_volcano.N_LABEL_PER_SIDE, 4)
        # Build a frame with 8 positive-beta and 8 negative-beta hits so the
        # per-side cap (4 each) is exercised in both directions.
        rows = []
        for i in range(8):
            rows.append(("v", f"POS_{i}", f"pos{i}", 0.5 + 0.1 * i, 10.0 ** -(4 + i)))
            rows.append(("v", f"NEG_{i}", f"neg{i}", -0.5 - 0.1 * i, 10.0 ** -(4 + i)))
        df = figure4_volcano._normalize(pd.DataFrame(
            rows, columns=["variant_tag", "phecode", "phecode_string",
                           "beta", "p_value"]))
        fig = figure4_volcano.plot_volcano(df)
        try:
            # adjustText may relocate labels, so assert on the selection count.
            self.assertEqual(len(fig.axes[0].texts), 2 * 4)
        finally:
            plt.close(fig)


    def test_v4_phecode_names_use_manual_and_bracket_shortening(self):
        from scripts import figure4_volcano

        self.assertEqual(
            figure4_volcano._display_phecode_name(
                "Irregular menstrual cycle/bleeding"
            ),
            "Irregular menstrual cycle",
        )
        self.assertEqual(
            figure4_volcano._display_phecode_name("Regular menstrual cycle/bleeding"),
            "Regular menstrual cycle",
        )
        self.assertEqual(
            figure4_volcano._display_phecode_name(
                "Dry eye syndrome [Tear film insufficiency]"
            ),
            "Dry eye syndrome",
        )
        self.assertEqual(
            figure4_volcano._display_phecode_name(
                "Hypokalemia [Hypopotassemia]"
            ),
            "Hypokalemia",
        )
        shortened = {
            "Patient's noncompliance with medical treatment and regimen": (
                "Patient noncompliance with medical treatment"
            ),
            "Disorders of mineral metabolism and mineral deficiencies": (
                "Disorders of mineral metabolism"
            ),
            "Degenerative disc disease": "Degenerative disc disease",
            "Pulmonary insufficiency and acute respiratory distress syndrome": (
                "Pulmonary insufficiency"
            ),
        }
        for source, expected in shortened.items():
            self.assertEqual(
                figure4_volcano._display_phecode_name(source), expected
            )

    def test_v5_uses_bare_beta_label_top_legend_and_requested_limits(self):
        from scripts import figure4_volcano

        df = figure4_volcano._normalize(self._frame())
        self.assertEqual(figure4_volcano.FONT_SCALE, 1.35)
        fig = figure4_volcano.plot_volcano(
            df,
            circle_meta=True,
            meta_keys={("1_100_A_G", "CV_401")},
            shorten_labels=True,
            x_limit=2.0,
            y_limit=10.0,
            beta_labels=True,
            legend_above=True,
            font_scale=figure4_volcano.FONT_SCALE,
            legend_frame=False,
            label_boxes=False,
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(tuple(ax.get_xlim()), (-2.0, 2.0))
            self.assertEqual(ax.get_ylim()[1], 10.0)
            self.assertEqual(ax.get_xlabel(), r"$\beta$")
            self.assertTrue(all(text.get_bbox_patch() is None for text in ax.texts))
            legend = ax.get_legend()
            self.assertEqual(legend.get_title().get_text(), r"$|\beta|$")
            self.assertEqual(legend._ncols, 5)
            self.assertGreater(legend.get_bbox_to_anchor().y0, 1.0)
            self.assertFalse(legend.get_frame_on())
            self.assertGreater(
                ax.xaxis.label.get_fontsize(), figure4_volcano.VOLCANO_FONT
            )
            self.assertTrue(
                all(
                    text.get_fontsize() > figure4_volcano.VOLCANO_LABEL
                    for text in legend.get_texts()
                )
            )
        finally:
            plt.close(fig)

    def test_v5_annotations_use_nonoverlapping_upper_lanes_and_visible_leaders(self):
        from matplotlib.text import Annotation
        from matplotlib.patches import FancyArrowPatch
        from scripts import figure4_volcano

        rows = []
        long_names = [
            "Alcohol abuse and dependence",
            "Irregular menstrual cycle",
            "Respiratory failure",
            "Diseases of hair and hair follicles",
            "Essential hypertension",
        ]
        for side, prefix in ((-1.0, "NEG"), (1.0, "POS")):
            for i, name in enumerate(long_names):
                rows.append((
                    f"variant_{prefix}_{i}",
                    f"{prefix}_{i}",
                    name,
                    side * (0.25 + 0.22 * i),
                    10.0 ** -(3.4 + 0.45 * i),
                ))
        frame = figure4_volcano._normalize(pd.DataFrame(
            rows,
            columns=[
                "variant_tag",
                "phecode",
                "phecode_string",
                "beta",
                "p_value",
            ],
        ))
        fig = figure4_volcano.plot_volcano(
            frame,
            shorten_labels=True,
            x_limit=2.0,
            y_limit=10.0,
            beta_labels=True,
            legend_above=True,
            font_scale=figure4_volcano.FONT_SCALE,
            legend_frame=False,
            label_boxes=False,
        )
        try:
            fig.canvas.draw()
            ax = fig.axes[0]
            annotations = [text for text in ax.texts if isinstance(text, Annotation)]
            self.assertEqual(len(annotations), 8)

            renderer = fig.canvas.get_renderer()
            boxes = [
                matplotlib.text.Text.get_window_extent(text, renderer).expanded(
                    1.02, 1.10
                )
                for text in annotations
            ]
            for i, box in enumerate(boxes):
                self.assertFalse(any(
                    box.overlaps(other) for other in boxes[i + 1:]
                ))

            for annotation in annotations:
                target_axes_y = ax.transAxes.inverted().transform(
                    ax.transData.transform(annotation.xy)
                )[1]
                self.assertGreater(annotation.xyann[1], target_axes_y)
                self.assertGreaterEqual(annotation.xyann[1], 0.48)

            leaders = [annotation.arrow_patch for annotation in annotations]
            self.assertEqual(len(leaders), len(annotations))
            self.assertTrue(all(
                isinstance(leader, FancyArrowPatch) for leader in leaders
            ))
            self.assertTrue(all(leader.get_linewidth() >= 2.0 for leader in leaders))
            self.assertTrue(all(not leader.get_clip_on() for leader in leaders))
            leader_lengths = []
            for leader in leaders:
                vertices = leader.get_transform().transform(
                    leader.get_path().vertices
                )
                leader_lengths.append(float(np.linalg.norm(vertices[-1] - vertices[0])))
            self.assertTrue(all(length >= 30.0 for length in leader_lengths))
        finally:
            plt.close(fig)

    def test_v5_volcano_output_preserves_the_fixed_canvas(self):
        from scripts import figure4_volcano

        fig = plt.figure(figsize=(11, 8.5))
        with (
            patch.object(figure4_volcano, "load_phewas_results", return_value=pd.DataFrame()),
            patch.object(figure4_volcano, "_meta_sig_keys", return_value=set()),
            patch.object(
                figure4_volcano, "plot_volcano", return_value=fig
            ) as plot_volcano,
            patch.object(figure4_volcano, "volcano_image_path", return_value=Path("v5.png")),
            patch.object(figure4_volcano.config, "FIGURE_FORMATS", ("png",)),
            patch.object(fig, "savefig") as save,
        ):
            figure4_volcano.render_volcano()

        self.assertEqual(save.call_args.kwargs["bbox_inches"], fig.bbox_inches)
        self.assertEqual(plot_volcano.call_args.kwargs["x_limit"], 2.0)
        self.assertFalse(plot_volcano.call_args.kwargs["label_boxes"])


class Figure3S9PanelTests(unittest.TestCase):
    def test_s9_source_contract_keeps_446_valid_rows(self):
        frame = discovery_frame()
        self.assertEqual(len(frame), 446)
        self.assertTrue(frame["maf"].between(0, 0.5, inclusive="right").all())
        self.assertTrue(frame["abs_beta"].ge(0).all())
        self.assertTrue(frame["phecode_string"].str.startswith("Test phenotype ").all())
        self.assertEqual(int(frame["category"].eq("SV-linked hit").sum()), 4)

    def test_missing_discovery_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.tsv")
            with patch.dict(
                os.environ,
                {figure4_s9.SOURCE_ENV: missing, "GOOGLE_PROJECT": "test-project"},
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "requires audited source data"
                ):
                    figure4_s9.load_source_data()

    def test_panel_b_rejects_phecode_ids_as_display_names(self):
        frame = discovery_frame()
        frame["phecode_string"] = frame["phecode"]
        with self.assertRaisesRegex(
            figure4_s9.figure_s9_phewas.S9ContractError,
            "phenotype string names",
        ):
            figure4_s9._validate_source_data(frame)


    def test_v5_panel_b_enlarges_all_text_and_removes_legend_frame(self):
        frame = discovery_frame().iloc[[0, -21, -1]].copy()
        fig = figure4_s9.figure_s9_phewas.plot_s9(
            frame,
            marker_scale=figure4_s9.MARKER_SCALE,
            font_scale=figure4_s9.FONT_SCALE,
            legend_frame=False,
        )
        try:
            ax = fig.axes[0]
            legend = ax.get_legend()
            self.assertFalse(legend.get_frame_on())
            self.assertGreater(ax.xaxis.label.get_fontsize(), 24)
            self.assertTrue(
                all(text.get_fontsize() > 18 for text in legend.get_texts())
            )
            self.assertEqual(len(ax.texts), 1)
            self.assertGreater(ax.texts[0].get_fontsize(), 13)
            fig.canvas.draw()
            legend_box = legend.get_window_extent()
            self.assertGreaterEqual(legend_box.x0, fig.bbox.x0)
            self.assertLessEqual(legend_box.x1, fig.bbox.x1)
        finally:
            plt.close(fig)




if __name__ == "__main__":
    unittest.main()
