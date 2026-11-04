import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba







class Figure2EnrichmentLabelTests(unittest.TestCase):
    """Panel H carries the annotation set in its x-label, not in a title."""

    def _enr(self):
        import numpy as np

        rng = np.random.default_rng(0)
        labels = ["GWAS", "eQTL", "sQTL"]
        return {
            "annotations": labels,
            "null": {a: rng.random(50) * 0.1 for a in labels},
            "observed": {a: 0.2 for a in labels},
            "stats": {a: {"perm_pval": 0.001, "fold_enrichment": 1.6}
                      for a in labels},
            "source": "cache",
        }

    def test_titleless_panels_name_the_annotation_set_on_the_x_axis(self):
        from scripts import figure3_enrichment

        fig, axes = plt.subplots(1, 3)
        try:
            figure3_enrichment.draw_panels(axes, self._enr(), show_titles=False)
            for ax, label in zip(axes, ("GWAS", "eQTL", "sQTL")):
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(
                    ax.get_xlabel(), f"% windows with {label}\nLD-linked hit"
                )
        finally:
            plt.close(fig)

    def test_standalone_strip_keeps_its_titles(self):
        from scripts import figure3_enrichment

        fig, axes = plt.subplots(1, 3)
        try:
            figure3_enrichment.draw_panels(axes, self._enr())
            for ax, label in zip(axes, ("GWAS", "eQTL", "sQTL")):
                self.assertTrue(ax.get_title().startswith(label))
                self.assertEqual(ax.get_xlabel(), "% windows with LD-linked hit")
        finally:
            plt.close(fig)

    def test_shared_legend_names_permutations(self):
        from scripts import figure3_enrichment

        fig, ax = plt.subplots()
        try:
            figure3_enrichment.add_shared_legend(fig, ax)
            self.assertEqual(
                [text.get_text() for text in ax.get_legend().get_texts()],
                ["Permutations", "Observed"],
            )
        finally:
            plt.close(fig)

    def test_panel_d_uses_supplied_null_colours_without_y_grid(self):
        from scripts import figure3_enrichment

        fig, axes = plt.subplots(1, 3)
        try:
            figure3_enrichment.draw_panels(
                axes,
                self._enr(),
                show_titles=False,
                null_colors=figure3_enrichment.D_NULL_PALETTE,
                show_y_grid=False,
            )
            for ax, label in zip(axes, ("GWAS", "eQTL", "sQTL")):
                expected = to_rgba(figure3_enrichment.D_NULL_PALETTE[label])
                self.assertTrue(ax.patches)
                for bar in ax.patches:
                    self.assertEqual(bar.get_facecolor(), expected)
                self.assertFalse(any(line.get_visible()
                                     for line in ax.get_ygridlines()))
        finally:
            plt.close(fig)


class Figure2GeneLabelFontTests(unittest.TestCase):
    def test_combined_locus_panels_reuse_complete_cache_without_vcf_access(self):
        from scripts import figure3_locus_combined

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for letter, gene in (("B", "LCT"), ("C", "SLC24A5")):
                stem = root / f"Figure_3{letter}_{gene}_combined"
                stem.with_suffix(".pdf").write_bytes(b"cached pdf")
                stem.with_suffix(".png").write_bytes(b"cached png")
            with (
                patch.object(
                    figure3_locus_combined.config,
                    "figure_dir",
                    return_value=root,
                ),
                patch.object(
                    figure3_locus_combined.figure3_ehh,
                    "on_workbench",
                    side_effect=AssertionError("VCF access must be skipped"),
                ),
            ):
                outputs = figure3_locus_combined.render_main_panels(
                    clean_run_plots=False,
                    panel_letters=("B", "C"),
                )

            self.assertEqual(
                outputs,
                [
                    root / "Figure_3B_LCT_combined.png",
                    root / "Figure_3C_SLC24A5_combined.png",
                ],
            )

    def test_combined_locus_gene_names_are_smaller(self):
        from scripts import figure3_ehh, figure3_locus_combined

        self.assertEqual(
            figure3_locus_combined.GENE_FONTSIZE_COMBINED,
            figure3_ehh.GENE_FONTSIZE - 6,
        )

    def test_draw_gene_track_honours_the_override(self):
        from scripts import figure3_ehh

        genes = {"LCT": {"tx_start": 10_000, "tx_end": 90_000,
                         "exons": [(20_000, 30_000)]}}
        fig, ax = plt.subplots()
        try:
            figure3_ehh.draw_gene_track(ax, genes, 0, 100_000, gene_fontsize=21)
            sizes = [t.get_fontsize() for t in ax.texts if t.get_text() == "LCT"]
            self.assertEqual(sizes, [21])
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
