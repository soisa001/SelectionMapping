# Repository structure

- `notebooks/Figure_1.ipynb` through `Figure_5.ipynb`: individual main-figure panels and related reference panels.
- `scripts/figure1_*` through `figure5_*`: panel rendering and analysis code, including required supporting modules.
- `scripts/common/`: shared input handling, configuration, plotting style, and panel output.
- `notebooks/susie_finemap_SNVs.ipynb` and the admixture notebooks: supporting analysis workflows.
- `tests/`: local checks for retained analyses and panel output.

Raw inputs, caches, and generated results belong in the populated analysis workspace. They are not bundled with this reference repository.

Set `AOU_EXCLUDED_SAMPLE_IDS_FILE` to the controlled cohort-exclusion manifest on Workbench (one ID per line, no header). The default is `data/raw/excluded_related_sample_ids.txt`. Use the same cohort exclusions as the existing analysis; keep this file in the authorized workspace.
