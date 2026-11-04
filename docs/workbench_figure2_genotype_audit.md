> Workbench workflow reference. Configure the required inputs in your authorized workspace before running these commands; consult the root README for the repository scope.

# Figure 2 representative genotype audit

Run after `scripts.figure2_loci.run(panel_only=True, force=False)` has completed.
This checks the representative variants from `figures/figure2_sources/loci_panel/fine_mapping_summary.tsv`
against the already-staged discovery VCF. It does not download inputs, refit models,
change existing analysis outputs, or emit individual-level results or plots.

```bash
%%bash
set -euo pipefail
repo_dir="${HOME}/SelectionMapping"
if [[ -d "${repo_dir}/.git" ]]; then
  git -C "${repo_dir}" pull --ff-only origin main
else
  git clone --branch main git@github.com:soisa001/SelectionMapping.git "${repo_dir}"
fi
cd "${repo_dir}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
mkdir -p analysis/figure_notebook_runs
uv run --link-mode copy --with-requirements requirements-figure2-loci.txt \
  python -u -m scripts.figure2_genotype_audit \
  --threads 4 --min-gp 0.90 --min-gq 20 --min-dp 10 \
  2>&1 | tee "analysis/figure_notebook_runs/genotype_audit_$(date -u +%Y%m%dT%H%M%SZ).log"
```

The saved run manifest supplies the exact local work directory, source VCF filename,
cohort hash, and minimum phenotype count. The saved discovery sample manifest supplies
the complete-covariate, unrelated sample set. The audit reapplies phenotype and sex
exclusions and checks resulting sample sizes, MAC, ALT-carrier counts and missingness
against the saved summary. It fails with a saved comparison report if they disagree.
An explicit `--vcf /already/local/input.vcf.gz` override is recorded in provenance;
it should be used only for an intentional comparison. An index is required.

Outputs are in `analysis/figure2/genotype_audit_panel/`:

- `genotype_counts.tsv`: per phenotype, for the analysis group, cases, and controls;
  GT missingness, partial calls, 0/0–0/1–1/1 counts, ALT allele count/number, ALT AF,
  minor allele count/frequency, and ALT carriers. Partial calls are fully excluded
  from the diploid denominator, matching the fine-map implementation.
- `summary_comparison.tsv`: agreement with the saved summary, case/control
  missingness Fisher test, FORMAT fields, site FILTER and INFO values. A source
  INFO frequency may use a different cohort, so it is not substituted for recomputed AF.
- `quality_sensitivity.tsv`: independently mask calls using the posterior of the
  **reported GT** from GP >= 0.90, GQ >= 20, or DP >= 10, then recompute counts and
  frequencies. These are diagnostic sensitivity thresholds, not automatic changes
  to the scientific analysis. Missing/invalid field values are excluded from that
  sensitivity denominator. Absent fields are marked `field_absent`, not zero.
- `quality_fields.tsv`: valid-value counts and distribution summaries for available
  GP, GQ, DP, DS and heterozygote AD allele balance; DS-based AF uses available valid
  dosages and has its own denominator. Absent fields have no distribution rows.
- `vcf_metadata.txt`: VCF metadata lines, excluding the sample-ID header.
- `manifest.json` and `audit.log`: parameters, paths, source fingerprints, cohort
  hash, output checksums, and status. The audit is deterministic, uses four workers
  by default, and validates written tables by reading them back.

The default discovery input is a KAGE/GLIMPSE imputed merged VCF. Zero GT missingness
in that input does not establish zero pre-imputation missingness or high confidence.
This audit cannot recover pre-imputation missingness from an imputed file. If quality
fields are absent, the reports say so; checking earlier-stage input would then be
necessary. GTs must be biallelic and diploid; malformed and duplicate exact records
fail rather than silently producing frequencies.

Run the relevant unit tests:

```bash
uv run --link-mode copy --with-requirements requirements-figure2-loci.txt --with pytest \
  python -m pytest tests/test_figure2_genotype_audit.py -q
```
