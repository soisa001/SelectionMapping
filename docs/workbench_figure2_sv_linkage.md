> Workbench workflow reference. Configure the required inputs in your authorized workspace before running these commands; consult the root README for the repository scope.

# All SVs linked to admixture fine-map minimum-p hits

For the new `loci_panel_sv` results, the scan runs automatically over the
**outlier boundaries ±100 kb**, saving CSV and TSV reports beside those
results. See the [panel-plus-SV workflow](workbench_figure2_panel_sv.md).
The standalone option `--window-basis outlier` selects that window; its summary
must contain `chrom`, `start`, and `end`. The legacy lead-centered mode below
remains the CLI default (`--window-basis lead`). Both modes report exact
REF/ALT, ALT AF, length and `sv_locus` (`chr:pos`) for all qualifying pairs.

The requested search is **r² > 0.5 within 100 kb of each minimum-p lead SNP**.
The input is the representative variant in each row of the completed Figure 2
panel-only summary. It is not restricted to one predefined SV per region or to
members of a credible set.

Run this after the existing Figure 2 locus workflow has completed:

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
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
output_dir="${repo_dir}/analysis/figure2/sv_linkage_panel"
log_dir="${repo_dir}/analysis/figure_notebook_runs"
mkdir -p "${output_dir}" "${log_dir}"
uv run --link-mode copy --with-requirements requirements-figure2-loci.txt \
  python -u -m scripts.figure2_sv_linkage \
  --threads 4 --flank-bp 100000 --r2-threshold 0.5 \
  --output "${output_dir}" \
  2>&1 | tee "${log_dir}/sv_linkage_$(date -u +%Y%m%dT%H%M%SZ).log"
```

This reads locally staged inputs referenced by the previous run's manifest
and writes the linkage reports on the Workbench VM.
The source is the same discovery VCF used for the SNP association, which by default
is the KAGE/GLIMPSE imputed merged VCF. This reports SVs represented in that source;
it does not discover additional SVs from sequencing reads. An already-local indexed
VCF can be supplied with `--vcf` for an intentional comparison; the override is
recorded, and the lead counts must still reproduce the original summary.

## Reports

All outputs stay in `analysis/figure2/sv_linkage_panel/` on Workbench:

- **`linked_svs.tsv` / `linked_svs.csv`**: every qualifying SV–lead–phenotype pair, sorted by linkage
  within phenotype/lead. Includes SV chromosome/POS, type, length, original VCF ID,
  exact REF/ALT and allele index/hash, distance, signed r, r², pairwise-complete r²,
  sample counts, ALT AF, carrier counts in cases/controls, missingness, and QC flags.
- **`locus_summary.tsv`**: every requested region/phenotype, including completed
  scans with zero SVs or zero qualifying links. Reports scanned, evaluable, and
  linked allele counts and whether a checkpoint was reused.
- **`all_sv_pairs.tsv`**: every structural allele scanned, including low/undefined
  LD, to make the threshold selection auditable.
- **`manifest.json`, `linkage.log`, `cache/`**: parameters, source and sample
  fingerprints, output checksums, completion/failure status, and reusable
  per-association checkpoints.

The same SV can appear once per phenotype because phenotype exclusions can change
the sample set and therefore its r². The manifest also records the number of
distinct linked SV alleles. `genes_in_finemap_window` is retained as regional
context; it does not assert that the SV overlaps those genes.

## Definitions

- Reference variant: `representative_variant`, with selection contract
  `minimum_score_p_across_finemap_window`. Its sample sizes, MAC, ALT-carrier counts,
  and missingness must reproduce the saved summary before its locus is scanned.
- Cohort: the same saved unrelated, complete-covariate discovery sample manifest,
  followed by the original phenotype and sex exclusions.
- Distance: same chromosome and `abs(SV_POS - lead_POS) <= 100000`, inclusive at
  either boundary. SVs anchored outside that range are outside this search even
  if their span overlaps it. The threshold comparison is strictly `r² > 0.5`.
- SV alleles: sequence-resolved insertions/deletions with length difference at
  least 50 bp, plus declared INS/DEL/DUP/INV/CNV/BND/TRA/CPX alleles. Known lengths
  below 50 bp are excluded except breakends/translocations. Declared structural
  alleles of unknown length are retained and flagged. Ordinary SNVs, small indels,
  spanning-deletion `*`, and `<NON_REF>` placeholders are not candidate SV alleles.
- Multiallelic records: each structural ALT is evaluated separately as 0/1/2 copies
  of that ALT. Other ALTs count as zero copies of the target allele. Thus
  `sv_target_alt_mac` is the minor count for target-ALT-versus-all-other-alleles,
  and `sv_n_zero_alt` does not necessarily mean homozygous VCF REF.
- Primary LD: squared Pearson correlation of ALT GT dosages, with missing values
  filled by each variant's mean **within that phenotype's analysis sample set**,
  matching Figure 2's existing `_r2_to_reference` preparation. It is not a
  covariate-adjusted association statistic or phased haplotype LD. The input GTs
  themselves may already be imputed. At least three pairwise-complete samples and
  nonzero variance are needed for a reported correlation.
- Sensitivity LD: `r2_pairwise_complete` uses only participants with complete GT
  calls at both variants. Partial GT calls count as missing. AF and carrier counts
  always use observed input GTs before the within-cohort mean filling.
- Direction: positive r means SV ALT is correlated with lead ALT; negative r means
  SV ALT is correlated with lead REF. The table labels this explicitly.
- Exhaustive threshold reporting: no additional MAC, carrier-count, missingness,
  hardmask, or VCF FILTER exclusion is applied to candidate SVs. Low MAC/carriers,
  high missingness, non-PASS filters, multiallelic sites, and unknown lengths are
  flagged. The flags do not remove qualifying links. This avoids interpreting a
  missing/filtered candidate as evidence of no linkage.

Checkpoints are reusable only when the lead summary row, cohort/phenotype mask,
source VCF and index fingerprints, sample order/header, scan bounds, and method
contract match, and the payload checksum is valid. Changing the reporting r²
threshold reuses completed scans. Repository commits are not cache keys. A failed
association stops combined report generation; consult `manifest.json` before using
older output tables. The computation is deterministic with four workers by default.

Relevant local tests:

```bash
uv run --link-mode copy --with-requirements requirements-figure2-loci.txt --with pytest \
  python -m pytest tests/test_figure2_sv_linkage.py tests/test_figure2_genotype_audit.py -q
```
