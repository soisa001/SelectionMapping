> Workbench workflow reference. Configure the required inputs in your authorized workspace before running these commands; consult the root README for the repository scope.

# Acute kidney failure lead genotype QC

The supplied `loci_panel_sv` summary identifies the minimum-p lead for
`GU_582.1` as **GRCh38 chr6:32479741:T:A (rs35007143)**. The same allele and
forward-strand coordinate are confirmed by the
[Ensembl population endpoint](https://rest.ensembl.org/variation/human/rs35007143?pops=1&content-type=application/json).
The public reference snapshot in `data/figure2_rs35007143_1kg_reference.tsv`
was checked against this API on 2026-09-11; only public reference data are bundled.

The user-provided aggregate summary implies:

| Quantity | Value |
|---|---:|
| Analysis samples / cases / controls | 8,379 / 719 / 7,660 |
| Called samples | 8,379 |
| A allele count / allele number | 2,180 / 16,758 |
| A allele frequency | 0.1300871 (13.0087%) |
| T/T / T/A / A/A | 6,779 / 1,020 / 580 |
| A-carrier frequency | 1,600 / 8,379 = 19.0954% |
| 1000 Genomes Phase 3 AFR A frequency | 318 / 1,322 = 24.0545% |
| 1000 Genomes Phase 3 EUR A frequency | 452 / 1,006 = 44.9304% |

MAC alone does not identify the ALT frequency. Here 1,600 ALT carriers permit
at most 3,200 ALT copies, so the alternative major-ALT count of 14,578 is
impossible. Thus A count = MAC = 2,180; A/A = 2,180 - 1,600 = 580 and
T/A = 2(1,600) - 2,180 = 1,020. Case/control allele frequencies cannot be
recovered from carrier counts alone; the command below computes them from GT.

At the observed AF, a homogeneous Hardy-Weinberg calculation predicts
1,896.41 heterozygotes and 141.79 A/A genotypes. The descriptive heterozygote
deficit statistic is 0.46214. Population structure/admixture, phenotype
selection, and technical issues can affect these numbers: this pooled
calculation is not a calibrated HWE significance test for this cohort and
does not establish a calling error.

The ancestry track is EUR-like local ancestry among the 1,025 LAI samples,
not A-allele frequency in these 8,379 discovery samples. A two-source mixture
using exactly the cited AFR/EUR frequencies would fall between 24.05% and
44.93%; the observed AF is below both. A simple allele flip gives 86.99%,
which also does not reconcile the estimates. This A/T SNP is strand-palindromic,
so compare exact build, position and forward-strand alleles, not rsID/MAF alone.

Run this on the existing Workbench VM. It selects only this phenotype and
requires the exact expected lead, reuses local input paths and sample manifests,
and writes aggregate tables on the Workbench VM using the existing fitted results.

```bash
%%bash
set -euo pipefail
export MPLBACKEND=Agg UV_LINK_MODE=copy
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ -d "$HOME/SelectionMapping/.git" ]; then
  git -C "$HOME/SelectionMapping" pull --ff-only
else
  git clone --branch main git@github.com:soisa001/SelectionMapping.git "$HOME/SelectionMapping"
fi
cd "$HOME/SelectionMapping"
mkdir -p analysis/figure_notebook_runs
uv run --with-requirements requirements-figure2-loci.txt python -u -m scripts.figure2_genotype_audit \
  --summary figures/figure2_sources/loci_panel_sv/fine_mapping_summary.tsv \
  --phecode GU_582.1 --variant chr6:32479741:T:A \
  --reference-frequencies data/figure2_rs35007143_1kg_reference.tsv \
  --output analysis/figure2/qc_acute_kidney_lead \
  --threads 4 --min-gp 0.9 --min-gq 20 --min-dp 10 \
  2>&1 | tee "analysis/figure_notebook_runs/qc_acute_kidney_$(date -u +%Y%m%dT%H%M%SZ).log"
```

Interpret the output in this order:

1. `summary_comparison.tsv`: the exact local GT record must reproduce sample,
   case, MAC, carrier and missingness counts. A mismatch fails explicitly.
2. `genotype_counts.tsv`: observed GT counts, ALT AF, MAF, called-allele
   denominators and descriptive heterozygosity for analysis/cases/controls.
3. `reference_frequency_comparison.tsv`: those groups compared against
   allele-matched Phase 3 AFR/EUR counts. No strand or REF/ALT swap is inferred.
4. `quality_sensitivity.tsv`: separate GP-called >=0.9, GQ >=20 and DP >=10
   sensitivity masks. These do not alter the original report or refit anything.
   Absent fields are labeled `field_absent`, not treated as high confidence.
5. `quality_fields.tsv`: DS-derived AF, GP summaries, heterozygote allele
   balance when AD exists, and `DS_minus_GT` mean/RMSE/disagreement counts.
   Inspect whether masking uncertain calls changes AF and the homozygote excess.
6. `vcf_metadata.txt`, `manifest.json`, `audit.log`: input pipeline metadata,
   source paths/fingerprints, parameters, selection and completion status.

The discovery source GTs may already be imputed. Zero missing GTs therefore
does not establish good original coverage or confident calls. If this audit
reproduces the discrepancy, compare against a separately validated local
pre-imputation/direct-call source and examine ancestry-stratified genotype
counts before attributing the signal to biology. Those inputs are not accessed
by this command. The supplied fit has no credible set and a maximum PIP of
0.1158; the lead is a minimum-p representative, not an established causal variant.

To query this allele separately in the **local FLARE ancestry-panel VCF**:

```bash
%%bash
set -euo pipefail
cd "$HOME/SelectionMapping"
uv run --with-requirements requirements-figure2-loci.txt python -u -m scripts.figure2_panel_allele_qc \
  --variant chr6:32479741:T:A --threads 4
```

Run after pulling the updated repository. This uses the source directory and
sample digest recorded in the panel-marker manifest, reads only the cached
canonical 1,025-person sample list, and queries exact T>A alleles at that POS.
It prints and saves `genotype_counts.tsv` and `manifest.json` under
`analysis/figure2/qc_acute_kidney_lai_panel/`. INFO annotations and their
header definitions are retained separately from GT-derived allele frequency.
An ancestry-only VCF lacking GT is reported as `GT_absent_AF_not_computable`;
AN1/AN2 are never treated as allele dosages. This command never stages inputs.
