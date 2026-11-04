> Workbench workflow reference. Configure the required inputs in your authorized workspace before running these commands; consult the root README for the repository scope.

# Figure 2 panel SNVs plus SVs

Use `run(panel_only=True, include_svs=True)` to create **loci_panel_sv** for
the same discovery-nominal associations selected by `loci_panel`.

The joint genotype matrix contains exact chromosome/POS/REF/ALT-matched
biallelic SNVs from the masked FLARE panel, plus discovery-VCF INS/DEL SVs
classified by the existing Figure 2 rules (at least 50 bp). Off-panel SNVs
and small indels are excluded. Both variant types use the same outlier window
plus 100 kb flanks, hardmask, discovery cohort, 20 PCs, phenotype/sex scope,
and observed-call missingness checks.

SVs require **observed MAC > 20**, computed after cohort/phenotype/sex
restriction and before imputation. Restored SVs have no extra ALT-carrier
cutoff; for example, MAC 21 in 11 carriers qualifies. Panel SNVs retain their
configured MAC and carrier thresholds (defaults: MAC > 20 and at least 20
ALT carriers). The saved parameters explicitly record the SV-specific rule.

SuSiE, logistic score tests, LD, PIPs and credible sets are fitted jointly
from this matrix. QC-passing SVs appear in the fine-map plots and the EUR-like
ancestry panel using the shared loci interpolation: linear interpolation
between observed LAI marker fractions, or the nearest boundary fraction
outside their range. This does not refit the upstream PLSR or regional PheWAS.

Outputs are separate from `loci` and `loci_panel`:

- `figures/figure2_sources/loci_panel_sv/`: QC summary and PNG/vector-PDF locus plots.
- `figures/figure2_sources/loci_panel_sv/figures/`: stacked figures, association results,
  all credible-set variants, and `sv_ancestry.tsv` with SV identities, MAC,
  interpolated fractions, flanking marker positions and assignment methods.
- `analysis/figure2/finemap_panel_sv/`: fitted tables and panel catalogs.
- `figures/figure2_sources/loci_panel_sv/sv_linkage/linked_svs.csv` (also `.tsv`):
  every SV/lead/phenotype pair with **r² > 0.5** within the **outlier start/end
  ±100 kb**, including `sv_locus` (`chr:pos`), exact `sv_ref`, `sv_alt`,
  `sv_alt_af`, `sv_length_bp`, SV type, lead identity/p-value/PIP, r/r²,
  sample counts and QC flags. `all_sv_pairs.tsv` retains all scanned candidates;
  `locus_summary.tsv` includes associations with zero qualifying links.

The linked-SV scan runs automatically after fitting/saving `loci_panel_sv`.
Its reference is the minimum score-test p-value variant used in the combined
fine-map plot (which can itself be an SV). The scan window is expanded from
the outlier boundaries, not from the lead position. SV POS must fall within
the inclusive bounds, even for spanning deletions. All structural ALT alleles
present in the local discovery VCF are scanned, including symbolic and
multiallelic alleles; candidates need not belong to the panel, fine map or
credible set. MAC/carrier/missingness/FILTER issues are flagged without
excluding a pair that passes r² > 0.5. Unknown SV lengths are reported as `NA`.

ALT AF uses observed input calls in the same phenotype/sex-restricted
discovery cohort. LD is squared Pearson correlation of ALT GT dosages after
within-cohort mean filling of missing calls; pairwise-complete r² is also
reported. Each phenotype has its own row and sample mask. This scan does not
change the fine-map variant filters above. It is deterministic and uses four
workers by default, with separate checkpoints, output checksums and a log.
Use `scan_linked_svs=False` only when intentionally skipping this final phase.

Fit contracts and stage names are separate. Existing sample manifests,
regional panel identity digests, parameters, and output integrity determine
whether to resume; a Git revision alone does not invalidate a fit. Unchanged
upstream inputs are reused, and interrupted/corrupt output can be repaired by
rerunning without `force=True`. Run the analysis on the same Workbench VM as
the existing Figure 2 results.

```bash
%%bash
set -euo pipefail
export MPLBACKEND=Agg UV_LINK_MODE=copy
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export AOU_FIGURE_SEED=20260715
export AOU_FIG4_FINEMAP_MIN_MAC=20
export AOU_FIG4_FINEMAP_MIN_CARRIERS=20
export AOU_FIG4_FINEMAP_MAX_MISSING=0.05
export AOU_LAI_DIR="${AOU_LAI_DIR:-$HOME/lai_masked}"

# Setup: update code, keeping existing inputs and analysis caches.
if [ -d "$HOME/SelectionMapping/.git" ]; then
  git -C "$HOME/SelectionMapping" pull --ff-only
else
  git clone --branch main git@github.com:soisa001/SelectionMapping.git "$HOME/SelectionMapping"
fi
cd "$HOME/SelectionMapping"
mkdir -p analysis/figure_notebook_runs

# Analysis: validate inputs, resume joint fits, then save and verify figures.
uv run --with-requirements requirements-figure2-loci.txt python -u - <<'PY' 2>&1 | tee "analysis/figure_notebook_runs/loci_panel_sv_$(date -u +%Y%m%dT%H%M%SZ).log"
from scripts.figure2_loci import run

result = run(panel_only=True, include_svs=True, force=False)
figures = result["two_panel_figures"]
print(f"Locus plots: {result['directory']}")
print(f"Stacked figures: {figures['directory']}")
print(f"QC summary: {result['summary']}")
print(f"Association results: {figures['results_table']}")
print(f"All credible-set variants: {figures['credible_set_table']}")
print(f"Panel provenance: {result['panel_manifest']}")
print(f"Linked SVs (r2 > 0.5): {result['linked_sv_table']}")
PY
```

The cell saves plots to disk on the Workbench VM. Set `AOU_LAI_DIR` if the existing masked
FLARE inputs live elsewhere. A successful first run creates the new results;
later calls resume verified fits and figures.

If the fits are already complete, run only the additional scan after pulling
the updated repository:

```bash
%%bash
set -euo pipefail
cd "$HOME/SelectionMapping"
uv run --with-requirements requirements-figure2-loci.txt python -u -m scripts.figure2_sv_linkage \
  --summary figures/figure2_sources/loci_panel_sv/fine_mapping_summary.tsv \
  --output figures/figure2_sources/loci_panel_sv/sv_linkage \
  --window-basis outlier --flank-bp 100000 --r2-threshold 0.5 --threads 4
```
