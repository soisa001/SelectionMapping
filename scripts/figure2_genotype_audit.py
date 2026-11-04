"""Audit existing Figure 2 representative genotypes without staging or refitting.

Run on Workbench after figure2_loci.run(panel_only=True). Only aggregate reports
are written. Source files and analysis caches are never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from . import admixture_inputs, admixture_phewas, figure2_1025
from .common import config

CONTRACT = 'figure5_representative_gt_quality_frequency_audit_v2'
LOG = logging.getLogger(__name__)


def fingerprint(path, *, sha=False):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    result = {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if sha:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result['sha256'] = digest.hexdigest()
    return result


def atomic_json(path, payload):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    json.loads(temp.read_text(encoding='utf-8'))
    temp.replace(path)


def atomic_table(path, frame):
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, sep='\t', index=False, na_rep='NA')
    checked = pd.read_csv(temp, sep='\t')
    if checked.shape != frame.shape or checked.columns.tolist() != frame.columns.tolist():
        raise ValueError(f'Table read-back failed: {path}')
    temp.replace(path)


def decode_gt(genotypes):
    """Treat partial calls as missing; refuse haploid or unsplit multiallelic GT."""
    if any(len(gt) != 3 for gt in genotypes):
        raise ValueError('Expected two allele indices and phase flag for every autosomal GT')
    alleles = np.asarray([gt[:2] for gt in genotypes], dtype=int)
    if not np.isin(alleles, [-1, 0, 1]).all():
        raise ValueError('Invalid allele index in a biallelic GT record')
    missing = (alleles < 0).any(axis=1)
    dosage = alleles.sum(axis=1).astype(float)
    dosage[missing] = np.nan
    return dosage, (alleles < 0).sum(axis=1) == 1


def genotype_stats(dosage):
    called = np.isfinite(dosage)
    n = len(dosage)
    n_called = int(called.sum())
    ac = int(dosage[called].sum())
    an = 2 * n_called
    af = ac / an if an else np.nan
    observed_het = int((dosage == 1).sum()) / n_called if n_called else np.nan
    expected_het = 2 * af * (1 - af)
    return {
        'n_samples': n, 'n_called': n_called, 'n_missing': n - n_called,
        'missing_rate': (n - n_called) / n if n else np.nan,
        'n_hom_ref': int((dosage == 0).sum()), 'n_het': int((dosage == 1).sum()),
        'n_hom_alt': int((dosage == 2).sum()), 'n_alt_carriers': int((dosage > 0).sum()),
        'alt_allele_count': ac, 'allele_number': an,
        'alt_af': ac / an if an else np.nan,
        'mac': min(ac, an - ac), 'maf': min(ac, an - ac) / an if an else np.nan,
        'observed_heterozygosity': observed_het,
        'pooled_hwe_expected_n_het': n_called * expected_het,
        'pooled_hwe_expected_n_hom_alt': n_called * af * af,
        'pooled_het_deficit_f': 1 - observed_het / expected_het if expected_het > 0 else np.nan,
    }


def format_values(record, field, n, width):
    """None means field absent; present-but-missing is retained as NaN."""
    if field not in record.FORMAT:
        return None
    value = record.format(field)
    if value is None:
        return np.full((n, width), np.nan)
    values = np.asarray(value, dtype=float).copy()
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (n, width):
        raise ValueError(f'Unexpected {field} shape: {values.shape}; expected {(n, width)}')
    values[(values < 0) | ~np.isfinite(values)] = np.nan
    return values


def quality_arrays(record, dosage):
    n = len(dosage)
    result = {}
    for field in ('GQ', 'DP', 'DS'):
        value = format_values(record, field, n, 1)
        if value is not None:
            result[field] = value[:, 0]
    if 'DS' in result:
        result['DS'][result['DS'] > 2] = np.nan
    gp = format_values(record, 'GP', n, 3)
    if gp is not None:
        valid = np.isfinite(gp).all(axis=1) & (gp <= 1).all(axis=1)
        valid &= np.isclose(gp.sum(axis=1), 1, atol=0.02, rtol=0)
        result['GP_max'] = np.where(valid, np.max(gp, axis=1), np.nan)
        posterior = np.full(n, np.nan)
        called = valid & np.isfinite(dosage)
        posterior[called] = gp[np.flatnonzero(called), dosage[called].astype(int)]
        result['GP_called'] = posterior
    ad = format_values(record, 'AD', n, 2)
    if ad is not None:
        valid = np.isfinite(ad).all(axis=1) & (ad.sum(axis=1) > 0) & (dosage == 1)
        balance = np.full(n, np.nan)
        balance[valid] = ad[valid, 1] / ad[valid].sum(axis=1)
        result['het_alt_balance'] = balance
    return result


def analysis_labels(sample_ids, retained_ids, sex, counts, phecode, scope, minimum_count):
    """Mirror phenotype_vector's case, ambiguous-control, and sex exclusions."""
    group = counts.loc[counts['phecode'] == phecode].groupby('person_id')['count'].sum()
    labels = np.zeros(len(sample_ids), dtype=float)
    labels[np.isin(sample_ids, group.index)] = np.nan
    labels[np.isin(sample_ids, group.index[group >= minimum_count])] = 1
    labels[~np.isin(sample_ids, list(retained_ids))] = np.nan
    if scope == 'Female':
        labels[sex != 0] = np.nan
    elif scope == 'Male':
        labels[sex != 1] = np.nan
    elif scope != 'Both':
        raise ValueError(f'Unknown sex scope: {scope}')
    return labels


def audit_record(record, expected, labels, thresholds):
    dosage, partial = decode_gt(record.genotypes)
    quality = quality_arrays(record, dosage)
    identity = {'variant': expected['representative_variant'], 'phecode': expected['phecode']}
    rows, sensitivity, diagnostics = [], [], []
    masks = {'analysis': np.isfinite(labels), 'cases': labels == 1, 'controls': labels == 0}
    for group, mask in masks.items():
        stats = {**identity, 'group': group, **genotype_stats(dosage[mask]),
                 'n_partial_gt': int(partial[mask].sum())}
        rows.append(stats)
        for field, threshold in thresholds.items():
            if field not in quality:
                sensitivity.append({**identity, 'group': group, 'field': field,
                                    'threshold': threshold, 'status': 'field_absent'})
                continue
            values = quality[field][mask]
            retained = np.isfinite(values) & (values >= threshold)
            masked = np.where(retained, dosage[mask], np.nan)
            sensitivity.append({**identity, 'group': group, 'field': field,
                                'threshold': threshold, 'status': 'evaluated',
                                'n_field_missing_or_invalid': int((~np.isfinite(values)).sum()),
                                'n_called_below_threshold': int((np.isfinite(dosage[mask]) &
                                                               np.isfinite(values) & (values < threshold)).sum()),
                                **genotype_stats(masked)})
        for field, values in quality.items():
            eligible = mask & (dosage == 1) if field == 'het_alt_balance' else mask
            valid = values[eligible][np.isfinite(values[eligible])]
            diagnostics.append({**identity, 'group': group, 'field': field,
                                'n_eligible': int(eligible.sum()),
                                'n_valid': len(valid), 'n_missing_or_invalid': int(eligible.sum()) - len(valid),
                                'mean': float(valid.mean()) if len(valid) else np.nan,
                                'p10': float(np.quantile(valid, 0.1)) if len(valid) else np.nan,
                                'median': float(np.median(valid)) if len(valid) else np.nan,
                                'p90': float(np.quantile(valid, 0.9)) if len(valid) else np.nan,
                                'dosage_af': float(valid.mean() / 2) if field == 'DS' and len(valid) else np.nan})
        if 'DS' in quality:
            paired = mask & np.isfinite(dosage) & np.isfinite(quality['DS'])
            delta = quality['DS'][paired] - dosage[paired]
            diagnostics.append({**identity, 'group': group, 'field': 'DS_minus_GT',
                                'n_valid': len(delta),
                                'mean': float(delta.mean()) if len(delta) else np.nan,
                                'rmse': float(np.sqrt(np.mean(delta ** 2))) if len(delta) else np.nan,
                                'n_abs_difference_gt_0_5': int((abs(delta) > .5).sum()),
                                'n_abs_difference_gt_1': int((abs(delta) > 1).sum())})
    all_stats, cases, controls = rows
    comparisons = {
        'n_samples': all_stats['n_samples'], 'n_cases': cases['n_samples'],
        'representative_mac': all_stats['mac'],
        'representative_n_carriers': all_stats['n_alt_carriers'],
        'representative_n_case_carriers': cases['n_alt_carriers'],
        'representative_n_control_carriers': controls['n_alt_carriers'],
        'representative_missing_rate': all_stats['missing_rate'],
    }
    mismatches = [key for key, value in comparisons.items()
                  if key in expected and not np.isclose(float(expected[key]), value, rtol=0, atol=1e-10)]
    differential = fisher_exact([[cases['n_missing'], cases['n_called']],
                                [controls['n_missing'], controls['n_called']]]).pvalue
    check = {**identity, 'matches_saved_summary': not mismatches,
             'mismatched_columns': ','.join(mismatches),
             'case_control_missingness_fisher_p': float(differential),
             'format_fields': ','.join(record.FORMAT),
             'filter': str(record.FILTER) if record.FILTER is not None else 'PASS_or_unfiltered',
             'info_json': json.dumps(dict(record.INFO), default=str)}
    return rows, sensitivity, diagnostics, check


def audit_variant(vcf_path, expected_rows, labels_by_row, thresholds):
    from cyvcf2 import VCF
    variant = expected_rows[0]['representative_variant']
    chrom, pos, ref, alt = variant.removeprefix('chr').split(':')
    reader = VCF(str(vcf_path))
    try:
        token = admixture_inputs.detect_vcf_chrom(reader, chrom)
        if token is None:
            raise ValueError(f'Chromosome missing from VCF: {chrom}')
        matches = [record for record in reader(f'{token}:{pos}-{pos}')
                   if record.POS == int(pos) and record.REF == ref and record.ALT == [alt]]
        if len(matches) != 1:
            raise ValueError(f'{variant}: expected one exact biallelic record, found {len(matches)}')
        record = matches[0]
        results = [audit_record(record, row, labels_by_row[(variant, row['phecode'])], thresholds)
                   for row in expected_rows]
        LOG.info('Audited %s (%d phenotype rows)', variant, len(expected_rows))
        return results
    finally:
        reader.close()


def load_cached_inputs(summary_path):
    """Read paths from the previous run, never invoke any staging helper."""
    run_path = summary_path.parent / 'run_manifest.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    params = run['parameters']
    root = figure2_1025.analysis_root()
    manifest_path = root / 'cohorts' / 'discovery_analysis_samples_1025.tsv'
    audit_path = root / 'cohorts' / 'discovery_cohort_audit_1025.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    ids = pd.read_csv(manifest_path, sep='\t', dtype=str)['person_id']
    if (ids.duplicated().any() or figure2_1025._sample_sha(ids) != audit['sample_sha256']
            or len(ids) != audit['retained_after_pca_and_complete_covariates']
            or audit['n_pcs'] != 20 or audit['internal_related_pairs'] != 0
            or params['discovery_cohort']['sample_sha256'] != audit['sample_sha256']):
        raise ValueError('Saved Figure 2 sample manifest and run provenance disagree')
    sources = audit['sources']
    paths = {key: Path(sources[key]).expanduser() for key in ('covariates', 'phenotypes')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required already-staged input is missing: {path}')
    work = Path(run['runtime']['work_directory']).expanduser()
    vcf = work / 'workspace_inputs' / 'discovery' / Path(params['discovery_vcf_source']).name
    return run, audit, set(ids), paths, vcf, [run_path, audit_path, manifest_path]


def cached_analysis_labels(table, sample_ids, retained, sources, minimum_count):
    """Build the same phenotype masks for genotype audits and regional linkage."""
    cov = pd.read_csv(sources['covariates'], sep='\t', dtype=str)
    cov = cov.rename(columns={admixture_phewas._id_column(cov.columns): 'person_id'})
    if cov['person_id'].duplicated().any():
        raise ValueError('Duplicate covariate sample IDs')
    sex = admixture_phewas._normalize_sex(cov['sex_at_birth'])
    sex.index = cov['person_id']
    if not pd.Index(sorted(retained)).isin(sex.dropna().index).all():
        raise ValueError('Saved retained cohort lacks valid sex in cached covariates')
    sex = sex.reindex(sample_ids).to_numpy(dtype=float)
    chunks = []
    for chunk in pd.read_csv(sources['phenotypes'], sep='\t', dtype={'person_id': str, 'phecode': str},
                             usecols=['person_id', 'phecode', 'count'], chunksize=250_000):
        chunks.append(chunk.loc[chunk['person_id'].isin(retained) & chunk['phecode'].isin(table['phecode'])])
    counts = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['person_id', 'phecode', 'count'])
    counts['count'] = pd.to_numeric(counts['count'], errors='coerce').fillna(0)
    return {(row['representative_variant'], row['phecode']): analysis_labels(
        sample_ids, retained, sex, counts, row['phecode'], row['sex_scope'], minimum_count)
        for row in table.to_dict('records')}


def load_representative_summary(summary, *, required=None, phecode=None):
    """Read rich summaries or recover minimum-p fields from compact run summaries."""
    from .figure2_locus_results import summary_fields

    summary = Path(summary)
    table = pd.read_csv(summary, sep='\t', dtype={'phecode': str})
    if phecode is not None:
        table = table.loc[table['phecode'].eq(str(phecode))]
    rows = []
    required = required or {'representative_variant', 'representative_selection', 'sex_scope',
                'representative_p_score', 'representative_pip', 'representative_mac',
                'representative_n_carriers', 'representative_n_case_carriers',
                'representative_n_control_carriers', 'representative_missing_rate'}
    for index, row in table.iterrows():
        if 'status' in row and row['status'] != 'ready':
            raise ValueError(f'Summary row {index} is not ready: {row.get("error", "")}')
        if not all(key in row and pd.notna(row[key]) for key in required):
            if pd.isna(row.get('fine_map_table')):
                raise ValueError(f'Summary row {index} lacks representative fields and fine_map_table')
            path = Path(row['fine_map_table'])
            if not path.is_absolute():
                path = summary.parent / path
            frame = pd.read_csv(path, sep='\t')
            if frame.empty or frame['variant_id'].duplicated().any():
                raise ValueError(f'Empty or duplicate-variant fine map: {path}')
            for key in ('n_samples', 'n_cases', 'sex_scope'):
                if key not in frame or frame[key].isna().any() or frame[key].nunique() != 1:
                    raise ValueError(f'Invalid {key} in {path}')
                value = frame[key].iloc[0]
                if key in row and pd.notna(row[key]) and row[key] != value:
                    raise ValueError(f'Summary/fine-map {key} mismatch: {path}')
                row[key] = value
            for key, value in summary_fields(frame).items():
                if key in row and pd.notna(row[key]) and row[key] != value:
                    raise ValueError(f'Summary/fine-map {key} mismatch: {path}')
                row[key] = value
            row['representative_source_sha256'] = fingerprint(path, sha=True)['sha256']
        rows.append(row)
    return pd.DataFrame(rows)


def reference_comparison(counts, reference_path):
    """Compare the exact forward-strand ALT; never silently flip A/T or C/G SNPs."""
    reference = pd.read_csv(reference_path, sep='\t', dtype={'variant': str})
    required = {'variant', 'genome_build', 'allele', 'population', 'alt_allele_count',
                'allele_number', 'source_url'}
    if not required.issubset(reference) or reference.empty or reference[list(required)].isna().any().any():
        raise ValueError('Incomplete allele-specific reference frequency table')
    if reference.duplicated(['variant', 'population']).any():
        raise ValueError('Duplicate reference variant/population')
    for row in reference.to_dict('records'):
        if row['genome_build'] != 'GRCh38' or row['variant'].split(':')[-1] != row['allele']:
            raise ValueError('Reference requires exact GRCh38 forward-strand ALT identity')
        ac, an = float(row['alt_allele_count']), float(row['allele_number'])
        if not (np.isfinite([ac, an]).all() and ac.is_integer() and an.is_integer() and 0 <= ac <= an and an > 0):
            raise ValueError('Invalid reference allele counts')
    if not set(counts['variant']).issubset(reference['variant']):
        raise ValueError('Reference lacks an exact selected variant; no position-only or strand-swapped matching')
    reference['reference_alt_af'] = reference['alt_allele_count'] / reference['allele_number']
    reference = reference.rename(columns={'alt_allele_count': 'reference_alt_allele_count',
                                           'allele_number': 'reference_allele_number'})
    result = counts.merge(reference, on='variant', how='left', validate='many_to_many')
    result['alt_af_minus_reference'] = result['alt_af'] - result['reference_alt_af']
    return result


def run_audit(*, summary, output, vcf=None, threads=4, min_gp=0.9, min_gq=20, min_dp=10,
              phecode=None, variant=None, reference_frequencies=None):
    from cyvcf2 import VCF
    if not 1 <= threads <= 32 or not 0 <= min_gp <= 1 or min_gq < 0 or min_dp < 0:
        raise ValueError('Invalid thread count or quality threshold')
    summary = Path(summary).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run, cohort, retained, sources, default_vcf, input_paths = load_cached_inputs(summary)
    vcf = Path(vcf or default_vcf).expanduser().resolve(strict=True)
    indexes = [Path(str(vcf) + ext) for ext in ('.tbi', '.csi')]
    index = next((path for path in indexes if path.is_file()), None)
    if index is None:
        raise FileNotFoundError(f'Indexed VCF required: {vcf}')
    required = {'phecode', 'representative_variant', 'sex_scope', 'n_samples', 'n_cases',
                'representative_mac', 'representative_n_carriers',
                'representative_n_case_carriers', 'representative_n_control_carriers',
                'representative_missing_rate'}
    table = load_representative_summary(summary, required=required, phecode=phecode)
    if table.empty:
        raise ValueError('No representative matches the requested phecode')
    if variant is not None:
        table = table.loc[table['representative_variant'].eq(variant)]
    if not required.issubset(table) or table.empty or table[list(required)].isna().any().any():
        raise ValueError('Incomplete representative-variant summary; rerun the prior loci job first')
    if table.duplicated(['representative_variant', 'phecode']).any():
        raise ValueError('Duplicate variant/phenotype rows in summary')
    header = VCF(str(vcf))
    try:
        sample_ids = np.asarray(header.samples, dtype=str)
        metadata = '\n'.join(line for line in header.raw_header.splitlines()
                             if line.startswith('##') and not line.startswith(('##SAMPLE=', '##PEDIGREE=')))
    finally:
        header.close()
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate VCF sample IDs')
    minimum_count = int(run['parameters']['min_phecode_count'])
    expected_rows = table.to_dict('records')
    labels = cached_analysis_labels(table, sample_ids, retained, sources, minimum_count)
    thresholds = {'GP_called': min_gp, 'GQ': min_gq, 'DP': min_dp}
    manifest = {
        'contract': CONTRACT, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running', 'threads': threads, 'quality_thresholds': thresholds,
        'minimum_phecode_count': minimum_count,
        'threshold_policy': 'Independent sensitivity masks; original GT report is unfiltered',
        'source_vcf': fingerprint(vcf), 'source_index': fingerprint(index),
        'configured_source_uri_from_previous_run': run['parameters']['discovery_vcf_source'],
        'vcf_override': str(vcf) != str(default_vcf.resolve()),
        'sample_manifest_sha256': cohort['sample_sha256'],
        'vcf_sample_order_sha256': hashlib.sha256('\n'.join(sample_ids).encode()).hexdigest(),
        'n_manifest_samples_absent_from_vcf': len(retained.difference(sample_ids)),
        'selection': {'phecode': phecode, 'variant': variant},
        'reference_frequencies': fingerprint(reference_frequencies, sha=True) if reference_frequencies else None,
        'pooled_heterozygosity_note': 'Descriptive pooled HWE expectation only; admixture/structure violates the homogeneous random-mating null',
        'source_files': [fingerprint(path, sha=True) for path in [summary, *input_paths]],
        'cached_covariates': fingerprint(sources['covariates']),
        'cached_phenotypes': fingerprint(sources['phenotypes']),
        'header_metadata_sha256': hashlib.sha256(metadata.encode()).hexdigest(),
        'audit_code': fingerprint(Path(__file__), sha=True),
        'rng': 'No random operations', 'pre_imputation_missingness': 'Not evaluated',
    }
    atomic_json(output / 'manifest.json', manifest)
    (output / 'vcf_metadata.txt').write_text(metadata + '\n', encoding='utf-8')
    LOG.info('Inputs validated; %d phenotype rows, %d variants; threads=%d; thresholds=%s',
             len(table), table['representative_variant'].nunique(), threads, thresholds)
    groups = [[row for row in expected_rows if row['representative_variant'] == variant]
              for variant in table['representative_variant'].drop_duplicates()]
    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(lambda group: audit_variant(vcf, group, labels, thresholds), groups))
        flat = [row for group in results for row in group]
        names = ['genotype_counts.tsv', 'quality_sensitivity.tsv', 'quality_fields.tsv', 'summary_comparison.tsv']
        outputs = []
        for i, name in enumerate(names):
            items = [result[i] for result in flat]
            frame = pd.DataFrame(items if i == 3 else [row for rows in items for row in rows])
            if frame.empty and i == 2:
                frame = pd.DataFrame(columns=['variant', 'phecode', 'group', 'field', 'n_valid'])
            path = output / name
            atomic_table(path, frame)
            outputs.append(fingerprint(path, sha=True))
        if reference_frequencies:
            comparison = reference_comparison(pd.DataFrame([row for result in flat for row in result[0]]),
                                              reference_frequencies)
            path = output / 'reference_frequency_comparison.tsv'
            atomic_table(path, comparison)
            outputs.append(fingerprint(path, sha=True))
        checks = [result[3] for result in flat]
        n_mismatches = sum(not row['matches_saved_summary'] for row in checks)
        manifest.update(status='summary_mismatch' if n_mismatches else 'complete',
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        n_summary_mismatches=n_mismatches, outputs=outputs)
        atomic_json(output / 'manifest.json', manifest)
        LOG.info('Audit reports saved to %s; summary mismatches=%d', output, n_mismatches)
        if n_mismatches:
            raise ValueError('GT audit differs from saved summary; inspect summary_comparison.tsv')
    except Exception as exc:
        if manifest['status'] == 'running':
            manifest.update(status='failed', error=str(exc))
            atomic_json(output / 'manifest.json', manifest)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=config.figure_dir('figure2_sources') / 'loci_panel' / 'fine_mapping_summary.tsv')
    parser.add_argument('--output', type=Path, default=figure2_1025.analysis_root() / 'genotype_audit_panel')
    parser.add_argument('--vcf', type=Path, help='Optional already-local indexed VCF; no downloads')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--min-gp', type=float, default=0.9)
    parser.add_argument('--min-gq', type=float, default=20)
    parser.add_argument('--min-dp', type=float, default=10)
    parser.add_argument('--phecode', help='Audit only this phenotype from the saved summary')
    parser.add_argument('--variant', help='Require this exact chr:pos:REF:ALT representative')
    parser.add_argument('--reference-frequencies', type=Path, help='Local allele-specific GRCh38 reference TSV')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / 'audit.log', encoding='utf-8')])
    LOG.info('Arguments: %s', vars(args))
    run_audit(**vars(args))


if __name__ == '__main__':
    main()
