"""All regional SVs linked to Figure 2 minimum-p representatives, from local inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import admixture_inputs, figure2_genotype_audit as audit

CONTRACT = 'figure5_all_sv_minp_ld_within_cohort_mean_imputation_windows_v2'
LOG = logging.getLogger(__name__)
SV_TYPES = {'INS', 'DEL', 'DUP', 'INV', 'CNV', 'BND', 'TRA', 'CPX'}
PAIR_COLUMNS = [
    'region_id', 'phecode', 'phecode_string', 'lead_variant', 'lead_p', 'lead_pip',
    'genes_in_finemap_window', 'window_start', 'window_end', 'window_basis',
    'outlier_chrom', 'outlier_start', 'outlier_end', 'n_samples', 'n_cases',
    'sv_id', 'sv_vcf_id', 'sv_chrom', 'sv_pos', 'sv_type', 'sv_length_bp',
    'sv_locus', 'sv_ref', 'sv_alt', 'sv_alt_index', 'sv_n_alts', 'sv_alleles_sha256',
    'distance_bp', 'r', 'r2', 'r2_pairwise_complete', 'n_pairwise_complete',
    'ld_status', 'ld_method', 'sv_alt_coupled_with_lead_allele',
    'lead_alt_af', 'lead_missing_rate', 'sv_alt_af', 'sv_target_alt_mac',
    'sv_allele_count', 'sv_allele_number', 'sv_n_called', 'sv_missing_rate',
    'sv_n_zero_alt', 'sv_n_one_alt', 'sv_n_two_alt', 'sv_n_alt_carriers',
    'sv_case_carriers', 'sv_control_carriers', 'sv_filter', 'qc_flags',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def info_allele(record, name, alt_index):
    value = record.INFO.get(name)
    if isinstance(value, (list, tuple)):
        if len(value) == len(record.ALT):
            return value[alt_index - 1]
        return value[0] if len(value) == 1 else None
    return value


def describe_sv(record, alt_index):
    """Support sequence and symbolic SV alleles, including multiallelic records."""
    alt = str(record.ALT[alt_index - 1])
    if alt in {'.', '*', '<NON_REF>', '<*>'}:
        return None
    kind = str(info_allele(record, 'SVTYPE', alt_index) or '').upper().split(':')[0]
    if '[' in alt or ']' in alt:
        kind = 'BND'
    elif alt.startswith('<') and alt.endswith('>'):
        symbol = alt[1:-1].upper().split(':')[0]
        if symbol in SV_TYPES:
            kind = symbol
        elif symbol.startswith('CN') and symbol[2:].isdigit():
            kind = 'CNV'
    length = info_allele(record, 'SVLEN', alt_index)
    try:
        number = abs(float(length))
        length = int(number) if np.isfinite(number) and number.is_integer() and number > 0 else None
    except (TypeError, ValueError):
        length = None
    sequence = not (alt.startswith('<') or '[' in alt or ']' in alt)
    difference = len(alt) - len(record.REF) if sequence else 0
    if length is None and sequence and difference:
        length = abs(difference)
    if length is None and kind in {'DEL', 'DUP', 'INV', 'CNV'}:
        try:
            length = max(0, int(record.INFO.get('END')) - record.POS) or None
        except (TypeError, ValueError):
            pass
    if kind not in SV_TYPES:
        if not sequence or abs(difference) < 50:
            return None
        kind = 'INS' if difference > 0 else 'DEL'
    if length is not None and length < 50 and kind not in {'BND', 'TRA'}:
        return None
    return kind, length


def alt_dosage(genotypes, alt_index, n_alts):
    if any(len(gt) != 3 for gt in genotypes):
        raise ValueError('Expected diploid autosomal GT for SV linkage')
    alleles = np.asarray([gt[:2] for gt in genotypes], dtype=int)
    if ((alleles < -1) | (alleles > n_alts)).any():
        raise ValueError('Invalid VCF allele index')
    dosage = (alleles == alt_index).sum(axis=1).astype(float)
    dosage[(alleles < 0).any(axis=1)] = np.nan
    return dosage


def pearson_r(x, y):
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else None


def dosage_linkage(lead, sv):
    """Primary LD matches Figure 2; also report pairwise-complete sensitivity."""
    paired = np.isfinite(lead) & np.isfinite(sv)
    result = {'r': None, 'r2': None, 'r2_pairwise_complete': None,
              'n_pairwise_complete': int(paired.sum()), 'ld_status': 'insufficient_complete_pairs'}
    if paired.sum() < 3:
        return result
    lead_fill = np.where(np.isfinite(lead), lead, np.nanmean(lead))
    sv_fill = np.where(np.isfinite(sv), sv, np.nanmean(sv))
    r = pearson_r(lead_fill, sv_fill)
    complete_r = pearson_r(lead[paired], sv[paired])
    result.update(r=r, r2=r * r if r is not None else None,
                  r2_pairwise_complete=complete_r * complete_r if complete_r is not None else None,
                  ld_status='defined' if r is not None else 'monomorphic')
    return result


def linked_pairs(rows, threshold):
    return [row for row in rows if row['r2'] is not None and row['r2'] > threshold]


def validate_lead(record, expected, labels):
    raw, _ = audit.decode_gt(record.genotypes)
    stats = audit.genotype_stats(raw[np.isfinite(labels)])
    values = {'n_samples': stats['n_samples'], 'n_cases': int((labels == 1).sum()),
              'representative_mac': stats['mac'],
              'representative_n_carriers': stats['n_alt_carriers'],
              'representative_n_case_carriers': int((raw[labels == 1] > 0).sum()),
              'representative_n_control_carriers': int((raw[labels == 0] > 0).sum()),
              'representative_missing_rate': stats['missing_rate']}
    mismatches = [key for key, value in values.items()
                  if not np.isclose(float(expected[key]), value, atol=1e-10, rtol=0)]
    if mismatches:
        raise ValueError(f"Lead differs from saved summary ({expected['phecode']}): {mismatches}")
    return raw[np.isfinite(labels)], stats


def scan_window(expected, flank_bp, window_basis):
    """Use VCF POS anchors and inclusive boundaries; never infer outlier bounds."""
    chrom, position, _, _ = expected['representative_variant'].removeprefix('chr').split(':')
    if window_basis == 'lead':
        return max(1, int(position) - flank_bp), int(position) + flank_bp
    if window_basis != 'outlier':
        raise ValueError(f'Unknown linkage window basis: {window_basis}')
    if str(expected.get('chrom', '')).removeprefix('chr') != chrom:
        raise ValueError('Outlier chromosome and lead chromosome disagree')
    try:
        start, end = (float(expected[key]) for key in ('start', 'end'))
        if not (np.isfinite([start, end]).all() and start.is_integer() and end.is_integer()
                and 1 <= start <= end):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('Valid outlier start/end coordinates required') from None
    bounds = max(1, int(start) - flank_bp), int(end) + flank_bp
    if not bounds[0] <= int(position) <= bounds[1]:
        raise ValueError('Lead lies outside the outlier linkage window')
    return bounds


def scan_association(vcf_path, expected, labels, sample_ids, flank_bp, window_basis='lead'):
    from cyvcf2 import VCF
    lead_id = expected['representative_variant']
    chrom, position, ref, alt = lead_id.removeprefix('chr').split(':')
    pos = int(position)
    start, end = scan_window(expected, flank_bp, window_basis)
    reader = VCF(str(vcf_path))
    rows = []
    seen = set()
    try:
        if list(reader.samples) != list(sample_ids):
            raise ValueError('VCF sample order changed during linkage scan')
        token = admixture_inputs.detect_vcf_chrom(reader, chrom)
        if token is None:
            raise ValueError(f'Chromosome absent from VCF: {chrom}')
        records = [r for r in reader(f'{token}:{pos}-{pos}')
                   if r.POS == pos and r.REF == ref and r.ALT == [alt]]
        if len(records) != 1:
            raise ValueError(f'{lead_id}: expected one exact biallelic lead, found {len(records)}')
        lead, lead_stats = validate_lead(records[0], expected, labels)
        selected = np.isfinite(labels)
        y = labels[selected]
        for record in reader(f'{token}:{start}-{end}'):
            # Indexed queries can return spanning records outside the POS window.
            if not start <= record.POS <= end:
                continue
            for allele_index in range(1, len(record.ALT or []) + 1):
                description = describe_sv(record, allele_index)
                if description is None:
                    continue
                kind, length = description
                target_alt = record.ALT[allele_index - 1]
                allele_hash = hashlib.sha256(f'{record.REF}\n{target_alt}'.encode()).hexdigest()
                identity = (record.POS, allele_hash)
                if identity in seen:
                    raise ValueError(f'Duplicate exact SV allele at chr{chrom}:{record.POS}; cannot merge records')
                seen.add(identity)
                dosage = alt_dosage(record.genotypes, allele_index, len(record.ALT))[selected]
                stats = audit.genotype_stats(dosage)
                ld = dosage_linkage(lead, dosage)
                flags = []
                if stats['missing_rate'] > .05:
                    flags.append('sv_missingness_gt_0.05')
                if stats['mac'] <= 20:
                    flags.append('sv_target_alt_mac_le_20')
                if stats['n_alt_carriers'] < 20:
                    flags.append('sv_alt_carriers_lt_20')
                if record.FILTER not in (None, '.', 'PASS'):
                    flags.append('vcf_filter_nonpass')
                if len(record.ALT) > 1:
                    flags.append('multiallelic_record')
                if length is None:
                    flags.append('sv_length_unavailable')
                row = {
                    'region_id': str(expected['region_id']), 'phecode': str(expected['phecode']),
                    'phecode_string': str(expected.get('phecode_string', '')),
                    'lead_variant': lead_id, 'lead_p': float(expected['representative_p_score']),
                    'lead_pip': float(expected['representative_pip']),
                    'genes_in_finemap_window': str(expected.get('genes_in_finemap_window') or ''),
                    'window_start': start, 'window_end': end, 'window_basis': window_basis,
                    'outlier_chrom': expected.get('chrom'),
                    'outlier_start': expected.get('start'), 'outlier_end': expected.get('end'),
                    'n_samples': len(lead), 'n_cases': int(y.sum()),
                    'sv_id': f'chr{chrom}:{record.POS}:{kind}:{length or "NA"}:{allele_hash[:16]}',
                    'sv_vcf_id': str(record.ID or '.'), 'sv_chrom': f'chr{chrom}',
                    'sv_pos': int(record.POS), 'sv_type': kind, 'sv_length_bp': length,
                    'sv_locus': f'chr{chrom}:{record.POS}',
                    'sv_ref': record.REF, 'sv_alt': target_alt,
                    'sv_alt_index': allele_index, 'sv_n_alts': len(record.ALT),
                    'sv_alleles_sha256': allele_hash, 'distance_bp': abs(record.POS - pos),
                    **ld, 'ld_method': 'ALT_GT_Pearson_r2_mean_imputed_within_analysis_cohort',
                    'sv_alt_coupled_with_lead_allele': ('ALT' if ld['r'] > 0 else 'REF')
                    if ld['r'] is not None and ld['r'] != 0 else 'UNRESOLVED',
                    'lead_alt_af': lead_stats['alt_af'], 'lead_missing_rate': lead_stats['missing_rate'],
                    'sv_alt_af': stats['alt_af'] if stats['allele_number'] else None,
                    'sv_target_alt_mac': stats['mac'], 'sv_allele_count': stats['alt_allele_count'],
                    'sv_allele_number': stats['allele_number'], 'sv_n_called': stats['n_called'],
                    'sv_missing_rate': stats['missing_rate'],
                    'sv_n_zero_alt': stats['n_hom_ref'], 'sv_n_one_alt': stats['n_het'],
                    'sv_n_two_alt': stats['n_hom_alt'], 'sv_n_alt_carriers': stats['n_alt_carriers'],
                    'sv_case_carriers': int((dosage[y == 1] > 0).sum()),
                    'sv_control_carriers': int((dosage[y == 0] > 0).sum()),
                    'sv_filter': str(record.FILTER) if record.FILTER is not None else 'PASS_or_unfiltered',
                    'qc_flags': ','.join(flags),
                }
                rows.append(row)
    finally:
        reader.close()
    return rows


def cached_scan(cache, contract, scan):
    """Cache only complete per-association scans; revisions do not invalidate LD."""
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['contract'] == contract and saved['rows_sha256'] == digest(saved['rows']):
                return saved['rows'], True
        except (ValueError, KeyError, TypeError):
            LOG.warning('Invalid linkage checkpoint; recomputing %s', cache.name)
    rows = scan()
    audit.atomic_json(cache, {'contract': contract, 'rows': rows, 'rows_sha256': digest(rows)})
    return rows, False


def write_report(path, frame):
    """Verify every saved cell, including long REF/ALT sequences."""
    separator = ',' if path.suffix == '.csv' else '\t'
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, sep=separator, index=False, na_rep='NA')
    checked = pd.read_csv(temporary, sep=separator, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(checked, frame.fillna('NA').astype(str).reset_index(drop=True))
    temporary.replace(path)


def run_locus_linkage(summary, output, *, threads=4):
    """Final logged phase of loci_panel_sv; read only already-staged inputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(output / 'linkage.log', encoding='utf-8')]
    if not LOG.hasHandlers():
        handlers.append(logging.StreamHandler())
    previous_level = LOG.level
    LOG.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOG.addHandler(handler)
    try:
        LOG.info('Phase=linked_SVs; summary=%s; output=%s; threads=%d; window=outlier +/-100000; r2>0.5',
                 summary, output, threads)
        return run_linkage(summary=summary, output=output, threads=threads,
                           flank_bp=100_000, r2_threshold=.5, window_basis='outlier')
    finally:
        for handler in handlers:
            LOG.removeHandler(handler)
            handler.close()
        LOG.setLevel(previous_level)


def run_linkage(*, summary, output, vcf=None, threads=4, flank_bp=100_000, r2_threshold=.5,
                window_basis='lead'):
    from cyvcf2 import VCF
    if (not 1 <= threads <= 32 or flank_bp < 0 or not 0 <= r2_threshold <= 1
            or window_basis not in {'lead', 'outlier'}):
        raise ValueError('Invalid linkage parameters')
    summary = Path(summary).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    cache_dir = output / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    run, cohort, retained, sources, default_vcf, inputs = audit.load_cached_inputs(summary)
    vcf = Path(vcf or default_vcf).expanduser().resolve(strict=True)
    index = next((Path(str(vcf) + ext) for ext in ('.tbi', '.csi') if Path(str(vcf) + ext).is_file()), None)
    if index is None:
        raise FileNotFoundError(f'Indexed input required: {vcf}')
    table = audit.load_representative_summary(summary)
    required = {'region_id', 'phecode', 'representative_variant', 'representative_selection',
                'sex_scope', 'n_samples', 'n_cases', 'representative_p_score', 'representative_pip',
                'representative_mac', 'representative_n_carriers', 'representative_n_case_carriers',
                'representative_n_control_carriers', 'representative_missing_rate'}
    if not required.issubset(table) or table.empty or table[list(required)].isna().any().any():
        raise ValueError('Incomplete Figure 2 representative summary')
    if not table['representative_selection'].eq('minimum_score_p_across_finemap_window').all():
        raise ValueError('Expected minimum-p representatives from the previous Figure 2 run')
    if table.duplicated(['region_id', 'phecode']).any():
        raise ValueError('Duplicate region/phenotype rows')
    table = table.astype(object).where(pd.notna(table), None)
    for expected in table.to_dict('records'):
        scan_window(expected, flank_bp, window_basis)
    reader = VCF(str(vcf))
    try:
        sample_ids = np.asarray(reader.samples, dtype=str)
        header_sha = hashlib.sha256(reader.raw_header.encode()).hexdigest()
    finally:
        reader.close()
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate VCF sample IDs')
    minimum_count = int(run['parameters']['min_phecode_count'])
    labels = audit.cached_analysis_labels(table, sample_ids, retained, sources, minimum_count)
    common = {
        'contract': CONTRACT, 'source_vcf': audit.fingerprint(vcf),
        'source_index': audit.fingerprint(index), 'vcf_header_sha256': header_sha,
        'sample_manifest_sha256': cohort['sample_sha256'], 'flank_bp': flank_bp,
        'window_basis': window_basis,
        'minimum_phecode_count': minimum_count, 'sv_min_known_length_bp': 50,
        'minimum_complete_pairs': 3,
        'ld_method': 'ALT_GT_Pearson_r2_mean_imputed_within_analysis_cohort',
        'candidate_filters': 'No additional MAC/carrier/missingness/hardmask/FILTER exclusions',
        'distance_rule': ('outlier_start - flank_bp <= SV_POS <= outlier_end + flank_bp, same chromosome'
                          if window_basis == 'outlier' else 'abs(SV_POS - lead_POS) <= flank_bp, same chromosome'),
    }
    manifest = {**common, 'status': 'running', 'threads': threads, 'r2_threshold': r2_threshold,
                'r2_operator': '>', 'rng': 'No random operations',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'configured_source_uri_from_previous_run': run['parameters']['discovery_vcf_source'],
                'vcf_override': str(vcf) != str(default_vcf.resolve()),
                'inputs': [audit.fingerprint(path, sha=True) for path in [summary, *inputs]],
                'cached_cohort_sources': {key: audit.fingerprint(path) for key, path in sources.items()},
                'code': audit.fingerprint(Path(__file__), sha=True),
                'cohort_helper_code': audit.fingerprint(Path(audit.__file__), sha=True)}
    audit.atomic_json(output / 'manifest.json', manifest)

    def one(expected):
        key = (expected['representative_variant'], expected['phecode'])
        label = labels[key]
        selection = hashlib.sha256(np.asarray(label, dtype='<f8').tobytes()).hexdigest()
        contract = {**common, 'expected_row': expected, 'phenotype_mask_sha256': selection}
        cache = cache_dir / f'{digest([expected["region_id"], expected["phecode"]])}.json'
        rows, reused = cached_scan(cache, contract,
                                  lambda: scan_association(vcf, expected, label, sample_ids, flank_bp, window_basis))
        selected = linked_pairs(rows, r2_threshold)
        stats = {'region_id': expected['region_id'], 'phecode': expected['phecode'],
                 'phecode_string': expected.get('phecode_string', ''), 'lead_variant': key[0],
                 'n_sv_alleles_scanned': len(rows), 'n_defined_ld': sum(r['r2'] is not None for r in rows),
                 'n_sv_r2_gt_threshold': len(selected), 'cache_reused': reused,
                 'status': 'complete', 'flank_bp': flank_bp, 'r2_threshold': r2_threshold}
        start, end = scan_window(expected, flank_bp, window_basis)
        stats.update(window_basis=window_basis, window_start=start, window_end=end)
        LOG.info('%s %s: scanned=%d linked=%d cached=%s', key[0], key[1], len(rows), len(selected), reused)
        return rows, stats

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(one, table.to_dict('records')))
        # Refuse to save combined results if source identity changed during scanning.
        if audit.fingerprint(vcf) != common['source_vcf'] or audit.fingerprint(index) != common['source_index']:
            raise ValueError('Source VCF or index changed while computing linkage')
        all_rows = [row for rows, _ in results for row in rows]
        all_frame = pd.DataFrame(all_rows, columns=PAIR_COLUMNS)
        hits = pd.DataFrame(linked_pairs(all_rows, r2_threshold), columns=PAIR_COLUMNS)
        if not hits.empty:
            hits = hits.sort_values(['phecode', 'lead_variant', 'r2', 'sv_pos', 'sv_id'],
                                    ascending=[True, True, False, True, True], kind='stable')
        outputs = []
        for filename, frame in [('all_sv_pairs.tsv', all_frame), ('linked_svs.tsv', hits),
                                ('linked_svs.csv', hits),
                                ('locus_summary.tsv', pd.DataFrame([stats for _, stats in results]))]:
            path = output / filename
            write_report(path, frame)
            outputs.append(audit.fingerprint(path, sha=True))
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        n_associations=len(results), n_linked_pairs=len(hits),
                        n_distinct_linked_sv_alleles=len({row['sv_id'] for row in linked_pairs(all_rows, r2_threshold)}),
                        outputs=outputs)
        audit.atomic_json(output / 'manifest.json', manifest)
        LOG.info('All SV linkage reports saved to %s', output)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        audit.atomic_json(output / 'manifest.json', manifest)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=audit.config.figure_dir('figure2_sources') / 'loci_panel' / 'fine_mapping_summary.tsv')
    parser.add_argument('--output', type=Path, default=audit.figure2_1025.analysis_root() / 'sv_linkage_panel')
    parser.add_argument('--vcf', type=Path, help='Optional already-local indexed VCF; no staging')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--flank-bp', type=int, default=100_000)
    parser.add_argument('--r2-threshold', type=float, default=.5)
    parser.add_argument('--window-basis', choices=['lead', 'outlier'], default='lead',
                        help='Center on the lead (legacy default), or expand the outlier start/end')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / 'linkage.log', encoding='utf-8')])
    LOG.info('Parameters: %s', vars(args))
    run_linkage(**vars(args))


if __name__ == '__main__':
    main()
