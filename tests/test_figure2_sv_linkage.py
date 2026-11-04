"""Synthetic-only coverage for exhaustive regional SV linkage reporting."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import figure2_sv_linkage as ld


class Record:
    ID = None
    FILTER = None

    def __init__(self, pos, ref, alts, dosages=None, info=None, genotypes=None):
        self.POS, self.REF, self.ALT = pos, ref, alts
        self.INFO = info or {}
        calls = {0: [0, 0, False], 1: [0, 1, False], 2: [1, 1, False]}
        self.genotypes = genotypes if genotypes is not None else [calls[d] for d in dosages]


def test_structural_allele_types_and_exact_size_threshold():
    assert ld.describe_sv(Record(100, 'A' * 51, ['A'], []), 1) == ('DEL', 50)
    assert ld.describe_sv(Record(100, 'A', ['A' * 51], []), 1) == ('INS', 50)
    assert ld.describe_sv(Record(100, 'A', ['A' * 50], []), 1) is None
    assert ld.describe_sv(Record(100, 'A', ['<INV>'], [], {'END': 300}), 1) == ('INV', 200)
    assert ld.describe_sv(Record(100, 'A', ['<DUP>'], []), 1) == ('DUP', None)
    assert ld.describe_sv(Record(100, 'A', ['A]chr2:50]'], []), 1) == ('BND', None)
    assert ld.describe_sv(Record(100, 'A', ['<NON_REF>'], []), 1) is None
    assert ld.describe_sv(Record(100, 'A', ['C', '<DEL>'], [], {'SVLEN': (1, -150)}), 2) == ('DEL', 150)


def test_multiallelic_target_dosages_and_partial_missing():
    dosage = ld.alt_dosage([[1, 1, False], [1, 2, False], [2, 2, False], [0, -1, False]], 2, 2)
    np.testing.assert_allclose(dosage, [0, 1, 2, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match='allele index'):
        ld.alt_dosage([[0, 3, False]], 2, 2)


def test_signed_ld_strict_cutoff_and_undefined_variants():
    lead = np.array([0., 0., 1., 1., 2., 2.])
    same = ld.dosage_linkage(lead, lead)
    opposite = ld.dosage_linkage(lead, 2 - lead)
    assert same['r'] == 1
    assert opposite['r'] == -1
    assert opposite['r2'] == 1
    assert ld.dosage_linkage(lead, np.zeros(6))['ld_status'] == 'monomorphic'
    assert ld.dosage_linkage(lead, np.array([0., 1., np.nan, np.nan, np.nan, np.nan]))['r2'] is None
    rows = [{'r2': .5}, {'r2': .50001}, {'r2': None}]
    assert ld.linked_pairs(rows, .5) == [rows[1]]


def test_missingness_sensitivity_uses_correct_denominators():
    lead = np.array([0., 1., 2., 0., 1., 2.])
    sv = np.array([0., 1., 2., np.nan, np.nan, 2.])
    result = ld.dosage_linkage(lead, sv)
    filled = np.array([0., 1., 2., 1.25, 1.25, 2.])
    assert result['r2'] == pytest.approx(np.corrcoef(lead, filled)[0, 1] ** 2)
    assert result['r2_pairwise_complete'] == pytest.approx(1)
    assert result['n_pairwise_complete'] == 4
    assert ld.audit.genotype_stats(sv)['allele_number'] == 8


def test_checkpoint_validation_and_reuse(tmp_path):
    path = tmp_path / 'checkpoint.json'
    calls = []
    def scan():
        calls.append(1)
        return [{'r2': .75}]
    contract = {'cohort': 'synthetic', 'flank_bp': 100_000}
    assert ld.cached_scan(path, contract, scan)[1] is False
    assert ld.cached_scan(path, contract, scan)[1] is True
    assert len(calls) == 1
    payload = json.loads(path.read_text())
    payload['rows'][0]['r2'] = .9
    path.write_text(json.dumps(payload))
    assert ld.cached_scan(path, contract, scan)[0][0]['r2'] == .75
    assert len(calls) == 2
    assert ld.cached_scan(path, {**contract, 'cohort': 'changed'}, scan)[1] is False


@pytest.fixture
def scan_input(tmp_path, monkeypatch):
    vcf = tmp_path / 'synthetic.vcf.gz'
    vcf.write_bytes(b'synthetic fake-reader fixture')
    (tmp_path / 'synthetic.vcf.gz.tbi').write_bytes(b'index fixture')
    expected = {
        'region_id': 'chr6_locus', 'phecode': 'P', 'phecode_string': 'Synthetic phenotype',
        'representative_variant': 'chr6:200001:T:A', 'sex_scope': 'Both',
        'representative_selection': 'minimum_score_p_across_finemap_window',
        'representative_p_score': .0001, 'representative_pip': .1,
        'n_samples': 6, 'n_cases': 3, 'representative_mac': 6,
        'representative_n_carriers': 4, 'representative_n_case_carriers': 1,
        'representative_n_control_carriers': 3, 'representative_missing_rate': 0.,
    }
    summary = tmp_path / 'summary.tsv'
    pd.DataFrame([expected]).to_csv(summary, sep='\t', index=False)
    dosage = [0, 0, 1, 1, 2, 2]
    records = [
        Record(200001, 'T', ['A'], dosage),
        Record(100001, 'A' * 51, ['A'], dosage),  # inclusive left boundary
        Record(300001, 'A', ['A' * 51], [2, 2, 1, 1, 0, 0]),  # inclusive right boundary
        Record(300002, 'A', ['A' * 51], dosage),  # outside, despite indexed overlap
        Record(200010, 'A', ['<INV>'], [0, 1, 2, 2, 1, 0], {'SVLEN': 100}),
        Record(200020, 'A' * 60, ['A'], [0] * 6),  # monomorphic, kept in all-pairs report
        Record(200030, 'A', ['C', 'A' * 51], genotypes=[
            [1, 1, False], [1, 1, False], [1, 2, False],
            [1, 2, False], [2, 2, False], [2, 2, False]]),
    ]
    ids = [f's{i}' for i in range(6)]
    calls = []

    class Reader:
        samples = ids
        raw_header = '##fileformat=VCFv4.2\n#CHROM\tPOS\n'
        seqnames = ['chr6']
        def __init__(self, path):
            pass
        def __call__(self, region):
            calls.append(region)
            return iter(records)
        def close(self):
            pass

    labels = np.array([1., 1., 1., 0., 0., 0.])
    monkeypatch.setitem(sys.modules, 'cyvcf2', SimpleNamespace(VCF=Reader))
    monkeypatch.setattr(ld.audit, 'load_cached_inputs', lambda path: (
        {'parameters': {'min_phecode_count': 2, 'discovery_vcf_source': 'gs://synthetic/source.vcf.gz'}},
        {'sample_sha256': 'synthetic'}, set(ids), {}, vcf, []))
    monkeypatch.setattr(ld.audit, 'cached_analysis_labels',
                        lambda *args: {('chr6:200001:T:A', 'P'): labels})
    return summary, tmp_path / 'output', expected, calls, records, labels, ids, vcf


def test_all_svs_reported_boundary_direction_and_no_carrier_filter(scan_input):
    summary, output, _, calls, _, _, _, _ = scan_input
    ld.run_linkage(summary=summary, output=output, threads=2)
    hits = pd.read_csv(output / 'linked_svs.tsv', sep='\t')
    assert set(hits['sv_pos']) == {100001, 300001, 200030}
    assert hits['r2'].gt(.5).all()
    assert hits['qc_flags'].str.contains('sv_alt_carriers_lt_20').all()
    assert hits.set_index('sv_pos').loc[300001, 'sv_alt_coupled_with_lead_allele'] == 'REF'
    assert hits.set_index('sv_pos').loc[200030, 'sv_alt_index'] == 2
    pairs = pd.read_csv(output / 'all_sv_pairs.tsv', sep='\t')
    assert len(pairs) == 5
    assert pairs.set_index('sv_pos').loc[200020, 'ld_status'] == 'monomorphic'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['n_linked_pairs'] == 3
    first = (output / 'linked_svs.tsv').read_bytes()
    previous_calls = len(calls)
    ld.run_linkage(summary=summary, output=output, threads=2)
    assert len(calls) == previous_calls
    assert (output / 'linked_svs.tsv').read_bytes() == first
    assert pd.read_csv(output / 'locus_summary.tsv', sep='\t')['cache_reused'].all()


def test_no_hits_writes_empty_report_with_headers(scan_input):
    summary, output, _, _, _, _, _, _ = scan_input
    ld.run_linkage(summary=summary, output=output, r2_threshold=1.)
    assert pd.read_csv(output / 'linked_svs.tsv', sep='\t').empty
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'


def test_changed_lead_counts_fail_instead_of_saving(scan_input):
    summary, output, expected, _, _, _, _, _ = scan_input
    expected['representative_mac'] = 5
    pd.DataFrame([expected]).to_csv(summary, sep='\t', index=False)
    with pytest.raises(ValueError, match='Lead differs'):
        ld.run_linkage(summary=summary, output=output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'
    assert not (output / 'linked_svs.tsv').exists()


def test_duplicate_exact_sv_alleles_rejected(scan_input):
    _, _, expected, _, records, labels, ids, vcf = scan_input
    records.append(records[1])
    with pytest.raises(ValueError, match='Duplicate exact SV'):
        ld.scan_association(vcf, expected, labels, ids, 100_000)


def test_outlier_window_reports_all_linked_svs_and_preserves_exact_alleles(scan_input):
    summary, output, expected, calls, records, _, _, _ = scan_input
    expected.update(chrom=6, start=150001, end=400001)
    pd.DataFrame([expected]).to_csv(summary, sep='\t', index=False)
    dosage = [0, 0, 1, 1, 2, 2]
    records.extend([
        Record(50001, 'A' * 101, ['A'], dosage),
        Record(500001, 'A', ['A' * 151], dosage),
        Record(50000, 'A' * 101, ['A'], dosage),
        Record(500002, 'A' * 101, ['A'], dosage),
    ])
    ld.run_locus_linkage(summary, output, threads=2)
    assert 'chr6:50001-500001' in calls
    hits = pd.read_csv(output / 'linked_svs.csv')
    pd.testing.assert_frame_equal(hits, pd.read_csv(output / 'linked_svs.tsv', sep='\t'))
    assert set(hits['sv_pos']) == {50001, 100001, 200030, 300001, 300002, 500001}
    assert hits['window_start'].eq(50001).all() and hits['window_end'].eq(500001).all()
    assert hits['window_basis'].eq('outlier').all() and hits['r2'].gt(.5).all()
    boundary = hits.set_index('sv_pos').loc[500001]
    assert boundary['sv_locus'] == 'chr6:500001'
    assert boundary['sv_ref'] == 'A' and boundary['sv_alt'] == 'A' * 151
    assert boundary['sv_length_bp'] == 150 and boundary['sv_alt_af'] == .5
    assert boundary['sv_target_alt_mac'] == 6  # Exhaustive scan, not fine-map eligibility.
    assert hits['outlier_start'].eq(150001).all() and hits['outlier_end'].eq(400001).all()
    previous = len(calls)
    (output / 'linked_svs.csv').write_text('corrupt report')
    ld.run_locus_linkage(summary, output, threads=2)
    assert len(calls) == previous
    assert len(pd.read_csv(output / 'linked_svs.csv')) == 6
    # Same checkpoint directory with a different window must perform a new scan.
    ld.run_linkage(summary=summary, output=output, window_basis='lead')
    assert len(calls) > previous
    assert set(pd.read_csv(output / 'linked_svs.csv')['sv_pos']) == {100001, 200030, 300001}


@pytest.mark.parametrize('changes', [{}, {'chrom': 7, 'start': 150001, 'end': 400001},
                                    {'chrom': 6, 'start': 400001, 'end': 150001},
                                    {'chrom': 6, 'start': 1.5, 'end': 400001}])
def test_outlier_scan_rejects_missing_or_invalid_boundaries(scan_input, changes):
    _, _, expected, _, _, _, _, _ = scan_input
    with pytest.raises(ValueError, match='Outlier chromosome|outlier start/end'):
        ld.scan_window({**expected, **changes}, 100_000, 'outlier')


@pytest.mark.parametrize('problem', [None, 'sample_count', 'representative', 'failed'])
def test_minimal_loci_panel_summary_recovers_representative_from_saved_fit(scan_input, problem):
    summary, output, expected, _, _, _, _, _ = scan_input
    fit = summary.parent / 'fit.tsv'
    pd.DataFrame([{
        'variant_id': expected['representative_variant'], 'pos': 200001,
        'beta_score': .2, 'se_score': .1, 'p_score': .0001, 'pip': .1,
        'mac': 6, 'n_carriers': 4, 'n_case_carriers': 1, 'n_control_carriers': 3,
        'missing_rate': 0., 'sv_type': '', 'sv_length_bp': 0, 'credible_set': '',
        'n_samples': 6, 'n_cases': 3, 'sex_scope': 'Both', 'n_pcs': 20,
    }]).to_csv(fit, sep='\t', index=False)
    minimal = {key: value for key, value in expected.items()
               if not key.startswith('representative_') and key != 'sex_scope'}
    minimal.update(fine_map_table=str(fit), status='ready')
    if problem == 'sample_count':
        minimal['n_samples'] = 7
    elif problem == 'representative':
        minimal['representative_variant'] = 'chr6:200002:T:A'
    elif problem == 'failed':
        minimal['status'] = 'failed'
    pd.DataFrame([minimal]).to_csv(summary, sep='\t', index=False)
    if problem:
        with pytest.raises(ValueError, match='mismatch|not ready'):
            ld.audit.load_representative_summary(summary)
        return
    hydrated = ld.audit.load_representative_summary(summary)
    assert hydrated.iloc[0]['representative_source_sha256'] == ld.audit.fingerprint(fit, sha=True)['sha256']
    ld.run_linkage(summary=summary, output=output)
    assert len(pd.read_csv(output / 'linked_svs.tsv', sep='\t')) == 3
