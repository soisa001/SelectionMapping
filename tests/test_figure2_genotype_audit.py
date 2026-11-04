"""Synthetic-only checks for Workbench genotype missingness and confidence audit."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import figure2_genotype_audit as audit


class Record:
    POS = 100
    REF = 'T'
    ALT = ['A']
    FILTER = None
    INFO = [('AF', 0.2)]

    def __init__(self, genotypes, fields=None):
        self.genotypes = genotypes
        self.fields = fields or {}
        self.FORMAT = ['GT', *self.fields]

    def format(self, field):
        return self.fields[field]


def test_partial_missing_and_major_alt_counts():
    gt = [[1, 1, True], [1, 1, False], [0, 1, False], [-1, 0, False], [-1, -1, False]]
    dosage, partial = audit.decode_gt(gt)
    assert partial.tolist() == [False, False, False, True, False]
    stats = audit.genotype_stats(dosage)
    assert stats['n_called'] == 3
    assert stats['n_missing'] == 2
    assert stats['allele_number'] == 6
    assert stats['alt_allele_count'] == 5
    assert stats['mac'] == 1
    assert stats['n_alt_carriers'] == 3
    assert stats['alt_af'] == pytest.approx(5 / 6)
    assert stats['maf'] == pytest.approx(1 / 6)
    assert np.isnan(audit.genotype_stats(np.array([np.nan]))['alt_af'])


@pytest.mark.parametrize('gt', [[[1, False]], [[0, 2, False]], [[-2, 0, False]]])
def test_rejects_unsupported_gt(gt):
    with pytest.raises(ValueError):
        audit.decode_gt(gt)


def test_called_gt_probability_not_maximum_and_invalid_fields():
    record = Record([[0, 0, False], [0, 1, False], [1, 1, False]], {
        'GP': [[0.02, 0.03, 0.95], [0.01, 0.98, 0.01], [-1, -1, -1]],
        'GQ': [[20], [-2147483648], [40]], 'DS': [[1.93], [1], [3]],
        'AD': [[10, 1], [4, 6], [1, 9]],
    })
    fields = audit.quality_arrays(record, np.array([0., 1., 2.]))
    assert fields['GP_max'][0] == .95
    assert fields['GP_called'][0] == .02
    assert fields['GP_called'][1] == .98
    assert np.isnan(fields['GP_called'][2])
    assert np.isnan(fields['GQ'][1])
    assert np.isnan(fields['DS'][2])
    assert fields['het_alt_balance'][1] == .6
    assert np.isnan(fields['het_alt_balance'][0])
    assert 'DP' not in fields
    with pytest.raises(ValueError, match='shape'):
        audit.quality_arrays(Record(record.genotypes, {'GP': [[.5, .5]] * 3}), np.array([0., 1., 2.]))


def test_cohort_and_sex_masks_match_existing_phenotype_rules():
    ids = np.array(['s0', 's1', 's2', 's3', 's4'])
    sex = np.array([0., 0., 0., 1., 0.])
    counts = pd.DataFrame({'person_id': ['s0', 's1', 's3', 's4'],
                           'phecode': ['P'] * 4, 'count': [2, 1, 3, 4]})
    labels = audit.analysis_labels(ids, {'s0', 's1', 's2', 's3'}, sex, counts, 'P', 'Female', 2)
    np.testing.assert_allclose(labels, [1, np.nan, 0, np.nan, np.nan], equal_nan=True)
    existing = audit.admixture_phewas.phenotype_vector(counts, ids, sex, 'Female')
    existing[-1] = np.nan
    np.testing.assert_allclose(labels, existing, equal_nan=True)


def test_quality_sensitivity_masks_uncertain_calls_without_changing_baseline():
    record = Record([[0, 0, False], [0, 1, False], [1, 1, False]], {
        'GP': [[0.02, 0.03, 0.95], [0.01, 0.98, 0.01], [-1, -1, -1]],
        'AD': [[10, 1], [4, 6], [1, 9]],
    })
    expected = {'representative_variant': 'chr6:100:T:A', 'phecode': 'P'}
    rows, sensitivity, diagnostics, _ = audit.audit_record(
        record, expected, np.array([1., 0., 0.]), {'GP_called': .9, 'GQ': 20})
    baseline = rows[0]
    assert baseline['n_called'] == 3
    assert baseline['alt_af'] == .5
    masked = next(row for row in sensitivity if row['group'] == 'analysis' and row['field'] == 'GP_called')
    assert masked['n_called'] == 1
    assert masked['n_missing'] == 2
    assert masked['n_called_below_threshold'] == 1
    assert masked['n_field_missing_or_invalid'] == 1
    balance = next(row for row in diagnostics if row['group'] == 'analysis' and row['field'] == 'het_alt_balance')
    assert balance['n_eligible'] == 1
    assert balance['n_missing_or_invalid'] == 0


@pytest.fixture
def cached_run(tmp_path, monkeypatch):
    root = tmp_path / 'analysis'
    cohorts = root / 'cohorts'
    cohorts.mkdir(parents=True)
    work = tmp_path / 'work'
    vcf = work / 'workspace_inputs' / 'discovery' / 'synthetic.vcf.gz'
    vcf.parent.mkdir(parents=True)
    vcf.write_bytes(b'synthetic mock reader input')
    (vcf.parent / (vcf.name + '.tbi')).write_bytes(b'synthetic index')
    ids = [f's{i}' for i in range(6)]
    sha = audit.figure2_1025._sample_sha(ids)
    pd.DataFrame({'person_id': ids}).to_csv(cohorts / 'discovery_analysis_samples_1025.tsv', sep='\t', index=False)
    cov = tmp_path / 'cov.tsv'
    phe = tmp_path / 'phe.tsv'
    pd.DataFrame({'person_id': ids, 'sex_at_birth': [0] * 6}).to_csv(cov, sep='\t', index=False)
    pd.DataFrame({'person_id': ['s0', 's1'], 'phecode': ['P', 'P'], 'count': [2, 2]}).to_csv(phe, sep='\t', index=False)
    cohort = {'sample_sha256': sha, 'retained_after_pca_and_complete_covariates': 6,
              'n_pcs': 20, 'internal_related_pairs': 0,
              'sources': {'covariates': str(cov), 'phenotypes': str(phe)}}
    (cohorts / 'discovery_cohort_audit_1025.json').write_text(json.dumps(cohort))
    summary_dir = tmp_path / 'loci_panel'
    summary_dir.mkdir()
    (summary_dir / 'run_manifest.json').write_text(json.dumps({
        'parameters': {'discovery_cohort': {'sample_sha256': sha}, 'min_phecode_count': 2,
                       'discovery_vcf_source': 'gs://synthetic/synthetic.vcf.gz'},
        'runtime': {'work_directory': str(work)},
    }))
    expected = {'representative_variant': 'chr6:100:T:A', 'phecode': 'P', 'sex_scope': 'Both',
                'n_samples': 6, 'n_cases': 2, 'representative_mac': 4,
                'representative_n_carriers': 3, 'representative_n_case_carriers': 1,
                'representative_n_control_carriers': 2, 'representative_missing_rate': 2 / 6}
    summary = summary_dir / 'fine_mapping_summary.tsv'
    pd.DataFrame([expected]).to_csv(summary, sep='\t', index=False)
    record = Record([[0, 1, False], [-1, -1, False], [1, 1, False],
                     [0, -1, False], [0, 0, False], [0, 1, False], [1, 1, False]])

    class Reader:
        samples = ids + ['outside_cohort']
        seqnames = ['chr6']
        raw_header = '##fileformat=VCFv4.2\n##source=synthetic\n#CHROM\ts0\n'

        def __init__(self, path):
            pass

        def __call__(self, region):
            assert region == 'chr6:100-100'
            return iter([record])

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'cyvcf2', SimpleNamespace(VCF=Reader))
    monkeypatch.setattr(audit.figure2_1025, 'analysis_root', lambda: root)
    # Any accidental staging violates this audit's offline contract.
    def no_staging(*args, **kwargs):
        raise AssertionError('Audit must never stage data')
    monkeypatch.setattr(audit.admixture_inputs, 'stage_file', no_staging)
    monkeypatch.setattr(audit.admixture_inputs, 'stage_cohort_sources', no_staging)
    return summary, tmp_path / 'output', expected


def test_cached_run_writes_verified_aggregates_and_is_repeatable(cached_run):
    summary, output, _ = cached_run
    audit.run_audit(summary=summary, output=output, threads=2)
    first = (output / 'genotype_counts.tsv').read_bytes()
    counts = pd.read_csv(output / 'genotype_counts.tsv', sep='\t').set_index('group')
    assert counts.loc['analysis', 'n_samples'] == 6
    assert counts.loc['analysis', 'n_missing'] == 2
    assert counts.loc['analysis', 'n_partial_gt'] == 1
    assert counts.loc['cases', 'n_missing'] == 1
    assert counts.loc['controls', 'n_missing'] == 1
    sensitivity = pd.read_csv(output / 'quality_sensitivity.tsv', sep='\t')
    assert sensitivity['status'].eq('field_absent').all()
    assert '#CHROM' not in (output / 'vcf_metadata.txt').read_text()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['n_summary_mismatches'] == 0
    audit.run_audit(summary=summary, output=output, threads=2)
    assert (output / 'genotype_counts.tsv').read_bytes() == first


def test_summary_disagreement_is_saved_and_fails(cached_run):
    summary, output, expected = cached_run
    expected['representative_mac'] = 5
    pd.DataFrame([expected]).to_csv(summary, sep='\t', index=False)
    with pytest.raises(ValueError, match='differs from saved summary'):
        audit.run_audit(summary=summary, output=output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'summary_mismatch'
    checks = pd.read_csv(output / 'summary_comparison.tsv', sep='\t')
    assert checks.loc[0, 'mismatched_columns'] == 'representative_mac'


def test_heterozygote_deficit_and_ds_disagreement_are_descriptive():
    stats = audit.genotype_stats(np.array([0.] * 6779 + [1.] * 1020 + [2.] * 580))
    assert stats['alt_af'] == pytest.approx(2180 / 16758)
    assert stats['pooled_hwe_expected_n_het'] == pytest.approx(1896.410072801)
    assert stats['pooled_het_deficit_f'] == pytest.approx(.46214164614)
    record = Record([[0, 0, False], [0, 1, False], [1, 1, False]], {'DS': [[1.9], [1], [2]]})
    _, _, diagnostics, _ = audit.audit_record(record,
        {'representative_variant': 'chr6:100:T:A', 'phecode': 'P'}, np.array([1., 0., 0.]), {})
    result = next(row for row in diagnostics if row['field'] == 'DS_minus_GT' and row['group'] == 'analysis')
    assert result['n_abs_difference_gt_1'] == 1
    assert result['mean'] == pytest.approx(1.9 / 3)


def test_focused_reference_audit_and_exact_allele_guard(cached_run):
    summary, output, expected = cached_run
    # Unrelated broken row must not interfere with a focused phenotype audit.
    other = {**expected, 'phecode': 'OTHER', 'representative_variant': None, 'fine_map_table': 'absent.tsv'}
    pd.DataFrame([expected, other]).to_csv(summary, sep='\t', index=False)
    reference = summary.parent / 'reference.tsv'
    ref = pd.DataFrame([{'variant': 'chr6:100:T:A', 'genome_build': 'GRCh38', 'allele': 'A',
                         'population': 'synthetic_AFR', 'alt_allele_count': 2,
                         'allele_number': 10, 'source_url': 'https://example.test/synthetic'}])
    ref.to_csv(reference, sep='\t', index=False)
    audit.run_audit(summary=summary, output=output, phecode='P', variant='chr6:100:T:A',
                    reference_frequencies=reference)
    counts = pd.read_csv(output / 'genotype_counts.tsv', sep='\t')
    assert set(counts['phecode']) == {'P'}
    comparison = pd.read_csv(output / 'reference_frequency_comparison.tsv', sep='\t')
    assert len(comparison) == 3
    assert comparison['reference_alt_af'].eq(.2).all()
    ref['allele'] = 'T'
    ref.to_csv(reference, sep='\t', index=False)
    with pytest.raises(ValueError, match='forward-strand ALT'):
        audit.reference_comparison(counts, reference)
    with pytest.raises(ValueError, match='Incomplete representative'):
        audit.run_audit(summary=summary, output=output, phecode='P', variant='chr6:101:T:A')
    with pytest.raises(ValueError, match='No representative matches'):
        audit.run_audit(summary=summary, output=output, phecode='ABSENT', variant='chr6:100:T:A')
