from types import SimpleNamespace
import json
import sys

import pandas as pd
import pytest

from scripts import figure2_panel_allele_qc as qc
from scripts.figure2_panel_allele_qc import record_summary


def test_panel_af_counts_gt_and_partial_missing():
    record = SimpleNamespace(REF='T', ALT=['A'], FORMAT=['GT', 'AN1', 'AN2'],
                             genotypes=[[0, 0, True], [0, 1, True], [1, 1, True], [0, -1, True]])
    row = record_summary(record, 'chr6:32479741:T:A', 4)
    assert row['alt_af'] == .5 and row['allele_number'] == 6
    assert row['n_hom_ref'] == row['n_het'] == row['n_hom_alt'] == 1
    assert row['n_partial_gt'] == 1
    with pytest.raises(ValueError, match='dimensions'):
        record_summary(record, 'chr6:32479741:T:A', 1025)


def test_ancestry_only_record_does_not_invent_genotype_frequency():
    record = SimpleNamespace(REF='T', ALT=['A'], FORMAT=['AN1', 'AN2'])
    row = record_summary(record, 'chr6:32479741:T:A', 1025)
    assert row['status'] == 'GT_absent_AF_not_computable'
    assert 'alt_af' not in row


@pytest.mark.parametrize('ref', ['T', 'A'])
def test_exact_panel_query_uses_saved_sample_manifest_and_local_source(tmp_path, monkeypatch, ref):
    ids = {f'synthetic_{i}' for i in range(1025)}
    sample_file = tmp_path / 'samples.txt'
    sample_file.write_text('\n'.join(sorted(ids)))
    monkeypatch.setattr(qc.samples, 'AOU_SAMPLE_LIST_LOCAL', sample_file)
    def local_samples(*, download):
        assert download is False
        return ids
    monkeypatch.setattr(qc.samples, 'load_aou_sample_list', local_samples)
    vcf = tmp_path / 'chr6.anc.vcf.gz'
    vcf.write_bytes(b'synthetic VCF reader fixture')
    (tmp_path / 'chr6.anc.vcf.gz.tbi').write_bytes(b'index')
    manifest = tmp_path / 'panel.json'
    manifest.write_text(json.dumps({'status': 'ready', 'source_directory': str(tmp_path),
                                   'sample_n': len(ids), 'sample_sha256': qc.corrected._sample_sha(ids)}))
    record = SimpleNamespace(POS=32479741, REF=ref, ALT=['A'], FORMAT=['GT', 'AN1', 'AN2'],
                             INFO={'AF': .9}, genotypes=[[0, 1, True]] * 1025)
    class Reader:
        seqnames = ['chr6']
        raw_header = '##INFO=<ID=AF,Number=A,Type=Float,Description="Original source AF">'
        def __init__(self, path, *, samples, threads):
            assert path == str(vcf) and threads == 4
            self.samples = samples
        def __call__(self, region):
            assert region == 'chr6:32479741-32479741'
            return iter([record])
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'cyvcf2', SimpleNamespace(VCF=Reader))
    options = dict(variant='chr6:32479741:T:A', panel_manifest=manifest, output=tmp_path / 'output')
    if ref != 'T':
        with pytest.raises(ValueError, match='Expected one exact'):
            qc.run(**options)
        return
    output = qc.run(**options)
    counts = pd.read_csv(output / 'genotype_counts.tsv', sep='\t')
    assert counts.loc[0, 'alt_af'] == .5  # Recomputed from panel GT, not INFO/AF=0.9.
    report = json.loads((output / 'manifest.json').read_text())
    assert report['sample_n'] == 1025 and json.loads(report['info_json'])['AF'] == .9
    assert len(report['info_definitions']) == 1
