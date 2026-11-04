"""Inspect one exact allele in the saved local FLARE panel; never stage inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import admixture_inputs, figure2_1025 as corrected, figure2_genotype_audit as audit
from .common import samples


def record_summary(record, variant, sample_n):
    result = {'variant': variant, 'group': 'LAI_panel', 'n_samples': sample_n,
              'ref': record.REF, 'alt': record.ALT[0], 'format_fields': ','.join(record.FORMAT)}
    if 'GT' not in record.FORMAT:
        return {**result, 'status': 'GT_absent_AF_not_computable'}
    dosage, partial = audit.decode_gt(record.genotypes)
    if len(dosage) != sample_n:
        raise ValueError('GT dimensions differ from the verified LAI sample manifest')
    return {**result, 'status': 'complete' if np.isfinite(dosage).any() else 'all_GT_missing',
            **audit.genotype_stats(dosage), 'n_partial_gt': int(partial.sum())}


def run(*, variant, panel_manifest, output, vcf=None, threads=4):
    from cyvcf2 import VCF

    chrom, pos, ref, alt = variant.removeprefix('chr').split(':')
    pos = int(pos)
    if not (chrom.isdigit() and 1 <= int(chrom) <= 22 and pos > 0
            and len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT'
            and ref != alt and 1 <= threads <= 32):
        raise ValueError('Expected an exact autosomal biallelic SNV and 1-32 threads')
    variant = f'chr{chrom}:{pos}:{ref}:{alt}'
    panel_manifest = Path(panel_manifest).expanduser().resolve(strict=True)
    panel = json.loads(panel_manifest.read_text())
    ids = samples.load_aou_sample_list(download=False)
    if not ids:
        raise FileNotFoundError(f'Existing local LAI sample list required: {samples.AOU_SAMPLE_LIST_LOCAL}')
    if (panel['status'] != 'ready' or len(ids) != 1025 or panel['sample_n'] != len(ids)
            or corrected._sample_sha(ids) != panel['sample_sha256']):
        raise ValueError('LAI sample list differs from the saved panel manifest')
    path = Path(vcf or Path(panel['source_directory']) / f'chr{chrom}.anc.vcf.gz').expanduser().resolve(strict=True)
    index = next((Path(str(path) + ext) for ext in ('.tbi', '.csi')
                  if Path(str(path) + ext).is_file() and Path(str(path) + ext).stat().st_size), None)
    if index is None:
        raise FileNotFoundError(f'Existing VCF index required: {path}')
    source = audit.fingerprint(path)
    index_source = audit.fingerprint(index)
    reader = VCF(str(path), samples=sorted(ids), threads=threads)
    try:
        if len(reader.samples) != len(ids) or set(reader.samples) != ids:
            raise ValueError('Panel VCF is missing required LAI samples')
        token = admixture_inputs.detect_vcf_chrom(reader, chrom)
        if token is None:
            raise ValueError(f'Chromosome {chrom} is absent from the panel VCF')
        records = [record for record in reader(f'{token}:{pos}-{pos}') if record.POS == pos]
        matches = [record for record in records if record.REF == ref and record.ALT == [alt]]
        if len(matches) != 1:
            alleles = [(record.REF, record.ALT) for record in records]
            raise ValueError(f'Expected one exact {variant}; found {len(matches)}. Alleles at POS: {alleles}')
        record = matches[0]
        row = record_summary(record, variant, len(ids))
        info = dict(record.INFO)
        definitions = []
        for line in reader.raw_header.splitlines():
            match = re.match(r'##INFO=<ID=([^,>]+)', line)
            if match and match.group(1) in info:
                definitions.append(line)
    finally:
        reader.close()
    if audit.fingerprint(path) != source or audit.fingerprint(index) != index_source:
        raise ValueError('VCF or index changed during the query')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit.atomic_table(output / 'genotype_counts.tsv', pd.DataFrame([row]))
    metadata = {'variant': variant, 'status': row['status'], 'source_vcf': source,
                'source_index': index_source, 'vcf_override': vcf is not None,
                'sample_n': len(ids), 'sample_sha256': corrected._sample_sha(ids),
                'sample_list_source': audit.fingerprint(samples.AOU_SAMPLE_LIST_LOCAL, sha=True),
                'panel_manifest': audit.fingerprint(panel_manifest, sha=True),
                'threads': threads, 'rng': 'No random operations',
                'checked_utc': datetime.now(timezone.utc).isoformat(),
                'info_json': json.dumps(info, default=str), 'info_definitions': definitions,
                'info_af_note': 'Copied source annotation; not recomputed for the selected 1025 samples',
                'genotype_counts': audit.fingerprint(output / 'genotype_counts.tsv', sha=True)}
    audit.atomic_json(output / 'manifest.json', metadata)
    print(f'Source panel VCF: {path}\nVerified LAI samples: {len(ids)}\nVariant: {variant}')
    print(pd.DataFrame([row]).to_string(index=False))
    print('Original INFO fields:', json.dumps(info, default=str, sort_keys=True))
    print('\n'.join(definitions))
    print(f'Reports: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='chr6:32479741:T:A')
    parser.add_argument('--panel-manifest', type=Path,
                        default=corrected.analysis_root() / 'finemap_panel_sv/panel_markers/run_manifest.json')
    parser.add_argument('--output', type=Path,
                        default=corrected.analysis_root() / 'qc_acute_kidney_lai_panel')
    parser.add_argument('--vcf', type=Path, help='Optional existing local panel VCF; never downloaded')
    parser.add_argument('--threads', type=int, default=4)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
