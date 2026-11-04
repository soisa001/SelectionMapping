"""Read exact SNV identities from the masked FLARE VCFs, without staging data."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from . import admixture_finemap as fm, admixture_inputs, admixture_lai
from . import figure2_1025 as corrected
from .figure2_locus_figures import MarkerScans

PANEL_CONTRACT = "masked_flare_snv_exact_alleles_observed_lai_markers_v1"
CATALOG_COLUMNS = ["variant_id", "chrom", "pos", "ref", "alt"]


def _read_window(reader, token, hit, markers, sample_n):
    """Check the indexed query against the full saved LAI scan, then select SNVs."""
    start, end = fm.expected_window(hit)
    chrom = admixture_inputs.normalize_chrom(hit["chrom"])
    rows, observed_positions, identifiers = [], set(), set()
    n_records, n_without_ancestry, n_non_snv = 0, 0, 0
    for record in reader(f"{token}:{start}-{end}"):
        pos = int(record.POS)
        if not start <= pos <= end:
            continue
        if admixture_inputs.normalize_chrom(record.CHROM) != chrom:
            raise ValueError("FLARE index returned the wrong chromosome")
        n_records += 1
        ancestry = [record.format(name) for name in ("AN1", "AN2")]
        if any(value is None for value in ancestry):
            n_without_ancestry += 1
            continue
        calls = [np.asarray(value).reshape(-1) for value in ancestry]
        if any(len(value) != sample_n for value in calls):
            raise ValueError(f"FLARE ancestry dimensions disagree with the sample manifest at {chrom}:{pos}")
        if not any((value >= 0).any() for value in calls):
            n_without_ancestry += 1
            continue
        observed_positions.add(pos)
        ref, alts = str(record.REF), record.ALT or []
        if (len(alts) != 1 or len(ref) != 1 or len(str(alts[0])) != 1
                or ref not in "ACGT" or str(alts[0]) not in "ACGT" or ref == str(alts[0])):
            n_non_snv += 1
            continue
        alt = str(alts[0])
        identifier = admixture_inputs.variant_id(chrom, pos, ref, alt)
        if identifier in identifiers:
            raise ValueError(f"Duplicate FLARE SNV identity in {hit['region_id']}: {identifier}")
        identifiers.add(identifier)
        rows.append(dict(variant_id=identifier, chrom=chrom, pos=pos, ref=ref, alt=alt))
    expected_positions = set(markers["pos"].astype(int))
    if observed_positions != expected_positions:
        raise ValueError(f"FLARE VCF/index and cached LAI marker positions disagree for {hit['region_id']}: "
                         f"missing={len(expected_positions - observed_positions)}, "
                         f"unexpected={len(observed_positions - expected_positions)}; "
                         "check AOU_LAI_DIR and the index/source files")
    if not rows:
        raise ValueError(f"No biallelic FLARE-panel SNVs in {hit['region_id']} with the configured flanks")
    return (pd.DataFrame(rows, columns=CATALOG_COLUMNS).sort_values(["pos", "variant_id"]).reset_index(drop=True),
            {"n_flare_records": n_records, "n_without_ancestry": n_without_ancestry,
             "n_non_snv": n_non_snv, "n_lai_marker_positions": len(expected_positions)})


def _write_catalog(frame, path):
    payload = frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tsv.tmp")
    temporary.write_bytes(payload)
    checked = pd.read_csv(temporary, sep="\t", keep_default_na=False)
    pd.testing.assert_frame_equal(checked, frame, check_dtype=False)
    temporary.replace(path)


class MarkerPanel:
    def __init__(self, regions, manifest_path, *, include_svs=False):
        self.regions, self.manifest_path = regions, manifest_path
        self.include_svs = include_svs
        self.variant_set = "flare_panel_snvs_svs" if include_svs else "flare_panel_snvs"

    def variants(self, hit):
        return self.regions[str(hit["region_id"])]["variants"]

    def parameters(self, hit):
        return self.regions[str(hit["region_id"])]["parameters"]

    def valid_frame(self, frame, hit):
        parameters = self.parameters(hit)
        is_sv = frame["sv_type"].fillna("").ne("")
        return (set(parameters).issubset(frame)
                and all(frame[key].eq(value).all() for key, value in parameters.items())
                and ((self.include_svs & is_sv)
                     | (~is_sv & frame["variant_id"].isin(self.variants(hit)))).all())

    @classmethod
    def prepare(cls, hits, ids, *, include_svs=False):
        """Re-query only relevant windows; reuse fits when marker identities match."""
        threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
        root = fm.analysis_dir() / "panel_markers"
        manifest_path = root / "run_manifest.json"
        variant_set = "flare_panel_snvs_svs" if include_svs else "flare_panel_snvs"
        manifest = {"panel_contract": PANEL_CONTRACT, "status": "running",
                    "variant_set": variant_set, "include_svs": include_svs,
                    "sv_selection": "discovery VCF INS/DELs; observed MAC > 20; no carrier cutoff; same hardmask/missingness/sex QC as loci" if include_svs else "excluded",
                    "started_utc": corrected._utc_now(), "threads": threads,
                    "sample_n": len(ids), "sample_sha256": corrected._sample_sha(ids),
                    "source_directory": str(Path(configured_panel_dir()).resolve()),
                    "window_pad_bp": fm.WINDOW_PAD_BP,
                    "matching": "exact normalized chromosome, POS, REF, ALT; no allele swaps or position-only matches",
                    "selection": "biallelic A/C/G/T SNVs at observed FLARE ancestry markers; discovery QC is applied later",
                    "integrity": "indexed FLARE positions must equal the full cached LAI positions in each window",
                    "cache_identity": "regional SNV identity digest and LAI sample manifest, not Git revision or VCF mtime",
                    "regions": {}}
        corrected._atomic_json(manifest_path, manifest)
        print(f"[figure2_sources][loci-panel] phase=marker_catalog; threads={threads}; "
              f"source={configured_panel_dir()}; output={root}", flush=True)
        regions = hits.drop_duplicates("region_id")

        def chromosome(group):
            chrom, chromosome_hits = group
            path = Path(configured_panel_dir()) / f"chr{int(chrom)}.anc.vcf.gz"
            indices = [Path(str(path) + suffix) for suffix in (".tbi", ".csi")]
            index = next((candidate for candidate in indices if candidate.is_file() and candidate.stat().st_size), None)
            if not path.is_file() or not path.stat().st_size or index is None:
                raise FileNotFoundError(f"Local masked FLARE VCF and index required: {path}. "
                                        "Set AOU_LAI_DIR to the existing lai_masked directory; this phase never downloads data.")
            VCF = admixture_lai._cyvcf2()
            reader = VCF(str(path), samples=list(ids))
            output = {}
            try:
                if len(reader.samples) != len(ids) or set(map(str, reader.samples)) != set(ids):
                    raise ValueError(f"FLARE VCF does not contain the required LAI sample manifest: {path}")
                token = admixture_inputs.detect_vcf_chrom(reader, chrom)
                if token is None:
                    raise ValueError(f"Chromosome {chrom} is absent from {path}")
                scans = MarkerScans(ids)
                source = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                          "mtime_ns": path.stat().st_mtime_ns, "index": corrected._output_fingerprint(index)}
                for _, hit in chromosome_hits.iterrows():
                    markers, marker_source = scans.region(hit)
                    frame, counts = _read_window(reader, token, hit, markers, len(ids))
                    catalog_path = root / f"{fm._safe(hit['region_id'])}.markers.tsv"
                    _write_catalog(frame, catalog_path)
                    variants = frozenset(frame["variant_id"])
                    digest = corrected._sha256_bytes(("\n".join(sorted(variants)) + "\n").encode("utf-8"))
                    parameters = {"variant_set": variant_set, "panel_contract": PANEL_CONTRACT,
                                  "panel_marker_sha256": digest, "panel_sample_sha256": manifest["sample_sha256"],
                                  "n_panel_markers_in_window": len(variants),
                                  "panel_catalog_path": str(catalog_path),
                                  "panel_match_rule": "exact_chr_pos_ref_alt"}
                    if include_svs:
                        parameters.update(sv_min_mac=20, sv_mac_operator=">", sv_min_carriers=0)
                    provenance = {"parameters": parameters, "source": source, "lai_source": marker_source,
                                  "catalog": corrected._output_fingerprint(catalog_path), **counts}
                    output[str(hit["region_id"])] = {"variants": variants, "parameters": parameters,
                                                    "provenance": provenance}
                    print(f"[figure2_sources][loci-panel] {hit['region_id']}; SNVs={len(variants)}; "
                          f"counts={json.dumps(counts, sort_keys=True)}; catalog={catalog_path}", flush=True)
            finally:
                reader.close()
            return output

        try:
            selected = {}
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for output in pool.map(chromosome, regions.groupby("chrom", sort=True)):
                    selected.update(output)
                    manifest["regions"].update({key: value["provenance"] for key, value in output.items()})
                    corrected._atomic_json(manifest_path, manifest)
            manifest.update(status="ready", n_regions=len(selected), completed_utc=corrected._utc_now())
            corrected._atomic_json(manifest_path, manifest)
            return cls(selected, manifest_path, include_svs=include_svs)
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            corrected._atomic_json(manifest_path, manifest)
            raise


def configured_panel_dir():
    from .common import config
    return Path(config.LAI_MASKED_DIR).expanduser()
