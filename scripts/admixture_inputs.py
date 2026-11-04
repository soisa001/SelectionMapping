"""Shared staging and VCF helpers for the admixture reference analysis pipeline."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, gcs


def work_dir() -> Path:
    path = Path(config.ADMIXTURE_WORK_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_cache() -> Path:
    path = work_dir() / "workspace_inputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_file(remote: str, relative_local: str | os.PathLike, *,
               force: bool = False, required: bool = True) -> Path | None:
    """Stage one bucket object beneath the admixture reference input cache."""
    local = input_cache() / Path(relative_local)
    if local.exists() and not force:
        return local
    if not remote or not gcs.available():
        if required:
            raise FileNotFoundError(
                f"admixture reference input is absent locally and bucket access is unavailable: {remote}"
            )
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[admixture_reference][input] staging {Path(remote).name} -> {local}", flush=True)
    started = time.perf_counter()
    result = gcs.gcs_cp(remote, local)
    if result.returncode != 0:
        if required:
            raise FileNotFoundError(f"Could not stage {remote}: {result.stderr.strip()}")
        return None
    size_mb = local.stat().st_size / (1024 ** 2) if local.exists() else float("nan")
    print(
        f"[admixture_reference][input] staged {Path(remote).name} "
        f"({size_mb:,.1f} MiB in {time.perf_counter() - started:,.1f} s)",
        flush=True,
    )
    return local


def stage_discovery_vcf(*, force: bool = False) -> Path:
    remote = config.ADMIXTURE_DISCOVERY_VCF_GS
    name = Path(remote).name
    local = stage_file(remote, Path("discovery") / name, force=force, required=True)
    assert local is not None
    stage_file(remote + ".tbi", Path("discovery") / (name + ".tbi"),
               force=force, required=True)
    return local


def stage_cohort_sources(label: str, *, force: bool = False) -> dict[str, Path]:
    if label == "discovery":
        paths = {
            "covariates": config.ADMIXTURE_DISCOVERY_COV_GS,
            "pca": config.ADMIXTURE_DISCOVERY_PCA_GS,
            "phenotypes": config.ADMIXTURE_DISCOVERY_PHE_GS,
        }
    elif label == "replication":
        paths = {
            "covariates": config.ADMIXTURE_REPLICATION_COV_GS,
            "pca": config.ADMIXTURE_REPLICATION_PCA_GS,
            "phenotypes": config.ADMIXTURE_REPLICATION_PHE_GS,
        }
    else:
        raise ValueError(f"Unknown admixture reference cohort: {label}")
    outputs = {}
    for kind, remote in paths.items():
        local = stage_file(remote, Path(label) / Path(remote).name,
                           force=force, required=True)
        assert local is not None
        outputs[kind] = local
    return outputs


def normalize_chrom(value) -> str:
    raw = str(value).strip().replace("chr", "")
    return f"chr{int(float(raw))}" if raw.replace(".", "", 1).isdigit() else f"chr{raw}"


def variant_id(chrom, pos, ref, alt) -> str:
    return f"{normalize_chrom(chrom)}:{int(pos)}:{str(ref)}:{str(alt)}"


def genotype_dosage(record) -> np.ndarray:
    """Return float ALT dosage from a cyvcf2 record, with missing calls as NaN."""
    genotypes = np.asarray(record.genotypes)
    if genotypes.ndim != 2 or genotypes.shape[1] < 2:
        raise ValueError(f"Unexpected genotype array at {record.CHROM}:{record.POS}")
    alleles = genotypes[:, :2].astype(float)
    missing = np.any(alleles < 0, axis=1)
    dosage = alleles.sum(axis=1)
    dosage[missing] = np.nan
    return dosage


def detect_vcf_chrom(vcf, chrom: int | str) -> str | None:
    bare = normalize_chrom(chrom).replace("chr", "")
    seqnames = set(vcf.seqnames)
    return next((token for token in (f"chr{bare}", bare) if token in seqnames), None)


def load_replication_shard_index(*, force: bool = False) -> pd.DataFrame:
    """Stage and normalize the ACAF interval-to-VCF shard index."""
    override = os.environ.get("AOU_FIG4_REPLICATION_SHARD_INDEX")
    if override:
        path = Path(override).expanduser()
    else:
        remote = config.ADMIXTURE_REPLICATION_SHARD_INDEX_GS
        path = stage_file(remote, Path("replication") / Path(remote).name,
                          force=force, required=False)
        if path is None:
            json_remote = remote.rsplit(".", 1)[0] + ".json"
            path = stage_file(json_remote,
                              Path("replication") / Path(json_remote).name,
                              force=force, required=True)
    if path is None or not path.exists():
        raise FileNotFoundError(
            "Replication requires an ACAF shard index. Set "
            "AOU_FIG4_REPLICATION_SHARD_INDEX or migrate "
            "phewas_acaf_replication/acaf_shard_index.tsv."
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for chrom, intervals in payload.items():
            for item in intervals:
                if isinstance(item, dict):
                    rows.append(item | {"chrom": item.get("chrom", chrom)})
                else:
                    start, end, vcf_path = item[:3]
                    rows.append({"chrom": chrom, "start": start,
                                 "end": end, "vcf_path": vcf_path})
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path, sep="\t")
    aliases = {
        "path": "vcf_path", "vcf": "vcf_path", "gcs_path": "vcf_path",
        "contig": "chrom", "chr": "chrom", "stop": "end",
    }
    frame = frame.rename(columns={old: new for old, new in aliases.items()
                                  if old in frame.columns and new not in frame.columns})
    required = {"chrom", "start", "end", "vcf_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ACAF shard index missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["chrom"] = frame["chrom"].map(normalize_chrom)
    frame[["start", "end"]] = frame[["start", "end"]].astype(int)
    return frame.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def shards_for_region(index: pd.DataFrame, chrom: int, start: int,
                      end: int) -> list[str]:
    label = normalize_chrom(chrom)
    rows = index[(index["chrom"] == label) &
                 (index["start"] <= int(end)) &
                 (index["end"] >= int(start))]
    return rows["vcf_path"].dropna().astype(str).drop_duplicates().tolist()


def normalize_acaf_remote(path: str) -> str:
    """Rewrite legacy controlled-bucket shard URLs to the current VWB root."""
    value = str(path)
    if not value.startswith("gs://"):
        return f"{config.ADMIXTURE_ACAF_GS_DIR.rstrip('/')}/{value.lstrip('/')}"
    marker = "/v8/wgs/short_read/snpindel/acaf_threshold/vcf/"
    if marker in value:
        suffix = value.split(marker, 1)[1]
        return f"{config.ADMIXTURE_ACAF_GS_DIR.rstrip('/')}/{suffix}"
    return value


def stage_acaf_shard(remote: str, *, force: bool = False) -> Path:
    normalized = normalize_acaf_remote(remote)
    local = stage_file(normalized, Path("replication") / "acaf" / Path(normalized).name,
                       force=force, required=True)
    assert local is not None
    index_candidates = [normalized + ".tbi"]
    if normalized.endswith(".vcf.bgz"):
        index_candidates.append(normalized[:-4] + ".tbi")
    staged_index = False
    for index_remote in index_candidates:
        index_local_name = Path(normalized).name + ".tbi"
        result = stage_file(index_remote,
                            Path("replication") / "acaf" / index_local_name,
                            force=force, required=False)
        if result is not None:
            staged_index = True
            break
    if not staged_index and not Path(str(local) + ".tbi").exists():
        raise FileNotFoundError(f"No tabix index could be staged for {normalized}")
    return local


def atomic_pickle_dump(payload, path: Path) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def pickle_load(path: Path):
    import pickle
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if str(exc.name).startswith("numpy._core"):
                raise RuntimeError(
                    "This cached admixture reference model was created with NumPy 2, but "
                    "the active kernel is running NumPy 1. Restore it with "
                    "`%pip install --upgrade --force-reinstall numpy==2.0.2`, "
                    "restart the kernel, and rerun the notebook. PheTK is now "
                    "installed in an isolated environment and will not "
                    "downgrade the kernel again."
                ) from exc
            raise


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
