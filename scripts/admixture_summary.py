"""Post-analysis diagnostics and focused annotation lookups for admixture reference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    admixture_data,
    admixture_inputs,
    admixture_lai,
    admixture_phewas,
    admixture_plrs,
)
from .common import config, style

MAGE_LINK_R2 = 0.5
QTL_CONTRACT = "figure4_min_p_mage_linked_r2_ge_0.5_v1"
QTL_COLUMNS = [
    "source", "variant_id", "is_minimum_p_variant", "r2_to_minimum_p_variant",
    "gene", "p_value", "q_value",
]
GWAS_P_THRESHOLD = 5e-8
GWAS_CONTRACT = "figure4_min_p_gwas_catalog_linked_r2_ge_0.5_p_le_5e-8_v1"
GWAS_COLUMNS = [
    "source", "linked_variant_id", "is_minimum_p_position",
    "r2_to_minimum_p_variant", "match_basis", "catalog_variant", "trait",
    "mapped_gene", "p_value", "study_accession", "pubmed_id",
    "gwas_catalog_url", "pubmed_url",
]
META_SUMMARY_COLUMNS = [
    "variant_tag", "chrom", "start", "end", "direction", "peak_z",
    "genes_in_region", "phecode", "phecode_string", "beta_disc", "se_disc",
    "p_disc", "n_disc", "beta_rep", "se_rep", "p_rep", "n_rep",
    "beta_meta", "se_meta", "p_meta", "same_direction", "rep_nominal",
    "meta_significant", "replicated_full", "disc_hit_threshold",
    "disc_hit_kind",
]


def analysis_dir() -> Path:
    path = admixture_lai.analysis_dir().parent / "summary"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_variant(chrom, pos, ref, alt) -> tuple[str, int, str, str]:
    chrom = admixture_inputs.normalize_chrom(chrom)
    pos = int(pos)
    ref, alt = str(ref).upper(), str(alt).upper()
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return chrom, pos, ref, alt


def _parse_variant(value: str) -> tuple[str, int, str, str] | None:
    parts = str(value).split(":", 3)
    if len(parts) != 4:
        return None
    try:
        return _normalize_variant(parts[0], parts[1], parts[2], parts[3])
    except (TypeError, ValueError):
        return None


def _linked_variants(frame: pd.DataFrame) -> tuple[str | None, dict]:
    if frame.empty or "variant_id" not in frame.columns:
        return None, {}
    minimum = frame.get("minimum_p_variant", pd.Series(False, index=frame.index))
    minimum = minimum.astype(str).str.lower().isin({"true", "1", "1.0"})
    if minimum.any():
        minimum_id = str(frame.loc[minimum, "variant_id"].iloc[0])
    elif "p_score" in frame.columns:
        p_value = pd.to_numeric(frame["p_score"], errors="coerce")
        minimum_id = str(frame.loc[p_value.idxmin(), "variant_id"])
    else:
        return None, {}
    r2 = pd.to_numeric(
        frame.get("r2_to_min_p", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    linked = {}
    for variant_id, value in zip(frame["variant_id"].astype(str), r2):
        key = _parse_variant(variant_id)
        if key is not None and np.isfinite(value) and value >= MAGE_LINK_R2:
            linked[key] = max(float(value), linked.get(key, -np.inf))
    minimum_key = _parse_variant(minimum_id)
    if minimum_key is not None:
        linked[minimum_key] = 1.0
    return minimum_id, linked


def _first_column(columns, candidates) -> str | None:
    lower = {str(column).lower(): column for column in columns}
    return next((lower[name.lower()] for name in candidates if name.lower() in lower), None)


def _scan_mage(path: Path | None, source: str, linked: dict,
               minimum_id: str | None) -> tuple[pd.DataFrame, dict]:
    status = {
        "source": source, "available": path is not None,
        "n_linked_associations": 0, "n_direct_associations": 0,
    }
    if path is None or not path.exists() or not linked:
        return pd.DataFrame(columns=QTL_COLUMNS), status
    header = pd.read_csv(path, sep="\t", nrows=0)
    aliases = {
        "chrom": _first_column(header.columns, ["variantChrom", "chrom", "chr"]),
        "pos": _first_column(header.columns, ["variantPosition", "pos", "position"]),
        "ref": _first_column(header.columns, ["variantRef", "ref"]),
        "alt": _first_column(header.columns, ["variantAlt", "alt"]),
        "gene": _first_column(header.columns, [
            "geneSymbol", "geneName", "geneId", "molecularTraitId", "phenotypeId",
        ]),
        "p": _first_column(header.columns, ["pval", "pvalue", "p_value", "p"]),
        "q": _first_column(header.columns, ["qval", "qvalue", "q_value", "q"]),
    }
    if any(aliases[name] is None for name in ("chrom", "pos", "ref", "alt")):
        status["available"] = False
        status["reason"] = "required variant columns absent"
        return pd.DataFrame(columns=QTL_COLUMNS), status
    linked_chroms = {key[0] for key in linked}
    linked_positions = {key[1] for key in linked}
    position_min, position_max = min(linked_positions), max(linked_positions)
    minimum_key = _parse_variant(minimum_id) if minimum_id else None
    rows = []
    for chunk in pd.read_csv(path, sep="\t", low_memory=False, chunksize=250_000):
        position = pd.to_numeric(chunk[aliases["pos"]], errors="coerce")
        chrom = chunk[aliases["chrom"]].map(admixture_inputs.normalize_chrom)
        candidate = chunk[
            chrom.isin(linked_chroms)
            & position.between(position_min - 50, position_max + 50)
        ].copy()
        if candidate.empty:
            continue
        if source == "MAGE eQTL" and aliases["q"] is not None:
            candidate = candidate[
                pd.to_numeric(candidate[aliases["q"]], errors="coerce") < 0.05
            ]
        for record in candidate.to_dict("records"):
            try:
                key = _normalize_variant(
                    record[aliases["chrom"]], record[aliases["pos"]],
                    record[aliases["ref"]], record[aliases["alt"]],
                )
            except (TypeError, ValueError):
                continue
            if key not in linked:
                continue
            variant_id = f"{key[0]}:{key[1]}:{key[2]}:{key[3]}"
            rows.append({
                "source": source,
                "variant_id": variant_id,
                "is_minimum_p_variant": key == minimum_key,
                "r2_to_minimum_p_variant": linked[key],
                "gene": record.get(aliases["gene"], "") if aliases["gene"] else "",
                "p_value": record.get(aliases["p"], np.nan) if aliases["p"] else np.nan,
                "q_value": record.get(aliases["q"], np.nan) if aliases["q"] else np.nan,
            })
    matches = pd.DataFrame(rows, columns=QTL_COLUMNS).drop_duplicates()
    status["n_linked_associations"] = int(len(matches))
    status["n_direct_associations"] = int(
        matches.get("is_minimum_p_variant", pd.Series(dtype=bool)).sum()
    )
    return matches, status


def _mage_lookup(frame: pd.DataFrame, *, force: bool = False
                 ) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    minimum_id, linked = _linked_variants(frame)
    output = analysis_dir() / "minimum_p_variant_mage_associations.tsv"
    status_path = analysis_dir() / "minimum_p_variant_mage_status.tsv"
    manifest_path = analysis_dir() / "minimum_p_variant_mage_manifest.json"
    contract = {
        "contract": QTL_CONTRACT, "minimum_p_variant": minimum_id,
        "n_linked_variants": len(linked), "r2_threshold": MAGE_LINK_R2,
    }
    if not force and output.exists() and status_path.exists() and manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == contract:
                cached_status = pd.read_csv(status_path, sep="\t")
                available = cached_status["available"].astype(str).str.lower().isin(
                    {"true", "1", "1.0"}
                )
                if available.all():
                    return pd.read_csv(
                        output, sep="\t", keep_default_na=False
                    ), cached_status, minimum_id
        except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
            pass
    if not linked:
        result = pd.DataFrame(columns=QTL_COLUMNS)
        status = pd.DataFrame([
            {
                "source": source, "available": False,
                "n_linked_associations": 0, "n_direct_associations": 0,
                "reason": "no current panel D fine-map cache",
            }
            for source in ("MAGE eQTL", "MAGE sQTL")
        ])
        result.to_csv(output, sep="\t", index=False)
        status.to_csv(status_path, sep="\t", index=False)
        manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        return result, status, minimum_id
    sources = [
        ("MAGE eQTL", config.ADMIXTURE_MAGE_EQTL_GS, "mage_eQTL.txt.gz"),
        ("MAGE sQTL", config.ADMIXTURE_MAGE_SQTL_GS, "mage_sQTL.txt.gz"),
    ]
    matches, statuses = [], []
    for label, remote, filename in sources:
        local = admixture_inputs.stage_file(
            remote, Path("annotation") / filename, force=force, required=False,
        )
        found, status = _scan_mage(local, label, linked, minimum_id)
        matches.append(found)
        statuses.append(status)
    result = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
        columns=QTL_COLUMNS
    )
    status = pd.DataFrame(statuses)
    result.to_csv(output, sep="\t", index=False)
    status.to_csv(status_path, sep="\t", index=False)
    manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return result, status, minimum_id


def _linked_positions(linked: dict) -> dict[tuple[str, int], tuple[str, float]]:
    """Collapse allele-aware fine-map LD onto GWAS Catalog positions."""
    positions: dict[tuple[str, int], tuple[str, float]] = {}
    for (chrom, pos, ref, alt), r2 in linked.items():
        key = (chrom, pos)
        if key not in positions or r2 > positions[key][1]:
            positions[key] = (f"{chrom}:{pos}:{ref}:{alt}", float(r2))
    return positions


def _scan_gwas(path: Path | None, linked: dict,
               minimum_id: str | None) -> tuple[pd.DataFrame, dict]:
    """Find genome-wide-significant Catalog rows at panel D-linked positions.

    The downloaded GWAS Catalog association table does not consistently expose
    reference and alternate alleles. Matching is therefore explicitly by
    normalized chromosome and position; the fine-map LD still determines which
    positions are eligible.
    """
    status = {
        "source": "GWAS Catalog", "available": path is not None,
        "n_linked_associations": 0, "n_direct_associations": 0,
        "match_basis": "chromosome_position",
        "p_value_threshold": GWAS_P_THRESHOLD,
    }
    linked_positions = _linked_positions(linked)
    if path is None or not path.exists() or not linked_positions:
        return pd.DataFrame(columns=GWAS_COLUMNS), status

    header = pd.read_csv(path, sep="\t", nrows=0)
    aliases = {
        "chrom": _first_column(header.columns, ["CHR_ID", "chrom", "chr"]),
        "pos": _first_column(
            header.columns, ["CHR_POS", "position", "pos", "base_pair_location"]
        ),
        "p": _first_column(
            header.columns, ["P-VALUE", "p_value", "pvalue", "pval", "p"]
        ),
        "variant": _first_column(header.columns, [
            "SNPS", "SNP_ID_CURRENT", "STRONGEST SNP-RISK ALLELE", "rsid",
        ]),
        "trait": _first_column(header.columns, [
            "DISEASE/TRAIT", "MAPPED_TRAIT", "trait", "phenotype",
        ]),
        "gene": _first_column(header.columns, [
            "MAPPED_GENE", "REPORTED GENE(S)", "gene", "mapped_gene",
        ]),
        "study": _first_column(
            header.columns, ["STUDY ACCESSION", "study_accession"]
        ),
        "pubmed": _first_column(header.columns, ["PUBMEDID", "pubmed_id"]),
    }
    if any(aliases[name] is None for name in ("chrom", "pos", "p")):
        status["available"] = False
        status["reason"] = "required chromosome, position, or p-value columns absent"
        return pd.DataFrame(columns=GWAS_COLUMNS), status

    linked_chroms = {key[0] for key in linked_positions}
    linked_bp = {key[1] for key in linked_positions}
    position_min, position_max = min(linked_bp), max(linked_bp)
    minimum_key = _parse_variant(minimum_id) if minimum_id else None
    minimum_position = minimum_key[:2] if minimum_key is not None else None
    usecols = list(dict.fromkeys(column for column in aliases.values() if column))
    rows = []
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, low_memory=False, usecols=usecols,
        chunksize=250_000,
    ):
        position = pd.to_numeric(chunk[aliases["pos"]], errors="coerce")
        p_value = pd.to_numeric(chunk[aliases["p"]], errors="coerce")
        chrom = chunk[aliases["chrom"]].map(admixture_inputs.normalize_chrom)
        keep = (
            chrom.isin(linked_chroms)
            & position.between(position_min, position_max)
            & p_value.le(GWAS_P_THRESHOLD)
        )
        candidate = chunk[keep].copy()
        if candidate.empty:
            continue
        candidate["_normalized_chrom"] = chrom[keep]
        candidate["_normalized_pos"] = position[keep].astype(int)
        candidate["_numeric_p"] = p_value[keep]
        for record in candidate.to_dict("records"):
            position_key = (
                str(record["_normalized_chrom"]), int(record["_normalized_pos"])
            )
            linked_value = linked_positions.get(position_key)
            if linked_value is None:
                continue
            linked_variant_id, r2 = linked_value
            study = str(record.get(aliases["study"], "")).strip() \
                if aliases["study"] else ""
            pubmed = str(record.get(aliases["pubmed"], "")).strip() \
                if aliases["pubmed"] else ""
            if study.lower() == "nan":
                study = ""
            if pubmed.lower() == "nan":
                pubmed = ""
            rows.append({
                "source": "GWAS Catalog",
                "linked_variant_id": linked_variant_id,
                "is_minimum_p_position": position_key == minimum_position,
                "r2_to_minimum_p_variant": r2,
                "match_basis": "chromosome_position",
                "catalog_variant": record.get(aliases["variant"], "")
                if aliases["variant"] else "",
                "trait": record.get(aliases["trait"], "")
                if aliases["trait"] else "",
                "mapped_gene": record.get(aliases["gene"], "")
                if aliases["gene"] else "",
                "p_value": record["_numeric_p"],
                "study_accession": study,
                "pubmed_id": pubmed,
                "gwas_catalog_url": (
                    f"https://www.ebi.ac.uk/gwas/studies/{study}" if study else ""
                ),
                "pubmed_url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pubmed}/" if pubmed else ""
                ),
            })
    matches = pd.DataFrame(rows, columns=GWAS_COLUMNS).drop_duplicates()
    status["n_linked_associations"] = int(len(matches))
    status["n_direct_associations"] = int(
        matches.get("is_minimum_p_position", pd.Series(dtype=bool)).sum()
    )
    return matches, status


def _gwas_lookup(frame: pd.DataFrame, *, force: bool = False
                 ) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    minimum_id, linked = _linked_variants(frame)
    output = analysis_dir() / "minimum_p_variant_gwas_associations.tsv"
    status_path = analysis_dir() / "minimum_p_variant_gwas_status.tsv"
    manifest_path = analysis_dir() / "minimum_p_variant_gwas_manifest.json"
    contract = {
        "contract": GWAS_CONTRACT, "minimum_p_variant": minimum_id,
        "n_linked_variants": len(linked), "r2_threshold": MAGE_LINK_R2,
        "p_value_threshold": GWAS_P_THRESHOLD,
    }
    if not force and output.exists() and status_path.exists() and manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == contract:
                cached_status = pd.read_csv(status_path, sep="\t")
                available = cached_status["available"].astype(str).str.lower().isin(
                    {"true", "1", "1.0"}
                )
                if available.all():
                    return pd.read_csv(
                        output, sep="\t", keep_default_na=False
                    ), cached_status, minimum_id
        except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
            pass
    if not linked:
        result = pd.DataFrame(columns=GWAS_COLUMNS)
        status = pd.DataFrame([{
            "source": "GWAS Catalog", "available": False,
            "n_linked_associations": 0, "n_direct_associations": 0,
            "match_basis": "chromosome_position",
            "p_value_threshold": GWAS_P_THRESHOLD,
            "reason": "no current panel D fine-map cache",
        }])
    else:
        local = admixture_inputs.stage_file(
            config.ADMIXTURE_GWAS_GS,
            Path("annotation") / "gwas_catalog_associations.tsv",
            force=force,
            required=False,
        )
        found, scan_status = _scan_gwas(local, linked, minimum_id)
        result = found
        status = pd.DataFrame([scan_status])
    result.to_csv(output, sep="\t", index=False)
    status.to_csv(status_path, sep="\t", index=False)
    manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return result, status, minimum_id


def _read_optional_table(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", **kwargs)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _phecode_string_lookup(meta: pd.DataFrame) -> dict[str, str]:
    """Recover PhecodeX descriptions from meta or raw discovery PheTK caches."""
    pieces = []
    description = _first_column(meta.columns, [
        "phecode_string", "phecode_description", "description", "meaning",
    ])
    if "phecode" in meta.columns and description is not None:
        pieces.append(meta[["phecode", description]].rename(
            columns={description: "phecode_string"}
        ))
    raw_dir = admixture_phewas.analysis_dir() / "phetk_runs" / "discovery"
    for path in sorted(raw_dir.glob("*.phewas.tsv")):
        try:
            header = pd.read_csv(path, sep="\t", nrows=0)
        except (OSError, pd.errors.EmptyDataError):
            continue
        phecode = _first_column(header.columns, ["phecode"])
        description = _first_column(header.columns, [
            "phecode_string", "phecode_description", "description", "meaning",
        ])
        if phecode is None or description is None:
            continue
        try:
            local = pd.read_csv(
                path, sep="\t", usecols=[phecode, description], dtype=str
            ).rename(columns={phecode: "phecode", description: "phecode_string"})
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        pieces.append(local)
    if not pieces:
        return {}
    descriptions = pd.concat(pieces, ignore_index=True)
    descriptions["phecode"] = descriptions["phecode"].astype(str).str.strip()
    descriptions["phecode_string"] = (
        descriptions["phecode_string"].fillna("").astype(str).str.strip()
    )
    descriptions = descriptions[
        descriptions["phecode"].ne("")
        & descriptions["phecode_string"].ne("")
        & descriptions["phecode_string"].str.lower().ne("nan")
    ].drop_duplicates("phecode")
    return dict(zip(descriptions["phecode"], descriptions["phecode_string"]))


def _genes_in_regions(regions: pd.DataFrame,
                      genes: pd.DataFrame) -> dict[str, str]:
    if regions.empty or genes.empty:
        return {}
    local_genes = genes.copy()
    required = {"chrom", "start", "end", "gene"}
    if not required.issubset(local_genes.columns):
        return {}
    local_genes["chrom"] = local_genes["chrom"].map(
        admixture_inputs.normalize_chrom
    )
    local_genes["start"] = pd.to_numeric(local_genes["start"], errors="coerce")
    local_genes["end"] = pd.to_numeric(local_genes["end"], errors="coerce")
    local_genes = local_genes.dropna(subset=["start", "end", "gene"])
    identifier = "region_id" if "region_id" in regions.columns else "variant_tag"
    result = {}
    for region in regions.itertuples(index=False):
        chrom = admixture_inputs.normalize_chrom(getattr(region, "chrom"))
        start, end = int(getattr(region, "start")), int(getattr(region, "end"))
        overlapping = local_genes[
            local_genes["chrom"].eq(chrom)
            & local_genes["start"].le(end)
            & local_genes["end"].ge(start)
        ].sort_values(["start", "end", "gene"])
        names = overlapping["gene"].astype(str).drop_duplicates().tolist()
        result[str(getattr(region, identifier))] = ",".join(names)
    return result


def _values(frame: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    column = _first_column(frame.columns, candidates)
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column]


def _boolean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _meta_summary_table(meta: pd.DataFrame, regions: pd.DataFrame,
                        genes: pd.DataFrame,
                        phecode_strings: dict[str, str]) -> pd.DataFrame:
    """Return the requested one-row-per-discovery-hit association table."""
    if meta.empty:
        return pd.DataFrame(columns=META_SUMMARY_COLUMNS)
    frame = meta.copy()
    region_key = _first_column(frame.columns, ["variant_tag", "region_id"])
    if region_key is None:
        return pd.DataFrame(columns=META_SUMMARY_COLUMNS)
    region_metadata = regions.copy()
    metadata_key = _first_column(region_metadata.columns, ["region_id", "variant_tag"])
    if metadata_key is not None:
        missing = [
            column for column in ("chrom", "start", "end", "direction", "peak_z")
            if column not in frame.columns and column in region_metadata.columns
        ]
        if missing:
            frame = frame.merge(
                region_metadata[[metadata_key, *missing]].rename(
                    columns={metadata_key: region_key}
                ),
                on=region_key,
                how="left",
            )
    region_ids = frame[region_key].astype(str)
    phecodes = _values(frame, ["phecode"], "").astype(str)
    gene_lookup = _genes_in_regions(regions, genes)
    existing_genes = _values(frame, ["genes_in_region"], "").fillna("").astype(str)
    annotated_genes = region_ids.map(gene_lookup).fillna("")
    genes_in_region = existing_genes.where(existing_genes.str.strip().ne(""), annotated_genes)
    existing_descriptions = _values(
        frame,
        ["phecode_string", "phecode_description", "description", "meaning"],
        "",
    ).fillna("").astype(str)
    recovered_descriptions = phecodes.map(phecode_strings).fillna("")
    descriptions = existing_descriptions.where(
        existing_descriptions.str.strip().ne(""), recovered_descriptions
    )

    beta_disc = pd.to_numeric(
        _values(frame, ["beta_disc", "beta_discovery"]), errors="coerce"
    )
    beta_rep = pd.to_numeric(
        _values(frame, ["beta_rep", "beta_replication"]), errors="coerce"
    )
    p_rep = pd.to_numeric(
        _values(frame, ["p_rep", "p_replication"]), errors="coerce"
    )
    same_direction_raw = _values(frame, ["same_direction"], np.nan)
    same_direction = _boolean(same_direction_raw)
    missing_direction = same_direction_raw.isna()
    same_direction.loc[missing_direction] = (
        np.sign(beta_disc[missing_direction]) == np.sign(beta_rep[missing_direction])
    ) & beta_rep[missing_direction].notna()
    rep_nominal_raw = _values(frame, ["rep_nominal"], np.nan)
    rep_nominal = _boolean(rep_nominal_raw)
    rep_nominal.loc[rep_nominal_raw.isna()] = p_rep[rep_nominal_raw.isna()] < 0.05
    meta_significant = _boolean(_values(frame, ["meta_significant"], False))
    replicated_full_raw = _values(
        frame, ["replicated_full", "replicated_meta_significant"], np.nan
    )
    replicated_full = _boolean(replicated_full_raw)
    replicated_full.loc[replicated_full_raw.isna()] = (
        same_direction[replicated_full_raw.isna()]
        & rep_nominal[replicated_full_raw.isna()]
        & meta_significant[replicated_full_raw.isna()]
    )
    thresholds = pd.to_numeric(
        _values(
            frame,
            ["disc_hit_threshold", "discovery_alpha"],
            admixture_phewas.DISCOVERY_P_THRESHOLD,
        ),
        errors="coerce",
    ).fillna(admixture_phewas.DISCOVERY_P_THRESHOLD)
    hit_kind = _values(frame, ["disc_hit_kind"], "").fillna("").astype(str)
    hit_kind = hit_kind.where(
        hit_kind.str.strip().ne(""),
        "fallback_1e-3",
    )

    output = pd.DataFrame({
        "variant_tag": region_ids,
        "chrom": _values(frame, ["chrom"], "").map(
            admixture_inputs.normalize_chrom
        ),
        "start": pd.to_numeric(_values(frame, ["start"]), errors="coerce"),
        "end": pd.to_numeric(_values(frame, ["end"]), errors="coerce"),
        "direction": _values(frame, ["direction"], ""),
        "peak_z": pd.to_numeric(_values(frame, ["peak_z"]), errors="coerce"),
        "genes_in_region": genes_in_region,
        "phecode": phecodes,
        "phecode_string": descriptions,
        "beta_disc": beta_disc,
        "se_disc": pd.to_numeric(
            _values(frame, ["se_disc", "se_discovery"]), errors="coerce"
        ),
        "p_disc": pd.to_numeric(
            _values(frame, ["p_disc", "p_discovery"]), errors="coerce"
        ),
        "n_disc": pd.to_numeric(
            _values(frame, ["n_disc", "n_discovery"]), errors="coerce"
        ),
        "beta_rep": beta_rep,
        "se_rep": pd.to_numeric(
            _values(frame, ["se_rep", "se_replication"]), errors="coerce"
        ),
        "p_rep": p_rep,
        "n_rep": pd.to_numeric(
            _values(frame, ["n_rep", "n_replication"]), errors="coerce"
        ),
        "beta_meta": pd.to_numeric(_values(frame, ["beta_meta"]), errors="coerce"),
        "se_meta": pd.to_numeric(_values(frame, ["se_meta"]), errors="coerce"),
        "p_meta": pd.to_numeric(_values(frame, ["p_meta"]), errors="coerce"),
        "same_direction": same_direction,
        "rep_nominal": rep_nominal,
        "meta_significant": meta_significant,
        "replicated_full": replicated_full,
        "disc_hit_threshold": thresholds,
        "disc_hit_kind": hit_kind,
    })
    return output[META_SUMMARY_COLUMNS].sort_values(
        "p_meta", na_position="last"
    ).reset_index(drop=True)


def _summary_statistics(regions: pd.DataFrame, models: pd.DataFrame,
                        discovery: pd.DataFrame, meta: pd.DataFrame,
                        qtl_status: pd.DataFrame, gwas_status: pd.DataFrame,
                        minimum_id: str | None
                        ) -> pd.DataFrame:
    lengths = pd.to_numeric(regions["end"], errors="coerce") - pd.to_numeric(
        regions["start"], errors="coerce"
    ) + 1
    r2 = pd.to_numeric(models["r2_holdout"], errors="coerce")
    retained = r2 > admixture_plrs.R2_MIN
    qtl_counts = dict(zip(
        qtl_status.get("source", pd.Series(dtype=str)),
        qtl_status.get("n_linked_associations", pd.Series(dtype=int)),
    ))
    gwas_linked = pd.to_numeric(
        gwas_status.get("n_linked_associations", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
    gwas_direct = pd.to_numeric(
        gwas_status.get("n_direct_associations", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
    discovery_phecodes = (
        discovery["phecode"].astype(str).nunique()
        if "phecode" in discovery.columns else 0
    )
    discovery_significant = (
        pd.to_numeric(discovery["p"], errors="coerce")
        < admixture_phewas.DISCOVERY_P_THRESHOLD
        if "p" in discovery.columns else pd.Series(False, index=discovery.index)
    )
    meta_significant = _boolean(
        meta.get("meta_significant", pd.Series(False, index=meta.index))
    )
    values = [
        ("outlier_regions", int(len(regions))),
        ("outlier_length_min_bp", int(lengths.min())),
        ("outlier_length_median_bp", float(lengths.median())),
        ("outlier_length_max_bp", int(lengths.max())),
        ("plrs_models_fitted", int(len(models))),
        ("plrs_models_r2_gt_0.5", int(retained.sum())),
        ("plrs_models_r2_gt_0.5_percent", float(retained.mean() * 100)),
        ("discovery_unique_phecodes_passing_filters", int(discovery_phecodes)),
        ("discovery_region_phecode_tests_passing_filters", int(len(discovery))),
        ("discovery_significant_region_phecode_pairs", int(discovery_significant.sum())),
        ("meta_analyzed_associations", int(len(meta))),
        ("meta_significant_associations", int(meta_significant.sum())),
        ("meta_significant_unique_phecodes", int(
            meta.loc[meta_significant, "phecode"].astype(str).nunique()
            if "phecode" in meta.columns else 0
        )),
        ("minimum_p_variant", minimum_id or "unavailable"),
        ("mage_eqtl_linked_associations", int(qtl_counts.get("MAGE eQTL", 0))),
        ("mage_sqtl_linked_associations", int(qtl_counts.get("MAGE sQTL", 0))),
        ("gwas_catalog_linked_associations", int(gwas_linked)),
        ("gwas_catalog_direct_associations", int(gwas_direct)),
    ]
    return pd.DataFrame(values, columns=["statistic", "value"])


def _model_plot_values(
    models: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return finite model sizes, held-out R2 values, and retention flags."""
    n_snvs = pd.to_numeric(models["n_snvs"], errors="coerce")
    r2 = pd.to_numeric(models["r2_holdout"], errors="coerce")
    finite = n_snvs.notna() & r2.notna()
    n_snvs, r2 = n_snvs[finite], r2[finite]
    retained = r2 > admixture_plrs.R2_MIN
    return n_snvs, r2, retained


def panel_plrs_r2_distribution(
    ax,
    models: pd.DataFrame,
    *,
    show_title: bool = False,
) -> None:
    """Draw the held-out PLRS R2 distribution for all regional models."""
    _, r2, _ = _model_plot_values(models)
    bins = min(24, max(6, int(np.sqrt(max(len(r2), 1)))))
    ax.hist(
        r2,
        bins=bins,
        color=style.PALETTE["blue"],
        edgecolor="white",
    )
    ax.axvline(
        admixture_plrs.R2_MIN,
        color=style.PALETTE["vermillion"],
        ls="--",
        lw=1.1,
    )
    ax.set_xlabel("Held-out PLRS $R^2$")
    ax.set_ylabel("Number of regions")
    if show_title:
        ax.set_title("PLRS $R^2$ distribution", fontsize=11)


def panel_plrs_model_size(
    ax,
    models: pd.DataFrame,
    *,
    show_title: bool = False,
) -> None:
    """Draw regional model size against held-out PLRS R2."""
    n_snvs, r2, retained = _model_plot_values(models)
    ax.scatter(
        n_snvs[~retained],
        r2[~retained],
        s=15,
        color="0.65",
        alpha=0.65,
        edgecolor="none",
        label=r"$R^2 \leq 0.5$",
    )
    ax.scatter(
        n_snvs[retained],
        r2[retained],
        s=19,
        color=style.PALETTE["blue"],
        alpha=0.8,
        edgecolor="none",
        label=f"$R^2 > 0.5$ (n={int(retained.sum())})",
    )
    ax.axhline(
        admixture_plrs.R2_MIN,
        color=style.PALETTE["vermillion"],
        ls="--",
        lw=1.0,
    )
    ax.set_xlabel("SNVs in regional PLRS model")
    ax.set_ylabel("Held-out PLRS $R^2$")
    if show_title:
        ax.set_title("Model size and held-out performance", fontsize=11)
    if len(r2) > 1 and n_snvs.nunique() > 1:
        correlation = float(np.corrcoef(n_snvs, r2)[0, 1])
        ax.text(
            0.02,
            0.98,
            f"Pearson r={correlation:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    ax.legend(loc="lower right", fontsize=7)


def _plot_models(models: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    panel_plrs_r2_distribution(axes[0], models, show_title=True)
    panel_plrs_model_size(axes[1], models, show_title=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(*, force: bool = False) -> dict:
    """Build summary plots and focused minimum-p MAGE/GWAS lookups."""
    regions = pd.read_csv(admixture_lai.analysis_dir() / "outlier_regions.tsv", sep="\t")
    models = pd.read_csv(admixture_plrs.analysis_dir() / "plrs_models_summary.tsv", sep="\t")
    discovery = _read_optional_table(
        admixture_phewas.analysis_dir() / "phewas.discovery.tsv.gz",
        dtype={"phecode": str, "region_id": str},
    )
    meta = _read_optional_table(
        admixture_phewas.analysis_dir() / "meta_analysis_results.tsv",
        dtype={"phecode": str, "region_id": str},
    )
    data = admixture_data.build()
    qtl, qtl_status, minimum_id = _mage_lookup(data["finemap"], force=force)
    gwas, gwas_status, gwas_minimum_id = _gwas_lookup(
        data["finemap"], force=force
    )
    minimum_id = minimum_id or gwas_minimum_id
    statistics = _summary_statistics(
        regions, models, discovery, meta, qtl_status, gwas_status, minimum_id
    )
    statistics_path = analysis_dir() / "summary_statistics.tsv"
    statistics.to_csv(statistics_path, sep="\t", index=False)
    meta_table = _meta_summary_table(
        meta,
        regions,
        data.get("finemap_genes", pd.DataFrame()),
        _phecode_string_lookup(meta),
    )
    meta_table_path = analysis_dir() / "discovery_replication_meta_summary.tsv"
    meta_table.to_csv(meta_table_path, sep="\t", index=False)
    plot_path = config.figure_dir("admixture_reference") / "Admixture_reference_summary_plrs.png"
    _plot_models(models, plot_path)
    return {
        "statistics": statistics,
        "phewas_meta_table": meta_table,
        "qtl_associations": qtl,
        "qtl_status": qtl_status,
        "gwas_associations": gwas,
        "gwas_status": gwas_status,
        "statistics_path": statistics_path,
        "phewas_meta_path": meta_table_path,
        "plot_path": plot_path,
        "qtl_path": analysis_dir() / "minimum_p_variant_mage_associations.tsv",
        "gwas_path": analysis_dir() / "minimum_p_variant_gwas_associations.tsv",
    }
