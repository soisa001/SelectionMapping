"""Figure 3 data stage -- genome-wide selection scan and candidate loci.

Caches intermediates under the established ``analysis/figure2/`` source path.

SAMPLE SET: the selection scan uses the canonical 1,025-sample unrelated AoU
analysis set, with exclusions defined by the controlled cohort manifest.

Data sources:
    - iHS: selscan v2.1 on the 1,025-sample AoU panel, normalized in 100
      allele-frequency bins; max |iHS| per 20 kb window.
    - sim p-values: 1,000 msprime coalescent sims under the inferred demography.
    - EHH / bifurcation: rehh at candidate loci (LCT shown).
    - pi: windowed nucleotide diversity at the locus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import config
from .common.genome import CHROM_SIZES

FIGURE_ID = "figure3"


def _adir():
    return config.analysis_dir(FIGURE_ID)


# Genes highlighted on the Figure 3A iHS Manhattan (labelled at their peak window).
SCAN_ANNOTATE_GENES = ("ANKRD31", "SLC24A5", "LCT", "RBFOX1", "ZNF74")
SCAN_SIG_PVAL = 0.01           # candidate windows: simulation p <= 0.01
SCAN_SLOT_WIDTH = 1_000_000    # fixed visual width per chromosome (position slots)
SV_HIT_EXPECTED_ROWS = 17
SV_HIT_EXPECTED_UNIQUE_ALLELES = 14
SV_LINK_MIN_R2 = 0.5
SV_NEAR_GENE_DISTANCE_SOURCE = (
    "UCSC hg38 RefGene GTF; "
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/"
    "hg38.refGene.gtf.gz"
)
_NEAR_GENE_CONSOLE_EMITTED = False

_SV_HIT_COLUMNS = (
    "selection_chrom",
    "region_start",
    "region_stop",
    "selection_peak_position",
    "selection_peak_ihs",
    "selection_window_gene",
    "sv_position",
    "sv_type",
    "sv_length_bp",
    "sv_ld_r2",
    "sv_relation",
    "sv_gene_label",
    "sv_gene_context",
    "sv_near_gene_distances_bp",
)


def _scan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw iHS-window TSV columns to the names the Figure 3A plot expects.

    Raw: chr, pos, max_IHS_value, sim_pval, gene_in_window (+ more). We keep the
    signed max iHS (y-axis), the simulation p-value, the position and the gene
    annotation, and add the fixed-width per-chromosome x layout used by the plot.
    """
    out = pd.DataFrame({
        "chrom_num": df["chr"].astype(str).str.replace("chr", "", regex=False).astype(int),
        "position": df["pos"].astype(int),
        "max_emp_ihs": df["max_IHS_value"].astype(float),
        "pval": df["sim_pval"].astype(float),
        "gene": df["gene_in_window"].astype("string"),
    })
    out = out.dropna(subset=["chrom_num", "position", "max_emp_ihs", "pval"])
    out = out.sort_values(["chrom_num", "position"]).reset_index(drop=True)

    # Normalise each chromosome's positions into a fixed-width visual slot so the
    # 22 autosomes lay out evenly left-to-right (matches the manuscript panel).
    cumulative = np.zeros(len(out))
    for chrom, sub in out.groupby("chrom_num"):
        pos = sub["position"].to_numpy(float)
        span = max(pos.max() - pos.min(), 1)
        cumulative[sub.index] = (chrom - 1) * SCAN_SLOT_WIDTH + (pos - pos.min()) / span * SCAN_SLOT_WIDTH
    out["cumulative_pos"] = cumulative
    out["candidate"] = out["pval"] <= SCAN_SIG_PVAL
    return out


def _build_scan() -> pd.DataFrame:
    """Load the per-window iHS selection scan for Figure 3A."""
    raw = pd.read_csv(config.IHS_WINDOWS_TSV, sep="\t", compression="infer")
    return _scan_columns(raw)


def _chrom_pos_ranges(scan: pd.DataFrame) -> dict:
    """Per-chromosome (pos_min, span) used for the fixed-width x layout."""
    ranges = {}
    for chrom, sub in scan.groupby("chrom_num"):
        pos = sub["position"].to_numpy(float)
        ranges[int(chrom)] = (pos.min(), max(pos.max() - pos.min(), 1))
    return ranges


def _to_cumulative(chrom_num: int, position: float, ranges: dict) -> float:
    """Map a genomic (chrom, position) to the panel's cumulative x coordinate."""
    pos_min, span = ranges.get(int(chrom_num), (position, 1))
    return (chrom_num - 1) * SCAN_SLOT_WIDTH + (position - pos_min) / span * SCAN_SLOT_WIDTH


def _build_gene_labels(scan: pd.DataFrame) -> pd.DataFrame:
    """Peak position + iHS for each highlighted gene, for the Figure 3A labels.

    Prefers the merged significant-regions TSV (precise ``max_IHS_position`` /
    ``max_emp_ihs`` per region, across its gene columns); falls back to the
    per-window scan for genes absent from that file. Positions are mapped into
    the panel's fixed-width x layout.
    """
    ranges = _chrom_pos_ranges(scan)
    regions = None
    if config.IHS_REGIONS_TSV.exists():
        regions = pd.read_csv(config.IHS_REGIONS_TSV, sep="\t", compression="infer")
        for c in ("max_emp_ihs", "max_IHS_position"):
            regions[c] = pd.to_numeric(regions[c], errors="coerce")
    gene_cols = ("gene_in_window", "exon_in_window", "intron_in_window", "SV_gene")

    rows = []
    for gene in SCAN_ANNOTATE_GENES:
        chrom_num = position = ihs = None
        cumulative_pos = None
        source = None
        # 1) Precise peak from the merged significant-regions TSV.
        if regions is not None:
            mask = pd.Series(False, index=regions.index)
            for col in gene_cols:
                if col in regions.columns:
                    mask |= regions[col].notna() & regions[col].astype(str).str.contains(
                        gene, regex=False, na=False)
            hit = regions[mask].dropna(subset=["max_emp_ihs", "max_IHS_position"])
            if not hit.empty:
                r = hit.loc[hit["max_emp_ihs"].abs().idxmax()]
                chrom_num = int(str(r["chrom"]).replace("chr", ""))
                position = float(r["max_IHS_position"])
                ihs = float(r["max_emp_ihs"])
                source = "regions"
        # 2) Fallback to the per-window scan.
        if chrom_num is None:
            sub = scan[scan["gene"].notna() & scan["gene"].str.contains(gene, regex=False)]
            if sub.empty:
                continue
            r = scan.loc[sub["max_emp_ihs"].abs().idxmax()]
            chrom_num = int(r["chrom_num"])
            position = float(r["position"])
            ihs = float(r["max_emp_ihs"])
            source = "windows"
        # ANKRD31 has several nearby significant windows. Anchor its label to
        # the actual plotted window point (rather than the within-region peak
        # coordinate), so the leader is exactly vertical above a visible dot.
        if gene == "ANKRD31":
            sub = scan[
                scan["gene"].notna()
                & scan["gene"].str.contains(gene, regex=False)
                & scan["candidate"]
            ]
            if not sub.empty:
                r = sub.loc[sub["max_emp_ihs"].abs().idxmax()]
                chrom_num = int(r["chrom_num"])
                position = float(r["position"])
                ihs = float(r["max_emp_ihs"])
                cumulative_pos = float(r["cumulative_pos"])
                source = "windows"
        rows.append({
            "gene": gene, "chrom_num": chrom_num, "position": position,
            "max_emp_ihs": ihs, "source": source,
            "cumulative_pos": (
                cumulative_pos if cumulative_pos is not None
                else _to_cumulative(chrom_num, position, ranges)
            ),
        })
    return pd.DataFrame(rows, columns=["gene", "chrom_num", "position",
                                       "max_emp_ihs", "source", "cumulative_pos"])


def _parse_near_gene_distances(value: object) -> dict[str, int]:
    """Parse ``GENE=distance`` pairs from the tracked S3 extraction."""
    if pd.isna(value) or not str(value).strip():
        return {}
    distances: dict[str, int] = {}
    for item in str(value).split(";"):
        try:
            gene, raw_distance = item.split("=", 1)
            distance = int(raw_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Figure 3A near-gene distances must use GENE=integer pairs: "
                f"{value!r}"
            ) from exc
        gene = gene.strip()
        if not gene or distance <= 0 or gene in distances:
            raise ValueError(
                "Figure 3A near-gene distances require unique gene names and "
                f"positive gaps: {value!r}"
            )
        distances[gene] = distance
    return distances


def _write_near_gene_distance_log(hits: pd.DataFrame) -> None:
    """Write and echo RefGene gaps for intergenic SV annotations."""
    intergenic = hits["sv_gene_context"].str.startswith("intergenic", na=False)
    rows = hits.loc[intergenic].drop_duplicates("sv_allele_key")
    lines = [
        f"source_table={config.IHS_SV_HITS_TSV.resolve()}",
        f"distance_source={SV_NEAR_GENE_DISTANCE_SOURCE}",
        (
            "distance_definition=minimum positive base-pair gap from the "
            "1-based SV position to the outer RefGene transcript span"
        ),
        "row_unit=unique intergenic SV x named nearby gene",
    ]
    console_lines = []
    for row in rows.itertuples(index=False):
        distances = _parse_near_gene_distances(row.sv_near_gene_distances_bp)
        for gene, distance in distances.items():
            message = (
                "near_gene_sv_distance "
                f"sv={row.selection_chrom}:{int(row.sv_position)} "
                f"sv_type={row.sv_type} gene={gene} distance_bp={distance}"
            )
            lines.append(message)
            console_lines.append(f"[figure3] {message}")

    log_path = _adir() / "figure3_sv_near_gene_distances.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    global _NEAR_GENE_CONSOLE_EMITTED
    if not _NEAR_GENE_CONSOLE_EMITTED:
        for message in console_lines:
            print(message)
        print(f"[figure3] near-gene SV distance log: {log_path.resolve()}")
        _NEAR_GENE_CONSOLE_EMITTED = True


def _build_sv_hit_annotations(scan: pd.DataFrame) -> pd.DataFrame:
    """Load and validate the 17 S3 selection hits with direct/linked SVs.

    The tracked source is a compact extraction of manuscript table S3. Its row
    unit is a selection region, not a unique SV: the 17 rows map to 14 unique SV
    alleles because PLEKHB2 and ANKRD31 each support multiple selection regions.
    Every circle is anchored to the exact plotted window in ``scan``; each SV
    allele receives only one gene label, at its strongest absolute-iHS hit.
    """
    path = config.IHS_SV_HITS_TSV
    if not path.exists():
        raise FileNotFoundError(f"Figure 3A SV-hit source not found: {path}")

    hits = pd.read_csv(path, sep="\t", dtype={"selection_chrom": "string"})
    missing = [column for column in _SV_HIT_COLUMNS if column not in hits.columns]
    if missing:
        raise ValueError(f"Figure 3A SV-hit source is missing columns: {missing}")
    if len(hits) != SV_HIT_EXPECTED_ROWS:
        raise ValueError(
            "Figure 3A expects "
            f"{SV_HIT_EXPECTED_ROWS} S3 SV-associated selection-hit rows; "
            f"found {len(hits)} in {path}"
        )

    for column in (
        "region_start",
        "region_stop",
        "selection_peak_position",
        "selection_peak_ihs",
        "sv_position",
        "sv_length_bp",
        "sv_ld_r2",
    ):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    required_numeric = (
        "region_start",
        "region_stop",
        "selection_peak_position",
        "selection_peak_ihs",
        "sv_position",
        "sv_length_bp",
    )
    if hits[list(required_numeric)].isna().any().any():
        bad = hits.index[hits[list(required_numeric)].isna().any(axis=1)].tolist()
        raise ValueError(f"Figure 3A SV-hit source has invalid numeric rows: {bad}")

    hits["chrom_num"] = (
        hits["selection_chrom"]
        .str.replace("chr", "", regex=False)
        .astype(int)
    )
    if not hits["chrom_num"].between(1, 22).all():
        raise ValueError("Figure 3A SV-hit source contains a non-autosomal chromosome")
    if not hits["sv_relation"].isin(("direct", "linked")).all():
        raise ValueError("Figure 3A sv_relation must be 'direct' or 'linked'")
    if hits["sv_gene_label"].fillna("").str.strip().eq("").any():
        raise ValueError("Every Figure 3A SV-associated hit needs a gene label")
    linked = hits["sv_relation"].eq("linked")
    if hits.loc[linked, "sv_ld_r2"].isna().any():
        raise ValueError("Every linked Figure 3A SV hit needs an LD r-squared value")
    if hits.loc[linked, "sv_ld_r2"].lt(SV_LINK_MIN_R2).any():
        raise ValueError(
            f"Every linked Figure 3A SV hit must have r-squared >= {SV_LINK_MIN_R2}"
        )

    allele_key = list(zip(hits["chrom_num"], hits["sv_position"].astype(int)))
    if len(set(allele_key)) != SV_HIT_EXPECTED_UNIQUE_ALLELES:
        raise ValueError(
            "Figure 3A expects "
            f"{SV_HIT_EXPECTED_UNIQUE_ALLELES} unique S3 SV alleles; found "
            f"{len(set(allele_key))}"
        )
    hits["sv_allele_key"] = [f"chr{chrom}:{position}" for chrom, position in allele_key]

    scan_index = scan.set_index(["chrom_num", "position"])
    if not scan_index.index.is_unique:
        raise ValueError("Figure 3A scan has duplicate chromosome/position rows")
    plotted_rows = []
    for row in hits.itertuples(index=False):
        key = (int(row.chrom_num), int(row.selection_peak_position))
        if key not in scan_index.index:
            raise ValueError(
                "Figure 3A S3 SV hit does not match a plotted scan window: "
                f"chr{key[0]}:{key[1]}"
            )
        plotted = scan_index.loc[key]
        if not bool(plotted["candidate"]):
            raise ValueError(
                "Figure 3A S3 SV hit is not significant at p <= "
                f"{SCAN_SIG_PVAL}: chr{key[0]}:{key[1]}"
            )
        if not np.isclose(
            float(row.selection_peak_ihs), float(plotted["max_emp_ihs"]), atol=0.001
        ):
            raise ValueError(
                "Figure 3A S3 SV-hit iHS disagrees with the plotted scan: "
                f"chr{key[0]}:{key[1]}"
            )
        plotted_rows.append(plotted)

    hits["max_emp_ihs"] = [float(row["max_emp_ihs"]) for row in plotted_rows]
    hits["cumulative_pos"] = [float(row["cumulative_pos"]) for row in plotted_rows]
    intergenic = hits["sv_gene_context"].str.startswith("intergenic", na=False)
    parsed_distances = hits["sv_near_gene_distances_bp"].map(
        _parse_near_gene_distances
    )
    for index in hits.index[intergenic]:
        expected_genes = set(str(hits.at[index, "sv_gene_label"]).split("/"))
        observed_genes = set(parsed_distances.at[index])
        if observed_genes != expected_genes:
            raise ValueError(
                "Figure 3A intergenic SV distance genes disagree with the "
                f"display label at row {index}: expected {sorted(expected_genes)}, "
                f"found {sorted(observed_genes)}"
            )
    if parsed_distances.loc[~intergenic].map(bool).any():
        raise ValueError(
            "Figure 3A intragenic SV rows must not carry a near-gene distance"
        )
    hits["display_gene_label"] = hits["sv_gene_label"].astype(str)
    intergenic_gene_display = hits.loc[intergenic, "sv_gene_label"].astype(
        str
    ).str.replace("/", "/\n", regex=False)
    hits.loc[intergenic, "display_gene_label"] = (
        "near\n" + intergenic_gene_display
    )
    hits["label_sv"] = False
    strongest = (
        hits.assign(_abs_ihs=hits["max_emp_ihs"].abs())
        .sort_values("_abs_ihs", ascending=False)
        .drop_duplicates("sv_allele_key")
        .index
    )
    hits.loc[strongest, "label_sv"] = True
    _write_near_gene_distance_log(hits)
    return hits.sort_values(["chrom_num", "selection_peak_position"]).reset_index(drop=True)


def build(clean_run_data: bool = False) -> dict:
    d = _adir()
    scan = _build_scan()
    scan.to_csv(d / "selection_scan.csv", index=False)
    sv_hits = _build_sv_hit_annotations(scan)
    return {
        "scan": scan,
        "annotate_genes": SCAN_ANNOTATE_GENES,
        "gene_labels": _build_gene_labels(scan),
        "sv_hits": sv_hits,
        "chrom_sizes": CHROM_SIZES,
    }
