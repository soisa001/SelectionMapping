"""Figure 3B-G -- EHH decay + iHS/pi at meta-significant loci (AoU panel).

Driven by the meta-significant phecodes from the meta-analysis results TSV. For
each unique meta-significant variant position:

    EHH decay +/- 100 kb + gene track (from the phased VCF)
    normalized iHS (pre-computed) + pi (from VCF, hardmasked) +/- 100 kb
            + gene track

iHS values are cross-referenced from the iHS output TSV where available.

WORKBENCH-GATED. The phased VCFs and pre-computed iHS output live under the
migration workspace mount (``config.MIGRATION_WORKSPACE``); every in-bucket path
is prefixed via ``config.migration_path``. ``scikit-allel`` and ``tabix`` are
required and imported lazily, so this module imports fine off the Workbench --
``render_locus_figures`` simply skips loci whose VCF is absent.

SAMPLE SET: the analysis uses the canonical 1,025 unrelated AoU individuals.

Adapted from the manuscript's standalone EHH/iHS/pi script; algorithms unchanged.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re
import subprocess

import numpy as np
import pandas as pd

from .common import config, samples

# --- Paths --------------------------------------------------------------------
# All Workbench inputs live under the migration workspace mount
# (``config.MIGRATION_WORKSPACE`` == ``~/workspace/rw-migration-aou-rw-a3e31f4d``,
# absolute). Each is env-overridable in case the layout differs.
VCF_PATTERN = config.migration_path("SV_biallelic/{chrom}.biallelic.vcf.gz")
IHS_PATTERN = config.migration_path("ihs_output_v4/{chrom}.ihs.out.100bins.norm.ann")
META_TSV = os.environ.get(
    "AOU_META_TSV",
    config.migration_path("AOU Phase 1 Tables (current) - meta-analysis results.tsv"))
GTF_PATH = os.environ.get("AOU_REFGENE_GTF", config.migration_path("hg38.refGene.gtf"))
HARDMASK_PATH = os.environ.get(
    "AOU_HARDMASK_BED", config.migration_path("hardmask.hg38.v4.over99.bed"))
# The linked iHS TSV is the committed asset by default (also in the workspace).
IHS_TSV_PATH = os.environ.get("AOU_IHS_LINKED_TSV", str(config.IHS_REGIONS_TSV))

# Window sizes / parameters.
EHH_HALF_WINDOW = 100_000     # +/- 100 kb for the EHH figures (2B/2D/2F)
IHSPI_HALF_WINDOW = 100_000   # +/- 100 kb for the iHS/pi figures (2C/2E/2G)
PI_BIN_SIZE = 2_000           # 2 kb bins for pi
MIN_MAF = 0.05

CACHE_DIR = config.analysis_dir("figure3") / "ehh_cache"
FONTSIZE = 36
TITLE_FONTSIZE = 38
TICK_FONTSIZE = 32
LEGEND_FONTSIZE = 32
GENE_FONTSIZE = 32
_FIGSIZE_LOCUS = (12, 8)
_SAVEFIG_KW = {"bbox_inches": "tight", "pad_inches": 0.03, "dpi": 150}

# Font profile applied while rendering each figure. The explicit label/title
# sizes below already scale with FONTSIZE; this also enlarges the tick labels
# and legend (which otherwise fall back to the small matplotlib defaults).
_RC = {
    "font.size": FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "figure.titlesize": TITLE_FONTSIZE,
}

# Showcase loci rendered as the main manuscript panels (Figure 3B/2C = LCT,
# Figure 3D/2E = SLC24A5, Figure 3F/2G = ANKRD31). Core variants are taken from
# the linked iHS regions TSV. The ANKRD31 max-iHS SNV is in high LD with the
# 4,455-bp SV at chr5:75,128,422.
MAIN_LOCI = [
    {"gene": "LCT", "chrom": "chr2", "core_pos": 135851076, "ref": "G", "alt": "A",
     "rs": "rs4988235", "ihs": 3.20, "ehh_panel": "2B", "ihspi_panel": "2C"},
    {"gene": "SLC24A5", "chrom": "chr15", "core_pos": 48141297, "ref": "A", "alt": "G",
     "rs": "rs2470102", "ihs": -3.59, "ehh_panel": "2D", "ihspi_panel": "2E"},
    {"gene": "ANKRD31", "chrom": "chr5", "core_pos": 75131912, "ref": "A", "alt": "C",
     "rs": "rs6453119", "ihs": -9.30224, "sv_pos": 75128422,
     "ehh_panel": "2F", "ihspi_panel": "2G"},
]


def on_workbench() -> bool:
    """True when at least one phased VCF is reachable (i.e. bucket is mounted)."""
    from .common import gcs
    if gcs.available():
        return True
    return os.path.exists(VCF_PATTERN.format(chrom="chr2"))


def _out_dir():
    d = config.figure_dir("figure3") / "loci"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_gene(gene) -> str:
    """First gene symbol from a (possibly ';'-separated) annotation, filename-safe."""
    if gene is None or (isinstance(gene, float) and np.isnan(gene)):
        return "NA"
    first = str(gene).split(";")[0].strip()
    safe = re.sub(r"[^A-Za-z0-9._-]", "", first)
    return safe or "NA"


def main_panel_image_paths(ext: str = "png") -> dict:
    """Map composite panel letter (B-G) -> showcase figure image path.

    LCT, SLC24A5, and ANKRD31 each contribute an EHH / iHS-pi pair produced by
    ``render_main_panels`` for embedding in the Figure 3 composite.
    """
    d = config.figure_dir("figure3")
    out = {}
    for loc in MAIN_LOCI:
        out[loc["ehh_panel"][-1]] = d / f"Figure_{loc['ehh_panel']}_{loc['gene']}_EHH.{ext}"
        out[loc["ihspi_panel"][-1]] = d / f"Figure_{loc['ihspi_panel']}_{loc['gene']}_iHS_pi.{ext}"
    return out


# --- 1. Locus selection from the meta-analysis + iHS TSVs ---------------------
def _parse_variant_tag(tag):
    parts = str(tag).strip().split("_", 3)
    if len(parts) < 4:
        return None, None, None, None
    try:
        return int(parts[0]), int(parts[1]), parts[2], parts[3]
    except Exception:
        return None, None, None, None


def _find_ihs_match(chrom, pos, ihs_df, window=20000):
    """iHS TSV row whose max_IHS_position is closest to pos within +/- window."""
    sub = ihs_df[(ihs_df["chrom"] == chrom)
                 & (ihs_df["max_IHS_position"] >= pos - window)
                 & (ihs_df["max_IHS_position"] <= pos + window)]
    if len(sub) == 0:
        return None
    sub = sub.copy()
    sub["dist"] = (sub["max_IHS_position"] - pos).abs()
    return sub.sort_values("dist").iloc[0]


def select_loci() -> pd.DataFrame:
    """Read the meta-analysis TSV, keep meta-significant variants, dedupe by
    position, and cross-reference iHS. One row per unique locus."""
    with open(META_TSV) as fh:
        header_row = next((i for i, line in enumerate(fh)
                           if "variant_tag" in line.split("\t")[0]), None)
    if header_row is None:
        raise RuntimeError(f"No 'variant_tag' header row in {META_TSV}")

    meta = pd.read_csv(META_TSV, sep="\t", skiprows=header_row, dtype=str)
    meta.columns = meta.columns.str.strip()
    meta = meta.dropna(subset=["variant_tag", "phecode", "meta_significant"])
    meta = meta[meta["meta_significant"].str.strip().str.upper() == "TRUE"].copy()

    meta[["CHR", "POS", "REF", "ALT"]] = meta["variant_tag"].apply(
        lambda t: pd.Series(_parse_variant_tag(t)))
    meta = meta.dropna(subset=["CHR", "POS"]).copy()
    meta["CHR"] = meta["CHR"].astype(int)
    meta["POS"] = meta["POS"].astype(int)
    meta["phecode"] = meta["phecode"].astype(str).str.strip()
    meta["phecode_name"] = (meta["phecode_string"].astype(str).str.strip()
                            if "phecode_string" in meta.columns else meta["phecode"])
    for c in ("p_meta", "beta_meta"):
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["chrom"] = "chr" + meta["CHR"].astype(str)

    groups = (meta.groupby(["chrom", "POS", "REF", "ALT"]).agg(
        phecodes=("phecode", lambda x: "; ".join(sorted(set(x)))),
        phecode_names=("phecode_name", lambda x: "; ".join(
            sorted(set(str(v) for v in x if pd.notna(v))))),
        gene_in_window=("gene_in_window",
                        lambda x: next((str(v) for v in x if pd.notna(v)), "")),
        min_p_meta=("p_meta", "min"),
        n_phecodes=("phecode", "nunique"),
    ).reset_index())

    ihs_df = pd.read_csv(IHS_TSV_PATH, sep="\t", dtype=str)
    for col in ["start", "stop", "max_emp_ihs", "max_IHS_position", "IHS_AF"]:
        if col in ihs_df.columns:
            ihs_df[col] = pd.to_numeric(ihs_df[col], errors="coerce")

    rows = []
    for _, vg in groups.iterrows():
        pos = int(vg["POS"])
        ihs_row = _find_ihs_match(vg["chrom"], pos, ihs_df)
        rows.append({
            "chrom": vg["chrom"], "core_pos": pos, "ref": vg["REF"], "alt": vg["ALT"],
            "phecodes": vg["phecodes"], "phecode_names": vg["phecode_names"],
            "gene": vg["gene_in_window"], "n_phecodes": int(vg["n_phecodes"]),
            "min_p_meta": vg["min_p_meta"],
            "ihs_val": float(ihs_row["max_emp_ihs"]) if ihs_row is not None else np.nan,
            "ihs_pos": int(ihs_row["max_IHS_position"]) if ihs_row is not None else pos,
            "ihs_rs": str(ihs_row.get("IHS_rs", "")) if ihs_row is not None else "",
        })
    return pd.DataFrame(rows)


# --- 2. GTF parsing -----------------------------------------------------------
def parse_gtf_region(gtf_path, chrom, start, end):
    """GTF -> {gene_name: longest-transcript {tx_start, tx_end, strand, exons}}."""
    genes = {}
    open_fn = gzip.open if str(gtf_path).endswith(".gz") else open
    with open_fn(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            gchr, _, feature, gstart, gend, _, strand, _, attr = parts
            gstart, gend = int(gstart), int(gend)
            if gchr != chrom or gend < start or gstart > end:
                continue
            m = re.search(r'gene_name "([^"]+)"', attr) or re.search(r'gene_id "([^"]+)"', attr)
            gene_name = m.group(1) if m else "unknown"
            m2 = re.search(r'transcript_id "([^"]+)"', attr)
            tx_id = m2.group(1) if m2 else "unknown"
            genes.setdefault(gene_name, {}).setdefault(
                tx_id, {"tx_start": None, "tx_end": None, "strand": strand, "exons": []})
            tx = genes[gene_name][tx_id]
            if feature == "transcript":
                tx["tx_start"], tx["tx_end"] = gstart, gend
            elif feature == "exon":
                tx["exons"].append((gstart, gend))

    result = {}
    for gname, txs in genes.items():
        best_tx, best_len = None, 0
        for tx in txs.values():
            s = tx["tx_start"] or (min(e[0] for e in tx["exons"]) if tx["exons"] else 0)
            e = tx["tx_end"] or (max(e[1] for e in tx["exons"]) if tx["exons"] else 0)
            tx["tx_start"], tx["tx_end"] = s, e
            if e - s > best_len:
                best_len, best_tx = e - s, tx
        if best_tx:
            result[gname] = best_tx
    return result


# --- 3. VCF loading (phased, MAF-filtered, restricted to the 1,025) -----------
def ensure_tabix(vcf_path):
    if not (os.path.exists(vcf_path + ".tbi") or os.path.exists(vcf_path + ".csi")):
        subprocess.run(["tabix", "-p", "vcf", vcf_path], check=True)


def _aou_1025_in_vcf(vcf_path):
    """Select AoU VCF samples using the canonical unrelated cohort manifest."""
    import allel
    ids = samples.load_aou_sample_list(download=True)   # 1,025 unrelated
    vcf_samples = list(allel.read_vcf_headers(vcf_path).samples)
    if ids is None:
        raise FileNotFoundError("The canonical AoU sample manifest is required for EHH analysis.")
    return [s for s in vcf_samples if s in ids]


def load_haplotypes_from_vcf(vcf_path, chrom, start, end, core_pos=None):
    """Read a phased VCF region for the 1,025 unrelated AoU samples, MAF >= 0.05
    (always keeping core_pos). Returns (positions, hap, refs, alts, n_raw)."""
    import allel
    ensure_tabix(vcf_path)
    keep_samples = _aou_1025_in_vcf(vcf_path)   # filter to the 1,025 (the canonical manifest)
    callset = allel.read_vcf(
        vcf_path,
        region=f"{chrom}:{start}-{end}",
        samples=keep_samples,
        fields=["variants/POS", "variants/REF", "variants/ALT",
                "calldata/GT", "variants/CHROM", "samples"],
    )
    if callset is None:
        raise ValueError(f"No variants in {vcf_path} for {chrom}:{start}-{end}")

    pos = callset["variants/POS"]
    hap = allel.GenotypeArray(callset["calldata/GT"]).to_haplotypes()
    refs = callset["variants/REF"]
    alts_raw = callset["variants/ALT"]
    alts = alts_raw[:, 0] if alts_raw.ndim == 2 else alts_raw
    n_raw = len(pos)

    af = np.asarray(hap, dtype=np.float64).mean(axis=1)
    maf = np.minimum(af, 1.0 - af)
    keep = maf >= MIN_MAF
    if core_pos is not None:
        keep |= (pos == core_pos)
    return pos[keep], np.asarray(hap)[keep, :], refs[keep], alts[keep], n_raw


# --- 4. EHH computation (cached) ----------------------------------------------
def compute_ehh_fast(hap, pos, core_idx):
    """EHH decay from the core outward via incremental group-splitting."""
    n_var, n_hap = hap.shape
    core_alleles = hap[core_idx, :]
    idx_anc = np.where(core_alleles == 0)[0]
    idx_der = np.where(core_alleles == 1)[0]

    def _ehh_direction(var_indices, sample_indices):
        n = len(sample_indices)
        if n <= 1 or len(var_indices) == 0:
            return np.ones(max(len(var_indices), 0))
        n_choose_2 = n * (n - 1) / 2.0
        ehh_vals = np.empty(len(var_indices))
        group_id = np.zeros(n, dtype=np.int64)
        next_id = 1
        for step, vi in enumerate(var_indices):
            alleles = hap[vi, sample_indices]
            new_group_id = np.empty(n, dtype=np.int64)
            new_counts, seen = {}, {}
            for j in range(n):
                key = (group_id[j], alleles[j])
                if key not in seen:
                    seen[key] = next_id
                    next_id += 1
                nid = seen[key]
                new_group_id[j] = nid
                new_counts[nid] = new_counts.get(nid, 0) + 1
            group_id = new_group_id
            sum_pairs = sum(c * (c - 1) / 2.0 for c in new_counts.values())
            ehh_vals[step] = sum_pairs / n_choose_2
        return ehh_vals

    left_idxs = list(range(core_idx - 1, -1, -1))
    right_idxs = list(range(core_idx + 1, n_var))
    ehh0_L = _ehh_direction(left_idxs, idx_anc) if left_idxs and len(idx_anc) > 1 else np.array([])
    ehh1_L = _ehh_direction(left_idxs, idx_der) if left_idxs and len(idx_der) > 1 else np.array([])
    ehh0_R = _ehh_direction(right_idxs, idx_anc) if right_idxs and len(idx_anc) > 1 else np.array([])
    ehh1_R = _ehh_direction(right_idxs, idx_der) if right_idxs and len(idx_der) > 1 else np.array([])
    pos_L = pos[left_idxs] if left_idxs else np.array([])
    pos_R = pos[right_idxs] if right_idxs else np.array([])
    return pos_L, ehh0_L, ehh1_L, pos_R, ehh0_R, ehh1_R


def _cache_key(chrom, core_pos, ref, alt):
    allele_str = f"{ref}_{alt}"
    if len(allele_str) > 40:
        allele_str = hashlib.md5(allele_str.encode()).hexdigest()[:12]
    return f"{chrom}_{core_pos}_{allele_str}"


def compute_or_load_ehh(chrom, core_pos, ref, alt, vcf_path, window_start, window_end):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    span_kb = (window_end - window_start) // 1000  # keyed by window so ranges don't clash
    cache_file = CACHE_DIR / f"{_cache_key(chrom, core_pos, ref, alt)}_w{span_kb}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return {k: data[k] for k in data.files}

    pos, hap, refs, alts, n_raw = load_haplotypes_from_vcf(
        vcf_path, chrom, window_start, window_end, core_pos=core_pos)

    core_matches = np.where(pos == core_pos)[0]
    if len(core_matches) == 0:
        dists = np.abs(pos.astype(np.int64) - core_pos)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= 50:
            core_matches = np.array([nearest])
        else:
            raise ValueError(f"Core SNP {core_pos} not found (nearest {pos[nearest]})")

    core_idx, match_type = None, "none"
    for ci in core_matches:
        vref, valt = refs[ci], alts[ci]
        if vref == ref and valt == alt:
            core_idx, match_type = ci, "direct"; break
        if vref == alt and valt == ref:
            core_idx, match_type = ci, "flip"; break
    if core_idx is None:
        core_idx = int(core_matches[0])
        match_type = f"pos_only(vcf={refs[core_idx]}/{alts[core_idx]})"

    n_anc = int(np.sum(hap[core_idx, :] == 0))
    n_der = int(np.sum(hap[core_idx, :] == 1))
    pos_L, ehh0_L, ehh1_L, pos_R, ehh0_R, ehh1_R = compute_ehh_fast(hap, pos, core_idx)
    result = {
        "pos_left": pos_L, "ehh0_left": ehh0_L, "ehh1_left": ehh1_L,
        "pos_right": pos_R, "ehh0_right": ehh0_R, "ehh1_right": ehh1_R,
        "core_pos": np.array([core_pos]), "match_type": np.array([match_type]),
        "n_variants": np.array([len(pos)]), "n_haplotypes": np.array([hap.shape[1]]),
        "n_anc": np.array([n_anc]), "n_der": np.array([n_der]),
    }
    np.savez_compressed(cache_file, **result)
    return result


# --- 5. Pre-computed normalized iHS -------------------------------------------
def load_norm_ihs(chrom, start, end):
    """Read the pre-computed normalized iHS for a region (region + MAF filter)."""
    ihs_file = IHS_PATTERN.format(chrom=chrom)
    if not os.path.exists(ihs_file):
        return np.array([]), np.array([])
    ihs_df = pd.read_csv(ihs_file, sep="\t", dtype=str)
    for c in ("pos", "freq", "norm_ihs"):
        ihs_df[c] = pd.to_numeric(ihs_df[c], errors="coerce")
    mask = ((ihs_df["pos"] >= start) & (ihs_df["pos"] <= end)
            & (ihs_df["freq"] >= MIN_MAF) & (ihs_df["freq"] <= 1.0 - MIN_MAF)
            & ihs_df["norm_ihs"].notna())
    sub = ihs_df[mask].sort_values("pos")
    return sub["pos"].values, sub["norm_ihs"].values


# --- 6. Hardmask + pi from VCF (cached) ---------------------------------------
def load_accessible_mask(hardmask_path, chrom, start, end):
    """Load the hardmask BED (negative mask: BED regions are removed). Returns a
    boolean array over [start, end), True = accessible."""
    length = end - start
    is_accessible = np.ones(length, dtype=bool)
    if not os.path.exists(hardmask_path):
        return is_accessible
    with open(hardmask_path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            bstart, bend = int(parts[1]), int(parts[2])
            os_pos, oe_pos = max(bstart, start) - start, min(bend, end) - start
            if os_pos < oe_pos:
                is_accessible[os_pos:oe_pos] = False
    return is_accessible


def compute_or_load_pi(chrom, core_pos, vcf_path):
    """Windowed pi over the configured locus range, normalized by accessible bases
    per window (hardmask); windows with <10% accessible bases -> NaN."""
    import allel
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{chrom}_{core_pos}_pi_masked_w{IHSPI_HALF_WINDOW}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return {k: data[k] for k in data.files}

    start = max(core_pos - IHSPI_HALF_WINDOW, 1)
    end = core_pos + IHSPI_HALF_WINDOW
    pos, hap, refs, alts, n_raw = load_haplotypes_from_vcf(
        vcf_path, chrom, start, end, core_pos=core_pos)

    acc_mask = load_accessible_mask(HARDMASK_PATH, chrom, start, end)
    var_accessible = np.array([acc_mask[p - start] for p in pos])
    pos, hap = pos[var_accessible], hap[var_accessible, :]

    ac = allel.HaplotypeArray(hap).count_alleles()
    mpd = allel.mean_pairwise_difference(ac)
    min_accessible_frac = 0.10

    pi_mids, pi_vals = [], []
    for ws in range(start, end, PI_BIN_SIZE):
        we = min(ws + PI_BIN_SIZE, end)
        bin_len = we - ws
        n_accessible = int(acc_mask[ws - start:we - start].sum())
        accessible_frac = n_accessible / bin_len if bin_len > 0 else 0
        wmask = (pos >= ws) & (pos < we)
        if n_accessible == 0 or accessible_frac < min_accessible_frac or int(wmask.sum()) == 0:
            pi_mids.append((ws + we) / 2.0); pi_vals.append(np.nan); continue
        pi_mids.append((ws + we) / 2.0)
        pi_vals.append(float(mpd[wmask].sum()) / n_accessible)

    result = {"pi_mids": np.array(pi_mids), "pi_vals": np.array(pi_vals),
              "pi_start": np.array([start]), "pi_end": np.array([end])}
    np.savez_compressed(cache_file, **result)
    return result


# --- 7. Gene track ------------------------------------------------------------
# Small non-coding RNA gene classes to hide from the track (declutter): microRNAs,
# sno/sn/sc/vault/Y RNAs, etc. Also drop any gene shorter than MIN_GENE_SPAN.
_SMALL_RNA_RE = re.compile(r"^(MIR|MIRLET|SNOR|SNORD|SNORA|SCARNA|RNU|RN7|RNA5|VTRNA|RNY)",
                           re.IGNORECASE)
MIN_GENE_SPAN = 1_000
GENE_TRACK_MIN_GAP_FRACTION = 0.24
GENE_TRACK_ROW_GAP = 3.0
SV_GENE_TRACK_MARKER_SIZE = 720
SV_GENE_TRACK_MARKER_COLOR = "#d95f02"
SV_GENE_TRACK_MARKER_LINEWIDTH = 2.4
SV_GENE_TRACK_TOP_PADDING = 6.5
SV_GENE_TRACK_MARKER_TOP_INSET = 0.65


def _is_small_gene(name, tx) -> bool:
    """True for tiny/small-RNA genes (MIR, snoRNA, ...) to exclude from the track."""
    if _SMALL_RNA_RE.match(str(name)):
        return True
    return (tx.get("tx_end", 0) - tx.get("tx_start", 0)) < MIN_GENE_SPAN


def draw_gene_track(ax, genes_dict, window_start, window_end, highlight_gene=None,
                    sv_pos=None, sv_type=None, gene_fontsize=None,
                    reserve_sv_marker_band=False):
    """Draw the gene track. ``gene_fontsize`` overrides ``GENE_FONTSIZE`` for the
    italic gene names (used by the composite locus panels, which are down-scaled
    when embedded and therefore want slightly smaller names)."""
    import matplotlib.pyplot as plt
    gene_fontsize = GENE_FONTSIZE if gene_fontsize is None else gene_fontsize
    ax.set_xlim(window_start, window_end)
    ax.set_ylabel("Genes", fontsize=FONTSIZE)
    ax.set_xlabel("Position (bp)", fontsize=FONTSIZE)
    def draw_sv_marker(y_value):
        if sv_pos is None:
            return
        normalized_type = str(sv_type).strip().upper() if sv_type is not None else "DEL"
        if normalized_type not in {"INS", "DEL"}:
            raise ValueError(f"Unsupported SV type {sv_type!r}; expected INS or DEL")
        ax.scatter(
            [sv_pos], [y_value],
            marker="^" if normalized_type == "INS" else "v",
            s=SV_GENE_TRACK_MARKER_SIZE,
            color=SV_GENE_TRACK_MARKER_COLOR,
            edgecolors="black", linewidths=SV_GENE_TRACK_MARKER_LINEWIDTH,
            zorder=7, clip_on=False,
        )

    if not genes_dict:
        ax.text(0.5, 0.5, "No genes in window", transform=ax.transAxes,
                ha="center", va="center", fontsize=TICK_FONTSIZE, color="gray")
        ax.set_ylim(-0.7, 3.6)
        draw_sv_marker(2.9)
        ax.set_yticks([])
        return
    genes_dict = {n: tx for n, tx in genes_dict.items() if not _is_small_gene(n, tx)}
    if not genes_dict:
        ax.text(0.5, 0.5, "No genes in window", transform=ax.transAxes,
                ha="center", va="center", fontsize=TICK_FONTSIZE, color="gray")
        ax.set_ylim(-0.7, 3.6)
        draw_sv_marker(2.9)
        ax.set_yticks([])
        return
    sorted_genes = sorted(genes_dict.items(), key=lambda x: x[1]["tx_start"])
    rows, gene_row = [], {}
    # Wide horizontal gap for same-row placement so the (large) gene names don't
    # collide left-to-right; closer genes stack onto separate rows instead.
    min_gap = (window_end - window_start) * GENE_TRACK_MIN_GAP_FRACTION
    for gname, tx in sorted_genes:
        placed = False
        for ri, row_ends in enumerate(rows):
            if tx["tx_start"] > row_ends[-1] + min_gap:
                row_ends.append(tx["tx_end"]); gene_row[gname] = ri; placed = True; break
        if not placed:
            rows.append([tx["tx_end"]]); gene_row[gname] = len(rows) - 1
    row_height = 0.6
    row_gap = GENE_TRACK_ROW_GAP
    # The supplementary SV marker is intentionally large.  Reserve a real
    # marker band rather than relying only on its data-coordinate centre; at
    # the compact source-panel height the old 3.6-unit padding allowed the
    # triangle itself to cover the highest gene label (for example SH3D21).
    top_padding = (
        SV_GENE_TRACK_TOP_PADDING
        if sv_pos is not None or reserve_sv_marker_band
        else 2.8
    )
    track_top = (len(rows) - 1) * row_gap + top_padding
    ax.set_ylim(-0.7, track_top)
    ax.set_yticks([])
    for gname, tx in sorted_genes:
        y = gene_row[gname] * row_gap + 0.5
        color = "black"
        lw = 1.8
        ax.plot([tx["tx_start"], tx["tx_end"]], [y, y], color=color, linewidth=lw,
                solid_capstyle="butt")
        min_exon_w = (window_end - window_start) * 0.002
        for es, ee in tx["exons"]:
            ax.add_patch(plt.Rectangle((es, y - row_height / 2), max(ee - es, min_exon_w),
                                       row_height, facecolor=color, edgecolor="none", alpha=0.85))
        label_x = np.clip((tx["tx_start"] + tx["tx_end"]) / 2,
                          window_start + (window_end - window_start) * 0.05,
                          window_end - (window_end - window_start) * 0.05)
        label_y = y + row_height / 2 + 0.35
        # Uniform (non-bold) gene labels; the highlighted gene is no longer bolded.
        ax.text(label_x, label_y, gname, ha="center",
                va="bottom", fontsize=gene_fontsize, fontstyle="italic", color=color,
                fontweight="normal")
    # Reserve a stable marker band at the very top of the gene axis, directly
    # beneath the pi track and above every transcript label.
    draw_sv_marker(track_top - SV_GENE_TRACK_MARKER_TOP_INSET)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# --- 8. Figure 3B: EHH + gene track ----------------------------------------------
def plot_ehh_figure(gene, chrom, core_pos, ref, alt, ihs_val, rs_id, vcf_path,
                    gtf_path, window_start, window_end, min_p_meta=np.nan,
                    show_title=False, sv_pos=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ehh = compute_or_load_ehh(chrom, core_pos, ref, alt, vcf_path, window_start, window_end)
    genes_dict = parse_gtf_region(gtf_path, chrom, window_start, window_end)
    fig, (ax_ehh, ax_gene) = plt.subplots(
        2, 1, figsize=_FIGSIZE_LOCUS,
        gridspec_kw={"height_ratios": [3.0, 1.6], "hspace": 0.05},
        sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04)
    # Not polarized: "Reference"/"Alternate" allele haplotypes, not ancestral/derived.
    # Join the left + right decay through the core (EHH = 1 at the core) so the
    # line is continuous -- no gap at the lead SNV.
    _ehh_lw = 4.0
    pl = np.asarray(ehh["pos_left"], float)
    pr = np.asarray(ehh["pos_right"], float)

    def _joined(y_left, y_right):
        yl = np.asarray(y_left, float)
        yr = np.asarray(y_right, float)
        x = np.concatenate([pl[::-1], [core_pos], pr])   # pos_left is core->far, reverse
        y = np.concatenate([yl[::-1], [1.0], yr])
        return x, y

    xr, yr_ref = _joined(ehh["ehh0_left"], ehh["ehh0_right"])
    xa, ya_alt = _joined(ehh["ehh1_left"], ehh["ehh1_right"])
    ax_ehh.plot(xr, yr_ref, color="#2471a3", lw=_ehh_lw, label="Ref. Hap.")
    ax_ehh.plot(xa, ya_alt, color="#c0392b", lw=_ehh_lw, label="Alt. Hap.")
    ax_ehh.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_ehh.plot(core_pos, 1.0, marker="v", color="black", ms=11, zorder=5)
    ax_ehh.set_xlim(window_start, window_end)
    ax_ehh.set_ylim(-0.02, 1.05)
    ax_ehh.set_ylabel("EHH", fontsize=FONTSIZE)
    if show_title:
        alt_disp = alt if len(str(alt)) <= 12 else str(alt)[:9] + "…"
        ihs_str = f"IHS = {ihs_val:.2f}" if np.isfinite(ihs_val) else ""
        meta_str = f"p_meta = {min_p_meta:.2g}" if np.isfinite(min_p_meta) else ""
        subtitle_parts = [s for s in [rs_id, f"{ref}>{alt_disp}", ihs_str, meta_str] if s]
        gene_str = f" ({gene})" if gene else ""
        ax_ehh.set_title(f"EHH decay — {chrom}:{core_pos:,}{gene_str}\n"
                         f"{' | '.join(subtitle_parts)}", fontsize=TITLE_FONTSIZE)
    if show_title:
        ax_ehh.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    else:
        ax_ehh.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=2.2,
            columnspacing=1.4,
            borderaxespad=0.0,
        )
    ax_ehh.grid(axis="y", alpha=0.3, lw=0.8)
    ax_ehh.tick_params(labelsize=TICK_FONTSIZE)
    for s in ("top", "right"):
        ax_ehh.spines[s].set_visible(False)
    draw_gene_track(ax_gene, genes_dict, window_start, window_end,
                    highlight_gene=gene, sv_pos=sv_pos)
    ax_gene.tick_params(labelsize=TICK_FONTSIZE)
    ax_gene.set_xticks([window_start, core_pos, window_end])
    ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.2f} Mb"))
    return fig


# --- 9. Figure 3C: iHS + pi + gene track -----------------------------------------
def plot_ihs_pi_figure(gene, chrom, core_pos, ihs_val, rs_id, vcf_path, gtf_path,
                       min_p_meta=np.nan, show_title=False, sv_pos=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ip_start = max(core_pos - IHSPI_HALF_WINDOW, 1)
    ip_end = core_pos + IHSPI_HALF_WINDOW
    ihs_pos, ihs_norm = load_norm_ihs(chrom, ip_start, ip_end)
    pi_data = compute_or_load_pi(chrom, core_pos, vcf_path)
    genes_dict = parse_gtf_region(gtf_path, chrom, ip_start, ip_end)

    fig = plt.figure(figsize=_FIGSIZE_LOCUS, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.04)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.7, 1.6], hspace=0.05)
    ax_ihs = fig.add_subplot(gs[0])
    ax_pi = fig.add_subplot(gs[1], sharex=ax_ihs)
    ax_gene = fig.add_subplot(gs[2], sharex=ax_ihs)

    if len(ihs_pos) > 0:
        ax_ihs.scatter(ihs_pos, np.abs(ihs_norm), s=28, color="#5b2c6f", alpha=0.75,
                       zorder=3, edgecolors="none")
        ax_ihs.vlines(ihs_pos, 0, np.abs(ihs_norm), colors="#5b2c6f", alpha=0.35, linewidth=1.0)
    ax_ihs.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_ihs.set_xlim(ip_start, ip_end)
    ax_ihs.set_ylim(bottom=0)
    ax_ihs.set_ylabel("|norm iHS|", fontsize=FONTSIZE)
    if show_title:
        gene_str = f" — {gene}" if gene else ""
        ihs_str = f"IHS={ihs_val:.2f}" if np.isfinite(ihs_val) else ""
        meta_str = f"p_meta={min_p_meta:.2g}" if np.isfinite(min_p_meta) else ""
        subtitle = "  ".join(s for s in [ihs_str, meta_str] if s)
        ax_ihs.set_title(f"|iHS|{gene_str} ({chrom}:{core_pos:,})\n"
                         f"± {IHSPI_HALF_WINDOW // 1000} kb,  MAF ≥ {MIN_MAF}"
                         + (f"  |  {subtitle}" if subtitle else ""),
                         fontsize=TITLE_FONTSIZE)
    ax_ihs.grid(axis="y", alpha=0.3, lw=0.8)
    for s in ("top", "right"):
        ax_ihs.spines[s].set_visible(False)
    ax_ihs.tick_params(labelbottom=False, labelsize=TICK_FONTSIZE)

    pi_mids, pi_vals = pi_data["pi_mids"], pi_data["pi_vals"]
    valid = ~np.isnan(pi_vals)
    if valid.sum() > 0:
        ax_pi.vlines(pi_mids[valid], 0, pi_vals[valid], colors="#117864", alpha=0.35, linewidth=1.0)
        ax_pi.scatter(pi_mids[valid], pi_vals[valid], s=40, color="#117864", zorder=3,
                      edgecolors="white", linewidths=0.8)
    ax_pi.axvline(core_pos, color="black", ls="--", lw=1.5, alpha=0.6)
    ax_pi.set_xlim(ip_start, ip_end)
    ax_pi.set_ylim(bottom=0)
    ax_pi.set_ylabel("π", fontsize=FONTSIZE)
    if show_title:
        ax_pi.set_title(f"Nucleotide diversity π\n({PI_BIN_SIZE // 1000} kb bins,  "
                        f"MAF ≥ {MIN_MAF},  hardmasked)", fontsize=FONTSIZE)
    ax_pi.grid(axis="y", alpha=0.3, lw=0.8)
    for s in ("top", "right"):
        ax_pi.spines[s].set_visible(False)
    ax_pi.tick_params(labelbottom=False, labelsize=TICK_FONTSIZE)

    draw_gene_track(ax_gene, genes_dict, ip_start, ip_end,
                    highlight_gene=gene, sv_pos=sv_pos)
    ax_gene.tick_params(labelsize=TICK_FONTSIZE)
    ax_gene.set_xticks([ip_start, core_pos, ip_end])
    ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.2f} Mb"))
    return fig


# --- 10. Orchestration --------------------------------------------------------
def render_locus_figures(clean_run_plots: bool = False, max_loci: int | None = None) -> list:
    """Generate Figure 3B (EHH) + Figure 3C (iHS/pi) per meta-significant locus.

    Workbench-gated: needs the phased VCFs + iHS output under the migration
    workspace. Loci whose VCF is absent are skipped. Returns the saved PNG paths.
    """
    import warnings

    import matplotlib.pyplot as plt
    if not on_workbench():
        raise FileNotFoundError("Required phased VCF inputs are unavailable; stage them on Workbench.")

    # scikit-allel is required to read the phased VCFs. Check it once, upfront,
    # with an actionable message -- otherwise every locus fails with an opaque
    # "No module named 'allel'" warning.
    try:
        import allel  # noqa: F401
    except ImportError:
        warnings.warn(
            "[figure3] scikit-allel is not installed -- the locus panels cannot read the "
            "phased VCFs. Install it (and the tabix CLI) on the Workbench:\n"
            "    pip install scikit-allel\n"
            "then re-run this cell.",
            stacklevel=2,
        )
        return []

    # Locus selection needs the meta-analysis TSV (+ the linked iHS TSV). These
    # are working-directory / asset files, not bucket data -- skip cleanly (don't
    # crash) if they aren't where expected. Override via AOU_META_TSV /
    # AOU_IHS_LINKED_TSV.
    missing = [p for p in (META_TSV, IHS_TSV_PATH) if not os.path.exists(p)]
    if missing:
        warnings.warn(
            "[figure3] locus selection skipped -- input(s) not found: "
            + "; ".join(missing)
            + ". Set AOU_META_TSV / AOU_IHS_LINKED_TSV to their locations.",
            stacklevel=2,
        )
        return []

    loci = select_loci()
    if max_loci is not None:
        loci = loci.head(max_loci)
    out_dir = _out_dir()
    saved = []
    with plt.rc_context(_RC):
        for _, row in loci.iterrows():
            chrom, core_pos = row["chrom"], int(row["core_pos"])
            vcf_path = VCF_PATTERN.format(chrom=chrom)
            if not os.path.exists(vcf_path):
                continue
            tag = f"{chrom}_{core_pos}_{_safe_gene(row['gene'])}"
            ehh_png = out_dir / f"ehh_{tag}.png"
            ihspi_png = out_dir / f"ihs_pi_{tag}.png"
            try:
                fig1 = plot_ehh_figure(row["gene"], chrom, core_pos, row["ref"], row["alt"],
                                       row["ihs_val"], row["ihs_rs"], vcf_path, GTF_PATH,
                                       max(core_pos - EHH_HALF_WINDOW, 1),
                                       core_pos + EHH_HALF_WINDOW, min_p_meta=row["min_p_meta"],
                                       show_title=True)
                fig1.savefig(ehh_png, **_SAVEFIG_KW)
                fig1.savefig(ehh_png.with_suffix(".pdf"), **_SAVEFIG_KW)
                plt.close(fig1)
                saved.append(ehh_png)
            except Exception as exc:  # keep going through the loci
                warnings.warn(f"[figure3] EHH failed for {tag}: {exc}", stacklevel=2)
            try:
                fig2 = plot_ihs_pi_figure(row["gene"], chrom, core_pos, row["ihs_val"],
                                          row["ihs_rs"], vcf_path, GTF_PATH,
                                          min_p_meta=row["min_p_meta"], show_title=True)
                fig2.savefig(ihspi_png, **_SAVEFIG_KW)
                fig2.savefig(ihspi_png.with_suffix(".pdf"), **_SAVEFIG_KW)
                plt.close(fig2)
                saved.append(ihspi_png)
            except Exception as exc:
                warnings.warn(f"[figure3] iHS/pi failed for {tag}: {exc}", stacklevel=2)
    return saved


def render_main_panels(clean_run_plots: bool = False) -> list:
    """Render the three manuscript showcase loci as paired panels 2B-2G.

    Each locus gets an EHH figure and an iHS/pi figure (the ``MAIN_LOCI`` spec),
    saved to ``figures/figure3/Figure_<panel>_<gene>_*.{pdf,png}``. Workbench-
    gated (phased VCFs + scikit-allel). Returns the saved PNG paths.
    """
    import warnings

    import matplotlib.pyplot as plt
    if not on_workbench():
        raise FileNotFoundError("Required phased VCF inputs are unavailable; stage them on Workbench.")
    try:
        import allel  # noqa: F401
    except ImportError:
        warnings.warn("[figure3] scikit-allel not installed -- run `pip install "
                      "scikit-allel` for Figure 3B-2G.", stacklevel=2)
        return []

    fig_dir = config.figure_dir("figure3")
    saved = []
    with plt.rc_context(_RC):
        for loc in MAIN_LOCI:
            chrom, core_pos, gene = loc["chrom"], loc["core_pos"], loc["gene"]
            vcf_path = VCF_PATTERN.format(chrom=chrom)
            if not os.path.exists(vcf_path):
                warnings.warn(f"[figure3] VCF not found for {gene}: {vcf_path}", stacklevel=2)
                continue
            ehh_stem = fig_dir / f"Figure_{loc['ehh_panel']}_{gene}_EHH"
            ihspi_stem = fig_dir / f"Figure_{loc['ihspi_panel']}_{gene}_iHS_pi"
            try:
                fig1 = plot_ehh_figure(gene, chrom, core_pos, loc["ref"], loc["alt"],
                                       loc["ihs"], loc["rs"], vcf_path, GTF_PATH,
                                       max(core_pos - EHH_HALF_WINDOW, 1),
                                       core_pos + EHH_HALF_WINDOW,
                                       sv_pos=loc.get("sv_pos"))
                for ext in config.FIGURE_FORMATS:
                    fig1.savefig(ehh_stem.with_suffix(f".{ext}"), **_SAVEFIG_KW)
                plt.close(fig1)
                saved.append(ehh_stem.with_suffix(".png"))
            except Exception as exc:
                warnings.warn(f"[figure3] {loc['ehh_panel']} EHH ({gene}) failed: {exc}",
                              stacklevel=2)
            try:
                fig2 = plot_ihs_pi_figure(gene, chrom, core_pos, loc["ihs"], loc["rs"],
                                          vcf_path, GTF_PATH, sv_pos=loc.get("sv_pos"))
                for ext in config.FIGURE_FORMATS:
                    fig2.savefig(ihspi_stem.with_suffix(f".{ext}"), **_SAVEFIG_KW)
                plt.close(fig2)
                saved.append(ihspi_stem.with_suffix(".png"))
            except Exception as exc:
                warnings.warn(f"[figure3] {loc['ihspi_panel']} iHS/pi ({gene}) failed: {exc}",
                              stacklevel=2)
    return saved
