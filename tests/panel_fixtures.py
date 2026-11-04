"""Small, explicit inputs for isolated renderer tests."""
import json

import numpy as np
import pandas as pd


def finemap_frame(locus):
    positions = locus["lead_pos"] + np.arange(-4, 5) * 10_000
    p_values = np.array([0.1, 0.03, 0.01, 0.001, 1e-8, 0.002, 0.01, 0.04, 0.2])
    return pd.DataFrame({
        "variant_id": [f"{locus['chrom']}_{pos}_A_G" for pos in positions],
        "chrom": locus["chrom"], "pos": positions, "is_sv": False, "sv_type": "",
        "pip": [0, 0, 0, 0.2, 0.6, 0.2, 0, 0, 0],
        "cs": ["", "", "", "L1", "L1", "L1", "", "", ""],
        "p_score": p_values, "neg_log_p": -np.log10(p_values),
        "r2_to_lead": [0.1, 0.2, 0.4, 0.8, 1, 0.8, 0.4, 0.2, 0.1],
        "_is_lead": positions == locus["lead_pos"],
    })


def discovery_frame():
    count = 446
    category = ["discovery hit"] * 425 + ["meta-significant"] * 17 + ["SV-linked hit"] * 4
    return pd.DataFrame({
        "variant_tag": [f"1_{1000 + i}_A_G" for i in range(count)],
        "phecode": [f"TEST_{i}" for i in range(count)],
        "phecode_string": [f"Test phenotype {i}" for i in range(count)],
        "maf": np.geomspace(0.001, 0.4, count), "abs_beta": np.linspace(0.1, 2, count),
        "beta": np.linspace(0.1, 2, count), "se": 0.1, "p": 1e-5,
        "category": category, "is_meta_significant": [c == "meta-significant" for c in category],
        "is_sv_linked": [c == "SV-linked hit" for c in category],
        "sv_annotation": [""] * 442 + [f"chr1:{1000 + i} (72 bp DEL)" for i in range(4)],
    })


def idw_frames():
    projected = pd.DataFrame({
        "sample_id": ["sample1", "sample2"], "pred_longitude": [3.9, -2.0],
        "pred_latitude": [7.4, 54.0], "region": "AoU",
    })
    sources = pd.DataFrame({
        "pop_code": ["YRI", "GBR"], "lon": [3.9, -2.0], "lat": [7.4, 54.0],
        "region": ["AFR", "EUR"], "source_project": "1000 Genomes", "n_reference": 1,
    })
    return projected, sources


def admixture_frame():
    from scripts.common.genome import CHROM_SIZES
    from scripts import figure2_sv
    start, end = 63_748_785, 63_837_789
    region = {"region_id": "test-region", "chrom": 9, "start": start, "end": end,
              "direction": "AFR-excess", "peak_z": -6, "n_markers": 3}
    positions = np.linspace(start - 100_000, end + 100_000, 9).astype(int)
    fractions = np.array([0.2, 0.19, 0.15, 0.08, 0.05, 0.08, 0.15, 0.19, 0.2])
    zoom = pd.DataFrame({"chrom": 9, "pos": positions, "eur_frac": fractions,
                         "eur_z": (fractions - 0.2) / 0.02,
                         "in_region": (positions >= start) & (positions <= end)})
    finemap = pd.DataFrame({
        "variant_id": [f"chr9:{p}:A:G" for p in positions], "chrom": "chr9", "pos": positions,
        "pip": [0, 0, 0.2, 0, 0.6, 0, 0.2, 0, 0], "is_sv": False, "sv_type": "",
        "sv_length_bp": 0, "beta_score": 0.5, "r2_to_min_p": [0.1, 0.2, 0.7, 0.8, 1, 0.8, 0.7, 0.2, 0.1],
        "credible_set": "CS1", "p_score": [0.1, 0.03, 1e-6, 0.001, 1e-8, 0.001, 1e-6, 0.03, 0.1],
        "minimum_p_variant": np.arange(9) == 4,
        "region_id": "test-region", "phecode": "401.1", "window_start": start - 100_000,
        "window_end": end + 100_000,
    })
    for index, kind in [(2, "INS"), (6, "DEL")]:
        finemap.loc[index, ["variant_id", "is_sv", "sv_type", "sv_length_bp"]] = [f"chr9:{positions[index]}:{kind}:80", True, kind, 80]
    finemap["neg_log10_p"] = -np.log10(finemap["p_score"])
    svs = pd.DataFrame([{
        "variant_id": finemap.loc[i, "variant_id"], "pos": positions[i], "sv_type": finemap.loc[i, "sv_type"],
        "sv_length_bp": 80, "overall_eur_frac": fractions[i], "nearest_lai_marker": positions[i],
        "ancestry_method": "overall_nearest_marker", "n_carrier_haplotypes": 30,
    } for i in (2, 6)])
    genome = pd.DataFrame([{"chrom": chrom, "pos": pos, "eur_frac": fraction,
                           "eur_z": (fraction - 0.2) / 0.02, "telomere_trimmed": False}
                          for chrom, size in CHROM_SIZES.items()
                          for pos, fraction in [(size // 3, 0.21), (size // 2, 0.2), (size * 2 // 3, 0.05 if chrom == 9 else 0.19)]])
    r2 = np.r_[np.linspace(-0.02, 0.49, 144), np.linspace(0.51, 0.8, 36)]
    return {
        "tracts": pd.DataFrame([{"sample_id": "test", "chrom": c, "hap": h, "start": 1,
                                  "end": size, "ancestry": "AFR" if h == 1 else "EUR"}
                                 for c, size in CHROM_SIZES.items() for h in (1, 2)]),
        "masks": pd.DataFrame(columns=["chrom", "start", "end"]),
        "genome_eur": genome, "region_zoom": zoom, "regions": pd.DataFrame([region]),
        "stats": {"mean_eur": 0.2, "sd_eur": 0.02, "z_threshold": 4.42},
        "display_region": region, "finemap": finemap, "finemap_hit": {**region, "phecode": "401.1", "p_meta": 1e-5},
        "finemap_genes": pd.DataFrame([{"chrom": "chr9", "start": start, "end": end, "gene": "TEST",
                                        "transcript_id": "TEST1", "strand": "+", "exons": json.dumps([[start, start + 1000], [end - 1000, end]])}]),
        "plrs_holdout": pd.DataFrame({"true_eur_dosage": [0.1, 0.3, 0.5, 0.9], "predicted_eur_dosage": [0.15, 0.35, 0.45, 0.85]}),
        "plrs_metrics": {"r2_holdout": 0.8, "rmse_holdout": 0.1, "r2_null": 0},
        "plrs_models": pd.DataFrame({"region_id": [f"test{i}" for i in range(len(r2))], "n_snvs": np.arange(len(r2)) + 5,
                                      "r2_holdout": r2, "retained_for_phewas": r2 > 0.5}),
        "chrom_sizes": CHROM_SIZES, "locus_window": (start - 100_000, end + 100_000),
        "regional_svs": figure2_sv.match_finemap_svs(svs, finemap),
    }
