"""Complete SuSiE credible-set membership and minimum-p representative statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_CONTRACT = "all_credible_members_minimum_score_p_representative_v1"
STAT_COLUMNS = ["variant_id", "pos", "beta_score", "se_score", "p_score", "pip",
                "mac", "n_carriers", "n_case_carriers", "n_control_carriers", "missing_rate",
                "is_sv", "sv_type", "sv_length_bp"]
MEMBER_COLUMNS = ["chrom", "start", "end", "region_id", "phecode", "phecode_string",
                  "credible_set_ids", *STAT_COLUMNS, "is_representative", "representative_variant",
                  "n_samples", "n_cases", "n_controls", "n_pcs", "sex_scope",
                  "window_start", "window_end", "analysis_contract", "fine_map_table"]


def memberships(frame):
    return frame["credible_set"].fillna("").astype(str).map(
        lambda value: ",".join(sorted({part.strip() for part in value.split(",") if part.strip()})))


def variant_statistics(row):
    required = set(STAT_COLUMNS) - {"is_sv", "sv_length_bp"}
    missing = required.difference(row.index)
    if missing:
        raise ValueError(f"Fine-map table is missing variant statistics: {sorted(missing)}")
    result = {column: row[column] for column in STAT_COLUMNS if column in row.index}
    result["sv_type"] = "" if pd.isna(row["sv_type"]) else str(row["sv_type"]).strip().upper()
    result["is_sv"] = result["sv_type"] in {"INS", "DEL"}
    result["sv_length_bp"] = row.get("sv_length_bp", np.nan) if result["is_sv"] else 0
    return result


def summary_fields(frame):
    """Represent the whole fine-map window by its minimum score-test p-value."""
    representative = frame.loc[frame["p_score"].idxmin()]
    sets = memberships(frame)
    members = frame.loc[sets.ne("")].sort_values(["pos", "variant_id"], kind="stable")
    result = {
        "credible_set_variants": ";".join(members["variant_id"].astype(str)),
        "representative_selection": "minimum_score_p_across_finemap_window",
        "representative_in_credible_set": bool(sets.loc[representative.name]),
        "representative_credible_set_ids": sets.loc[representative.name],
    }
    for key, value in variant_statistics(representative).items():
        result[f"representative_{'variant' if key == 'variant_id' else key}"] = value
    return result


def member_rows(frame, hit, table):
    """One row per distinct credible-set variant, preserving all set memberships."""
    representative = frame.loc[frame["p_score"].idxmin(), "variant_id"]
    sets = memberships(frame)
    members = frame.loc[sets.ne("")].sort_values(["pos", "variant_id"], kind="stable")
    rows = []
    for index, variant in members.iterrows():
        row = {key: hit[key] for key in ("chrom", "start", "end", "region_id", "phecode", "phecode_string")}
        row["chrom"] = "chr" + str(row["chrom"]).removeprefix("chr")
        row.update(variant_statistics(variant))
        row.update({key: variant[key] for key in ("n_samples", "n_cases", "n_pcs", "sex_scope",
                                                  "window_start", "window_end", "analysis_contract")})
        row.update(credible_set_ids=sets.loc[index],
                   is_representative=variant["variant_id"] == representative,
                   representative_variant=representative,
                   n_controls=int(variant["n_samples"] - variant["n_cases"]), fine_map_table=str(table))
        rows.append(row)
    return rows


def write_members(rows, csv_path: Path, tsv_path: Path):
    """Write both formats atomically, including headers when no sets exist."""
    frame = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    if frame.duplicated(["region_id", "phecode", "variant_id"]).any():
        raise ValueError("Duplicate association/variant in credible-set table")
    numeric = [column for column in STAT_COLUMNS if column not in {"variant_id", "is_sv", "sv_type"}]
    for path, separator in ((csv_path, ","), (tsv_path, "\t")):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, sep=separator, index=False)
        checked = pd.read_csv(temporary, sep=separator, keep_default_na=False, float_precision="round_trip",
                              dtype={"region_id": str, "phecode": str, "variant_id": str})
        for column in numeric:
            checked[column] = pd.to_numeric(checked[column].replace("", np.nan), errors="raise")
        pd.testing.assert_frame_equal(checked, frame, check_dtype=False, rtol=1e-12, atol=0)
        temporary.replace(path)
