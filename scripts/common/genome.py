"""Shared genome-coordinate helpers for Manhattan-style panels.

Used by the selection scan (Figure 3) and the genome-wide local-ancestry track
(admixture reference). Lives here so figures don't import private helpers from each other.
"""
from __future__ import annotations

# hg38 autosome lengths (bp), for laying chromosomes out left-to-right.
CHROM_SIZES: dict[int, int] = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555, 5: 181538259,
    6: 170805979, 7: 159345973, 8: 145138636, 9: 138394717, 10: 133797422,
    11: 135086622, 12: 133275309, 13: 114364328, 14: 107043718, 15: 101991189,
    16: 90338345, 17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}


def genome_coords(df, chrom_sizes: dict[int, int] = CHROM_SIZES):
    """Cumulative x offsets so chromosomes lay out left-to-right.

    ``df`` needs ``chrom`` and ``pos`` columns. Returns (x, ticks, labels):
    the genome-wide x coordinate per row, and the per-chromosome tick positions
    and labels.
    """
    offset, offsets, ticks, labels = 0, {}, [], []
    for chrom, size in chrom_sizes.items():
        offsets[chrom] = offset
        ticks.append(offset + size / 2)
        labels.append(str(chrom))
        offset += size
    x = df["pos"].to_numpy() + df["chrom"].map(offsets).to_numpy()
    return x, ticks, labels
