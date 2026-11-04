"""Figure 1B — ADMIXTURE (k=2) on the merged AoU + 1000 Genomes fileset.

Naive ADMIXTURE on the admixed AoU cohort alone is unstable, so we run it on a
larger fileset that includes 1000 Genomes reference samples (which anchor the
two components), then split the results for plotting.

Pipeline (runs on the Workbench):
  1. Obtain the merged PLINK fileset (AoU + 1000G; ``srWGS_4193_sample.plink``).
     The study's ADMIXTURE input is intergenic SNVs LD-pruned in Hail with
     hl.ld_prune(r2=0.1, bp_window_size=50000); the exact SNV count is reported
     by ``count_snps`` at run time.
  2. Rewrite the .bim chromosome codes to integers for ADMIXTURE.
  3. Run ADMIXTURE k=2 on ALL samples -> a .Q proportions matrix.
  4. (Downstream, in figure1_data) drop the excluded AoU samples and split
     into the AoU-only plot and the AoU + 1kg AFR/EUR plot.

``load_admixture_frame`` parses .Q + .fam into raw per-sample components; the
AoU/1kg annotation and the 2-sample removal happen in figure1_data. All
inputs/outputs are git-ignored (AoU-derived).

Mirrors the manual starter (ADMIXTURE v1.4.0; ``sed`` chromosome fix;
``admixture <bed> -j16 2`` on ``srWGS_4193_sample.plink``).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from .common import config

# --- Inputs / outputs (git-ignored) -------------------------------------------
# The merged PLINK fileset AND the ADMIXTURE .Q live under data/raw/admixture.
# The .Q is technically a derived artifact, but ADMIXTURE k=2 is an expensive
# one-time Workbench run whose output the manuscript treats as a primary result;
# we deliberately hold it at the source tier so it is NOT wiped by
# clean_run_data (which clears analysis/figure1). Everything the figure code
# recomputes cheaply from it — the annotated/split frames, the summary — is
# written to analysis/figure1 by figure1_data.
ADMIXTURE_DIR = config.RAW_DATA_DIR / "admixture"
# Merged AoU + 1000G fileset (bed/bim/fam share this stem).
MERGED_NAME = "srWGS_4193_sample.plink"
MERGED_STEM = ADMIXTURE_DIR / MERGED_NAME
# GCS source (override via AOU_MERGED_PLINK_STEM if the path differs). Best guess
# is the same bucket as the AoU-only fileset; drop the fileset in locally if not.
GS_MERGED_STEM = os.environ.get(
    "AOU_MERGED_PLINK_STEM",
    f"gs://rw-migration-aou-rw-a3e31f4d/Hail/{MERGED_NAME}",
)
K = 2

ADMIXTURE_VERSION = "1.4.0"
ADMIXTURE_TARBALL = f"admixture_linux-{ADMIXTURE_VERSION}.tar.gz"
ADMIXTURE_URL = f"https://dalexander.github.io/admixture/binaries/{ADMIXTURE_TARBALL}"


def q_path(stem: Path = MERGED_STEM, k: int = K) -> Path:
    return stem.with_name(f"{stem.name}.{k}.Q")


def fam_path(stem: Path = MERGED_STEM) -> Path:
    return stem.with_name(f"{stem.name}.fam")


def bim_path(stem: Path = MERGED_STEM) -> Path:
    return stem.with_name(f"{stem.name}.bim")


def outputs_exist() -> bool:
    return q_path().exists() and fam_path().exists()


def on_workbench() -> bool:
    """True when the tools needed for the live pipeline are present."""
    from .common import gcs
    return gcs.available()


# --- Pure helpers (unit-testable without plink/admixture) ----------------------
def read_fam(path: Path) -> pd.DataFrame:
    """Read a PLINK ``.fam`` (FID IID PID MID SEX PHENO)."""
    return pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["FID", "IID", "PID", "MID", "SEX", "PHENO"], dtype=str,
    )


def count_snps(bim: Path = None) -> int:
    """Number of SNVs = number of lines in the ``.bim``."""
    bim = bim or bim_path()
    with open(bim) as fh:
        return sum(1 for _ in fh)


def fix_bim_chromosomes(bim: Path) -> None:
    """Rewrite chromosome codes to integers for ADMIXTURE (chrM/X/Y -> 25/23/24,
    strip a leading ``chr``)."""
    lines = bim.read_text().splitlines()
    fixed = []
    for ln in lines:
        parts = ln.split("\t") if "\t" in ln else ln.split()
        c = parts[0]
        c = {"chrM": "25", "chrX": "23", "chrY": "24"}.get(c, c)
        if c.startswith("chr"):
            c = c[3:]
        parts[0] = c
        fixed.append("\t".join(parts))
    bim.write_text("\n".join(fixed) + "\n")


def load_admixture_frame(stem: Path = MERGED_STEM, k: int = K) -> pd.DataFrame | None:
    """Parse ``.Q`` + ``.fam`` into sample_id + raw components comp1..compk.

    Returns None if the ADMIXTURE output is absent. The AoU/1kg annotation and
    the removal of the cohort exclusions happen in figure1_data.
    """
    qp, fp = q_path(stem, k), fam_path(stem)
    if not (qp.exists() and fp.exists()):
        return None
    q = pd.read_csv(qp, sep=r"\s+", header=None)
    ids = read_fam(fp)["IID"].astype(str).to_numpy()
    if len(ids) != len(q):
        raise ValueError(f"admixture: {len(q)} Q rows vs {len(ids)} fam samples")
    out = pd.DataFrame({"sample_id": ids})
    for i in range(q.shape[1]):
        out[f"comp{i + 1}"] = q[i].to_numpy()
    return out


# --- Live pipeline (Workbench) -------------------------------------------------
def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def download_merged_fileset() -> None:
    """Copy the merged bed/bim/fam from GCS into ADMIXTURE_DIR (Workbench).

    Uses ``gcloud storage cp`` with a billing project (requester-pays aware).
    """
    from .common import gcs
    if not gcs.available():
        raise RuntimeError("gcloud not found; run on the Workbench.")
    ADMIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("bed", "bim", "fam"):
        dst = MERGED_STEM.with_name(f"{MERGED_NAME}.{ext}")
        r = gcs.gcs_cp(f"{GS_MERGED_STEM}.{ext}", dst)
        if r.returncode != 0:
            raise RuntimeError(
                f"gcloud storage cp failed for {MERGED_NAME}.{ext} "
                f"(rc={r.returncode}): {r.stderr.strip()[:300]}"
            )


def download_admixture(dest: Path = ADMIXTURE_DIR) -> Path:
    """Fetch + extract the ADMIXTURE binary (v1.4.0); return its path."""
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / ADMIXTURE_TARBALL
    if not tar.exists():
        _run(["wget", "-q", ADMIXTURE_URL, "-O", str(tar)])
    _run(["tar", "-xzf", str(tar), "-C", str(dest)])
    for name in ("admixture", "admixture32"):
        hits = list(dest.glob(f"**/{name}"))
        if hits:
            hits[0].chmod(0o755)
            return hits[0]
    raise RuntimeError("admixture binary not found after extraction")


def run_admixture(threads: int = 16, k: int = K) -> Path:
    """Run ADMIXTURE k=2 on ALL merged samples; return the ``.Q`` path."""
    admix = download_admixture()
    bed = MERGED_STEM.with_name(f"{MERGED_NAME}.bed")
    # ADMIXTURE writes <bedbasename>.<k>.Q into the working directory.
    _run([str(admix), bed.name, "-j%d" % threads, str(k)], cwd=ADMIXTURE_DIR)
    produced = q_path()
    if not produced.exists():
        raise RuntimeError(f"ADMIXTURE did not produce {produced}")
    return produced


def build_admixture(threads: int = 16, k: int = K) -> dict:
    """End-to-end on the Workbench: obtain merged fileset -> fix bim -> ADMIXTURE.

    Runs on ALL samples (AoU + 1000G). The excluded AoU samples are removed
    downstream (figure1_data), not here. Returns
    {'n_samples', 'n_snps', 'q_path'}.
    """
    if not MERGED_STEM.with_name(f"{MERGED_NAME}.bed").exists():
        download_merged_fileset()
    fix_bim_chromosomes(bim_path())
    produced = run_admixture(threads=threads, k=k)
    return {
        "n_samples": len(read_fam(fam_path())),
        "n_snps": count_snps(),
        "q_path": str(produced),
    }
