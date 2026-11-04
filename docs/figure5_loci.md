# Figure 5 locus panels

`notebooks/Figure_5.ipynb` calls the standard-locus and SV-linked workflows. Both use existing input manifests and verified caches. Outputs are written under `figures/figure5/loci/` as PNG/PDF pairs, with run manifests.

On the authorized analysis VM, install the Figure 5 runtime through `uv`. For ADH, ZMYM1, and the other standard loci, run the restartable `stage`, `selection`, and `compose` phases. For THRAP3 and other SV-linked loci, the phases are `tasks`, `selection`, `finemap`, and `compose`. Use `--help` for input-path overrides.

```bash
uv run --python 3.12 --with-requirements requirements-figure5.txt \
  python -m scripts.figure5_locus_composites --phase compose --threads 4
uv run --python 3.12 --with-requirements requirements-figure5.txt \
  python -m scripts.figure5_sv_locus_composites --phase compose --threads 4
```

Each locus panel combines PheWAS, regional association, selection tracks, and gene annotations as applicable. Additional locus examples are retained.
