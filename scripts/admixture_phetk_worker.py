"""Execute one admixture reference PheTK job inside its isolated environment."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: admixture_phetk_worker.py CONFIG.json")
    config_path = Path(sys.argv[1])
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    parallelization = payload.pop("parallelization")
    n_workers = int(payload.pop("n_workers"))

    from phetk.phewas import PheWAS

    print(
        f"[admixture_reference][phetk-worker] method={payload['method']}, "
        f"phecodes={len(payload.get('phecode_to_process') or [])}, "
        f"workers={n_workers}",
        flush=True,
    )
    phewas = PheWAS(**payload)
    phewas.run(parallelization=parallelization, n_workers=n_workers)


if __name__ == "__main__":
    main()
