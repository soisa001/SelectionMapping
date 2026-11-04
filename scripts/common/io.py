"""Idempotency helpers.

The whole pipeline is built around two flags that flow from the notebooks down
into every script:

    clean_run_data   -- when False (default), reuse cached intermediates in
                        analysis/ if present; when True, recompute them.
    clean_run_plots  -- when False (default), reuse figures on disk if present;
                        when True, redraw them.

``clean_run_data=True`` implies redrawing plots as well, since the data they
depend on changed. Callers pass both flags explicitly; the helpers here only
decide "recompute or reuse" given a target path and a flag.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def needs_build(target: Path | list[Path], clean_run: bool) -> bool:
    """Return True if ``target`` must be (re)built.

    Build is required when the flag forces a clean run or when any target path
    is missing.
    """
    targets = [target] if isinstance(target, Path) else list(target)
    if clean_run:
        return True
    return any(not t.exists() for t in targets)


def cached_frame(
    path: Path,
    clean_run: bool,
    build: Callable[[], pd.DataFrame],
    *,
    expected_columns: "list[str] | tuple[str, ...] | None" = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Load a cached CSV DataFrame, or build+save it when stale.

    ``build`` is invoked when the cache is missing, ``clean_run`` is set, or the
    cached frame is missing any of ``expected_columns`` (a schema-drift guard:
    an old cache written by earlier code is transparently rebuilt instead of
    crashing a downstream consumer).
    """
    if not needs_build(path, clean_run):
        df = pd.read_csv(path, **read_kwargs)
        if expected_columns is None or set(expected_columns).issubset(df.columns):
            return df
        # Stale schema -> fall through and rebuild.
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def cached_array(
    path: Path,
    clean_run: bool,
    build: Callable[[], np.ndarray],
) -> np.ndarray:
    """Load a cached ``.npy`` array, or build+save it when stale."""
    if needs_build(path, clean_run):
        arr = np.asarray(build())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr
    return np.load(path, allow_pickle=False)


def cached_json(
    path: Path,
    clean_run: bool,
    build: Callable[[], dict],
) -> dict:
    """Load a cached JSON blob (e.g. fitted parameters), or build+save it."""
    if needs_build(path, clean_run):
        obj = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True))
        return obj
    return json.loads(path.read_text())
