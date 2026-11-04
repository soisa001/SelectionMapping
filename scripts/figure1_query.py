"""BigQuery extraction for Figure 1C — AoU sampling by US state.

This module runs ONLY on the All of Us Researcher Workbench, where the CDR
BigQuery dataset and the ``WORKSPACE_CDR`` / ``GOOGLE_PROJECT`` environment
variables are available. It pulls the state of residence for the long-read WGS
cohort and aggregates to per-state counts. Nothing here runs (or is needed) for
local plotting from cached data.

Data Use Agreement note
------------------------
All of Us participant-level data must not leave the workbench. This module and
its caller (``figure1_data``) write both a sample-level table (person_id ->
state) and a state summary to ``data/raw/`` for reuse, but those files are
git-ignored (see ``data/README.md``) and must not be committed. Raw counts may
be kept on disk; the small-count disclosure rule (no cells of 1-19 shown; render
as >= 20) is applied only at PLOT time.

Ported from the R ``usmap`` starter (bigrquery), which selected
``state_of_residence`` for participants with ``has_lr_whole_genome_variant = 1``.
"""
from __future__ import annotations

import os

import pandas as pd

from .common import geo, samples

# Env vars that must be present for a live query.
REQUIRED_ENV = ("WORKSPACE_CDR", "GOOGLE_PROJECT")


def on_workbench() -> bool:
    """True when the AoU Workbench BigQuery environment is available."""
    return all(os.environ.get(k) for k in REQUIRED_ENV)


def _state_query() -> str:
    cdr = os.environ["WORKSPACE_CDR"]
    return f"""
        SELECT
            person.person_id,
            state_concept.concept_name AS state_name
        FROM `{cdr}.person` person
        LEFT JOIN `{cdr}.concept` state_concept
            ON person.state_of_residence_concept_id = state_concept.concept_id
        WHERE person.person_id IN (
            SELECT DISTINCT person_id
            FROM `{cdr}.cb_search_person`
            WHERE has_lr_whole_genome_variant = 1
        )
    """


def _read_gbq(sql: str) -> pd.DataFrame:
    """Run a standard-SQL query, tolerating pandas/pandas-gbq API differences.

    Prefer ``pandas_gbq.read_gbq`` (the supported entry point); fall back to
    ``pandas.read_gbq`` only if pandas-gbq is not installed. ``pandas.read_gbq``
    is deprecated and emits a FutureWarning, hence the ordering.
    """
    project = os.environ.get("GOOGLE_PROJECT")
    use_storage = "BIGQUERY_STORAGE_API_ENABLED" in os.environ
    try:
        import pandas_gbq
        return pandas_gbq.read_gbq(sql, project_id=project, use_bqstorage_api=use_storage)
    except ImportError:
        pass
    read_gbq = getattr(pd, "read_gbq", None)
    if read_gbq is not None:
        return read_gbq(sql, dialect="standard", project_id=project,
                        use_bqstorage_api=use_storage)
    raise ImportError(
        "BigQuery pull needs pandas-gbq (or a pandas with read_gbq). "
        "Install pandas-gbq."
    )


def fetch_state_samples() -> pd.DataFrame:
    """Query the LR-WGS cohort and return SAMPLE-LEVEL state of residence.

    Returns a DataFrame with columns ``person_id`` and ``state`` (2-letter),
    with rows whose state could not be mapped to a US state/DC dropped. Counts
    are aggregated by the caller.
    """
    if not on_workbench():
        raise RuntimeError(
            "fetch_state_samples requires the AoU Workbench BigQuery environment "
            f"({', '.join(REQUIRED_ENV)} must be set)."
        )
    df = _read_gbq(_state_query())
    # Restrict to the canonical 1,025 good-sample list (post-VWB migration).
    ids = samples.load_aou_sample_list(download=True)
    if ids is not None:
        df = df[df["person_id"].astype(str).isin(ids)]
    else:
        import warnings
        warnings.warn(
            "[figure1_query] AOU_sample_list.txt unavailable; state counts are "
            "over all LR-WGS persons, not restricted to the 1,025 set.",
            stacklevel=2,
        )
    df["state"] = df["state_name"].map(geo.state_name_to_abbrev)
    return (
        df.dropna(subset=["state"])[["person_id", "state"]]
        .reset_index(drop=True)
    )
