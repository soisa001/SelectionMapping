"""US and world geometry helpers for geographic figure panels.

Loads the vendored US-states boundary file (``assets/us_states.geojson``; each
feature carries a 2-letter postal ``id`` and full ``name``) as a GeoDataFrame,
reprojected to an equal-area CRS suitable for a national choropleth.
"""
from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache

from . import config

# Non-contiguous states, excluded by default so the CONUS-Albers extent is clean.
NON_CONTIGUOUS = ("AK", "HI")
# US Contiguous Albers Equal Area (good for a lower-48 choropleth).
CONUS_ALBERS_EPSG = 5070

# Full state (+ DC) name -> 2-letter postal code.
STATE_NAME_TO_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}
_ABBREV_SET = set(STATE_NAME_TO_ABBREV.values())
# Longest names first so "West Virginia" matches before "Virginia".
_NAMES_BY_LEN = sorted(STATE_NAME_TO_ABBREV, key=len, reverse=True)


def state_name_to_abbrev(name: str) -> str | None:
    """Map an AoU state concept name to a 2-letter code, robustly.

    Handles full names ("California"), prefixed names ("State of Residence:
    California", "PIIState: CA"), and bare 2-letter codes. Returns None if no
    US state/DC can be identified.
    """
    if not isinstance(name, str):
        return None
    low = name.lower()
    for full in _NAMES_BY_LEN:
        if full.lower() in low:
            return STATE_NAME_TO_ABBREV[full]
    for tok in re.findall(r"\b[A-Za-z]{2}\b", name):
        if tok.upper() in _ABBREV_SET:
            return tok.upper()
    return None


def ensure_proj_data() -> None:
    """Point PROJ at pyproj's bundled database when the env path is broken.

    Some conda envs (e.g. the AoU Workbench) set PROJ_LIB/PROJ_DATA to a path
    that doesn't contain ``proj.db``, which makes reprojection fail with
    ``proj_create_from_database ... failed``. If the configured path is missing
    but pyproj ships its own database, redirect PROJ to it. Best-effort/quiet.
    """
    try:
        import pyproj
        data_dir = pyproj.datadir.get_data_dir()
    except Exception:
        return
    if not data_dir or not os.path.exists(os.path.join(data_dir, "proj.db")):
        return
    current = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if current and os.path.exists(os.path.join(current, "proj.db")):
        return  # existing path is valid; leave it alone
    os.environ["PROJ_LIB"] = data_dir
    os.environ["PROJ_DATA"] = data_dir
    try:
        pyproj.datadir.set_data_dir(data_dir)
    except Exception:
        pass


# Redirect PROJ as early as possible, before any read/transform.
ensure_proj_data()


@lru_cache(maxsize=None)
def _read(path_str: str):
    import geopandas as gpd  # imported lazily so the base install stays light
    return gpd.read_file(path_str)


def load_world():
    """Return world country polygons (Natural Earth) in lon/lat for map panels."""
    return _read(str(config.WORLD_COUNTRIES_GEOJSON))


@lru_cache(maxsize=None)
def load_world_robinson():
    """Return vendored world polygons in the global Robinson projection."""
    world = load_world().copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    ensure_proj_data()
    return world.to_crs(config.WORLD_ROBINSON_CRS)


def project_lonlat(longitude, latitude):
    """Project longitude/latitude arrays into the Figure 1 Robinson CRS."""
    import geopandas as gpd

    points = gpd.GeoSeries(
        gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    ).to_crs(config.WORLD_ROBINSON_CRS)
    return points.x.to_numpy(), points.y.to_numpy()


def load_us_states(exclude: tuple[str, ...] = NON_CONTIGUOUS, epsg: int | None = CONUS_ALBERS_EPSG):
    """Return US state polygons as a GeoDataFrame with a 2-letter ``state`` col.

    ``exclude`` drops states by postal code before reprojection. Alaska is
    dropped by default (per the figure spec); Hawaii is also dropped so the
    CONUS-Albers extent stays clean. Pass ``exclude=("AK",)`` to keep Hawaii.

    Reprojects to ``epsg`` (CONUS Albers by default). If reprojection fails
    (e.g. a broken PROJ install that ``ensure_proj_data`` couldn't repair), the
    geometry is returned in its native lon/lat CRS with a warning so the map
    still renders. Pass ``epsg=None`` to skip reprojection entirely.
    """
    gdf = _read(str(config.US_STATES_GEOJSON)).rename(columns={"id": "state"}).copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    if exclude:
        gdf = gdf[~gdf["state"].isin(set(exclude))].copy()
    if epsg is None:
        return gdf
    ensure_proj_data()
    try:
        return gdf.to_crs(epsg=epsg)
    except Exception as exc:  # pragma: no cover - env-dependent
        warnings.warn(
            f"[geo] reprojection to EPSG:{epsg} failed ({exc}); "
            "rendering in lon/lat (EPSG:4326) instead.",
            stacklevel=2,
        )
        return gdf
