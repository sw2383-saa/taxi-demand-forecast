"""
zones.py
--------
Taxi-zone lookup utilities for translating ``LocationID`` (the
integer used in the trip-record parquet files) into human-readable
borough, zone, and service-zone labels.

The data behind this module is the official NYC TLC Taxi Zone
Lookup CSV at::

    https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

which is linked directly from the same NYC TLC trip-record landing
page that the rest of the package targets:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
Without this lookup the package can still run end-to-end -- every
analysis stage produces correct numbers using zone IDs alone. But
the model output, the visualisations, and the README all become
much easier to interpret once those bare integers are paired with
their real-world names ("zone 138" becomes "zone 138 (LaGuardia
Airport)"). The translation is also useful for any downstream
analysis that wants to aggregate by borough rather than by zone.

The lookup table itself is small enough (265 rows) that we ship a
copy at ``data/taxi_zone_lookup.csv`` inside the package
distribution, so downstream code does not need network access to
use it.
"""

from __future__ import annotations

from pathlib import Path
from importlib.resources import files
from typing import Dict, Optional, Union

import pandas as pd


# Default location of the lookup CSV that ships with the package.
# We resolve it via ``importlib.resources`` rather than via
# ``__file__`` arithmetic because the latter only works in editable
# installs (``pip install -e .``), where ``__file__.parent.parent``
# happens to be the repo root that contains ``data/``. In a normal
# install (``pip install .``, ``pip install <wheel>``, or
# ``pip install --target``), the package gets copied into
# ``site-packages/taxi_demand/`` and there is no sibling ``data/``
# directory anywhere — only files declared as package-data and
# physically located under ``taxi_demand/`` get installed. By
# bundling the CSV at ``taxi_demand/data/taxi_zone_lookup.csv``
# and reading it via ``files("taxi_demand")``, we get the same
# behavior across every install mode (editable, wheel, --target,
# zipapp).
DEFAULT_LOOKUP_PATH = Path(
    str(files("taxi_demand").joinpath("data/taxi_zone_lookup.csv"))
)

# Official URL for the lookup CSV, kept here as a module-level
# constant so callers can re-download it if needed.
TAXI_ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)

# The four columns the official CSV ships with, in canonical order.
# We keep this as a module-level constant so tests can introspect it
# and so a future schema change at NYC TLC produces a clear error
# message rather than a confusing downstream failure.
EXPECTED_COLUMNS = ["LocationID", "Borough", "Zone", "service_zone"]


def load_zone_lookup(
    path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Load the Taxi Zone Lookup CSV from disk.

    The CSV is expected to have the four columns listed in
    :data:`EXPECTED_COLUMNS`, in any order. We do not silently
    rename or reorder columns: a missing column raises
    ``ValueError`` with a clear message, matching the
    strict-validation pattern used throughout the package.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Path to the lookup CSV. If None, the bundled copy at
        ``data/taxi_zone_lookup.csv`` (relative to the package
        root) is used. This makes the function "just work" out of
        the box for the team's default workflow.

    Returns
    -------
    pd.DataFrame
        DataFrame with the four expected columns in canonical
        order, with ``LocationID`` cast to ``int64``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the requested path.
    ValueError
        If the file is missing any of the expected columns or
        contains a non-integer ``LocationID``.
    """
    if path is None:
        path = DEFAULT_LOOKUP_PATH
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Taxi Zone Lookup CSV not found at {p}. Either pass the "
            f"correct path explicitly or download the file from "
            f"{TAXI_ZONE_LOOKUP_URL} into the package's data/ directory."
        )

    df = pd.read_csv(
        p,
        # The official CSV uses the literal string "N/A" as a real
        # value for zones 264 and 265 (catch-all entries for
        # "Unknown" and "Outside of NYC"). Without keep_default_na
        # = False, pandas would coerce those literal strings into
        # NaN, which would silently mislabel any trip ending at
        # zone 265 as "missing borough" rather than as the
        # well-defined "Outside of NYC" catch-all.
        keep_default_na=False,
        na_values=[""],
    )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Zone lookup CSV is missing required columns: {missing}. "
            f"Expected: {EXPECTED_COLUMNS}; got: {list(df.columns)}."
        )

    # Cast LocationID to a real integer dtype. The NYC TLC publishes
    # this column as an integer already, but we coerce defensively
    # so downstream code can rely on the dtype.
    try:
        df["LocationID"] = df["LocationID"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "LocationID column must contain integers."
        ) from exc

    return df[EXPECTED_COLUMNS].copy()


class ZoneResolver:
    """Hash-table-backed resolver from ``LocationID`` to zone metadata.

    Looking up a zone's borough, zone name, or service-zone is an
    O(1) operation because the underlying structure is a Python
    ``dict``. The course covered hash maps as the canonical
    "expected O(1) lookup" data structure in Week 5; we apply the
    same pattern here so a typical end-of-pipeline report (e.g.
    "label the top 10 busiest zones with their borough names") runs
    in O(K) rather than O(K * 265).

    The class is intentionally read-only after construction. If the
    NYC TLC lookup table changes, callers should rebuild the
    resolver from a fresh ``load_zone_lookup`` rather than mutating
    an existing one.
    """

    def __init__(self, lookup_df: pd.DataFrame):
        """Build internal lookup tables.

        Parameters
        ----------
        lookup_df : pd.DataFrame
            The DataFrame returned by :func:`load_zone_lookup`.

        Raises
        ------
        ValueError
            If the DataFrame is missing any of the required columns.
        """
        missing = [c for c in EXPECTED_COLUMNS if c not in lookup_df.columns]
        if missing:
            raise ValueError(
                f"lookup_df is missing required columns: {missing}"
            )

        # Three parallel dicts keyed by LocationID. We could store a
        # single dict whose value is a (borough, zone, service_zone)
        # tuple, but keeping them separate makes the most common
        # access pattern (".borough(123)") more direct and lets us
        # return ``None`` from a single lookup without unpacking.
        self._borough: Dict[int, str] = {}
        self._zone: Dict[int, str] = {}
        self._service_zone: Dict[int, str] = {}

        for _, row in lookup_df.iterrows():
            loc_id = int(row["LocationID"])
            self._borough[loc_id] = str(row["Borough"])
            self._zone[loc_id] = str(row["Zone"])
            self._service_zone[loc_id] = str(row["service_zone"])

        # Cache the set of valid IDs for fast membership tests.
        self._known_ids = set(self._borough.keys())

    def known_ids(self) -> set:
        """Return the set of ``LocationID`` values present in the lookup.

        Useful for tests that want to assert "every zone the
        official table publishes is resolvable", and for analyses
        that need to know which IDs are real before doing any
        lookups.
        """
        return set(self._known_ids)

    def borough(self, location_id: int) -> Optional[str]:
        """Return the borough name for a ``LocationID``, or None if unknown.

        Returning None on miss (rather than raising) is a deliberate
        choice: trip records occasionally contain zone IDs that are
        not in the published lookup (e.g. due to data-collection
        errors at the dispatch base), and we want such cases to
        surface as missing labels in the output rather than as
        runtime crashes deep inside a feature pipeline.
        """
        return self._borough.get(int(location_id))

    def zone(self, location_id: int) -> Optional[str]:
        """Return the zone name for a ``LocationID``, or None if unknown."""
        return self._zone.get(int(location_id))

    def service_zone(self, location_id: int) -> Optional[str]:
        """Return the service-zone label for a ``LocationID``, or None."""
        return self._service_zone.get(int(location_id))

    def annotate(
        self,
        df: pd.DataFrame,
        id_column: str = "PULocationID",
        prefix: str = "PU",
    ) -> pd.DataFrame:
        """Append borough / zone / service-zone columns to a DataFrame.

        The returned DataFrame is a copy; the input is not mutated.
        The new columns are named ``{prefix}_borough``,
        ``{prefix}_zone``, ``{prefix}_service_zone`` so the same
        resolver can be applied twice (once with prefix ``"PU"`` for
        pickup zone, once with prefix ``"DO"`` for dropoff zone)
        without column-name collisions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with an ``id_column`` containing zone IDs.
        id_column : str
            Name of the column holding the zone IDs. Defaults to
            ``"PULocationID"`` to match the column name in the
            HVFHV trip-record schema.
        prefix : str
            Prefix for the new columns. Defaults to ``"PU"``.

        Returns
        -------
        pd.DataFrame
            A copy of ``df`` with three new columns.

        Raises
        ------
        ValueError
            If ``id_column`` is not present in ``df``.
        """
        if id_column not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column {id_column!r}."
            )

        result = df.copy()
        # Use Int64 (capital I, nullable integer) so rows with NaN
        # in the id column don't crash the int conversion.
        ids = result[id_column].astype("Int64")
        result[f"{prefix}_borough"] = ids.map(
            lambda x: self._borough.get(int(x)) if pd.notna(x) else None
        )
        result[f"{prefix}_zone"] = ids.map(
            lambda x: self._zone.get(int(x)) if pd.notna(x) else None
        )
        result[f"{prefix}_service_zone"] = ids.map(
            lambda x: self._service_zone.get(int(x)) if pd.notna(x) else None
        )
        return result

    def label(self, location_id: int) -> str:
        """Return a human-readable single-line label for a zone.

        The format is ``"zone {id} ({zone_name}, {borough})"`` for
        known IDs, or ``"zone {id} (unknown)"`` for IDs that are
        not in the lookup. This is the format used in README
        examples and in the optional zone-annotated visualisations.

        Parameters
        ----------
        location_id : int
            The TLC LocationID.

        Returns
        -------
        str
            One-line human-readable label.
        """
        loc_id = int(location_id)
        zone = self._zone.get(loc_id)
        borough = self._borough.get(loc_id)
        if zone is None or borough is None:
            return f"zone {loc_id} (unknown)"
        return f"zone {loc_id} ({zone}, {borough})"
