"""
scraper.py
----------
Optional helper that parses the official NYC TLC trip-record HTML
landing page and returns the list of monthly parquet download URLs.

The page is::

    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The CloudFront URLs follow a fixed pattern, so for the team's primary
workflow (download January 2026) it is perfectly fine to construct the
URL by hand inside :mod:`loader`. This module exists for the secondary
workflow of *discovering* what is currently published -- for instance,
when running the pipeline a few months later, you want to know which
months are now available without editing code. Parsing the rendered
HTML with BeautifulSoup mirrors the approach the course took in
Homework 4's ``stock_parser.py``.

Per the course-forum guidance from Professor Zhang in #81, this module
deliberately does **no** filtering or cleaning of the URLs it finds:
the caller decides which year/month/taxi type to keep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from bs4 import BeautifulSoup


TLC_TRIP_DATA_PAGE_URL = (
    "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
)

# Regex for the ``href`` attribute of the parquet anchor tags. We
# capture (taxi_type, year, month) so we can build a structured record.
# The taxi-type alternation lists every type the TLC publishes, but
# for this project we only consume ``fhvhv``.
_PARQUET_URL_PATTERN = re.compile(
    r"^(?P<base>https?://[^/]+/trip-data/"
    r"(?P<taxi_type>yellow|green|fhv|fhvhv)_tripdata_"
    r"(?P<year>\d{4})-(?P<month>\d{2})\.parquet)$",
    re.IGNORECASE,
)

SUPPORTED_TAXI_TYPES = ("yellow", "green", "fhv", "fhvhv")


@dataclass(frozen=True)
class TripDataLink:
    """A single ``(taxi_type, year, month)`` parquet download link.

    Frozen so the dataclass is hashable, which makes it trivial to
    de-duplicate in a ``set``.
    """

    taxi_type: str
    year: int
    month: int
    url: str


def extract_trip_data_links(html_text: str) -> List[TripDataLink]:
    """Parse a TLC HTML page and return every parquet download link.

    Anchors that do not match the parquet URL pattern are silently
    skipped, mirroring the "tolerate noise" stance from forum #81. We
    also de-duplicate URLs (the rendered TLC page sometimes lists the
    same file under multiple anchors when a section is collapsed and
    expanded) and sort the result by ``(taxi_type, year, month)`` so
    the output is deterministic across runs.

    Parameters
    ----------
    html_text : str
        Rendered HTML of the TLC trip-record-data page.

    Returns
    -------
    list of TripDataLink
        One entry per matching parquet anchor.

    Raises
    ------
    ValueError
        If ``html_text`` is not a string. (We intentionally do *not*
        raise on empty input; an empty page should produce an empty
        list, not an error.)
    """
    if not isinstance(html_text, str):
        raise ValueError("html_text must be a string.")
    if not html_text.strip():
        return []

    soup = BeautifulSoup(html_text, "html.parser")
    seen_urls: set = set()
    out: List[TripDataLink] = []

    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        match = _PARQUET_URL_PATTERN.match(href.strip())
        if match is None:
            continue
        url = match.group("base")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(
            TripDataLink(
                taxi_type=match.group("taxi_type").lower(),
                year=int(match.group("year")),
                month=int(match.group("month")),
                url=url,
            )
        )

    out.sort(key=lambda link: (link.taxi_type, link.year, link.month))
    return out


def filter_links(
    links: Sequence[TripDataLink],
    *,
    taxi_types: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    months: Optional[Sequence[int]] = None,
) -> List[TripDataLink]:
    """Filter a list of :class:`TripDataLink` records.

    All filter arguments are optional and AND-combined; passing
    ``None`` disables the corresponding filter.

    Parameters
    ----------
    links : sequence of TripDataLink
        Input links, typically the output of
        :func:`extract_trip_data_links`.
    taxi_types : sequence of str, keyword-only
        Allowed taxi types. Each entry must be a member of
        :data:`SUPPORTED_TAXI_TYPES` (case-insensitive).
    year : int, keyword-only
        Restrict to a single year.
    months : sequence of int, keyword-only
        Restrict to specific months (each in 1..12).

    Returns
    -------
    list of TripDataLink
        Filtered links, preserving the input order.

    Raises
    ------
    ValueError
        On unknown taxi type or out-of-range month.
    """
    if taxi_types is not None:
        normalised_types = {t.lower() for t in taxi_types}
        unknown = normalised_types - set(SUPPORTED_TAXI_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown taxi types: {sorted(unknown)}. Allowed: "
                f"{SUPPORTED_TAXI_TYPES}."
            )
    else:
        normalised_types = None

    if months is not None:
        for m in months:
            if not isinstance(m, int) or not (1 <= m <= 12):
                raise ValueError("months must be integers in 1..12.")
        month_set = set(months)
    else:
        month_set = None

    out: List[TripDataLink] = []
    for link in links:
        if normalised_types is not None and link.taxi_type not in normalised_types:
            continue
        if year is not None and link.year != int(year):
            continue
        if month_set is not None and link.month not in month_set:
            continue
        out.append(link)
    return out
