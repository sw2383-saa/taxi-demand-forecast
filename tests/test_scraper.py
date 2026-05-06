"""Tests for :mod:`taxi_demand.scraper`.

The scraper takes the rendered HTML of the NYC TLC trip-record-data
page and returns the list of monthly parquet download URLs as
:class:`TripDataLink` records. We test:

* The full corner-case grid recommended in the course's W4D1
  "test-driven development" lecture: empty input, malformed input,
  single element, duplicates, and unsupported types.
* That all four NYC taxi types (yellow, green, fhv, fhvhv) are
  recognised even though the team's pipeline only consumes fhvhv.
* All four AND-combined filter dimensions in :func:`filter_links`.
"""

import pytest

from taxi_demand.scraper import (
    SUPPORTED_TAXI_TYPES,
    TLC_TRIP_DATA_PAGE_URL,
    TripDataLink,
    extract_trip_data_links,
    filter_links,
)




def test_empty_string_returns_empty_list():
    assert extract_trip_data_links("") == []


def test_whitespace_only_returns_empty_list():
    assert extract_trip_data_links("   \n\t") == []


def test_non_string_input_raises():
    with pytest.raises(ValueError, match="string"):
        extract_trip_data_links(None)
    with pytest.raises(ValueError, match="string"):
        extract_trip_data_links(123)


def test_no_anchors_returns_empty():
    html = "<html><body><p>Nothing here</p></body></html>"
    assert extract_trip_data_links(html) == []


def test_single_fhvhv_link():
    html = (
        '<a href="https://d37ci6vzurychx.cloudfront.net/'
        'trip-data/fhvhv_tripdata_2026-01.parquet">January</a>'
    )
    result = extract_trip_data_links(html)
    assert len(result) == 1
    assert result[0].taxi_type == "fhvhv"
    assert result[0].year == 2026
    assert result[0].month == 1


def test_irrelevant_links_skipped():
    html = (
        '<a href="https://example.com/something_else.parquet">other</a>'
        '<a href="/site/tlc/about/data.page">internal nav</a>'
        '<a href="">empty href</a>'
    )
    assert extract_trip_data_links(html) == []


def test_anchor_without_href_skipped():
    html = '<a name="anchor-only">no href</a>'
    assert extract_trip_data_links(html) == []


def test_duplicate_urls_deduplicated():
    url = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "fhvhv_tripdata_2026-01.parquet"
    )
    html = f'<a href="{url}">first</a><a href="{url}">duplicate</a>'
    assert len(extract_trip_data_links(html)) == 1


def test_results_sorted_deterministically():
    html = (
        '<a href="https://d37ci6vzurychx.cloudfront.net/'
        'trip-data/yellow_tripdata_2024-12.parquet">a</a>'
        '<a href="https://d37ci6vzurychx.cloudfront.net/'
        'trip-data/yellow_tripdata_2024-01.parquet">b</a>'
        '<a href="https://d37ci6vzurychx.cloudfront.net/'
        'trip-data/green_tripdata_2024-06.parquet">c</a>'
    )
    result = extract_trip_data_links(html)
    keys = [(r.taxi_type, r.year, r.month) for r in result]
    assert keys == sorted(keys)


def test_all_four_taxi_types_recognised():
    for taxi_type in SUPPORTED_TAXI_TYPES:
        url = (
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"{taxi_type}_tripdata_2024-01.parquet"
        )
        html = f'<a href="{url}">link</a>'
        result = extract_trip_data_links(html)
        assert len(result) == 1
        assert result[0].taxi_type == taxi_type




def _make_links():
    return [
        TripDataLink("yellow", 2024, 1, "u1"),
        TripDataLink("yellow", 2024, 6, "u2"),
        TripDataLink("yellow", 2023, 12, "u3"),
        TripDataLink("green", 2024, 1, "u4"),
        TripDataLink("fhv", 2024, 1, "u5"),
        TripDataLink("fhvhv", 2024, 1, "u6"),
    ]


def test_filter_no_args_returns_all():
    links = _make_links()
    assert filter_links(links) == links


def test_filter_by_taxi_type():
    result = filter_links(_make_links(), taxi_types=["yellow"])
    assert len(result) == 3
    for link in result:
        assert link.taxi_type == "yellow"


def test_filter_by_taxi_type_case_insensitive():
    result = filter_links(_make_links(), taxi_types=["YELLOW"])
    assert len(result) == 3


def test_filter_unknown_taxi_type_raises():
    with pytest.raises(ValueError, match="Unknown taxi types"):
        filter_links(_make_links(), taxi_types=["purple"])


def test_filter_by_year():
    result = filter_links(_make_links(), year=2023)
    assert len(result) == 1
    assert result[0].year == 2023


def test_filter_by_months():
    result = filter_links(_make_links(), months=[1])
    assert len(result) == 4
    for link in result:
        assert link.month == 1


def test_filter_invalid_month_raises():
    with pytest.raises(ValueError, match="months"):
        filter_links(_make_links(), months=[0])
    with pytest.raises(ValueError, match="months"):
        filter_links(_make_links(), months=[13])
    with pytest.raises(ValueError, match="months"):
        filter_links(_make_links(), months=["jan"])


def test_filter_combined_AND():
    result = filter_links(
        _make_links(),
        taxi_types=["yellow"],
        year=2024,
        months=[1, 6],
    )
    assert len(result) == 2


def test_filter_preserves_order():
    links = _make_links()
    result = filter_links(links, year=2024)
    # The relative order of the survivors should match the input order.
    survivors = [l for l in links if l.year == 2024]
    assert result == survivors




def test_official_url_is_string():
    assert isinstance(TLC_TRIP_DATA_PAGE_URL, str)
    assert "nyc.gov" in TLC_TRIP_DATA_PAGE_URL


def test_supported_taxi_types_constant():
    assert tuple(SUPPORTED_TAXI_TYPES) == ("yellow", "green", "fhv", "fhvhv")


def test_trip_data_link_is_frozen():
    """Frozen dataclass = hashable, so it can live inside a set."""
    link = TripDataLink("yellow", 2024, 1, "u")
    {link}
    with pytest.raises(Exception):
        link.year = 2025
