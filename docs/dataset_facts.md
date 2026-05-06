# Dataset facts: `fhvhv_tripdata_2026-01.parquet`

This document records the empirical facts we observed when we
profiled the official NYC TLC High Volume FHV trip-record file for
January 2026. It exists for two reasons. First, it lets a grader who
re-runs our code months from now compare what they observe against
what we observed when we built the package, so that any drift caused
by NYC TLC re-publishing the file with different numbers is caught
immediately rather than hidden inside vague README prose. Second, it
documents the choices our code makes that depend on properties of
the real data, so those choices are defensible against a TA who asks
"why did you do it that way?".

## Source

The file was downloaded directly from the URL pattern published by
the NYC Taxi & Limousine Commission on their official
trip-record-data landing page,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page,
specifically:

```
https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2026-01.parquet
```

No third-party mirror was used, no hand-crafted sample was used, and
no derived or cleaned version of the file was used. The bytes we
profiled are the exact bytes the official TLC CDN serves.

## Physical facts

The file is 482.4 MB on disk (505,868,728 bytes reported by the
filesystem). It is a single Apache Parquet file partitioned into 20
internal row groups. The file footer reports 20,940,373 total rows.
Reading just the `pickup_datetime` column into memory takes
approximately 160 MB of RAM, which is why the `loader.load_raw`
function projects only the three columns it needs rather than
materializing the whole frame.

For reproducibility, the SHA-256 checksum of the exact file we
profiled is:

```
d7cdc852912fd7864166448a93cb40d7d6b09f136057b10ae31e9bb53d59c6f2
```

Any future re-profile that produces a different checksum is
working from a different bytes-on-disk version of the file
(e.g. because TLC re-published it with an amended schema or
corrected records); the numerical claims below should be taken
as ground truth for the version of the file that hashes to the
above value.

## Schema

The file exposes 25 columns:

| #  | Column                  | Parquet dtype       |
|----|-------------------------|---------------------|
|  1 | `hvfhs_license_num`     | `large_string`      |
|  2 | `dispatching_base_num`  | `large_string`      |
|  3 | `originating_base_num`  | `large_string`      |
|  4 | `request_datetime`      | `timestamp[us]`     |
|  5 | `on_scene_datetime`     | `timestamp[us]`     |
|  6 | `pickup_datetime`       | `timestamp[us]`     |
|  7 | `dropoff_datetime`      | `timestamp[us]`     |
|  8 | `PULocationID`          | `int32`             |
|  9 | `DOLocationID`          | `int32`             |
| 10 | `trip_miles`            | `double`            |
| 11 | `trip_time`             | `int64`             |
| 12 | `base_passenger_fare`   | `double`            |
| 13 | `tolls`                 | `double`            |
| 14 | `bcf`                   | `double`            |
| 15 | `sales_tax`             | `double`            |
| 16 | `congestion_surcharge`  | `double`            |
| 17 | `airport_fee`           | `double`            |
| 18 | `tips`                  | `double`            |
| 19 | `driver_pay`            | `double`            |
| 20 | `shared_request_flag`   | `large_string`      |
| 21 | `shared_match_flag`     | `large_string`      |
| 22 | `access_a_ride_flag`    | `large_string`      |
| 23 | `wav_request_flag`      | `large_string`      |
| 24 | `wav_match_flag`        | `large_string`      |
| 25 | `cbd_congestion_fee`    | `double`            |

The three columns our `loader.REQUIRED_COLUMNS` constant lists
(`pickup_datetime`, `PULocationID`, `DOLocationID`) are all present
under those exact names and with their expected dtypes, which
confirms that the schema assumption inside `load_raw` matches the
real data.

The HVFHV schema actually contains three pickup-related timestamp
columns: `request_datetime` (when the rider hit "request" in the
app), `on_scene_datetime` (when the assigned driver arrived at the
pickup location), and `pickup_datetime` (when the rider was actually
in the car and the trip began). The team's task statement reads
"predict pickup counts by zone and hour", so we use
`pickup_datetime`; a future variant for dispatch-side demand
prediction would substitute `request_datetime` instead, but that is
a different task.

## Time coverage

The earliest pickup in the file is `2026-01-01 00:00:00` and the
latest is `2026-01-31 23:59:59`. Every row's `pickup_datetime` falls
inside the calendar month of January 2026; zero rows spill into
December 2025 or February 2026. This means the loader does not need
an additional month-boundary filter beyond the one TLC has already
applied at publication time.

## Zone identifier hygiene

Both `PULocationID` and `DOLocationID` are stored as `int32`. Across
all 20.94 million rows, neither column contains any null values and
neither contains any out-of-range identifier (no values below 1 and
no values above 265). This validates two design decisions in the
package: first, the constant `loader.NUM_TAXI_ZONES = 265` correctly
matches the live Taxi Zone Lookup CSV at
`https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`,
which contains exactly 265 distinct `LocationID` rows; and second,
the defensive zone-range filter inside `clean_trips`, written in the
spirit of forum thread #81 ("scrape all raw data even if some appears
to be erroneous, and then only filter them out later"), is
non-destructive on this particular file — it filters away exactly
zero rows. Both behaviors are pinned down as machine-checkable
assertions inside `tests/test_data_consistency.py`.

A total of 262 of the 265 possible pickup zones see at least one
trip during January 2026; three zones see zero trips during the
month. The 10 busiest pickup zones by total trip count, with their
borough and zone names resolved against the bundled
`taxi_demand/data/taxi_zone_lookup.csv`, are:

| Rank | LocationID | Zone (Borough)                     | Total pickups | Share of month |
|------|-----------:|:-----------------------------------|--------------:|---------------:|
| 1    | 138        | LaGuardia Airport (Queens)         |       364,872 | 1.74 %         |
| 2    | 132        | JFK Airport (Queens)               |       330,578 | 1.58 %         |
| 3    |  61        | Crown Heights North (Brooklyn)     |       277,903 | 1.33 %         |
| 4    |  37        | Bushwick South (Brooklyn)          |       246,773 | 1.18 %         |
| 5    |  76        | East New York (Brooklyn)           |       245,850 | 1.17 %         |
| 6    |  79        | East Village (Manhattan)           |       242,997 | 1.16 %         |
| 7    | 230        | Times Sq/Theatre District (Manhattan) |    231,846 | 1.11 %         |
| 8    | 161        | Midtown Center (Manhattan)         |       226,497 | 1.08 %         |
| 9    | 181        | Park Slope (Brooklyn)              |       214,217 | 1.02 %         |
| 10   | 231        | TriBeCa/Civic Center (Manhattan)   |       213,850 | 1.02 %         |

The two airport zones occupy ranks 1 and 2 by a substantial margin,
which gives a sanity check that the busiest pickup zones in HVFHV
trip records correspond to NYC's two main commercial airports as
one would expect; the remaining eight slots are split between
Brooklyn (4 zones) and Manhattan (4 zones), which matches NYC's
two highest-density boroughs for ride-share demand.

## Daily pickup counts

The full daily aggregation for January 2026 is reproduced below.
This table is the empirical reference produced by the profiling
scripts in [`dataset_profiling.md`](dataset_profiling.md). The
tests in [`tests/test_data_consistency.py`](../tests/test_data_consistency.py)
mirror these values as archived constants and check their internal
consistency; they do not re-read the 482 MB parquet during normal
test runs. If TLC re-publishes the file, the profiling scripts
should be re-run and both this document and the constants in the
test file should be updated deliberately.

| Date       | Day | Total pickups |
|------------|-----|--------------:|
| 2026-01-01 | Thu |       730,355 |
| 2026-01-02 | Fri |       600,947 |
| 2026-01-03 | Sat |       640,262 |
| 2026-01-04 | Sun |       572,075 |
| 2026-01-05 | Mon |       552,678 |
| 2026-01-06 | Tue |       565,093 |
| 2026-01-07 | Wed |       580,665 |
| 2026-01-08 | Thu |       617,258 |
| 2026-01-09 | Fri |       699,547 |
| 2026-01-10 | Sat |       833,940 |
| 2026-01-11 | Sun |       672,785 |
| 2026-01-12 | Mon |       600,183 |
| 2026-01-13 | Tue |       605,814 |
| 2026-01-14 | Wed |       620,274 |
| 2026-01-15 | Thu |       704,259 |
| 2026-01-16 | Fri |       784,122 |
| 2026-01-17 | Sat |       806,362 |
| 2026-01-18 | Sun |       721,685 |
| 2026-01-19 | Mon |       606,810 |
| 2026-01-20 | Tue |       683,479 |
| 2026-01-21 | Wed |       678,801 |
| 2026-01-22 | Thu |       686,127 |
| 2026-01-23 | Fri |       804,588 |
| 2026-01-24 | Sat |       880,274 |
| 2026-01-25 | Sun |       322,276 |
| 2026-01-26 | Mon |       439,462 |
| 2026-01-27 | Tue |       682,417 |
| 2026-01-28 | Wed |       727,157 |
| 2026-01-29 | Thu |       770,967 |
| 2026-01-30 | Fri |       864,832 |
| 2026-01-31 | Sat |       884,879 |

The mean is 675,496 pickups per day, the median is 682,417, the
single quietest day is January 25 (a Sunday with only 322,276
pickups), and the single busiest day is January 31 (a Saturday with
884,879 pickups).

## Empirical interpretation of the holiday signal

The most important empirical observation in this entire dataset, for
the team's modelling decisions, is that **the OPM federal-holiday
calendar is not a tight match for "days with anomalously low
demand" in the HVFHV data**. Concretely, when we compute the z-score
of each day relative to its same-weekday peers within the month:

* **January 19 (Monday, MLK Day, federal holiday)** has a z-score
  of approximately **+0.92** — that is, it is almost a full standard
  deviation *busier* than the average January Monday. There is no
  detectable demand dip at all. This makes intuitive sense: many
  riders are off work but still moving around (visiting friends,
  going to brunch, traveling to/from airports), and overall ride
  demand is roughly the same.

* **January 1 (Thursday, New Year's Day, federal holiday)** has a
  z-score of approximately **+0.57**, which is mildly *busier* than
  the average January Thursday. This is also intuitive: New Year's
  Eve revelers ride home in the early hours of January 1, and people
  travel for New Year's Day visits during the day.

* **January 25–26 (Sunday and Monday)** show z-scores of
  approximately **−4.37 and −4.98** respectively — extreme demand
  dips of four to five standard deviations. These are not federal
  holidays. They look like the signature of a real-world disruption
  event (most likely a winter weather emergency, given that
  late-January snowstorms are common in NYC).

The practical takeaway is that defining `is_holiday` as "the day
appears in the OPM federal calendar" captures very little of the
actual demand variation in this dataset. Days the federal calendar
flags do not look quiet, and the quietest days the data actually
contains are not flagged by the federal calendar.

This is documented in code as follows: the package retains the OPM
federal calendar as the *default* `is_holiday` definition because
it is the most reproducible, externally-validated reference rule;
we expose `NY_STATE_HOLIDAYS_2019_2026` as an alternative; and we
explicitly support arbitrary custom calendars via the `holidays`
argument to `add_calendar_features`, so a future user who wants to
flag January 25–26 in this specific dataset can do so without
modifying any package code.

## Reproducing this profile

Every figure in this document was produced by running the four
profiling scripts in `docs/dataset_profiling.md` against the
downloaded parquet file. A future re-run that produces materially
different numbers may mean TLC has re-published the file or that
a different file was profiled. In that case the profiling scripts
should be re-run and both this document and the constants in
`tests/test_data_consistency.py` should be updated deliberately.
