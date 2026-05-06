# Analysis: January 2026 NYC HVFHV Demand Patterns

This short report turns the raw empirical observations recorded in
[`docs/dataset_facts.md`](docs/dataset_facts.md) into substantive
findings about how ride-share demand actually behaves in New York
City. Every claim below is anchored either to a number we measured
directly from the official NYC TLC HVFHV January 2026 parquet or to
an external authoritative source we cite explicitly.

## Finding 1: federal holidays don't look like holidays in this data

When we computed each day's pickup count z-score against its
same-weekday peers within January 2026, the two federal holidays in
the month behaved quite differently from how a naive holiday-as-dip
model would predict. Martin Luther King, Jr. Day (Monday, January
19) registered a z-score of **+0.92** — almost a full standard
deviation *busier* than the typical January Monday. New Year's Day
(Thursday, January 1) registered **+0.57**, also above its weekday
peers. Neither shows the kind of demand collapse that drives the
holiday signal in commuter-style data sets.

This inverts the usual intuition that "people travel less on
holidays". For ride-share specifically, the pattern is consistent
with the structural facts of the service: New Year's Eve revelers
take rides home in the early hours of January 1, holiday visitors
move around the city throughout the day, and MLK Day brings out
both leisure travelers and people heading to celebratory events. The
implication for our forecasting model is that an `is_holiday`
indicator built from the OPM federal calendar — which is what most
off-the-shelf calendar-feature libraries provide — adds essentially
no signal. The package preserves the federal calendar as the
default for reproducibility but exposes `add_calendar_features`'s
`holidays=` argument so a user with a better-fit calendar (one
derived empirically from the data, for instance) can substitute it
in one line.

## Finding 2: the largest demand dips are weather events

The two single quietest days in the entire month were Sunday
January 25 (322,276 pickups, z = **-4.37**) and Monday January 26
(439,462 pickups, z = **-4.98**) — drops of more than four standard
deviations from the same-weekday baseline. Neither date is a
holiday on any calendar we considered.

Cross-referencing with the National Weather Service's official
post-event summary, January 25-26 was a major winter storm — the
largest to hit the NYC area since the January 31-February 2, 2021
storm. Snow and sleet accumulations reached 8-13 inches near the
coast and 12-17 inches inland, with near-whiteout conditions
during the heaviest precipitation on January 25. The NYC
Department of Sanitation issued a "Snow Alert" — its own press
release describes this as the Department's "higher level"
snow-fighting notification (as opposed to the "lower level"
Winter Operations Advisory) — starting at 1:00 AM Sunday January
25, with plows and salt spreaders pre-deployed across the five
boroughs.

The signal in our HVFHV data is the demand-side counterpart to
this disruption. Riders did not vanish; they simply had nowhere to
be. This is a much stronger driver of demand than any holiday in
the month, which suggests that any production-grade forecasting
system in this domain would benefit substantially from a real-time
weather feed feature — something well outside the scope of this
project but worth flagging as the natural next step.

## Finding 3: the busiest pickup zones match NYC's geography

The 10 busiest pickup zones for the month, ranked by total trip
count and resolved against the bundled
[`taxi_demand/data/taxi_zone_lookup.csv`](taxi_demand/data/taxi_zone_lookup.csv), are
LaGuardia Airport, JFK Airport, Crown Heights North, Bushwick
South, East New York, East Village, Times Sq/Theatre District,
Midtown Center, Park Slope, and TriBeCa/Civic Center. The two
airports occupy ranks 1 and 2 with a combined 695,450 pickups (about
3.3% of the month's total). The remaining eight slots split four to
Brooklyn (Crown Heights North, Bushwick South, East New York, Park
Slope) and four to Manhattan (East Village, Times Sq, Midtown
Center, TriBeCa). Notably, no Bronx, Queens (non-airport), or Staten
Island zone appears in the top ten.

This distribution is consistent with the known structure of NYC's
ride-share market. Yellow-zone Manhattan and the high-density
Brooklyn neighborhoods that have grown into ride-share-heavy
districts dominate volumes, while the airports — which generate
disproportionately many trips per zone because each zone is small
and traffic-concentrated — top the table. The fact that we recover
this real-world structure from raw HVFHV data without any
geographic priors is a useful sanity check that the loader,
aggregator, and zone resolver are all wired up correctly.

## Implications for the forecasting model

The lag-based regression model in `taxi_demand.model.DemandForecaster`
is paired with a naive lag-24h baseline that we report on every
evaluation, so any user can verify whether the trained model adds
real value over "predict that this hour looks like the same hour
yesterday". On the synthetic-pipeline run shipped in
`examples/run_pipeline.py`, the trained model improves on that
baseline across MAE, RMSE, and MAPE. On real January 2026 data,
the same evaluation utilities can be run to check whether the
trained model improves over the baseline; the exact figures
depend on the chosen feature set and split, which is why we
expose the comparison through `evaluate_model` rather than
quoting fixed numbers in the README.
But the analysis above suggests that lag features alone cannot
capture the two largest sources of variation we actually see in the
data: weather shocks and the inverse holiday signal. A future
extension of this project would substitute the federal-calendar
`is_holiday` indicator with a data-derived one (any day where the
within-month z-score of demand falls below some threshold) and add
a weather-feed feature pulling daily NYC snowfall and precipitation
totals from NOAA. Both extensions are straightforward given the
package's `add_calendar_features(..., holidays=...)` and
`build_features(..., add_calendar=True)` extension points.

## Sources

Real-data findings in this document come from direct profiling of
`fhvhv_tripdata_2026-01.parquet` downloaded from the NYC TLC's
official CloudFront endpoint linked at
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>.
The full numerical archive is in [`docs/dataset_facts.md`](docs/dataset_facts.md).

The January 25-26 winter storm description is sourced from the
NWS New York Office's post-event summary at
<https://www.weather.gov/okx/20260125_26> and the NYC Department
of Sanitation Snow Alert at
<https://www.nyc.gov/site/dsny/news/26-006/dsny-issues-snow-alert-sunday-january-25-2026-1-am>.

Federal holiday rules follow 5 U.S.C. §6103 as published by the US
Office of Personnel Management. Zone names follow the official NYC
TLC Taxi Zone Lookup at
<https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv>.
