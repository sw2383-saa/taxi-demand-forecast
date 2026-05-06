# taxi-demand-forecast

## Team

- Emily Wang (sw2383)
- Yuhe Jiang (yj596)
- Runhua Cao (rc993)

## Purpose

This package builds an end-to-end forecasting pipeline for hourly NYC
ride-share pickup demand at the pickup-zone level. Given the New York City
Taxi & Limousine Commission's High Volume For-Hire Vehicle (HVFHV) trip
records, the package downloads the raw parquet, cleans it, aggregates it
into hourly per-zone pickup counts, engineers lag and calendar features,
and trains scikit-learn forecasting models that we evaluate against a
naive lag-24h baseline.

The project's deliverable is the engineering pipeline itself, not a
particular accuracy number. We score every model with MAE, RMSE, and
optionally MAPE, and we offer a time-series-aware cross-validation
splitter so users can evaluate without leaking information from future
hours into the past.

### A note on the modelling-approach choice

Our original task plan listed two alternative modelling approaches
for this project: a SARIMA-style classical time-series model, and
a regression with engineered lag features. We chose the second
path, and we want to record that choice transparently here rather
than burying it.

The decision came down to fit-for-purpose. HVFHV demand exhibits
very strong *double seasonality* — both an hour-of-day cycle and a
day-of-week cycle, with the two interacting (Saturday 6 PM does
not behave like Wednesday 6 PM, and Wednesday 6 PM does not behave
like Wednesday 4 AM). Capturing this with a classical SARIMA model
requires either a SARIMAX specification with Fourier-term
exogenous regressors or a hierarchical seasonal-naive structure;
either route carries substantially more research-grade modelling
work than a course final project budgets for, and the gain — given
that this course's grading is explicitly on code quality rather
than predictive performance (forum thread #96) — would be modest.
The lag-regression path, by contrast, lets the same kinds of
seasonal effects be expressed as `lag_24h`, `lag_168h`, and
calendar features that a linear or tree-based learner can fit in
seconds, fits naturally inside the scikit-learn `Pipeline` idiom
the course covered in Weeks 12-13, and produces an evaluation
story that is easy to read.

To make the trade-off honest we built in a guardrail: every
evaluation reports a *naive lag-24h baseline* alongside the
trained model. That baseline says "predict that this hour will
look exactly like the same hour 24 hours ago", which is the
strongest non-trivial benchmark a SARIMA-style approach would
need to beat to justify itself. By reporting the baseline on
every evaluation we let the user verify whether the trained
model adds real value under their chosen split and feature set,
rather than asking the reader to take any quoted improvement
on faith. A future continuation of the project, with more
modelling time available, would naturally extend in the SARIMAX
direction and use the same baseline as the floor to beat.

## Dataset

We use the official NYC TLC HVFHV trip records, published monthly at
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. The
parquet files live on a CloudFront CDN at the URL pattern

```
https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{YYYY}-{MM}.parquet
```

The default workflow uses January 2026 (`fhvhv_tripdata_2026-01.parquet`)
as our fixed, reproducible reference month: every empirical figure in
this project is anchored to that specific file. The numerical facts
are archived in [`docs/dataset_facts.md`](docs/dataset_facts.md) and
mirrored as Python constants in `tests/test_data_consistency.py`, so
accidental edits to the documented assumptions are caught by tests.
If TLC re-publishes the file, the profiling scripts in
[`docs/dataset_profiling.md`](docs/dataset_profiling.md) should be
re-run and the recorded checksum and facts updated deliberately.
We have downloaded this file directly from the official TLC
CloudFront URL above and run a full profiling pass on its contents.
In short, the file is
482.4 MB on disk, contains 20,940,373 trip records spanning the entire month
(`2026-01-01 00:00:00` through `2026-01-31 23:59:59`), is structured as 25
columns over 20 row groups, and exposes zero null `PULocationID` /
`DOLocationID` values and zero out-of-range zone identifiers. With respect
to the limited validation checks used in this project — non-null required
columns and zone IDs in `[1, 265]` — this month behaves as already clean:
`clean_trips` removes exactly zero rows, a fact that is locked down by an
empirical test. This narrow result does not override TLC's broader caveat
that submitted trip records are not guaranteed to be complete or error-free;
our `clean_trips` filter is defensive in spirit (per forum thread #81 it lets
us "keep the raw load and the cleaning step separate"), so the same code
remains useful on future months that may not be as well-formed.

The `loader` module downloads the file lazily on demand, caches it on
disk, and reads only the three columns we actually need (`pickup_datetime`,
`PULocationID`, `DOLocationID`) so the pipeline remains memory-friendly
even though the monthly HVFHV file is roughly 500 MB on disk. The
downloader includes retry-with-backoff and an atomic-write pattern, both
of which are lessons from the course's Week 4 treatment of resilient
pipelines.

A note on column choice: the HVFHV schema actually contains three
distinct timestamp columns — `request_datetime` (when the rider
requested the ride), `on_scene_datetime` (when the driver arrived), and
`pickup_datetime` (when the rider was actually picked up). Because the
team's task statement says "predict pickup counts by zone and hour", we
deliberately use `pickup_datetime`. A future variant aimed at
**dispatch-side demand prediction** would want to swap in
`request_datetime` instead, but for the assignment as defined,
`pickup_datetime` is the correct field.

The `taxi_demand.scraper` module is provided as an *optional* helper. It
parses the TLC HTML landing page with BeautifulSoup and returns the full
list of available monthly download URLs, so the same code can pick up
new months as they are published without code changes. You do not need
to use it for the default January 2026 workflow; the URL pattern above
is constructed directly inside `loader.download` for that case.

### Taxi Zone Lookup table

Alongside the trip-record parquet files, the official NYC TLC page
links a small CSV that maps each `LocationID` (the zone integer that
appears in the trip records) to its borough, zone name, and
service-zone label. The file lives at
`https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` and
contains 265 rows covering the full range of valid `LocationID` values
(with 264 reserved for "Unknown" and 265 reserved for "Outside of
NYC"). We ship a copy of this CSV at
[`taxi_demand/data/taxi_zone_lookup.csv`](taxi_demand/data/taxi_zone_lookup.csv)
*inside* the installed package itself, so `load_zone_lookup` can find
it via `importlib.resources` regardless of how the package was
installed (editable, normal `pip install`, wheel, or
`pip install --target`) and without needing network access at runtime.

The `taxi_demand.zones` module provides `load_zone_lookup` (loads the
bundled CSV) and a `ZoneResolver` class that exposes `O(1)` lookups
from `LocationID` to borough, zone, or service-zone, plus an
`annotate` helper that joins those labels onto a trip DataFrame in
one call. See the Usage section below for an example.

## Install

```bash
git clone https://github.com/sw2383-saa/taxi-demand-forecast.git
cd taxi-demand-forecast
pip install -e .
# Or, with the development tooling:
pip install -e .[dev]
```

The package targets Python 3.9 or newer. The hard runtime dependencies are
listed in both `pyproject.toml` and `requirements.txt`.

## Usage

The pipeline reads top-to-bottom: download → load → engineer features →
train → evaluate. The simplest end-to-end script that uses the
backwards-compatible `DemandForecaster` class looks like this:

```python
from taxi_demand import (
    DemandForecaster, evaluate_model, load_month, build_features
)

raw_df = load_month(year=2026, month=1)
# fill_missing_hours=True ensures lag_24h means "24 hours earlier"
# rather than "24 records earlier" -- important for any zone with
# hour-level gaps in its pickup stream.
feature_df = build_features(raw_df, fill_missing_hours=True)

# Time-respecting split: cut at a specific hour so every (zone, hour)
# row before the cut goes to train and every row at-or-after the cut
# goes to test. We split on unique hours so the same hour is never
# present in both folds. Splitting by `iloc[:80%]` directly would
# silently bias the split because `build_features` returns rows in
# (zone_id, hour) order, not pure hour order.
hours = sorted(feature_df["hour"].unique())
cut_hour = hours[int(len(hours) * 0.8)]
train_df = feature_df[feature_df["hour"] < cut_hour]
test_df  = feature_df[feature_df["hour"] >= cut_hour]

model = DemandForecaster()
model.fit(train_df)

results = evaluate_model(model, test_df)
print(results)
# {'model_mae': ..., 'model_rmse': ..., 'baseline_mae': ..., 'baseline_rmse': ...}

model.save("model.joblib")
```

If you prefer the multi-model variant that the project plan
originally called for (Pipeline + StandardScaler + three models),
use `MultiModelForecaster` or the lower-level `train_models` helper:

```python
from taxi_demand import (
    build_features, load_month, MultiModelForecaster
)

raw_df = load_month(year=2026, month=1)
feature_df = build_features(
    raw_df,
    lags=(1, 24, 168),       # 1h, 24h, 1-week lags
    rolling_windows=(24,),    # 24h causal rolling mean & std
    add_calendar=True,        # hour-of-day, day-of-week, weekend, holiday
    fill_missing_hours=True,  # complete (zone, hour) grid for correct lag semantics
)

# Drop warm-up rows whose lag values are NaN, then do a
# time-respecting split (cut on a specific hour so the same
# hour is never in both folds).
feature_df = feature_df.dropna()
hours = sorted(feature_df["hour"].unique())
cut_hour = hours[int(len(hours) * 0.8)]
train_df = feature_df[feature_df["hour"] < cut_hour]
test_df  = feature_df[feature_df["hour"] >= cut_hour]

multi = MultiModelForecaster(
    feature_columns=[
        "lag_1h", "lag_24h", "lag_168h",
        "roll_mean_24h", "roll_std_24h",
        "hour_of_day", "day_of_week", "is_weekend", "is_holiday",
    ],
)
multi.fit(train_df)

# Compare all three families on the test fold using MAE / RMSE.
# This is the pattern that powers Stage 5 of examples/run_pipeline.py
# and produces a comparable output here.
from taxi_demand.evaluate import mae, rmse
test_clean = test_df.dropna(subset=multi.feature_columns + ["lag_24h"])
y_true = test_clean["demand"].to_numpy()
print(f"{'model':<16s} {'MAE':>10s} {'RMSE':>10s}")
print("-" * 38)
print(f"{'baseline-lag24':<16s} "
      f"{mae(y_true, test_clean['lag_24h']):>10.4f} "
      f"{rmse(y_true, test_clean['lag_24h']):>10.4f}")
for name in multi.models_:
    preds = multi.predict(test_clean, model_name=name)
    print(f"{name:<16s} {mae(y_true, preds):>10.4f} {rmse(y_true, preds):>10.4f}")
```

To do feature engineering in parallel across pickup zones (useful for the
full month, which has hundreds of zones), wrap any per-zone function with
`parallel_apply_per_zone`:

```python
from taxi_demand import aggregate, load_month, parallel_apply_per_zone
from taxi_demand.features import add_lags, add_rolling_features

raw_df = load_month(year=2026, month=1)
hourly = aggregate(raw_df, fill_missing_hours=True)

def per_zone_features(zone_df):
    zone_df = add_lags(zone_df)
    zone_df = add_rolling_features(zone_df, windows=(24,))
    return zone_df

featured = parallel_apply_per_zone(hourly, per_zone_features, n_workers=4)
```

### Annotating zones with human-readable names

Trip records use integer `LocationID` values, but model output and
plots are much easier to interpret with real zone names. The
`ZoneResolver` class joins the bundled lookup CSV onto any DataFrame
with a zone-id column:

```python
from taxi_demand import (
    ZoneResolver, load_zone_lookup, aggregate, load_month
)

raw_df = load_month(year=2026, month=1)
hourly = aggregate(raw_df, fill_missing_hours=True)

resolver = ZoneResolver(load_zone_lookup())
hourly_with_names = resolver.annotate(
    hourly,
    id_column="zone_id",
    prefix="zone",
)
print(hourly_with_names.head())
#    zone_id   hour                 demand   zone_borough  zone_zone           zone_service_zone
# 0  1        2026-01-01 00:00:00  ...      EWR            Newark Airport      EWR
# 1  4        2026-01-01 00:00:00  ...      Manhattan      Alphabet City       Yellow Zone
# ...

# A single-zone label, e.g. for plot titles or report tables:
print(resolver.label(132))   # zone 132 (JFK Airport, Queens)
print(resolver.label(138))   # zone 138 (LaGuardia Airport, Queens)
```

In our January 2026 reference month, the two busiest pickup zones
were 138 (LaGuardia Airport) and 132 (JFK Airport), accounting for
roughly 3.3% of all pickups in the month combined — a sanity check
that the dataset's real-world structure matches NYC's actual
ride-share volume distribution.

### Running the example pipeline

The fastest way to see the package run from end to end is to
execute the bundled example script. After installing the package
in editable mode (`pip install -e .` or `pip install -e .[dev]`,
as documented in the Install section above), run from the
repo root:

```bash
python examples/run_pipeline.py            # synthetic mode, ~5 seconds
python examples/run_pipeline.py --real-data # real January 2026 data, 1-3 minutes
```

The default synthetic mode generates a small in-memory HVFHV-shaped
dataset and pushes it through every stage of the pipeline (load,
clean, aggregate, feature-engineer, train, evaluate, rank), so a
reader can verify that all the plumbing works without needing to
download the full 482 MB parquet file. The `--real-data` mode
substitutes the official NYC TLC January 2026 file for the
synthetic input and produces the same shape of output on real data.

For a discussion of what the real data actually looks like and
what we learned from it, see [`ANALYSIS.md`](ANALYSIS.md).

To run the test suite and check coverage:

```bash
pytest -v
python -m coverage run -m pytest
python -m coverage report
```

## Repository layout

```
taxi-demand-forecast/
├── taxi_demand/
│   ├── __init__.py        public API
│   ├── loader.py          download + load_raw + clean_trips + load
│   ├── scraper.py         optional BeautifulSoup parser for TLC HTML
│   ├── zones.py           Taxi Zone Lookup loader + ZoneResolver
│   ├── features.py        aggregate + lag / rolling / calendar features
│   ├── parallel.py        multiprocessing.Pool wrapper for per-zone funcs
│   ├── model.py           DemandForecaster + MultiModelForecaster + Pipeline
│   ├── evaluate.py        MAE / RMSE / MAPE + naive baseline + time-series CV
│   ├── visualize.py       plot_forecast + plot_demand_heatmap
│   └── data/
│       └── taxi_zone_lookup.csv   bundled zone-id-to-name lookup
├── examples/
│   └── run_pipeline.py    end-to-end pipeline demo (synthetic + real-data)
├── tests/                 pytest test suite (>80% line and branch coverage)
├── data/                  cache directory for downloaded parquet files
├── docs/
│   ├── dataset_facts.md       empirical snapshot of Jan 2026 data
│   └── dataset_profiling.md   reproducible profiling scripts
├── ANALYSIS.md            findings about NYC ride-share patterns
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
├── requirements.txt
└── .gitignore
```

## Acknowledgement of AI assistance

We used generative AI tools for brainstorming,
drafting boilerplate, and scaffolding tests. All code, analysis, and
final design decisions were reviewed and edited by the team, and we
take responsibility for the correctness and quality of the submitted
work.

## License

Released under the [MIT License](LICENSE). The NYC TLC trip-record
dataset itself is published by the New York City Taxi & Limousine
Commission and remains subject to its own terms of use.
