"""Public API for the ``taxi_demand`` package.

The names re-exported here are the stable public surface. Anything
not listed in ``__all__`` is considered internal and may change
without notice.
"""

from .evaluate import (
    evaluate_model,
    mae,
    mape,
    naive_baseline,
    rmse,
    time_series_cv_split,
)
from .features import (
    NY_STATE_HOLIDAYS_2019_2026,
    US_FEDERAL_HOLIDAYS_2019_2026,
    add_calendar_features,
    add_lags,
    add_rolling_features,
    aggregate,
    build_features,
)
from .loader import (
    DEFAULT_USER_AGENT,
    HVFHV_URL,
    NUM_TAXI_ZONES,
    REQUIRED_COLUMNS,
    clean_trips,
    download,
    load,
    load_month,
    load_raw,
)
from .model import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_RANDOM_STATE,
    SUPPORTED_MODELS,
    DemandForecaster,
    MultiModelForecaster,
    make_pipeline,
    train_models,
)
from .parallel import parallel_apply_per_zone
from .scraper import (
    SUPPORTED_TAXI_TYPES,
    TLC_TRIP_DATA_PAGE_URL,
    TripDataLink,
    extract_trip_data_links,
    filter_links,
)
from .visualize import plot_demand_heatmap, plot_forecast
from .zones import (
    DEFAULT_LOOKUP_PATH,
    EXPECTED_COLUMNS,
    TAXI_ZONE_LOOKUP_URL,
    ZoneResolver,
    load_zone_lookup,
)

__version__ = "0.2.0"

__all__ = [
    "evaluate_model",
    "mae",
    "mape",
    "naive_baseline",
    "rmse",
    "time_series_cv_split",
    "NY_STATE_HOLIDAYS_2019_2026",
    "US_FEDERAL_HOLIDAYS_2019_2026",
    "add_calendar_features",
    "add_lags",
    "add_rolling_features",
    "aggregate",
    "build_features",
    "DEFAULT_USER_AGENT",
    "HVFHV_URL",
    "NUM_TAXI_ZONES",
    "REQUIRED_COLUMNS",
    "clean_trips",
    "download",
    "load",
    "load_month",
    "load_raw",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_RANDOM_STATE",
    "SUPPORTED_MODELS",
    "DemandForecaster",
    "MultiModelForecaster",
    "make_pipeline",
    "train_models",
    "parallel_apply_per_zone",
    "SUPPORTED_TAXI_TYPES",
    "TLC_TRIP_DATA_PAGE_URL",
    "TripDataLink",
    "extract_trip_data_links",
    "filter_links",
    "plot_demand_heatmap",
    "plot_forecast",
    "DEFAULT_LOOKUP_PATH",
    "EXPECTED_COLUMNS",
    "TAXI_ZONE_LOOKUP_URL",
    "ZoneResolver",
    "load_zone_lookup",
    "__version__",
]
