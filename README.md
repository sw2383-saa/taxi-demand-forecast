# taxi-demand-forecast

## Purpose

Predicting hourly Uber/Lyft pickup demand by NYC taxi zone using Jan 2026 HVFHV (High Volume For-Hire Vehicle) trip data. Built as a Python package with a full model training and evaluation pipeline.

## Dataset

NYC TLC High Volume FHV, January 2026:
```
https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2026-01.parquet
```
The data is downloaded automatically by the loader — do not commit the parquet file to the repo.

## Install

```bash
git clone https://github.com/sw2383-saa/taxi-demand-forecast.git
cd taxi-demand-forecast
pip install -e .
```

## Usage

```python
from taxi_demand import DemandForecaster, evaluate_model
from taxi_demand.loader import load_month
from taxi_demand.features import build_features

# Download and load January 2026 HVFHV data
raw_df = load_month(year=2026, month=1)

# Aggregate into hourly zone demand + lag features
feature_df = build_features(raw_df)

# Train/test split (80/20 by time)
split_idx = int(len(feature_df) * 0.8)
train_df = feature_df.iloc[:split_idx]
test_df = feature_df.iloc[split_idx:]

# Train model
model = DemandForecaster()
model.fit(train_df)

# Evaluate
results = evaluate_model(model, test_df)
print(results)
# {'model_mae': ..., 'model_rmse': ..., 'baseline_mae': ..., 'baseline_rmse': ...}

# Optional: save model
model.save("model.joblib")
```
