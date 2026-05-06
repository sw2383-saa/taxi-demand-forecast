# Dataset profiling scripts

This document lists the four small Python scripts that, when run
against a freshly-downloaded `fhvhv_tripdata_2026-01.parquet`,
reproduce every numerical fact recorded in
[`dataset_facts.md`](dataset_facts.md). They are kept here as plain
markdown rather than as runnable `.py` files because they are
diagnostic / archival in nature — they only need to be re-executed
when the team wants to refresh the empirical snapshot, not on every
package install.

To re-run them, copy any block below into a Python session whose
working directory contains the downloaded
`fhvhv_tripdata_2026-01.parquet` file.

## Script 1: file size, basic identity, and SHA-256 checksum

```python
import hashlib
from pathlib import Path

DEST = Path("./fhvhv_tripdata_2026-01.parquet")
print(f"File size: {DEST.stat().st_size:,} bytes "
      f"({DEST.stat().st_size / 1024 / 1024:.1f} MB)")

# Compute SHA-256 in 1 MB streaming chunks so we don't load the
# whole 482 MB file into memory just to hash it. The resulting
# hex digest is the value archived in dataset_facts.md and is the
# single canonical identifier for the exact bytes-on-disk version
# of the file we profiled.
h = hashlib.sha256()
with DEST.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(f"SHA-256: {h.hexdigest()}")
```

## Script 2: schema inspection (footer-only read)

```python
import pyarrow.parquet as pq
from pathlib import Path

DEST = Path("./fhvhv_tripdata_2026-01.parquet")

schema = pq.read_schema(DEST)
print(f"Number of columns: {len(schema.names)}")
for i, (name, ftype) in enumerate(zip(schema.names, schema.types), 1):
    print(f"  {i:2d}. {name:<35s} {ftype}")

metadata = pq.read_metadata(DEST)
print(f"Total rows: {metadata.num_rows:,}")
print(f"Row groups: {metadata.num_row_groups}")
```

## Script 3: daily aggregation and z-score-based holiday signal

```python
import numpy as np
import pandas as pd
from pathlib import Path

DEST = Path("./fhvhv_tripdata_2026-01.parquet")

df = pd.read_parquet(DEST, columns=["pickup_datetime"])
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['pickup_datetime'].min()} to "
      f"{df['pickup_datetime'].max()}")

daily = (
    df.assign(date=df["pickup_datetime"].dt.date)
      .groupby("date").size().sort_index()
)
print("Pickup count by date:")
for d, n in daily.items():
    weekday = pd.Timestamp(d).day_name()[:3]
    print(f"  {d}  {weekday}  {n:>10,}")

# Z-score per day vs. same-weekday peers within the month
print("Z-scores (negative = unusually quiet):")
by_wd = {}
for d, n in daily.items():
    by_wd.setdefault(pd.Timestamp(d).day_name(), []).append((d, n))

for d, n in daily.items():
    wd = pd.Timestamp(d).day_name()
    peers = [c for dd, c in by_wd[wd] if dd != d]
    mean, std = np.mean(peers), np.std(peers, ddof=1)
    z = (n - mean) / std if std > 0 else 0
    print(f"  {d}: z={z:+.2f}")
```

## Script 4: zone-id range validation and top-zone summary

```python
import pandas as pd
from pathlib import Path

DEST = Path("./fhvhv_tripdata_2026-01.parquet")

df = pd.read_parquet(
    DEST, columns=["pickup_datetime", "PULocationID", "DOLocationID"]
)

for col in ("PULocationID", "DOLocationID"):
    print(f"{col}:")
    print(f"  dtype          {df[col].dtype}")
    print(f"  null count     {df[col].isnull().sum():,}")
    print(f"  min            {df[col].min()}")
    print(f"  max            {df[col].max()}")
    print(f"  unique values  {df[col].nunique()}")
    out = ((df[col] < 1) | (df[col] > 265)).sum()
    print(f"  out-of-range   {out:,}")

print("Top 10 busiest pickup zones:")
for zone_id, count in df["PULocationID"].value_counts().head(10).items():
    print(f"  Zone {zone_id:>3}: {count:>10,}")
```
