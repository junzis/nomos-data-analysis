"""
Ingest VEMMIS enhanced surveillance data for EHAM departures.

- Reads daily VEMMIS CSV files (March 2025).
- Filters to EHAM departures only.
- Extracts climb segments (ALT < 5000 ft).
- Keeps relevant columns: flight tracking + EHS parameters.
- Exports to parquet.
"""

import glob

import pandas as pd

files = sorted(glob.glob("data/vemmis_202503/vemmis_202503*.csv"))
print(f"Found {len(files)} VEMMIS files")

# Read and concatenate all daily files
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# Filter to EHAM departures only
df = df[df["ADEP"] == "EHAM"].copy()
print(f"EHAM departure rows: {len(df)}")
print(f"Unique flights: {df['FLIGHT_ID'].nunique()}")

# Parse timestamps
df["actual_time"] = pd.to_datetime(df["actual_time"])

# Keep relevant columns
cols = [
    "FLIGHT_ID",
    "CALLSIGN",
    "ICAO_ACTYPE",
    "ADEP",
    "DEST",
    "ALT",
    "EHS_IAS",
    "EHS_TAS",
    "EHS_ROCD",
    "EHS_HDG",
    "EHS_MACH",
    "actual_time",
    "lat",
    "lon",
]
df = df[cols].copy()

# Sort by flight and time
df = df.sort_values(["FLIGHT_ID", "actual_time"]).reset_index(drop=True)

# Keep only the departure segment (first 500s from each flight's start)
df["t_from_start"] = df.groupby("FLIGHT_ID")["actual_time"].transform(
    lambda x: (x - x.min()).dt.total_seconds()
)
df = df[df["t_from_start"] <= 500].copy()
df = df.drop(columns="t_from_start")
print(f"Rows within 500s of departure: {len(df)}")

# Remove flights that never reach 5000 ft within 500s (incomplete climbs)
max_alt = df.groupby("FLIGHT_ID")["ALT"].max()
full_departures = max_alt[max_alt >= 5000].index
df = df[df["FLIGHT_ID"].isin(full_departures)].copy()
print(f"Flights reaching 5000 ft within 500s: {len(full_departures)}")

# Filter to climb segments: keep only rows below 5000 ft
df = df[df["ALT"] < 5000].copy()
print(f"Rows below 5000 ft: {len(df)}")

# Drop flights with too few data points (need at least 10 points for a climb)
flight_counts = df.groupby("FLIGHT_ID").size()
valid_flights = flight_counts[flight_counts >= 10].index
df = df[df["FLIGHT_ID"].isin(valid_flights)].copy()
print(f"Flights with >= 10 data points: {df['FLIGHT_ID'].nunique()}")

# Export
df.to_parquet("data/vemmis_departures.parquet", index=False)
print(f"Saved to data/vemmis_departures.parquet")
