"""
Combine cleaned flight trajectory CSV files for 2025 into a single DataFrame,
process and filter by flight flag, and export results to Parquet files.

Steps:
- Reads all CSVs matching 'data/cleaned/2025*.csv'.
- Ensures consistent dtypes for columns.
- Concatenates all data into one DataFrame.
- Adds 'icao24' column (copied from 'callsign') for `traffic` library
- Original labels "Departure" and "Arrival" are inverted in the CSV data, they need to be swapped
- Filter out correct 'Departure' and 'Arrival' flights and writes to Parquet files.
"""

# %%
import glob

import pandas as pd

# %%

files = sorted(glob.glob("data/cleaned/2025*.csv"))

# Define expected data types for each column
# 'timestamp' will be parsed as date, so removed from dtype below
dtype = {
    "flight_id": "str",
    "timestamp": "str",
    "latitude": "float",
    "longitude": "float",
    "track": "float",
    "altitude": "float",
    "callsign": "str",
    "registration": "str",
    "typecode": "str",
    "flag": "str",
}

names = list(dtype.keys())
dtype.pop("timestamp")

# %%

# Read and concatenate all CSV files into a single DataFrame
# Each file is read with consistent column names and dtypes

df = pd.concat(
    [
        pd.read_csv(
            f,
            header=0,
            names=names,
            dtype=dtype,
            parse_dates=["timestamp"],
        )
        for f in files
    ],
    ignore_index=True,
).eval("icao24=callsign")

# Original labels "Departure" and "Arrival" are inverted in the CSV data
df.query("flag=='Departure'").drop("flag", axis=1).to_parquet(
    "data/arrival_trajectories.parquet",
    index=False,
)
df.query("flag=='Arrival'").drop("flag", axis=1).to_parquet(
    "data/departure_trajectories.parquet",
    index=False,
)


# %%
