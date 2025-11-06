"""
Extract NADP segments from departure trajectories and save to Parquet.

- Loads departure trajectories from Parquet file.
- Defines a function to extract the NADP segment from each flight:
    - Drops the 'track' column and computes cumulative distance.
    - Renames columns for groundspeed and track.
    - Calculates vertical rate (feet per minute).
    - Identifies climb phase and selects segment between 30 and 5000 ft altitude.
    - Keeps only segments shorter than 3 minutes, resampled every 2 seconds.
- Applies extraction to all flights in parallel.
- Saves the resulting NADP segments to a new Parquet file.
"""

# %%
import pandas as pd
from traffic.core import Flight, Traffic

# %%
t = Traffic.from_file("data/departure_trajectories.parquet")


# %%
def extract_nadp_segment(flight: Flight) -> Flight:
    """
    Extract the NADP segment from a flight trajectory.
    Returns the segment if it is shorter than 3 minutes, else None.
    """
    flight = (
        flight.drop(columns="track")
        .cumulative_distance()
        .rename(
            columns={
                "compute_gs": "groundspeed",
                "compute_track": "track",
            }
        )
        .assign(
            vertical_rate=lambda x: x.altitude.diff().bfill()
            / x.timestamp.diff().dt.total_seconds().bfill()
            * 60
        )
        .phases()
    )

    # Get start and stop timestamps for climb phase
    start_climb = flight.query("phase=='CLIMB'").timestamp_min
    stop_climb = flight.query("phase=='CLIMB'").timestamp_max

    flight = flight.between(start_climb, stop_climb).query("30<altitude<5000")

    if flight is not None and flight.shorter_than("3min"):
        return flight.resample("2s").drop(columns=["phase", "track_unwrapped"])
    else:
        return None


# %%
# t = t.sample(5000)

# Apply NADP segment extraction to all flights with parallel process
t_nadp = t.pipe(extract_nadp_segment).eval(24, desc="processing")

# %%

t_nadp.to_parquet("data/nadp_trajectories.parquet")
