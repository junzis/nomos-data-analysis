"""
Extract V2 and NADP milestone features from VEMMIS departure data.

For each flight:
- Extracts V2 (IAS at rotation, ~35ft).
- Computes IAS and ROCD at milestone altitudes (800, 1500, 3000 ft).
- Computes mean ROCD in altitude bands (800-1500, 1500-3000 ft).
- Builds altitude-indexed IAS and ROCD curves (every 100ft, 200-4000ft).
- Exports features to parquet.
"""

import numpy as np
import pandas as pd


def interpolate_at_altitude(group, column, target_alt):
    """Interpolate a column value at a specific altitude using linear interpolation."""
    alts = group["ALT"].values
    vals = group[column].values

    # Remove NaN pairs
    mask = ~(np.isnan(alts) | np.isnan(vals))
    alts, vals = alts[mask], vals[mask]

    if len(alts) < 2:
        return np.nan

    # Check target is within range
    if target_alt < alts.min() or target_alt > alts.max():
        return np.nan

    return float(np.interp(target_alt, alts, vals))


def extract_v2(group):
    """
    Extract V2 proxy from the initial climb segment.

    Uses the minimum IAS in the 200-800ft altitude band as the V2 baseline.
    This captures the stabilized initial climb speed rather than the rotation
    speed (Vr), which is typically slightly higher due to the pitch-up transient.
    """
    df = group.sort_values("actual_time").reset_index(drop=True)

    # Find points in the 200-800ft band
    climb_band = df[(df["ALT"] >= 200) & (df["ALT"] <= 800)]
    if climb_band.empty:
        return np.nan

    ias_values = climb_band["EHS_IAS"].dropna()
    if ias_values.empty:
        return np.nan

    return ias_values.min()


def extract_flight_features(flight_id, group):
    """Extract all milestone features for one flight."""
    group = group.sort_values("actual_time").reset_index(drop=True)

    # Extract V2
    v2 = extract_v2(group)
    if pd.isna(v2) or v2 < 80 or v2 > 250:
        return None  # Invalid V2, skip flight

    # Filter to airborne only for milestone extraction
    airborne = group[group["ALT"] >= 50].copy()
    if len(airborne) < 5:
        return None

    # Check altitude range covers the milestones
    alt_max = airborne["ALT"].max()
    if alt_max < 3000:
        return None  # Flight doesn't reach 3000ft in our data

    # IAS at milestone altitudes
    ias_800 = interpolate_at_altitude(airborne, "EHS_IAS", 800)
    ias_1500 = interpolate_at_altitude(airborne, "EHS_IAS", 1500)
    ias_3000 = interpolate_at_altitude(airborne, "EHS_IAS", 3000)

    # Delta IAS (relative to V2)
    delta_ias_800 = ias_800 - v2 if not pd.isna(ias_800) else np.nan
    delta_ias_1500 = ias_1500 - v2 if not pd.isna(ias_1500) else np.nan
    delta_ias_3000 = ias_3000 - v2 if not pd.isna(ias_3000) else np.nan

    # Mean ROCD in altitude bands
    band_low = airborne[(airborne["ALT"] >= 800) & (airborne["ALT"] < 1500)]
    band_high = airborne[(airborne["ALT"] >= 1500) & (airborne["ALT"] < 3000)]

    mean_rocd_800_1500 = band_low["EHS_ROCD"].mean() if len(band_low) > 0 else np.nan
    mean_rocd_1500_3000 = band_high["EHS_ROCD"].mean() if len(band_high) > 0 else np.nan

    # Skip if any critical feature is missing
    if any(pd.isna(x) for x in [delta_ias_800, delta_ias_1500, delta_ias_3000,
                                  mean_rocd_800_1500, mean_rocd_1500_3000]):
        return None

    # Altitude-indexed curves (every 100ft from 200 to 4000)
    alt_grid = np.arange(200, 3600, 100)
    ias_curve = [interpolate_at_altitude(airborne, "EHS_IAS", a) for a in alt_grid]
    rocd_curve = [interpolate_at_altitude(airborne, "EHS_ROCD", a) for a in alt_grid]

    # Convert curves to delta_ias (relative to V2)
    delta_ias_curve = [v - v2 if not pd.isna(v) else np.nan for v in ias_curve]

    actype = group["ICAO_ACTYPE"].iloc[0]
    callsign = str(group["CALLSIGN"].iloc[0]).strip()
    airline = callsign[:3]
    start = airborne["actual_time"].iloc[0]

    return {
        "flight_id": flight_id,
        "typecode": actype,
        "callsign": callsign,
        "airline": airline,
        "start": start,
        "v2": v2,
        "delta_ias_800": delta_ias_800,
        "delta_ias_1500": delta_ias_1500,
        "delta_ias_3000": delta_ias_3000,
        "mean_rocd_800_1500": mean_rocd_800_1500,
        "mean_rocd_1500_3000": mean_rocd_1500_3000,
        # Store curves as lists for delta computation later
        **{f"delta_ias_alt_{a}": v for a, v in zip(alt_grid, delta_ias_curve)},
        **{f"rocd_alt_{a}": v for a, v in zip(alt_grid, rocd_curve)},
    }


# --- Main ---

df = pd.read_parquet("data/vemmis_departures.parquet")
print(f"Loaded {df['FLIGHT_ID'].nunique()} flights")

results = []
for flight_id, group in df.groupby("FLIGHT_ID"):
    feat = extract_flight_features(flight_id, group)
    if feat is not None:
        results.append(feat)

df_features = pd.DataFrame(results)
print(f"Extracted features for {len(df_features)} flights")
print(f"Dropped {df['FLIGHT_ID'].nunique() - len(df_features)} flights (missing data)")

# Summary statistics
print(f"\nV2 statistics:")
print(df_features["v2"].describe())
print(f"\nDelta IAS at 1500ft (key NADP discriminator):")
print(df_features["delta_ias_1500"].describe())
print(f"\nMean ROCD 800-1500ft:")
print(df_features["mean_rocd_800_1500"].describe())

df_features.to_parquet("data/nadp_features.parquet", index=False)
print(f"\nSaved to data/nadp_features.parquet")
