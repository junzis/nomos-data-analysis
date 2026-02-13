"""
Classify flights into NADP1/NADP2 by matching against ICAO reference profiles,
and compute delta scores measuring deviation from the matched standard.

- Defines ICAO reference profiles as piecewise-linear functions of altitude.
- Classifies each flight by Euclidean distance to reference milestone features.
- Computes per-flight delta scores as RMS residual from matched reference curve.
- Generates visualizations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({"font.size": 14})

# --- ICAO Reference Profiles ---
# Piecewise-linear reference curves as functions of altitude.
# delta_ias = IAS - V2 at each altitude
# rocd = vertical rate (ft/min) at each altitude

# Reference altitudes for curves (every 100ft from 200 to 4000)
ALT_GRID = np.arange(200, 4100, 100)


def _interp_profile(alt_breakpoints, values, alt_grid):
    """Interpolate a piecewise-linear profile onto a regular altitude grid."""
    return np.interp(alt_grid, alt_breakpoints, values)


# NADP1: maintain V2+15kt until 3000ft, then accelerate
NADP1_DELTA_IAS = _interp_profile(
    [0, 800, 3000, 4000],
    [0, 15, 15, 70],
    ALT_GRID,
)
NADP1_ROCD = _interp_profile(
    [0, 800, 3000, 4000],
    [1800, 1800, 1700, 1400],
    ALT_GRID,
)

# NADP2: accelerate from 800ft onward
NADP2_DELTA_IAS = _interp_profile(
    [0, 800, 1500, 3000, 4000],
    [0, 15, 45, 85, 90],
    ALT_GRID,
)
NADP2_ROCD = _interp_profile(
    [0, 800, 1500, 3000, 4000],
    [1800, 1800, 1500, 1400, 1200],
    ALT_GRID,
)

# Reference milestone feature vectors (for classification)
# [delta_ias_800, delta_ias_1500, delta_ias_3000, mean_rocd_800_1500, mean_rocd_1500_3000]
NADP1_REF_FEATURES = np.array([15, 15, 15, 1750, 1700])
NADP2_REF_FEATURES = np.array([15, 45, 85, 1650, 1450])

# Normalization weights (so speed in kt and ROCD in ft/min are comparable)
# Speed range ~0-90kt, ROCD range ~1200-1800 ft/min
FEATURE_WEIGHTS = np.array([1 / 30, 1 / 30, 1 / 30, 1 / 200, 1 / 200])


def classify_flight(row):
    """Classify a flight as NADP1 or NADP2 by distance to reference features."""
    features = np.array([
        row["delta_ias_800"],
        row["delta_ias_1500"],
        row["delta_ias_3000"],
        row["mean_rocd_800_1500"],
        row["mean_rocd_1500_3000"],
    ])

    dist_nadp1 = np.sqrt(np.sum((FEATURE_WEIGHTS * (features - NADP1_REF_FEATURES)) ** 2))
    dist_nadp2 = np.sqrt(np.sum((FEATURE_WEIGHTS * (features - NADP2_REF_FEATURES)) ** 2))

    return "nadp1" if dist_nadp1 < dist_nadp2 else "nadp2"


def compute_delta(row, nadp_type):
    """Compute RMS delta between flight's altitude curves and matched reference."""
    # Get flight's altitude-indexed curves
    delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]
    rocd_cols = [f"rocd_alt_{a}" for a in ALT_GRID]

    flight_delta_ias = row[delta_ias_cols].values.astype(float)
    flight_rocd = row[rocd_cols].values.astype(float)

    # Select reference curves
    if nadp_type == "nadp1":
        ref_delta_ias = NADP1_DELTA_IAS
        ref_rocd = NADP1_ROCD
    else:
        ref_delta_ias = NADP2_DELTA_IAS
        ref_rocd = NADP2_ROCD

    # Compute residuals only where flight data exists (not NaN)
    ias_mask = ~np.isnan(flight_delta_ias)
    rocd_mask = ~np.isnan(flight_rocd)

    if ias_mask.sum() < 5 or rocd_mask.sum() < 5:
        return np.nan, np.nan, np.nan

    ias_residuals = flight_delta_ias[ias_mask] - ref_delta_ias[ias_mask]
    rocd_residuals = flight_rocd[rocd_mask] - ref_rocd[rocd_mask]

    # Normalize: IAS residuals in kt, ROCD in ft/min — scale to comparable units
    delta_ias_rms = np.sqrt(np.mean(ias_residuals ** 2))
    delta_rocd_rms = np.sqrt(np.mean(rocd_residuals ** 2))

    # Combined delta (normalize each dimension by typical range)
    delta_score = np.sqrt((delta_ias_rms / 30) ** 2 + (delta_rocd_rms / 200) ** 2)

    return delta_score, delta_ias_rms, delta_rocd_rms


# --- Main ---

df = pd.read_parquet("data/nadp_features.parquet")
print(f"Loaded features for {len(df)} flights")

# Classify each flight
df["nadp_type"] = df.apply(classify_flight, axis=1)

# Compute delta scores
deltas = df.apply(lambda row: compute_delta(row, row["nadp_type"]), axis=1)
df["delta_score"] = deltas.apply(lambda x: x[0])
df["delta_ias_rms"] = deltas.apply(lambda x: x[1])
df["delta_rocd_rms"] = deltas.apply(lambda x: x[2])

# Print results
nadp1 = df[df["nadp_type"] == "nadp1"]
nadp2 = df[df["nadp_type"] == "nadp2"]
print(f"\nClassification results:")
print(f"  NADP1: {len(nadp1)} flights ({len(nadp1)/len(df)*100:.1f}%)")
print(f"  NADP2: {len(nadp2)} flights ({len(nadp2)/len(df)*100:.1f}%)")
print(f"\nDelta score statistics:")
print(f"  NADP1 mean delta: {nadp1['delta_score'].mean():.3f}")
print(f"  NADP2 mean delta: {nadp2['delta_score'].mean():.3f}")

# Export results (milestone features + classification + delta)
result_cols = [
    "flight_id", "icao_actype", "v2", "nadp_type",
    "delta_score", "delta_ias_rms", "delta_rocd_rms",
    "delta_ias_800", "delta_ias_1500", "delta_ias_3000",
    "mean_rocd_800_1500", "mean_rocd_1500_3000",
]
df[result_cols].to_csv("data/nadp_results.csv", index=False)
print(f"\nSaved results to data/nadp_results.csv")


# --- Visualizations ---

# Plot 1: Sample altitude vs time (colored by NADP type)
# Reload raw data for trajectory plots
df_raw = pd.read_parquet("data/vemmis_departures.parquet")

fig, ax = plt.subplots(figsize=(10, 6))
sample_ids = df.sample(min(400, len(df)), random_state=42)
for _, row in sample_ids.iterrows():
    fid = row["flight_id"]
    color = "tab:blue" if row["nadp_type"] == "nadp1" else "tab:red"
    fdata = df_raw[df_raw["FLIGHT_ID"] == fid].sort_values("actual_time")
    if fdata.empty:
        continue
    t0 = fdata["actual_time"].iloc[0]
    ts = (fdata["actual_time"] - t0).dt.total_seconds()
    ax.plot(ts, fdata["ALT"], lw=0.5, color=color, alpha=0.2)

ax.plot([], [], color="tab:blue", label="NADP1")
ax.plot([], [], color="tab:red", label="NADP2")
ax.legend()
ax.set_xlabel("Time from start (s)")
ax.set_ylabel("Altitude (ft)")
ax.set_title("Sample departure trajectories by NADP type")
ax.set_ylim(0, 5000)
plt.tight_layout()
plt.savefig("plots/01_trajectories_by_nadp.png", dpi=150)
print("Saved plots/01_trajectories_by_nadp.png")


# Plot 2: IAS vs altitude with reference curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for _, row in nadp1.sample(min(200, len(nadp1)), random_state=42).iterrows():
    ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
    ax1.plot(ias_vals, ALT_GRID, lw=0.5, color="tab:blue", alpha=0.15)
ax1.plot(NADP1_DELTA_IAS, ALT_GRID, "k-", lw=2.5, label="NADP1 reference")
ax1.set_xlabel("IAS - V2 (kt)")
ax1.set_ylabel("Altitude (ft)")
ax1.set_title("NADP1 flights")
ax1.legend()
ax1.set_xlim(-20, 120)

for _, row in nadp2.sample(min(200, len(nadp2)), random_state=42).iterrows():
    ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
    ax2.plot(ias_vals, ALT_GRID, lw=0.5, color="tab:red", alpha=0.15)
ax2.plot(NADP2_DELTA_IAS, ALT_GRID, "k-", lw=2.5, label="NADP2 reference")
ax2.set_xlabel("IAS - V2 (kt)")
ax2.set_title("NADP2 flights")
ax2.legend()
ax2.set_xlim(-20, 120)

plt.suptitle("Speed profiles vs ICAO reference")
plt.tight_layout()
plt.savefig("plots/02_speed_profiles_vs_reference.png", dpi=150)
print("Saved plots/02_speed_profiles_vs_reference.png")


# Plot 3: Delta score distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(nadp1["delta_score"].dropna(), bins=50, alpha=0.6, label="NADP1", color="tab:blue")
ax.hist(nadp2["delta_score"].dropna(), bins=50, alpha=0.6, label="NADP2", color="tab:red")
ax.set_xlabel("Delta score (deviation from reference)")
ax.set_ylabel("Number of flights")
ax.set_title("Distribution of delta scores by NADP type")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_delta_distribution.png", dpi=150)
print("Saved plots/03_delta_distribution.png")


# Plot 4: Example flights vs reference (3 best + 3 worst per type)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col_idx, (nadp_label, nadp_df, ref_ias, ref_rocd) in enumerate([
    ("NADP1", nadp1, NADP1_DELTA_IAS, NADP1_ROCD),
    ("NADP2", nadp2, NADP2_DELTA_IAS, NADP2_ROCD),
]):
    valid = nadp_df.dropna(subset=["delta_score"]).sort_values("delta_score")
    best = valid.head(3)
    worst = valid.tail(3)

    # Top row: delta_ias curves
    ax = axes[0, col_idx]
    ax.plot(ref_ias, ALT_GRID, "k--", lw=2, label="Reference")
    for _, row in best.iterrows():
        vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "g-", lw=1, alpha=0.7)
    for _, row in worst.iterrows():
        vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "r-", lw=1, alpha=0.7)
    ax.set_title(f"{nadp_label}: IAS - V2")
    ax.set_xlabel("IAS - V2 (kt)")
    ax.set_ylabel("Altitude (ft)")
    ax.plot([], [], "g-", label="Best 3")
    ax.plot([], [], "r-", label="Worst 3")
    ax.legend(fontsize=10)

    # Bottom row: ROCD curves
    ax = axes[1, col_idx]
    ax.plot(ref_rocd, ALT_GRID, "k--", lw=2, label="Reference")
    for _, row in best.iterrows():
        vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "g-", lw=1, alpha=0.7)
    for _, row in worst.iterrows():
        vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "r-", lw=1, alpha=0.7)
    ax.set_title(f"{nadp_label}: ROCD")
    ax.set_xlabel("ROCD (ft/min)")
    ax.set_ylabel("Altitude (ft)")
    ax.legend(fontsize=10)

plt.suptitle("Best vs worst matching flights against reference profiles")
plt.tight_layout()
plt.savefig("plots/04_reference_comparison.png", dpi=150)
print("Saved plots/04_reference_comparison.png")
