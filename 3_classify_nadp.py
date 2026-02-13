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
import seaborn as sns

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {"nadp1": "#4C72B0", "nadp2": "#DD5145", "unknown": "#999999"}

# --- ICAO Reference Speed Profiles ---
# Piecewise-linear reference curves: delta_ias (IAS - V2) as function of altitude.
# V2 baseline is min IAS in 200-800ft band, so delta ≈ 0 in initial climb.

# Reference altitudes for curves (every 100ft from 200 to 3500)
# Capped at 3500ft to avoid QNH->QNE transition zone above ~4000ft
ALT_GRID = np.arange(200, 3600, 100)


def _interp_profile(alt_breakpoints, values, alt_grid):
    """Interpolate a piecewise-linear profile onto a regular altitude grid."""
    return np.interp(alt_grid, alt_breakpoints, values)


# NADP1: constant speed (~V2) until 3000ft, then accelerate
NADP1_DELTA_IAS = _interp_profile(
    [0, 800, 3000, 3500],
    [0, 0, 0, 25],
    ALT_GRID,
)

# NADP2: constant speed until 800ft, then accelerate
NADP2_DELTA_IAS = _interp_profile(
    [0, 800, 1500, 3000, 3500],
    [0, 0, 30, 70, 75],
    ALT_GRID,
)

# Reference milestone feature vectors for classification (IAS only)
# [delta_ias_800, delta_ias_1500, delta_ias_3000]
NADP1_REF_IAS = np.array([0, 0, 0])
NADP2_REF_IAS = np.array([0, 30, 70])

# Normalization weights for IAS features
IAS_WEIGHTS = np.array([1 / 30, 1 / 30, 1 / 30])

# Maximum allowed ratio (closer / farther) to classify a flight.
# Ratio ranges from 0 (one reference is perfect match) to 1 (equidistant).
# E.g., 0.6 means the flight must be at least 40% closer to one reference
# than the other; above this ratio, the flight is marked as unknown.
SEPARATION_THRESHOLD = 0.6


def classify_flight(row):
    """Classify a flight as NADP1, NADP2, or unknown based on distance separation.

    Computes distance to both NADP1 and NADP2 reference IAS profiles.
    Only classifies if the two distances are sufficiently different.
    """
    ias_features = np.array([
        row["delta_ias_800"],
        row["delta_ias_1500"],
        row["delta_ias_3000"],
    ])

    dist_nadp1 = np.sqrt(np.sum((IAS_WEIGHTS * (ias_features - NADP1_REF_IAS)) ** 2))
    dist_nadp2 = np.sqrt(np.sum((IAS_WEIGHTS * (ias_features - NADP2_REF_IAS)) ** 2))

    closer = min(dist_nadp1, dist_nadp2)
    farther = max(dist_nadp1, dist_nadp2)

    # If the two distances are too similar, mark as unknown
    if farther == 0 or closer / farther > SEPARATION_THRESHOLD:
        return ("unknown", dist_nadp1, dist_nadp2)

    if dist_nadp1 < dist_nadp2:
        return ("nadp1", dist_nadp1, dist_nadp2)
    else:
        return ("nadp2", dist_nadp1, dist_nadp2)


def compute_delta(row, nadp_type):
    """Compute RMS delta between flight's IAS curve and matched reference."""
    delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]
    flight_delta_ias = row[delta_ias_cols].values.astype(float)

    ref_delta_ias = NADP1_DELTA_IAS if nadp_type == "nadp1" else NADP2_DELTA_IAS

    # Compute residuals only where flight data exists (not NaN)
    ias_mask = ~np.isnan(flight_delta_ias)
    if ias_mask.sum() < 5:
        return np.nan, np.nan

    ias_residuals = flight_delta_ias[ias_mask] - ref_delta_ias[ias_mask]
    delta_ias_rms = np.sqrt(np.mean(ias_residuals ** 2))
    delta_score = delta_ias_rms / 30

    return delta_score, delta_ias_rms


# --- Main ---

df = pd.read_parquet("data/nadp_features.parquet")
print(f"Loaded features for {len(df)} flights")

# Classify each flight
classifications = df.apply(classify_flight, axis=1)
df["nadp_type"] = classifications.apply(lambda x: x[0])
df["dist_nadp1"] = classifications.apply(lambda x: x[1])
df["dist_nadp2"] = classifications.apply(lambda x: x[2])

# Compute delta scores against closest reference (for classified flights)
# For unknown flights, compute against whichever is closer
def _best_type(row):
    if row["nadp_type"] != "unknown":
        return row["nadp_type"]
    return "nadp1" if row["dist_nadp1"] < row["dist_nadp2"] else "nadp2"

deltas = df.apply(lambda row: compute_delta(row, _best_type(row)), axis=1)
df["delta_score"] = deltas.apply(lambda x: x[0])
df["delta_ias_rms"] = deltas.apply(lambda x: x[1])

# Print results
# Compute separation ratio for diagnostics
df["separation_ratio"] = df[["dist_nadp1", "dist_nadp2"]].min(axis=1) / df[["dist_nadp1", "dist_nadp2"]].max(axis=1)

nadp1 = df[df["nadp_type"] == "nadp1"]
nadp2 = df[df["nadp_type"] == "nadp2"]
unknown = df[df["nadp_type"] == "unknown"]
print(f"\nClassification results (separation threshold = {SEPARATION_THRESHOLD}):")
print(f"  NADP1:   {len(nadp1)} flights ({len(nadp1)/len(df)*100:.1f}%)")
print(f"  NADP2:   {len(nadp2)} flights ({len(nadp2)/len(df)*100:.1f}%)")
print(f"  Unknown: {len(unknown)} flights ({len(unknown)/len(df)*100:.1f}%)")
print(f"\nDelta score (IAS RMS / 30) statistics:")
if len(nadp1) > 0:
    print(f"  NADP1 mean delta: {nadp1['delta_score'].mean():.3f}")
if len(nadp2) > 0:
    print(f"  NADP2 mean delta: {nadp2['delta_score'].mean():.3f}")
if len(unknown) > 0:
    print(f"  Unknown mean delta: {unknown['delta_score'].mean():.3f}")
print(f"\nSeparation ratio (closer/farther, lower = more distinct):")
print(f"  Classified mean: {df[df.nadp_type != 'unknown']['separation_ratio'].mean():.3f}")
print(f"  Unknown mean:    {unknown['separation_ratio'].mean():.3f}")

# Export results (milestone features + classification + delta)
result_cols = [
    "flight_id", "icao_actype", "v2", "nadp_type",
    "delta_score", "delta_ias_rms",
    "delta_ias_800", "delta_ias_1500", "delta_ias_3000",
    "mean_rocd_800_1500", "mean_rocd_1500_3000",
]
df[result_cols].to_csv("data/nadp_results.csv", index=False)
print(f"\nSaved results to data/nadp_results.csv")


# --- Visualizations ---

df_raw = pd.read_parquet("data/vemmis_departures.parquet")


# Plot 1: Sample altitude vs time (colored by NADP type)
def _plot_trajectories(ax, flight_ids_df, df_raw, color, max_n=200):
    """Plot altitude vs time-from-liftoff trajectories on a given axis."""
    sample = flight_ids_df.sample(min(max_n, len(flight_ids_df)), random_state=42)
    for _, row in sample.iterrows():
        fdata = df_raw[df_raw["FLIGHT_ID"] == row["flight_id"]].sort_values("actual_time")
        if fdata.empty:
            continue
        airborne = fdata[fdata["ALT"] >= 50]
        if airborne.empty:
            continue
        t0 = airborne["actual_time"].iloc[0]
        fdata = fdata[fdata["actual_time"] >= t0]
        ts = (fdata["actual_time"] - t0).dt.total_seconds()
        ax.plot(ts, fdata["ALT"], lw=0.5, color=color, alpha=0.2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

_plot_trajectories(ax1, nadp1, df_raw, PALETTE["nadp1"])
ax1.set_xlabel("Time from liftoff (s)")
ax1.set_ylabel("Altitude (ft)")
ax1.set_title(f"NADP1 ({len(nadp1)} flights)")
ax1.set_ylim(0, 5000)

_plot_trajectories(ax2, nadp2, df_raw, PALETTE["nadp2"])
ax2.set_xlabel("Time from liftoff (s)")
ax2.set_title(f"NADP2 ({len(nadp2)} flights)")

plt.suptitle("Departure trajectories by NADP type")
plt.tight_layout()
plt.savefig("plots/01_trajectories_by_nadp.png", dpi=150)
print("Saved plots/01_trajectories_by_nadp.png")


# Plot 2: IAS and ROCD vs altitude (2x2: top=IAS, bottom=ROCD)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

nadp1_sample = nadp1.sample(min(200, len(nadp1)), random_state=42)
nadp2_sample = nadp2.sample(min(200, len(nadp2)), random_state=42)

for _, row in nadp1_sample.iterrows():
    ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
    axes[0, 0].plot(ias_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp1"], alpha=0.15)
    rocd_vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
    axes[1, 0].plot(rocd_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp1"], alpha=0.15)

for _, row in nadp2_sample.iterrows():
    ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
    axes[0, 1].plot(ias_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp2"], alpha=0.15)
    rocd_vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
    axes[1, 1].plot(rocd_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp2"], alpha=0.15)

axes[0, 0].plot(NADP1_DELTA_IAS, ALT_GRID, "k-", lw=2.5, label="NADP1 reference")
axes[0, 0].set_title("NADP1: IAS - V2")
axes[0, 0].set_xlabel("IAS - V2 (kt)")
axes[0, 0].set_ylabel("Altitude (ft)")
axes[0, 0].legend()
axes[0, 0].set_xlim(-20, 120)

axes[0, 1].plot(NADP2_DELTA_IAS, ALT_GRID, "k-", lw=2.5, label="NADP2 reference")
axes[0, 1].set_title("NADP2: IAS - V2")
axes[0, 1].set_xlabel("IAS - V2 (kt)")
axes[0, 1].legend()
axes[0, 1].set_xlim(-20, 120)

axes[1, 0].set_title("NADP1: ROCD")
axes[1, 0].set_xlabel("ROCD (ft/min)")
axes[1, 0].set_ylabel("Altitude (ft)")

axes[1, 1].set_title("NADP2: ROCD")
axes[1, 1].set_xlabel("ROCD (ft/min)")

plt.suptitle("Speed and climb rate profiles by NADP type")
plt.tight_layout()
plt.savefig("plots/02_speed_profiles_vs_reference.png", dpi=150)
print("Saved plots/02_speed_profiles_vs_reference.png")


# Plot 3: Separation ratio distribution (NADP1 drawn on top)
fig, ax = plt.subplots(figsize=(10, 6))
for label in ["unknown", "nadp2", "nadp1"]:
    subset = df[df["nadp_type"] == label]["separation_ratio"].dropna()
    sns.histplot(subset, bins=50, alpha=0.5, label=label.upper(), color=PALETTE[label], ax=ax)
ax.axvline(SEPARATION_THRESHOLD, color="k", ls="--", lw=1.5, label=f"Threshold ({SEPARATION_THRESHOLD})")
ax.set_xlabel("Separation ratio (0 = perfect match, 1 = equidistant)")
ax.set_ylabel("Number of flights")
ax.set_title("Distance separation between NADP1 and NADP2 references")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_separation_ratio.png", dpi=150)
print("Saved plots/03_separation_ratio.png")


# Plot 3b: Varying separation thresholds
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5), sharey=True)

for ax, thresh in zip(axes, thresholds):
    n1, n2, nu = 0, 0, 0
    for _, row in df.iterrows():
        d1, d2 = row["dist_nadp1"], row["dist_nadp2"]
        closer, farther = min(d1, d2), max(d1, d2)
        if farther == 0 or closer / farther > thresh:
            nu += 1
        elif d1 < d2:
            n1 += 1
        else:
            n2 += 1
    total = n1 + n2 + nu
    bars = ax.barh(
        ["NADP1", "NADP2", "Unknown"], [n1, n2, nu],
        color=[PALETTE["nadp1"], PALETTE["nadp2"], PALETTE["unknown"]],
    )
    ax.set_title(f"Threshold = {thresh}")
    ax.set_xlim(0, len(df) * 1.15)
    for bar, count in zip(bars, [n1, n2, nu]):
        ax.text(count + len(df) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/total*100:.0f}%)", va="center", fontsize=10)

axes[0].set_ylabel("Category")
plt.suptitle("Classification results at varying separation thresholds")
plt.tight_layout()
plt.savefig("plots/03b_threshold_sensitivity.png", dpi=150)
print("Saved plots/03b_threshold_sensitivity.png")


# Plot 4: Example flights vs reference (3 best + 3 worst per type, IAS + ROCD)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for col_idx, (nadp_label, nadp_df, ref_ias) in enumerate([
    ("NADP1", nadp1, NADP1_DELTA_IAS),
    ("NADP2", nadp2, NADP2_DELTA_IAS),
]):
    valid = nadp_df.dropna(subset=["delta_score"]).sort_values("delta_score")
    best = valid.head(3)
    worst = valid.tail(3)

    # Top row: IAS
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
    ax.plot([], [], "g-", label="Best 3")
    ax.plot([], [], "r-", label="Worst 3")
    ax.legend(fontsize=10)

    # Bottom row: ROCD
    ax = axes[1, col_idx]
    for _, row in best.iterrows():
        vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "g-", lw=1, alpha=0.7)
    for _, row in worst.iterrows():
        vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "r-", lw=1, alpha=0.7)
    ax.set_title(f"{nadp_label}: ROCD")
    ax.set_xlabel("ROCD (ft/min)")

axes[0, 0].set_ylabel("Altitude (ft)")
axes[1, 0].set_ylabel("Altitude (ft)")
plt.suptitle("Best vs worst matching flights (IAS and ROCD)")
plt.tight_layout()
plt.savefig("plots/04_reference_comparison.png", dpi=150)
print("Saved plots/04_reference_comparison.png")


# Plot 5: Classification feature space (delta_ias_1500 vs delta_ias_3000)
fig, ax = plt.subplots(figsize=(10, 8))
for label in ["unknown", "nadp1", "nadp2"]:
    subset = df[df["nadp_type"] == label]
    ax.scatter(
        subset["delta_ias_1500"], subset["delta_ias_3000"],
        c=PALETTE[label], label=label.upper(), s=8, alpha=0.3,
    )
# Mark reference points
ax.scatter(*NADP1_REF_IAS[1:], marker="*", s=300, c="k", zorder=10, edgecolors="white", linewidths=1)
ax.scatter(*NADP2_REF_IAS[1:], marker="*", s=300, c="k", zorder=10, edgecolors="white", linewidths=1)
ax.annotate("NADP1 ref", NADP1_REF_IAS[1:], fontsize=11, fontweight="bold",
            xytext=(8, 8), textcoords="offset points")
ax.annotate("NADP2 ref", NADP2_REF_IAS[1:], fontsize=11, fontweight="bold",
            xytext=(8, 8), textcoords="offset points")
ax.set_xlabel("IAS - V2 at 1500 ft (kt)")
ax.set_ylabel("IAS - V2 at 3000 ft (kt)")
ax.set_title("Classification feature space")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig("plots/05_feature_space.png", dpi=150)
print("Saved plots/05_feature_space.png")


# Plot 6: NADP type breakdown by aircraft type (top 15 most common)
top_types = df["icao_actype"].value_counts().head(15).index
df_top = df[df["icao_actype"].isin(top_types)].copy()

ct = pd.crosstab(df_top["icao_actype"], df_top["nadp_type"], normalize="index")
ct = ct.reindex(columns=["nadp1", "nadp2", "unknown"], fill_value=0)
ct = ct.loc[ct["nadp2"].sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 7))
ct.plot.barh(
    stacked=True, color=[PALETTE["nadp1"], PALETTE["nadp2"], PALETTE["unknown"]],
    ax=ax,
)
ax.set_xlabel("Proportion of flights")
ax.set_ylabel("Aircraft type")
ax.set_title("NADP classification by aircraft type (top 15)")
ax.legend(
    ["NADP1", "NADP2", "Unknown"],
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncol=3, frameon=False,
)
counts = df_top["icao_actype"].value_counts()
for i, actype in enumerate(ct.index):
    ax.text(1.02, i, f"n={counts[actype]}", va="center", fontsize=10)
ax.set_xlim(0, 1.15)
plt.tight_layout()
plt.savefig("plots/06_actype_breakdown.png", dpi=150, bbox_inches="tight")
print("Saved plots/06_actype_breakdown.png")


# Plot 7: Mean IAS and ROCD profiles by NADP type with confidence bands
fig, (ax_ias, ax_rocd) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

for label, ref_ias in [("nadp1", NADP1_DELTA_IAS), ("nadp2", NADP2_DELTA_IAS)]:
    subset = df[df["nadp_type"] == label]
    ias_curves = np.array([[row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
                            for _, row in subset.iterrows()])
    rocd_curves = np.array([[row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
                             for _, row in subset.iterrows()])

    # IAS
    mean_ias = np.nanmean(ias_curves, axis=0)
    ax_ias.plot(mean_ias, ALT_GRID, color=PALETTE[label], lw=2, label=f"{label.upper()} mean")
    ax_ias.fill_betweenx(ALT_GRID, np.nanpercentile(ias_curves, 25, axis=0),
                          np.nanpercentile(ias_curves, 75, axis=0),
                          color=PALETTE[label], alpha=0.2, label=f"{label.upper()} IQR")
    ax_ias.plot(ref_ias, ALT_GRID, color=PALETTE[label], lw=2, ls="--", label=f"{label.upper()} reference")

    # ROCD
    mean_rocd = np.nanmean(rocd_curves, axis=0)
    ax_rocd.plot(mean_rocd, ALT_GRID, color=PALETTE[label], lw=2, label=f"{label.upper()} mean")
    ax_rocd.fill_betweenx(ALT_GRID, np.nanpercentile(rocd_curves, 25, axis=0),
                           np.nanpercentile(rocd_curves, 75, axis=0),
                           color=PALETTE[label], alpha=0.2, label=f"{label.upper()} IQR")

ax_ias.set_xlabel("IAS - V2 (kt)")
ax_ias.set_ylabel("Altitude (ft)")
ax_ias.set_title("Speed profiles")
ax_ias.legend(loc="lower right")
ax_ias.set_xlim(-20, 120)

ax_rocd.set_xlabel("ROCD (ft/min)")
ax_rocd.set_title("Climb rate profiles")
ax_rocd.legend(loc="lower right")

plt.suptitle("Mean profiles with IQR by NADP type")
plt.tight_layout()
plt.savefig("plots/07_mean_profiles.png", dpi=150)
print("Saved plots/07_mean_profiles.png")
