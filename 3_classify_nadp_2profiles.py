"""
Classify flights into NADP1/NADP2 by matching against ICAO reference profiles,
and compute delta scores measuring deviation from the matched standard.

This is the 2-profile classifier (NADP1 vs NADP2).
For NADP2 sub-type classification (800/1000/1500), see 4_classify_nadp_4profiles.py.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--all-months", action="store_true",
                    help="Label plots/output for all months (data selection is done in step 1)")
args = parser.parse_args()

PLOT_DIR = "plots/2profiles"
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {"nadp1": "#4C72B0", "nadp2": "#DD5145", "unknown": "#999999"}

# --- ICAO Reference Speed Profiles ---
ALT_GRID = np.arange(200, 3600, 100)


def _interp_profile(alt_breakpoints, values, alt_grid):
    """Interpolate a piecewise-linear profile onto a regular altitude grid."""
    return np.interp(alt_grid, alt_breakpoints, values)


# NADP1: constant speed (~V2) until 3000ft, then accelerate
# NADP2: linear ramp from 800ft to 70 kt above V2 at 3000ft
REF_CURVES = {
    "nadp1": _interp_profile([0, 800, 3000, 3500], [0, 0, 0, 25], ALT_GRID),
    "nadp2": _interp_profile([0, 800, 3000, 3500], [0, 0, 70, 75], ALT_GRID),
}

THRESHOLD = 0.6


# --- Main ---

df = pd.read_parquet("data/nadp_features.parquet")
print(f"Loaded features for {len(df)} flights")

# Vectorized classification: extract all flight IAS curves as a matrix
delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]
flight_matrix = df[delta_ias_cols].values.astype(float)  # (n_flights, n_alts)
valid_counts = np.sum(~np.isnan(flight_matrix), axis=1)  # per-flight valid points

# Build reference matrix: (n_refs, n_alts)
ref_labels = list(REF_CURVES.keys())  # ["nadp1", "nadp2"]
ref_matrix = np.array([REF_CURVES[k] for k in ref_labels])

# Compute RMS distance to each reference (NaN-aware)
# For each flight and reference, compute mean squared residual only over valid points
n_flights = len(df)
dist_all = np.full((n_flights, len(ref_labels)), np.nan)
for j, ref_curve in enumerate(ref_matrix):
    residuals = flight_matrix - ref_curve  # (n_flights, n_alts), NaN where flight is NaN
    dist_all[:, j] = np.sqrt(np.nanmean(residuals ** 2, axis=1))

dist_nadp1 = dist_all[:, 0]
dist_nadp2 = dist_all[:, 1]

# Classification logic
closer = np.minimum(dist_nadp1, dist_nadp2)
farther = np.maximum(dist_nadp1, dist_nadp2)
ratio = np.where(farther > 0, closer / farther, 1.0)

nadp_type = np.where(
    (valid_counts < 5) | (ratio > THRESHOLD) | (farther == 0),
    "unknown",
    np.where(dist_nadp1 < dist_nadp2, "nadp1", "nadp2"),
)

df["nadp_type"] = nadp_type
df["dist_nadp1"] = dist_nadp1
df["dist_nadp2"] = dist_nadp2
df["separation_ratio"] = ratio

# Vectorized delta score: RMS to matched reference
matched_ref = np.where(
    (nadp_type == "nadp1") | ((nadp_type == "unknown") & (dist_nadp1 < dist_nadp2)),
    0, 1,  # index into ref_matrix
)
matched_curve = ref_matrix[matched_ref]  # (n_flights, n_alts)
residuals = flight_matrix - matched_curve
delta_ias_rms = np.sqrt(np.nanmean(residuals ** 2, axis=1))
df["delta_score"] = delta_ias_rms / 30
df["delta_ias_rms"] = delta_ias_rms

nadp1 = df[df["nadp_type"] == "nadp1"]
nadp2 = df[df["nadp_type"] == "nadp2"]
unknown = df[df["nadp_type"] == "unknown"]

print(f"\nClassification results (threshold = {THRESHOLD}):")
print(f"  NADP1:       {len(nadp1)} flights ({len(nadp1)/len(df)*100:.1f}%)")
print(f"  NADP2:       {len(nadp2)} flights ({len(nadp2)/len(df)*100:.1f}%)")
print(f"  Unknown:     {len(unknown)} flights ({len(unknown)/len(df)*100:.1f}%)")

print(f"\nAmong classified: {len(nadp2)/(len(nadp1)+len(nadp2))*100:.1f}% NADP2")

print(f"\nDelta score (IAS RMS / 30) statistics:")
if len(nadp1) > 0:
    print(f"  NADP1 mean delta: {nadp1['delta_score'].mean():.3f}")
if len(nadp2) > 0:
    print(f"  NADP2 mean delta: {nadp2['delta_score'].mean():.3f}")
if len(unknown) > 0:
    print(f"  Unknown mean delta: {unknown['delta_score'].mean():.3f}")
print(f"\nSeparation ratio (closest/second-closest, lower = more distinct):")
classified = df[df.nadp_type != "unknown"]
print(f"  Classified mean: {classified['separation_ratio'].mean():.3f}")
print(f"  Unknown mean:    {unknown['separation_ratio'].mean():.3f}")

# Export results
result_cols = [
    "flight_id", "typecode", "callsign", "airline", "start", "v2", "nadp_type",
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
        keep = ts <= 300
        ax.plot(ts[keep], fdata["ALT"][keep], lw=0.5, color=color, alpha=0.2)

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
plt.savefig(f"{PLOT_DIR}/01_trajectories_by_nadp.png", dpi=150)
print(f"Saved {PLOT_DIR}/01_trajectories_by_nadp.png")


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

axes[0, 0].plot(REF_CURVES["nadp1"], ALT_GRID, "k-", lw=2.5, label="NADP1 reference")
axes[0, 0].set_title("NADP1: IAS - V2")
axes[0, 0].set_xlabel("IAS - V2 (kt)")
axes[0, 0].set_ylabel("Altitude (ft)")
axes[0, 0].legend()
axes[0, 0].set_xlim(-20, 120)

axes[0, 1].plot(REF_CURVES["nadp2"], ALT_GRID, "k-", lw=2.5, label="NADP2 reference")
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
plt.savefig(f"{PLOT_DIR}/02_speed_profiles_vs_reference.png", dpi=150)
print(f"Saved {PLOT_DIR}/02_speed_profiles_vs_reference.png")


# Plot 3: Separation ratio distribution (separate subplots per type)
TYPE_ORDER = ["nadp1", "nadp2", "unknown"]
fig, axes = plt.subplots(len(TYPE_ORDER), 1, figsize=(10, 7), sharex=True, sharey=True)
for ax, label in zip(axes, TYPE_ORDER):
    subset = df[df["nadp_type"] == label]["separation_ratio"].dropna()
    sns.histplot(subset, bins=50, color=PALETTE[label], ax=ax)
    ax.axvline(THRESHOLD, color="k", ls="--", lw=1.5, label=f"threshold={THRESHOLD}")
    ax.set_ylabel(f"{label.upper()}\ncount")
    ax.set_xlabel("")
axes[0].set_title("Separation ratio distribution by NADP type")
axes[0].legend()
axes[-1].set_xlabel("Separation ratio (0 = perfect match, 1 = equidistant)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_separation_ratio.png", dpi=150)
print(f"Saved {PLOT_DIR}/03_separation_ratio.png")


# Plot 3b: Threshold sensitivity
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

# Reuse precomputed distances for vectorized threshold sweep
_ratio = df["separation_ratio"].values
_d1 = df["dist_nadp1"].values
_d2 = df["dist_nadp2"].values
_vc = valid_counts

_bar_labels = ["NADP1", "NADP2", "Unknown"]
_bar_keys = ["nadp1", "nadp2", "unknown"]
_bar_colors = [PALETTE["nadp1"], PALETTE["nadp2"], PALETTE["unknown"]]

for ax, thresh in zip(axes.flat, thresholds):
    cats = np.where((_vc < 5) | (_ratio > thresh) | (np.maximum(_d1, _d2) == 0),
                    "unknown", np.where(_d1 < _d2, "nadp1", "nadp2"))
    counts = [int((cats == k).sum()) for k in _bar_keys]
    total = sum(counts)
    bars = ax.barh(_bar_labels, counts, color=_bar_colors)
    title = f"threshold={thresh}"
    if thresh == THRESHOLD:
        title += " (current)"
    ax.set_title(title)
    ax.set_xlim(0, len(df) * 1.15)
    for bar, count in zip(bars, counts):
        ax.text(count + len(df) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/total*100:.0f}%)", va="center", fontsize=9)

for ax in axes[:, 0]:
    ax.set_ylabel("Category")
plt.suptitle("Classification results at varying thresholds")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03b_threshold_sensitivity.png", dpi=150)
print(f"Saved {PLOT_DIR}/03b_threshold_sensitivity.png")


# Plot 4: Example flights vs reference (3 best + 3 worst per category, IAS + ROCD)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for col_idx, (cat_label, cat_df) in enumerate([("NADP1", nadp1), ("NADP2", nadp2)]):
    valid = cat_df.dropna(subset=["delta_score"]).sort_values("delta_score")
    best = valid.head(3)
    worst = valid.tail(3)

    # Top row: IAS
    ax = axes[0, col_idx]
    ref_key = "nadp1" if col_idx == 0 else "nadp2"
    ax.plot(REF_CURVES[ref_key], ALT_GRID, "k--", lw=1.5, alpha=0.5)
    for _, row in best.iterrows():
        vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "g-", lw=1, alpha=0.7)
    for _, row in worst.iterrows():
        vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
        ax.plot(vals, ALT_GRID, "r-", lw=1, alpha=0.7)
    ax.set_title(f"{cat_label}: IAS - V2")
    ax.set_xlabel("IAS - V2 (kt)")
    ax.plot([], [], "k--", label="Reference")
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
    ax.set_title(f"{cat_label}: ROCD")
    ax.set_xlabel("ROCD (ft/min)")

axes[0, 0].set_ylabel("Altitude (ft)")
axes[1, 0].set_ylabel("Altitude (ft)")
plt.suptitle("Best vs worst matching flights (IAS and ROCD)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_reference_comparison.png", dpi=150)
print(f"Saved {PLOT_DIR}/04_reference_comparison.png")


# Plot 5: Classification feature space (delta_ias_1500 vs delta_ias_3000)
fig, ax = plt.subplots(figsize=(10, 8))
_max_scatter = 5000
for label in ["unknown", "nadp1", "nadp2"]:
    subset = df[df["nadp_type"] == label]
    if len(subset) > _max_scatter:
        subset = subset.sample(_max_scatter, random_state=42)
    ax.scatter(
        subset["delta_ias_1500"], subset["delta_ias_3000"],
        c=PALETTE[label], label=label.upper(), s=8, alpha=0.3,
    )
# Mark reference points
i_1500 = np.searchsorted(ALT_GRID, 1500)
i_3000 = np.searchsorted(ALT_GRID, 3000)
for name, key in [("NADP1", "nadp1"), ("NADP2", "nadp2")]:
    curve = REF_CURVES[key]
    ax.scatter(curve[i_1500], curve[i_3000], marker="*", s=300, c="k",
               zorder=10, edgecolors="white", linewidths=1)
    ax.annotate(name, (curve[i_1500], curve[i_3000]), fontsize=9, fontweight="bold",
                xytext=(8, 8), textcoords="offset points")
ax.set_xlabel("IAS - V2 at 1500 ft (kt)")
ax.set_ylabel("IAS - V2 at 3000 ft (kt)")
ax.set_title("Classification feature space")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_feature_space.png", dpi=150)
print(f"Saved {PLOT_DIR}/05_feature_space.png")


# Plot 6: NADP type breakdown by aircraft type (top 15 most common)
top_types = df["typecode"].value_counts().head(15).index
df_top = df[df["typecode"].isin(top_types)].copy()

ct = pd.crosstab(df_top["typecode"], df_top["nadp_type"], normalize="index")
ct = ct.reindex(columns=["nadp1", "nadp2", "unknown"], fill_value=0)
ct = ct.loc[reversed(top_types)]

fig, ax = plt.subplots(figsize=(12, 7))
ct.plot.barh(stacked=True, color=[PALETTE["nadp1"], PALETTE["nadp2"], PALETTE["unknown"]], ax=ax)
ax.set_xlabel("Proportion of flights")
ax.set_ylabel("Aircraft type")
ax.set_title("NADP classification by aircraft type (top 15)")
ax.legend(
    ["NADP1", "NADP2", "Unknown"],
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncol=3, frameon=False,
)
counts_by_type = df_top["typecode"].value_counts()
for i, actype in enumerate(ct.index):
    ax.text(1.02, i, f"n={counts_by_type[actype]}", va="center", fontsize=10)
ax.set_xlim(0, 1.15)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_actype_breakdown.png", dpi=150, bbox_inches="tight")
print(f"Saved {PLOT_DIR}/06_actype_breakdown.png")


# Plot 7: Mean IAS and ROCD profiles by NADP category with confidence bands
fig, (ax_ias, ax_rocd) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

for label in ["nadp1", "nadp2"]:
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
plt.savefig(f"{PLOT_DIR}/07_mean_profiles.png", dpi=150)
print(f"Saved {PLOT_DIR}/07_mean_profiles.png")


# Plot 9: NADP type breakdown by airline (airlines with >2 flights/day avg)
airline_counts = df["airline"].value_counts()
n_days = (df["start"].max() - df["start"].min()).days + 1
top_airlines = airline_counts[airline_counts > 2 * n_days].index
df_top_al = df[df["airline"].isin(top_airlines)].copy()

ct_al = pd.crosstab(df_top_al["airline"], df_top_al["nadp_type"], normalize="index")
ct_al = ct_al.reindex(columns=["nadp1", "nadp2", "unknown"], fill_value=0)
ct_al = ct_al.loc[reversed(top_airlines)]

fig, ax = plt.subplots(figsize=(12, 13))
ct_al.plot.barh(stacked=True, color=[PALETTE["nadp1"], PALETTE["nadp2"], PALETTE["unknown"]], ax=ax)
ax.set_xlabel("Proportion of flights")
ax.set_ylabel("Airline (ICAO code)")
ax.set_title(f"NADP classification by airline ({len(top_airlines)} airlines, >2 flights/day)")
ax.legend(
    ["NADP1", "NADP2", "Unknown"],
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncol=3, frameon=False,
)
counts_by_airline = df_top_al["airline"].value_counts()
for i, airline in enumerate(ct_al.index):
    ax.text(1.02, i, f"n={counts_by_airline[airline]}", va="center", fontsize=10)
ax.set_xlim(0, 1.15)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_airline_breakdown.png", dpi=150, bbox_inches="tight")
print(f"Saved {PLOT_DIR}/09_airline_breakdown.png")
