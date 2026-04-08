"""
Classify flights into NADP1/NADP2 by matching against ICAO reference profiles,
and compute delta scores measuring deviation from the matched standard.

- Defines ICAO reference profiles as piecewise-linear functions of altitude.
- Classifies each flight by Euclidean distance to reference milestone features.
- Computes per-flight delta scores as RMS residual from matched reference curve.
- Generates visualizations.
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

PLOT_DIR = "plots/4profiles"
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {"nadp1": "#4C72B0", "nadp2": "#DD5145", "unknown": "#999999"}
SUBTYPE_PALETTE = {"nadp2-800": "#DD5145", "nadp2-1000": "#FF7F0E", "nadp2-1500": "#8C564B"}
SUBTYPE_LABELS = {"nadp2-800": "NADP2-800", "nadp2-1000": "NADP2-1000", "nadp2-1500": "NADP2-1500"}

# --- ICAO Reference Speed Profiles ---
# Piecewise-linear reference curves: delta_ias (IAS - V2) as function of altitude.
# V2 baseline is min IAS in 200-800ft band, so delta ≈ 0 in initial climb.

# Reference altitudes for curves (every 100ft from 200 to 3500)
# Capped at 3500ft to avoid QNH->QNE transition zone above ~4000ft
ALT_GRID = np.arange(200, 3600, 100)


def _interp_profile(alt_breakpoints, values, alt_grid):
    """Interpolate a piecewise-linear profile onto a regular altitude grid."""
    return np.interp(alt_grid, alt_breakpoints, values)


# All reference curves: NADP1 + three NADP2 sub-types
# NADP1: constant speed (~V2) until 3000ft, then accelerate
# NADP2 variants: linear ramp from acceleration altitude to 70 kt at 3000ft
REF_CURVES = {
    "nadp1": _interp_profile([0, 800, 3000, 3500], [0, 0, 0, 25], ALT_GRID),
    "nadp2-800": _interp_profile([0, 800, 3000, 3500], [0, 0, 70, 75], ALT_GRID),
    "nadp2-1000": _interp_profile([0, 1000, 3000, 3500], [0, 0, 70, 75], ALT_GRID),
    "nadp2-1500": _interp_profile([0, 1500, 3000, 3500], [0, 0, 70, 75], ALT_GRID),
}

# Separation ratio thresholds (closest / second-closest RMS).
# Stricter for NADP1 (tight cluster), more relaxed for NADP2 variants.
NADP1_THRESHOLD = 0.4
NADP2_THRESHOLD = 0.9


def classify_flight(row):
    """Classify a flight by full-curve RMS distance to all reference profiles.

    Computes RMS distance to each of the 4 references (NADP1, NADP2-800,
    NADP2-1000, NADP2-1500). First decides NADP1 vs NADP2 using separation
    ratio between NADP1 distance and best NADP2 distance. Then picks the
    closest NADP2 sub-type.
    """
    delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]
    flight_ias = row[delta_ias_cols].values.astype(float)
    mask = ~np.isnan(flight_ias)
    if mask.sum() < 5:
        return "unknown", np.nan, np.nan

    # Compute RMS distance to each reference
    distances = {}
    for label, ref_curve in REF_CURVES.items():
        residuals = flight_ias[mask] - ref_curve[mask]
        distances[label] = np.sqrt(np.mean(residuals ** 2))

    dist_nadp1 = distances["nadp1"]
    # Best NADP2 sub-type
    nadp2_dists = {k: v for k, v in distances.items() if k.startswith("nadp2")}
    best_nadp2_label = min(nadp2_dists, key=nadp2_dists.get)
    dist_nadp2 = nadp2_dists[best_nadp2_label]

    closer = min(dist_nadp1, dist_nadp2)
    farther = max(dist_nadp1, dist_nadp2)

    if farther == 0:
        return "unknown", closer, farther

    ratio = closer / farther

    if dist_nadp1 < dist_nadp2:
        if ratio > NADP1_THRESHOLD:
            return "unknown", dist_nadp1, dist_nadp2
        return "nadp1", dist_nadp1, dist_nadp2
    else:
        if ratio > NADP2_THRESHOLD:
            return "unknown", dist_nadp1, dist_nadp2
        return best_nadp2_label, dist_nadp1, dist_nadp2


# --- Main ---

df = pd.read_parquet("data/nadp_features.parquet")
print(f"Loaded features for {len(df)} flights")

# Classify each flight (single stage: all 4 types at once)
classifications = df.apply(classify_flight, axis=1)
df["nadp_type"] = classifications.apply(lambda x: x[0])
df["dist_nadp1"] = classifications.apply(lambda x: x[1])
df["dist_nadp2"] = classifications.apply(lambda x: x[2])

# Derive high-level type (nadp1 vs nadp2) and separation ratio
df["nadp_category"] = df["nadp_type"].apply(
    lambda x: "nadp1" if x == "nadp1" else ("nadp2" if x.startswith("nadp2") else "unknown")
)
df["separation_ratio"] = df[["dist_nadp1", "dist_nadp2"]].min(axis=1) / df[["dist_nadp1", "dist_nadp2"]].max(axis=1)

# Compute delta score: RMS to the matched reference curve
def _compute_delta(row):
    ref_key = row["nadp_type"]
    if ref_key == "unknown":
        # Use closest family
        ref_key = "nadp1" if row["dist_nadp1"] < row["dist_nadp2"] else "nadp2-1000"
    ref_curve = REF_CURVES[ref_key]
    delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]
    flight_ias = row[delta_ias_cols].values.astype(float)
    mask = ~np.isnan(flight_ias)
    if mask.sum() < 5:
        return np.nan, np.nan
    residuals = flight_ias[mask] - ref_curve[mask]
    rms = np.sqrt(np.mean(residuals ** 2))
    return rms / 30, rms

deltas = df.apply(_compute_delta, axis=1)
df["delta_score"] = deltas.apply(lambda x: x[0])
df["delta_ias_rms"] = deltas.apply(lambda x: x[1])

nadp1 = df[df["nadp_category"] == "nadp1"]
nadp2 = df[df["nadp_category"] == "nadp2"]
unknown = df[df["nadp_category"] == "unknown"]

print(f"\nClassification results (NADP1 threshold = {NADP1_THRESHOLD}, NADP2 threshold = {NADP2_THRESHOLD}):")
print(f"  NADP1:       {len(nadp1)} flights ({len(nadp1)/len(df)*100:.1f}%)")
print(f"  NADP2:       {len(nadp2)} flights ({len(nadp2)/len(df)*100:.1f}%)")
print(f"  Unknown:     {len(unknown)} flights ({len(unknown)/len(df)*100:.1f}%)")

# NADP2 sub-type breakdown
print(f"\nNADP2 sub-type breakdown:")
for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]:
    sub = df[df["nadp_type"] == st]
    if len(nadp2) > 0:
        print(f"  {st}: {len(sub)} flights ({len(sub)/len(nadp2)*100:.1f}% of NADP2, mean RMS = {sub['delta_ias_rms'].mean():.1f} kt)")

print(f"\nDelta score (IAS RMS / 30) statistics:")
if len(nadp1) > 0:
    print(f"  NADP1 mean delta: {nadp1['delta_score'].mean():.3f}")
if len(nadp2) > 0:
    print(f"  NADP2 mean delta: {nadp2['delta_score'].mean():.3f}")
if len(unknown) > 0:
    print(f"  Unknown mean delta: {unknown['delta_score'].mean():.3f}")
print(f"\nSeparation ratio (closest/second-closest, lower = more distinct):")
classified = df[df.nadp_category != "unknown"]
print(f"  Classified mean: {classified['separation_ratio'].mean():.3f}")
print(f"  Unknown mean:    {unknown['separation_ratio'].mean():.3f}")

# Export results
result_cols = [
    "flight_id", "typecode", "callsign", "airline", "start", "v2", "nadp_type", "nadp_category",
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
plt.savefig(f"{PLOT_DIR}/01_trajectories_by_nadp.png", dpi=150)
print(f"Saved {PLOT_DIR}/01_trajectories_by_nadp.png")


# Plot 2: IAS and ROCD vs altitude (2x2: top=IAS, bottom=ROCD)
# Use high-contrast colors for NADP2 traces so sub-types are distinguishable at low alpha
TRACE_COLORS = {"nadp2-800": "#E41A1C", "nadp2-1000": "#4DAF4A", "nadp2-1500": "#7B68EE"}
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

nadp1_sample = nadp1.sample(min(200, len(nadp1)), random_state=42)

for _, row in nadp1_sample.iterrows():
    ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
    axes[0, 0].plot(ias_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp1"], alpha=0.15)
    rocd_vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
    axes[1, 0].plot(rocd_vals, ALT_GRID, lw=0.5, color=PALETTE["nadp1"], alpha=0.15)

for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]:
    st_df = df[df["nadp_type"] == st]
    st_sample = st_df.sample(min(70, len(st_df)), random_state=42)
    for _, row in st_sample.iterrows():
        ias_vals = [row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
        axes[0, 1].plot(ias_vals, ALT_GRID, lw=0.5, color=TRACE_COLORS[st], alpha=0.15)
        rocd_vals = [row.get(f"rocd_alt_{a}", np.nan) for a in ALT_GRID]
        axes[1, 1].plot(rocd_vals, ALT_GRID, lw=0.5, color=TRACE_COLORS[st], alpha=0.15)

axes[0, 0].plot(REF_CURVES["nadp1"], ALT_GRID, "k-", lw=2.5, label="NADP1 reference")
axes[0, 0].set_title("NADP1: IAS - V2")
axes[0, 0].set_xlabel("IAS - V2 (kt)")
axes[0, 0].set_ylabel("Altitude (ft)")
axes[0, 0].legend()
axes[0, 0].set_xlim(-20, 120)

for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]:
    axes[0, 1].plot(REF_CURVES[st], ALT_GRID, lw=2, ls="--",
                     color=TRACE_COLORS[st], label=SUBTYPE_LABELS[st])
axes[0, 1].set_title("NADP2: IAS - V2")
axes[0, 1].set_xlabel("IAS - V2 (kt)")
axes[0, 1].legend(fontsize=9)
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
TYPE_ORDER = ["nadp1", "nadp2-800", "nadp2-1000", "nadp2-1500", "unknown"]
TYPE_COLORS = {**{"nadp1": PALETTE["nadp1"], "unknown": PALETTE["unknown"]}, **SUBTYPE_PALETTE}
fig, axes = plt.subplots(len(TYPE_ORDER), 1, figsize=(10, 10), sharex=True, sharey=True)
for ax, label in zip(axes, TYPE_ORDER):
    subset = df[df["nadp_type"] == label]["separation_ratio"].dropna()
    sns.histplot(subset, bins=50, color=TYPE_COLORS[label], ax=ax)
    ax.axvline(NADP1_THRESHOLD, color=PALETTE["nadp1"], ls="--", lw=1.5)
    ax.axvline(NADP2_THRESHOLD, color=PALETTE["nadp2"], ls="--", lw=1.5)
    display = SUBTYPE_LABELS.get(label, label.upper())
    ax.set_ylabel(f"{display}\ncount")
    ax.set_xlabel("")
axes[0].set_title("Separation ratio distribution by NADP type")
axes[-1].set_xlabel("Separation ratio (0 = perfect match, 1 = equidistant)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_separation_ratio.png", dpi=150)
print(f"Saved {PLOT_DIR}/03_separation_ratio.png")


# Plot 3b: Varying asymmetric separation thresholds (re-classify from stored RMS)
threshold_pairs = [(0.3, 0.7), (0.3, 0.9), (0.4, 0.7), (0.4, 0.9), (0.5, 0.8), (0.5, 0.9)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

# Pre-compute per-flight: best label, best RMS, second RMS (reuse classify logic)
_delta_ias_cols = [f"delta_ias_alt_{a}" for a in ALT_GRID]

def _classify_with_thresholds(row, t1, t2):
    flight_ias = row[_delta_ias_cols].values.astype(float)
    mask = ~np.isnan(flight_ias)
    if mask.sum() < 5:
        return "unknown"
    distances = {}
    for label, ref_curve in REF_CURVES.items():
        residuals = flight_ias[mask] - ref_curve[mask]
        distances[label] = np.sqrt(np.mean(residuals ** 2))
    d1 = distances["nadp1"]
    nadp2_dists = {k: v for k, v in distances.items() if k.startswith("nadp2")}
    best_n2_label = min(nadp2_dists, key=nadp2_dists.get)
    d2 = nadp2_dists[best_n2_label]
    closer, farther = min(d1, d2), max(d1, d2)
    if farther == 0:
        return "unknown"
    ratio = closer / farther
    if d1 < d2:
        return "unknown" if ratio > t1 else "nadp1"
    else:
        return "unknown" if ratio > t2 else best_n2_label

_bar_labels = ["NADP1", "NADP2-800", "NADP2-1000", "NADP2-1500", "Unknown"]
_bar_keys = ["nadp1", "nadp2-800", "nadp2-1000", "nadp2-1500", "unknown"]
_bar_colors = [PALETTE["nadp1"], SUBTYPE_PALETTE["nadp2-800"], SUBTYPE_PALETTE["nadp2-1000"],
               SUBTYPE_PALETTE["nadp2-1500"], PALETTE["unknown"]]

for ax, (t1, t2) in zip(axes.flat, threshold_pairs):
    cats = df.apply(lambda row: _classify_with_thresholds(row, t1, t2), axis=1)
    counts = [int((cats == k).sum()) for k in _bar_keys]
    total = sum(counts)
    bars = ax.barh(_bar_labels, counts, color=_bar_colors)
    is_current = (t1 == NADP1_THRESHOLD and t2 == NADP2_THRESHOLD)
    title = f"NADP1={t1}, NADP2={t2}"
    if is_current:
        title += " (current)"
    ax.set_title(title)
    ax.set_xlim(0, len(df) * 1.15)
    for bar, count in zip(bars, counts):
        ax.text(count + len(df) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/total*100:.0f}%)", va="center", fontsize=9)

for ax in axes[:, 0]:
    ax.set_ylabel("Category")
plt.suptitle("Classification results at varying asymmetric thresholds")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03b_threshold_sensitivity.png", dpi=150)
print(f"Saved {PLOT_DIR}/03b_threshold_sensitivity.png")


# Plot 4: Example flights vs reference (3 best + 3 worst per category, IAS + ROCD)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for col_idx, (cat_label, cat_df) in enumerate([("NADP1", nadp1), ("NADP2", nadp2)]):
    valid = cat_df.dropna(subset=["delta_score"]).sort_values("delta_score")
    best = valid.head(3)
    worst = valid.tail(3)

    # Top row: IAS — plot matched reference for each flight
    ax = axes[0, col_idx]
    refs_plotted = set()
    for _, row in pd.concat([best, worst]).iterrows():
        ref = REF_CURVES.get(row["nadp_type"], REF_CURVES["nadp2-800"])
        if row["nadp_type"] not in refs_plotted:
            ax.plot(ref, ALT_GRID, "k--", lw=1.5, alpha=0.5)
            refs_plotted.add(row["nadp_type"])
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
for label in ["unknown", "nadp1", "nadp2-800", "nadp2-1000", "nadp2-1500"]:
    subset = df[df["nadp_type"] == label]
    color = TYPE_COLORS[label]
    display = SUBTYPE_LABELS.get(label, label.upper())
    ax.scatter(
        subset["delta_ias_1500"], subset["delta_ias_3000"],
        c=color, label=display, s=8, alpha=0.3,
    )
# Mark reference points (interpolate at 1500 and 3000 from each curve)
i_1500 = np.searchsorted(ALT_GRID, 1500)
i_3000 = np.searchsorted(ALT_GRID, 3000)
ref_markers = {
    "NADP1": REF_CURVES["nadp1"],
    "NADP2-800": REF_CURVES["nadp2-800"],
    "NADP2-1000": REF_CURVES["nadp2-1000"],
    "NADP2-1500": REF_CURVES["nadp2-1500"],
}
for name, curve in ref_markers.items():
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

_type_cols = ["nadp1", "nadp2-800", "nadp2-1000", "nadp2-1500", "unknown"]
_type_display = ["NADP1", "NADP2-800", "NADP2-1000", "NADP2-1500", "Unknown"]
_type_colors = [PALETTE["nadp1"], SUBTYPE_PALETTE["nadp2-800"], SUBTYPE_PALETTE["nadp2-1000"],
                SUBTYPE_PALETTE["nadp2-1500"], PALETTE["unknown"]]

ct = pd.crosstab(df_top["typecode"], df_top["nadp_type"], normalize="index")
ct = ct.reindex(columns=_type_cols, fill_value=0)
ct = ct.loc[reversed(top_types)]

fig, ax = plt.subplots(figsize=(12, 7))
ct.plot.barh(stacked=True, color=_type_colors, ax=ax)
ax.set_xlabel("Proportion of flights")
ax.set_ylabel("Aircraft type")
ax.set_title("NADP classification by aircraft type (top 15)")
ax.legend(
    _type_display,
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncol=5, frameon=False,
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
    subset = df[df["nadp_category"] == label]
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


# Plot 8: NADP2 sub-type speed vs distance from takeoff

def _cumulative_dist_km(lats, lons):
    """Cumulative along-track haversine distance in km."""
    R = 6371.0
    dists = np.zeros(len(lats))
    for i in range(1, len(lats)):
        dlat = np.radians(lats[i] - lats[i - 1])
        dlon = np.radians(lons[i] - lons[i - 1])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lats[i - 1])) * np.cos(np.radians(lats[i])) * np.sin(dlon / 2) ** 2
        dists[i] = dists[i - 1] + R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return dists

# Build distance-indexed IAS profiles per NADP2 flight (vectorized)
DIST_GRID = np.arange(0, 21, 0.5)  # 0 to 20 km in 0.5 km steps
dist_alt_profiles = {st: [] for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]}

nadp2_flights = df[df["nadp_category"] == "nadp2"].dropna(subset=["v2"])
sample = nadp2_flights.sample(min(3000, len(nadp2_flights)), random_state=42)
sample_ids = set(sample["flight_id"])
type_lookup = sample.set_index("flight_id")["nadp_type"].to_dict()
v2_lookup = sample.set_index("flight_id")["v2"].to_dict()

raw_sub = df_raw[df_raw["FLIGHT_ID"].isin(sample_ids)].sort_values(["FLIGHT_ID", "actual_time"])

for fid, grp in raw_sub.groupby("FLIGHT_ID"):
    if len(grp) < 5:
        continue
    airborne = grp[grp["ALT"] >= 50]
    if airborne.empty:
        continue
    lats, lons = grp["lat"].values, grp["lon"].values
    dists = _cumulative_dist_km(lats, lons)
    # Offset so distance=0 at first airborne point
    airborne_idx = airborne.index[0] - grp.index[0]
    dists = dists - dists[airborne_idx]
    alts = grp["ALT"].values
    valid = ~np.isnan(alts) & ~np.isnan(dists)
    if valid.sum() < 5:
        continue
    interp_alt = np.interp(DIST_GRID, dists[valid], alts[valid],
                           left=np.nan, right=np.nan)
    dist_alt_profiles[type_lookup[fid]].append(interp_alt)

fig, (ax_ias, ax_dist, ax_bar) = plt.subplots(1, 3, figsize=(20, 8),
                                               gridspec_kw={"width_ratios": [2, 2, 1]})

# Left: IAS-V2 vs altitude (original)
for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]:
    subset = df[df["nadp_type"] == st]
    ias_curves = np.array([[row.get(f"delta_ias_alt_{a}", np.nan) for a in ALT_GRID]
                            for _, row in subset.iterrows()])
    mean_ias = np.nanmean(ias_curves, axis=0)
    ax_ias.plot(mean_ias, ALT_GRID, color=SUBTYPE_PALETTE[st], lw=2,
                label=f"{SUBTYPE_LABELS[st]} mean (n={len(subset)})")
    ax_ias.fill_betweenx(ALT_GRID,
                          np.nanpercentile(ias_curves, 25, axis=0),
                          np.nanpercentile(ias_curves, 75, axis=0),
                          color=SUBTYPE_PALETTE[st], alpha=0.15)
    ax_ias.plot(REF_CURVES[st], ALT_GRID, color=SUBTYPE_PALETTE[st], lw=2, ls="--")

ax_ias.set_xlabel("IAS - V2 (kt)")
ax_ias.set_ylabel("Altitude (ft)")
ax_ias.set_title("Speed profiles")
ax_ias.legend(loc="lower right", fontsize=9)
ax_ias.set_xlim(-10, 100)

# Middle: distance vs altitude
for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]:
    curves = np.array(dist_alt_profiles[st])
    if len(curves) == 0:
        continue
    mean_alt = np.nanmean(curves, axis=0)
    ax_dist.plot(DIST_GRID, mean_alt, color=SUBTYPE_PALETTE[st], lw=2,
                 label=f"{SUBTYPE_LABELS[st]}")
    ax_dist.fill_between(DIST_GRID,
                          np.nanpercentile(curves, 25, axis=0),
                          np.nanpercentile(curves, 75, axis=0),
                          color=SUBTYPE_PALETTE[st], alpha=0.15)

ax_dist.set_xlabel("Distance from takeoff (km)")
ax_dist.set_ylabel("Altitude (ft)")
ax_dist.set_title("Altitude vs distance")
ax_dist.legend(loc="upper left", fontsize=9)
ax_dist.set_ylim(0, 5000)

# Bar chart: sub-type counts
counts = [len(df[df["nadp_type"] == st]) for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]]
n_nadp2 = sum(counts)
bars = ax_bar.barh(
    [SUBTYPE_LABELS[st] for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]],
    counts,
    color=[SUBTYPE_PALETTE[st] for st in ["nadp2-800", "nadp2-1000", "nadp2-1500"]],
)
for bar, count in zip(bars, counts):
    ax_bar.text(count + n_nadp2 * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/n_nadp2*100:.0f}%)", va="center", fontsize=11)
ax_bar.set_xlabel("Number of flights")
ax_bar.set_title("NADP2 sub-type distribution")
ax_bar.set_xlim(0, max(counts) * 1.25)

plt.suptitle("NADP2 sub-type analysis (acceleration altitude variants)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_nadp2_subtypes.png", dpi=150)
print(f"Saved {PLOT_DIR}/08_nadp2_subtypes.png")


# Plot 9: NADP type breakdown by airline (top 15 most common)
airline_counts = df["airline"].value_counts()
n_days = (df["start"].max() - df["start"].min()).days + 1
top_airlines = airline_counts[airline_counts > 2 * n_days].index
df_top_al = df[df["airline"].isin(top_airlines)].copy()

ct_al = pd.crosstab(df_top_al["airline"], df_top_al["nadp_type"], normalize="index")
ct_al = ct_al.reindex(columns=_type_cols, fill_value=0)
ct_al = ct_al.loc[reversed(top_airlines)]

fig, ax = plt.subplots(figsize=(12, 13))
ct_al.plot.barh(stacked=True, color=_type_colors, ax=ax)
ax.set_xlabel("Proportion of flights")
ax.set_ylabel("Airline (ICAO code)")
ax.set_title(f"NADP classification by airline ({len(top_airlines)} airlines, >2 flights/day)")
ax.legend(
    _type_display,
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncol=5, frameon=False,
)
counts_by_airline = df_top_al["airline"].value_counts()
for i, airline in enumerate(ct_al.index):
    ax.text(1.02, i, f"n={counts_by_airline[airline]}", va="center", fontsize=10)
ax.set_xlim(0, 1.15)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_airline_breakdown.png", dpi=150, bbox_inches="tight")
print(f"Saved {PLOT_DIR}/09_airline_breakdown.png")
