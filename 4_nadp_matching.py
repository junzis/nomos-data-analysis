"""

This script processes aircraft climb trajectory data to classify flights 
into NADP1 and NADP2 departure procedures using unsupervised clustering. 
It loads trajectory data, extracts the significant climb phase for each flight, 
and applies KMeans clustering to group similar climb profiles. 

The clusters are then mapped to NADP types based on vertical rate characteristics, 
and the results are visualized and exported for further analysis.

Steps:

- Loads NADP trajectories from Parquet file.
- Extracts the significant climb phase (800-3000 ft) for each flight, resampling and adding time-from-takeoff.
- Prepares altitude data for clustering and applies KMeans to group into two clusters.
- Merges cluster labels with flight data.
- Determines which cluster is NADP1 or NADP2 based on mean vertical rate below 2500 ft.
- Exports mapping of flight IDs to NADP type as CSV.
- Merges NADP type back to flight data.
- Plots colored altitude profiles for each NADP type for visual comparison.
"""

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from traffic.core import Flight, Traffic

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# %%
t = Traffic.from_file("data/nadp_trajectories.parquet")
print(f"Total flights loaded from NADP trajectories: {len(t)}")


def extract_significant_phase(flight: Flight) -> Flight:
    """
    Extracts the significant phase of a flight between 800 and 3000 altitude.
    Adds a timestamp column in seconds and resamples the data.
    """
    return (
        flight.query("800<altitude<3000")
        .assign(ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds())
        .resample(100)
    )


t_initial_climb = t.pipe(extract_significant_phase).eval(24, desc="processing")

print(f"Flights with valid significant phase (800-3000 ft): {len(t_initial_climb)}")

# %%
# Plot initial sample of trajectories
fig, ax = plt.subplots(figsize=(10, 6))
for f in t_initial_climb.sample(400):
    ax.plot(f.data.ts, f.data.altitude, lw=0.5, color="k", alpha=0.2)
ax.set_xlabel("Time from start of segment (s)")
ax.set_ylabel("Altitude (ft)")
ax.set_title("Sample of 400 climb trajectories (800-3000 ft)")
plt.tight_layout()
# plt.savefig("plots/01_sample_trajectories.png", dpi=150)


# %%
# Prepare data for clustering
X = np.array(
    [flight.data["altitude"].to_numpy().flatten() for flight in t_initial_climb]
)

print(f"\nDesign matrix shape: {X.shape}")
print(f"  - Number of flights: {X.shape[0]}")
print(f"  - Features per flight (resampled altitude points): {X.shape[1]}")

model = KMeans(n_clusters=2, random_state=42)

labels = model.fit_predict(X)

print(f"\nKMeans clustering completed.")
print(f"Cluster 0 count: {np.sum(labels == 0)}")
print(f"Cluster 1 count: {np.sum(labels == 1)}")

# %%
# Merge cluster labels with flight data
t_nadp = t_initial_climb.merge(pd.DataFrame(dict(flight_id=t.flight_ids, cluster_label=labels)))

#%%
# Determine which cluster is NADP1 and NADP2 based on mean vertical rate before FL25
vs_0_mean = t_nadp.query("altitude<2500 and cluster_label==0").data.vertical_rate.mean()
vs_1_mean = t_nadp.query("altitude<2500 and cluster_label==1").data.vertical_rate.mean()

print(f"\nVertical rate heuristic for NADP mapping:")
print(f"Mean vertical rate below 2500 ft for cluster 0: {vs_0_mean:.1f} ft/min")
print(f"Mean vertical rate below 2500 ft for cluster 1: {vs_1_mean:.1f} ft/min")

if vs_0_mean > vs_1_mean:
    nadp_mapping = {0: "nadp1", 1: "nadp2"}
    color_mapping = {0: "blue", 1: "red"}
    print("Mapping: Cluster 0 -> NADP1 (higher vertical rate), Cluster 1 -> NADP2")
else:
    nadp_mapping = {1: "nadp1", 0: "nadp2"}
    color_mapping = {1: "blue", 0: "red"}
    print("Mapping: Cluster 1 -> NADP1 (higher vertical rate), Cluster 0 -> NADP2")

# Assign colors and NADP types
colors = list(map(color_mapping.get, labels))
nadps = list(map(nadp_mapping.get, labels))

# DataFrame with flight IDs and NADP types
df_nadp = pd.DataFrame(dict(flight_id=t.flight_ids, nadp=nadps))

df_nadp.to_csv("data/flight_id_nadp.csv", index=False)

# Print final statistics
nadp1_count = df_nadp[df_nadp.nadp == "nadp1"].shape[0]
nadp2_count = df_nadp[df_nadp.nadp == "nadp2"].shape[0]
print(f"\nFinal NADP classification:")
print(f"NADP1 flights: {nadp1_count} ({nadp1_count/len(df_nadp)*100:.1f}%)")
print(f"NADP2 flights: {nadp2_count} ({nadp2_count/len(df_nadp)*100:.1f}%)")
print(f"Total classified: {len(df_nadp)}")

t_nadp = t_initial_climb.merge(df_nadp)

# %%
# Plot combined trajectories colored by NADP type
fig, ax = plt.subplots(figsize=(10, 6))
for i, f in enumerate(t_nadp.query("altitude>800")[:500]):
    ax.plot(f.data.ts, f.data.altitude, lw=1, color=colors[i], alpha=0.2)

ax.plot(0, 0, color="tab:blue", label="NADP 1")
ax.plot(0, 0, color="tab:red", label="NADP 2")
ax.set_ylim(500)
ax.legend()
ax.set_xlabel("Time from take-off (s)")
ax.set_ylabel("Altitude (ft)")
ax.set_title("Classified trajectories colored by NADP type (sample of 500)")
plt.tight_layout()
# plt.savefig("plots/02_classified_trajectories_combined.png", dpi=150)


# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

for i, f in enumerate(t_nadp.query("nadp=='nadp1' and 800<altitude<3000")[:500]):
    ax1.plot(f.data.ts, f.data.altitude, lw=1, color="tab:blue", alpha=0.2)

for i, f in enumerate(t_nadp.query("nadp=='nadp2' and 800<altitude<3000")[:500]):
    ax2.plot(f.data.ts, f.data.altitude, lw=1, color="tab:red", alpha=0.2)

ax1.set_ylabel("NADP 1 : altitude (ft)")
ax2.set_ylabel("NADP 2 : altitude (ft)")

ax2.set_xlabel("time from take-off (s)")
ax1.set_title("NADP1 vs NADP2 Trajectory Comparison")

plt.tight_layout()
# plt.savefig("plots/03_nadp_comparison.png", dpi=150)

# %%
