Connected to .venv (Python 3.12.12)

Total flights loaded from NADP trajectories: 93216
Flights with valid significant phase (800-3000 ft): 93216
Saved: plots/01_sample_trajectories.png

Design matrix shape: (93216, 100)
  - Number of flights: 93216
  - Features per flight (resampled altitude points): 100

KMeans clustering completed.
Cluster 0 count: 41506
Cluster 1 count: 51710

Vertical rate heuristic for NADP mapping:
Mean vertical rate below 2500 ft for cluster 0: 1718.7 ft/min
Mean vertical rate below 2500 ft for cluster 1: 1790.7 ft/min
Mapping: Cluster 1 -> NADP1 (higher vertical rate), Cluster 0 -> NADP2

Final NADP classification:
NADP1 flights: 51710 (55.5%)
NADP2 flights: 41506 (44.5%)
Total classified: 93216
Saved: plots/02_classified_trajectories_combined.png
Saved: plots/03_nadp_comparison.png

=== Processing complete ===