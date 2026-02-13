# NOMOS: NADP profile identification from enhanced surveillance data

**Author:** Junzi Sun

## What this does

This repository classifies departure flights at Amsterdam Schiphol Airport (EHAM) as NADP1 or NADP2 based on their speed profiles. Each flight is matched against ICAO standard reference curves, and a delta score measures how far it deviates from the matched standard.

We use VEMMIS enhanced surveillance data (Mode S EHS), which gives us indicated airspeed (IAS), rate of climb/descent (ROCD), and other parameters directly from the aircraft, starting from the ground roll. This means we can classify based on actual speed behavior rather than inferring it from altitude shape.

## Background

NADP1 and NADP2 are two ICAO departure climb procedures that reduce noise exposure at different distances from the runway.

In **NADP1** (close-in), the aircraft holds takeoff configuration and a near-constant speed (about V2 + 10-20 kt) all the way to 3000 ft, then accelerates and retracts flaps. The result is a steeper, slower climb that is quieter close to the airport.

In **NADP2** (distant), the aircraft begins accelerating and retracting flaps from 800 ft. This gives a shallower, faster climb that is quieter farther out.

Below 800 ft, both procedures are identical. The difference shows up in the speed profile between 800 and 3000 ft: NADP1 stays flat, NADP2 ramps up.

![ICAO NADP reference speed profiles](plots/00_reference_profiles.png)

Schiphol recommends NADP2 for most departures, and literature reports about 80% NADP2 adoption. Our results are consistent with this.

## Data

Input is VEMMIS enhanced surveillance CSV files, one per day. Each row is a surveillance point:

| Column | Description |
|--------|-------------|
| `FLIGHT_ID` | Unique flight identifier |
| `ICAO_ACTYPE` | Aircraft type code (e.g. B738, A320) |
| `ALT` | QNH-corrected altitude (ft) |
| `EHS_IAS` | Indicated airspeed from Mode S EHS (kt) |
| `EHS_ROCD` | Rate of climb/descent from Mode S EHS (ft/min) |
| `actual_time` | Timestamp |

Place data files in `data/vemmis_202503/` with naming pattern `vemmis_YYYYMMDD.csv`.

## Method

### V2 extraction

V2 (takeoff safety speed) is different for every flight because it depends on weight, flap setting, and temperature. We approximate V2 as the minimum IAS in the 200-800 ft altitude band. This is the stabilized initial climb speed after the pitch-up transient that follows rotation, and it gives us a consistent baseline for computing speed deltas across flights.

### Why we use speed only, not climb rate

We classify on indicated airspeed alone and ignore ROCD entirely.

The main reason is that NADP procedures are defined by speed behavior: when to accelerate and when to retract flaps (800 ft for NADP2, 3000 ft for NADP1). Climb rate differences between the two procedures are a secondary consequence of these speed and configuration choices, not the defining feature.

There are also practical data quality issues. IAS comes directly from the aircraft's air data computer via Mode S downlink. ROCD is derived from barometric altitude differences over short time intervals, which makes it noisy. Typical oscillations are 500-1500 ft/min around the mean, and this swamps the ~500 ft/min systematic difference between procedures.

ROCD also varies a lot with aircraft type and weight. A heavy B777 and a light E190 can both follow NADP2 but climb at very different rates because of their thrust-to-weight ratios. IAS relative to V2 normalizes most of this out, because both types show the same acceleration pattern relative to their own V2.

We confirmed this empirically. When we included ROCD in the classification metric, noise dominated and the results were wrong (the NADP1/NADP2 ratio was inverted). Dropping ROCD and classifying on IAS alone gave results matching the expected ~80% NADP2 at Schiphol.

ROCD is still computed and shown in the plots for comparison. It does reveal useful aerodynamic patterns, like the characteristic ROCD dip at 1000-1500 ft in NADP2 flights where the aircraft trades climb rate for speed during flap retraction.

### Classification

We extract speed features at three altitudes that matter for distinguishing NADP procedures:

| Feature | Description |
|---------|-------------|
| `delta_ias_800` | IAS minus V2 at 800 ft |
| `delta_ias_1500` | IAS minus V2 at 1500 ft |
| `delta_ias_3000` | IAS minus V2 at 3000 ft |

These are compared to ICAO reference vectors using weighted Euclidean distance:

- NADP1 reference: `[0, 0, 0]` (no acceleration until 3000 ft)
- NADP2 reference: `[0, 30, 70]` (progressive acceleration from 800 ft)

A flight is only classified if the two distances are different enough. Specifically, the ratio of the closer distance to the farther distance must be below 0.6. Above that, the flight is marked as "unknown" rather than forced into a category.

![Classification feature space](plots/05_feature_space.png)

### Delta score

After classification, we compare each flight's full speed profile (IAS minus V2 at every 100 ft from 200 to 3500 ft) against its matched ICAO reference curve. The delta score is the RMS residual divided by 30 kt:

```
delta_score = RMS(flight_ias - reference_ias) / 30
```

Zero means the flight perfectly follows the reference; higher values mean more deviation.

## Pipeline

Three scripts, run in order:

| Script | Input | Output | What it does |
|--------|-------|--------|-------------|
| `1_ingest_vemmis.py` | `data/vemmis_202503/*.csv` | `data/vemmis_departures.parquet` | Filters to EHAM departures, keeps climb segments below 5000 ft within 500 s of departure |
| `2_extract_features.py` | `data/vemmis_departures.parquet` | `data/nadp_features.parquet` | Extracts V2, milestone features, and altitude-indexed speed/ROCD curves per flight |
| `3_classify_nadp.py` | `data/nadp_features.parquet` | `data/nadp_results.csv` + `plots/` | Classifies flights, computes delta scores, generates plots |

```bash
uv run python 1_ingest_vemmis.py
uv run python 2_extract_features.py
uv run python 3_classify_nadp.py
```

## Results

19,250 EHAM departures from March 2025:

| Category | Flights | Percentage |
|----------|---------|------------|
| NADP1 | 752 | 3.9% |
| NADP2 | 16,031 | 83.3% |
| Unknown | 2,467 | 12.8% |

Among classified flights, 95.5% follow NADP2.

### Departure trajectories

NADP1 flights climb more steeply. NADP2 flights have a more gradual, consistent profile:

![Departure trajectories by NADP type](plots/01_trajectories_by_nadp.png)

### Speed and climb rate profiles

Top row: IAS minus V2 versus altitude, with ICAO reference curves. NADP1 flights hold nearly constant speed through 3000 ft; NADP2 flights accelerate from 800 ft. Bottom row: the corresponding ROCD profiles. The NADP2 dip around 1000-1500 ft is where the aircraft trades climb rate for speed during flap retraction:

![Speed and ROCD profiles](plots/02_speed_profiles_vs_reference.png)

### Mean profiles

Mean speed and ROCD with interquartile range. The two procedures separate cleanly, and the ICAO reference curves sit close to the observed means:

![Mean profiles with IQR](plots/08_mean_profiles.png)

### Separation ratio

Most classified flights have separation ratios well below 0.6. Flights above the threshold end up as "unknown":

![Separation ratio distribution](plots/03_separation_ratio.png)

### Delta score distribution

IAS RMS deviation from reference. NADP1 flights match their reference a bit more closely (median ~8 kt) than NADP2 flights (median ~11 kt):

![Delta score violin plot](plots/06_delta_violin.png)

### Aircraft type breakdown

Wide-bodies (B77W, B789) have more NADP1 departures. Narrowbodies (B738, E295, B737) are almost all NADP2:

![NADP by aircraft type](plots/07_actype_breakdown.png)

### Best and worst matches

Three best and three worst flights per type, compared to the reference profiles for IAS and ROCD:

![Best vs worst reference comparison](plots/04_reference_comparison.png)

## Output

`data/nadp_results.csv`, one row per flight:

| Column | Description |
|--------|-------------|
| `flight_id` | Unique flight identifier |
| `icao_actype` | Aircraft type code |
| `v2` | Extracted V2 proxy (kt) |
| `nadp_type` | `nadp1`, `nadp2`, or `unknown` |
| `delta_score` | Normalized RMS deviation from reference (0 = perfect) |
| `delta_ias_rms` | Raw IAS RMS deviation (kt) |
| `delta_ias_800` | IAS minus V2 at 800 ft (kt) |
| `delta_ias_1500` | IAS minus V2 at 1500 ft (kt) |
| `delta_ias_3000` | IAS minus V2 at 3000 ft (kt) |
| `mean_rocd_800_1500` | Mean ROCD in 800-1500 ft band (ft/min) |
| `mean_rocd_1500_3000` | Mean ROCD in 1500-3000 ft band (ft/min) |

## Limitations

The QNH-corrected ALT column has a discontinuity around 4000 ft from the QNH-to-QNE barometric transition. We cap the analysis at 3500 ft to stay below this.

The V2 proxy (minimum IAS in 200-800 ft) is an approximation. It won't exactly match the V2 from aircraft performance tables, but it is consistent enough across flights for classification purposes.

About 13% of flights can't be confidently classified. Some of these may follow modified procedures, some may have been affected by wind or ATC constraints, and some just have noisy speed data.

Mode S ROCD is noisy by nature, which is why we don't use it for classification. It is still reported in the output and plots for reference.

The reference profiles and thresholds are calibrated for Schiphol. Other airports may need different values.
