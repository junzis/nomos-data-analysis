# NOMOS: NADP profile identification from enhanced surveillance data

**Author:** Junzi Sun

This repository classifies departure flights at Amsterdam Schiphol Airport (EHAM) into NADP1 and NADP2 noise abatement procedures based on their speed profiles during initial climb. It uses VEMMIS enhanced surveillance data (Mode S EHS), which provides indicated airspeed directly from the aircraft. Each flight is matched against ICAO reference curves, and an extended classifier further distinguishes three NADP2 sub-types by acceleration start altitude (800, 1000, or 1500 ft).

Applied to 127,237 departures from March--August 2025, the classifier finds 97.5% NADP2 adoption among classified flights.

For the full methodology, results, and discussion, see [report.md](report.md).

## Data

Input is VEMMIS enhanced surveillance CSV files, one per day. Place data files in `data/vemmis_YYYYMM/` directories with naming pattern `vemmis_YYYYMMDD.csv`.

Each row is a surveillance point with columns including `FLIGHT_ID`, `CALLSIGN`, `ICAO_ACTYPE`, `ALT`, `EHS_IAS`, `EHS_ROCD`, `actual_time`, `lat`, `lon`.

## Pipeline

Scripts, run in order:

| Script | Description | Output |
|--------|-------------|--------|
| `1_ingest_vemmis.py` | Reads daily VEMMIS CSVs, filters to EHAM departures, keeps climb segments below 5000 ft. Use `--all-months` to process all available months. | `data/vemmis_departures.parquet` |
| `2_extract_features.py` | Extracts V2, speed/ROCD features, and altitude-indexed curves per flight. | `data/nadp_features.parquet` |
| `3_classify_nadp_2profiles.py` | 2-profile classifier (NADP1 vs NADP2). | `data/nadp_results.csv`, `plots/2profiles/` |
| `4_classify_nadp_4profiles.py` | 4-profile classifier with NADP2 sub-types (800/1000/1500 ft). | `data/nadp_results.csv`, `plots/4profiles/` |

```bash
# Single month (March 2025)
uv run python 1_ingest_vemmis.py
uv run python 2_extract_features.py
uv run python 3_classify_nadp_2profiles.py
uv run python 4_classify_nadp_4profiles.py

# All available months
uv run python 1_ingest_vemmis.py --all-months
uv run python 2_extract_features.py
uv run python 3_classify_nadp_2profiles.py
uv run python 4_classify_nadp_4profiles.py
```

## Output

`data/nadp_results.csv`, one row per flight:

| Column | Description |
|--------|-------------|
| `flight_id` | Unique flight identifier |
| `typecode` | Aircraft type code (e.g. B738, A320) |
| `callsign` | Flight callsign (e.g. KLM1234) |
| `airline` | ICAO airline code (first 3 characters of callsign) |
| `start` | Departure timestamp (first airborne point) |
| `v2` | Extracted V2 proxy (kt) |
| `nadp_type` | `nadp1`, `nadp2-800`, `nadp2-1000`, `nadp2-1500`, or `unknown` |
| `nadp_category` | High-level category: `nadp1`, `nadp2`, or `unknown` |
| `delta_score` | Normalized RMS deviation from reference (0 = perfect) |
| `delta_ias_rms` | Raw IAS RMS deviation (kt) |
| `delta_ias_800` | IAS minus V2 at 800 ft (kt) |
| `delta_ias_1500` | IAS minus V2 at 1500 ft (kt) |
| `delta_ias_3000` | IAS minus V2 at 3000 ft (kt) |
| `mean_rocd_800_1500` | Mean ROCD in 800-1500 ft band (ft/min) |
| `mean_rocd_1500_3000` | Mean ROCD in 1500-3000 ft band (ft/min) |
