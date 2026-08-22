# AeroFlow — Traffic Analysis Agent

Extracts road-user trajectories from aerial drone footage and derives traffic insight from them.

**Level 1 write-up:** [`docs/level1_writeup.md`](docs/level1_writeup.md)

## Pipeline

```
video → track_visdrone.py → stitch_tracks.py ×2 → interpolate_gaps.py → visualize.py
```

- **Detector** — YOLOv8s fine-tuned on VisDrone2019 (`mshamrai/yolov8s-visdrone`), 1280 px, conf 0.55
- **Tracker** — BoT-SORT + ReID, tuned in `configs/botsort_visdrone.yaml`
- **Post-processing** — fragment stitching, then gap interpolation
- **Classes** — pedestrian, motorcycle, car, LGV, truck, bus

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

The VisDrone weights download to `models/best.pt`:

```bash
venv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mshamrai/yolov8s-visdrone', filename='best.pt', local_dir='models')"
```

## Run

```bash
# 1. detect + track
venv\Scripts\python src\track_visdrone.py --video data\raw_video\Intersection_Merged_convert_4k.mp4 ^
    --out data\output\tracks.csv --device 0 --imgsz 1280 --conf 0.55

# 2. re-link fragmented tracks (long-gap/tight, then short-gap/loose)
venv\Scripts\python src\stitch_tracks.py --in data\output\tracks.csv --out data\output\p1.csv --max-gap 240 --max-dist 100
venv\Scripts\python src\stitch_tracks.py --in data\output\p1.csv --out data\output\p2.csv --max-gap 45 --max-dist 220

# 3. fill detection dropouts
venv\Scripts\python src\interpolate_gaps.py --in data\output\p2.csv --out data\output\tracks_smooth.csv

# 4. annotated video
venv\Scripts\python src\visualize.py --video data\raw_video\Intersection_Merged_convert_4k.mp4 ^
    --tracks data\output\tracks_smooth.csv --out data\output\annotated.mp4 --trail 200
```

## Output schema

`frame, track_id, raw_class, taxonomy_class, conf, x1, y1, x2, y2, cx, cy, interpolated`

Both the raw detector label and the mapped taxonomy label are kept. `interpolated` marks rows
reconstructed through a detection dropout rather than observed — filter these out if you need
detections only.

## Results (60 s validation clip)

| Metric | Baseline | Final |
|---|---|---|
| Track IDs | 1,303 | 380 |
| Tracks ≤3 frames | 39.8 % | 13.9 % |
| Median track duration | 6 frames | 166 frames |
| Track continuity | 0.648 | 0.918 |

See the write-up for how these were achieved and the limitations that remain.

## Level 2 — Object-Level Insight

**Level 2 write-up:** [`docs/level2_writeup.md`](docs/level2_writeup.md)

Adds fine-grained classification and real-unit kinematics to the Level 1 trajectory table.

### Level 2 pipeline

```
intersection_tracks_smooth.csv  ──┐
multiroad_tiled_smooth.csv      ──┤
                                   level2_pipeline.py
                                    ├── calibrate_scale.py    (px/m from vehicle widths)
                                    ├── consolidate_class.py  (stable label + subclass)
                                    └── compute_kinematics.py (velocity/accel/heading)
                                   │
           intersection_tracks_l2.csv   multiroad_tiled_l2.csv
```

```bash
venv\Scripts\python src\level2_pipeline.py          # both datasets
venv\Scripts\python src\level2_pipeline.py --intersection
venv\Scripts\python src\level2_pipeline.py --multiroad
```

### Level 2 output schema

```
frame, track_id, raw_class, taxonomy_class, conf,
x1, y1, x2, y2, cx, cy, interpolated, class,
track_class, subclass,
smoothed_cx, smoothed_cy,
vx_ms, vy_ms, speed_kmh, accel_ms2, heading_deg
```

| New column    | Unit | Description |
|---|---|---|
| `track_class` | —    | Stable per-track class (confidence-weighted majority vote) |
| `subclass`    | —    | `HGV` / `rigid_truck` for truck tracks; NaN elsewhere |
| `smoothed_cx` | px   | Savitzky-Golay smoothed centroid x |
| `smoothed_cy` | px   | Savitzky-Golay smoothed centroid y |
| `vx_ms`       | m/s  | East-component velocity |
| `vy_ms`       | m/s  | South-component velocity |
| `speed_kmh`   | km/h | Scalar speed |
| `accel_ms2`   | m/s² | Scalar acceleration |
| `heading_deg` | °    | Direction of travel (0 = north, clockwise) |

### Level 2 results

**Scale calibration** (vehicle-width prior method):

| Scene        | px/m  | m/px   | Implied AGL |
|---|---|---|---|
| Intersection | 50.65 | 0.0197 | ~40 m       |
| Multi-road   | 14.89 | 0.0672 | ~133 m      |

**Speed by class — Intersection (signalised queue):**

| Class      | Median (km/h) | p95 (km/h) |
|---|---|---|
| motorcycle | 6.6           | 19.4       |
| car        | 0.8           | 14.2       |
| pedestrian | 1.2           | 12.6       |
| LGV        | 1.8           | 16.7       |
| truck      | 0.5           | 4.9        |
| bus        | 0.2           | 2.0        |

**Speed by class — Multi-road (mixed arterial):**

| Class      | Median (km/h) | p95 (km/h) |
|---|---|---|
| motorcycle | 22.0          | 77.5       |
| bus        | 11.5          | 26.2       |
| car        | 4.2           | 27.6       |
| pedestrian | 2.0           | 43.4*      |
| LGV        | 1.1           | 22.5       |
| truck      | 1.8           | 20.6       |

*p95 includes noise from short tracks near tile boundaries; median is unaffected.
