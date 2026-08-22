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
