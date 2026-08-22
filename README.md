# AeroFlow — Traffic Analysis Agent

Extracts road-user trajectories from drone footage and derives traffic insight from them.

## Level 1: Detection & Tracking

Pipeline: YOLOv8 detector + ByteTrack multi-object tracker, run on 4K drone footage,
producing per-frame trajectories (track_id, class, bbox, centroid) as CSV.

### Setup
```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### Run detection + tracking
```bash
venv\Scripts\python src\track.py --video data\raw_video\Intersection_Merged.MP4 --out data\output\intersection_tracks.csv
```

### Visualize (sanity-check identity persistence)
```bash
venv\Scripts\python src\visualize.py --video data\raw_video\Intersection_Merged.MP4 --tracks data\output\intersection_tracks.csv --out data\output\intersection_annotated.mp4
```

### Known limitations (Level 1)
- Base model uses stock COCO classes (car/truck/bus/motorcycle/person). LGV vs HGV vs truck
  distinction requires a fine-tuned classifier head or a model trained on aerial vehicle data
  (e.g. VisDrone) — planned as a follow-up.
- Pretrained on ground-level COCO imagery; top-down drone perspective may reduce recall on
  small/distant objects. Fine-tuning on a labeled subset of this footage would improve this.

### Output schema
`data/output/*.csv`: `frame, track_id, class, conf, x1, y1, x2, y2, cx, cy`

This is the foundation trajectory data used by all later levels (interaction metrics,
turning counts, spatial grounding, network reasoning).
