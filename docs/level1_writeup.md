# Level 1: Detection & Tracking — Write-up

## Approach
- **Detector**: YOLOv8s (Ultralytics), pretrained on COCO, run at 1280px inference size on
  4K (3840x2160 @ 30fps) drone footage.
- **Tracker**: BoT-SORT with ReID enabled (`configs/botsort_aerial.yaml`), tuned for aerial
  traffic: higher confidence thresholds to suppress detector flicker, larger track buffer
  (60 frames) to survive occlusion, ReID appearance matching for identity recovery after
  crossing paths.
- **Class mapping**: COCO classes collapsed to the required taxonomy — car, truck, bus,
  motorcycle, pedestrian (LGV/HGV split noted as a limitation below).
- **Output**: per-frame CSV (`frame, track_id, class, conf, x1, y1, x2, y2, cx, cy`) — the
  trajectory foundation for later levels.

## Key iteration: fixing identity persistence
An initial run with default ByteTrack settings and conf=0.25 produced catastrophic ID churn
(14,000+ unique track IDs within the first 2 minutes of footage, for what should be a few
hundred vehicles total). Root cause: default thresholds were tuned for ground-level, larger,
higher-contrast objects — aerial/top-down vehicles are small, low-contrast, and the detector's
confidence flickers frame to frame, causing the tracker to treat the same object as a new
track repeatedly.

Fix: raised `track_high_thresh` (0.25→0.35), `new_track_thresh` (0.25→0.4), detector
`conf` (0.25→0.35) to suppress flicker-driven false starts, increased `track_buffer`
(30→60) to bridge longer occlusions, and enabled BoT-SORT's ReID appearance matching
(`with_reid: True`) to re-associate identities after crossing paths. Validated on a 300-frame
sample: max track ID dropped from >14,000 (comparable window) to 258 — roughly a 5x
reduction in identity churn.

## Known limitations
- Base model uses stock COCO classes; LGV vs HGV vs generic truck requires a fine-tuned
  classifier or a model trained on aerial vehicle data (e.g. VisDrone) — not done given time
  constraints.
- Pretrained on ground-level COCO imagery; may still under-detect very small/distant objects
  in the top-down view compared to a model fine-tuned on this footage.
- **Motorcycle recall is measurably weak**: on the Intersection run, motorcycles account for
  515/200,971 detections (71 unique tracks) at 0.43 average confidence, vs. 183,408 car
  detections at 0.61 average confidence. Motorcycles have a much smaller top-down silhouette
  than the ground-level, side-on motorcycle images COCO was trained on, so detections flicker
  near the confidence threshold and tracks fragment or drop. Fixing this needs either a
  lower per-class confidence threshold for motorcycles specifically, higher-resolution tiled
  inference (e.g. SAHI) at the cost of speed, or fine-tuning on aerial motorcycle examples —
  none attempted here given time constraints.
- Full quantitative ID-switch benchmarking against ground truth wasn't performed (no
  annotations provided); the fix was validated qualitatively via track-count sanity checks
  and visual review of the annotated output video.

## Deliverables
- `src/track.py` — detection + tracking pipeline
- `src/visualize.py` — overlays trajectories on the source video for visual verification
- `configs/botsort_aerial.yaml` — tuned tracker config
- `data/output/intersection_tracks.csv` — trajectory output
- `data/output/intersection_annotated.mp4` — annotated video proof
