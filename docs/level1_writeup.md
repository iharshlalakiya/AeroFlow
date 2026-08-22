# Level 1 — Detection & Tracking

**AeroFlow: a traffic analysis agent for aerial video**
Repository: https://github.com/iharshlalakiya/AeroFlow

---

## 1. What this level had to do

Detect and track every road user in drone footage of an urban intersection, classify by mode,
and hold a stable identity through occlusion, crossing paths and long dwell times. The output
is a per-frame trajectory table — the foundation every later level builds on.

Footage: `Intersection_Merged` — 3840×2160 @ 29.97 fps, 11,971 frames (6 min 39 s), a dense
signalised intersection in Pune, India. No annotations provided.

Hardware constraint worth stating up front: a **GTX 1650 (4 GB VRAM)**. That shaped every
decision below — model size, inference resolution, and whether an idea was affordable at all.

---

## 2. Final pipeline

```
video
  └─> track_visdrone.py     detection + tracking      → raw trajectories
        └─> stitch_tracks.py     re-link fragments (2 passes)
              └─> interpolate_gaps.py  fill dropouts
                    └─> visualize.py   annotated video
```

**Detector** — YOLOv8s fine-tuned on VisDrone2019 (`mshamrai/yolov8s-visdrone`), run at
1280 px, conf 0.55. Chosen over stock COCO weights because VisDrone is aerial-native and its
label set maps far better onto the required taxonomy — critically it has `van` (→ LGV) and
`motor`, two categories COCO cannot express for this task.

| VisDrone class | Our taxonomy |
|---|---|
| pedestrian | pedestrian |
| car | car |
| van | **LGV** |
| truck | truck |
| bus | bus |
| motor | motorcycle |

`bicycle`, `people`, `tricycle`, `awning-tricycle` are excluded — measured as false-positive
prone on this footage.

**Tracker** — BoT-SORT with ReID (`configs/botsort_visdrone.yaml`), `match_thresh` 0.9,
`appearance_thresh` 0.85, `track_buffer` 60, sparse-optical-flow global motion compensation
to absorb drone drift.

**Post-processing** — two passes described in §3.

**Output schema** — `data/output/intersection_tracks_smooth.csv`:

```
frame, track_id, raw_class, taxonomy_class, conf, x1, y1, x2, y2, cx, cy, interpolated
```

Both the raw model label and the mapped taxonomy label are kept, so no information is
discarded by the mapping. `interpolated` flags rows that were reconstructed rather than
detected — downstream levels can filter them out if they need detections only.

---

## 3. The engineering that mattered: measuring the failure instead of guessing it

The first run was unusable — **1,303 track IDs in 60 seconds**, with trails that broke
constantly. Tuning thresholds helped but plateaued:

| Iteration | Tracks | ≤3-frame tracks | Median duration |
|---|---|---|---|
| v1 baseline | 1,303 | 39.8 % | 6 frames |
| v2 higher thresholds, noisy classes dropped | 742 | 33.3 % | 12 |
| v3 + ReID, conf 0.55, 1280 px | 693 | 30.9 % | 15 |
| v4 + strict match/appearance | 607 | 28.3 % | 22 |

Four rounds of tuning and still 28 % of tracks lasted under 3 frames. The visible symptom was
identities changing when two boxes overlapped, so the obvious diagnosis was ID swapping
during occlusion. **That diagnosis was wrong**, and testing it was what unlocked the rest.

**Test 1 — are identities actually swapping?** A genuine swap has a signature: two tracks
jump position in the same frame, each landing where the other just was. `src/fix_swaps.py`
searches for that signature. It found **zero** such events. The data also showed why: mean
per-frame displacement is 2.8 px, 99th percentile 16 px. Nothing teleports. There are no swaps.

**Test 2 — do tracks die during overlaps?** For every track ending mid-video, check whether
another box overlapped it at that moment. Only **14.8 %** did. The other 85 % were objects
sitting alone in clear view.

The real failure mode was therefore **detector dropout**, not tracker confusion: the detector
intermittently loses a plainly visible object, the track dies, and a new ID is born when it
reappears. Overlap was a coincidence of dense traffic, not the cause.

That reframing pointed at two fixes that tuning could never deliver:

**Fix 1 — stitch fragments (`stitch_tracks.py`).** If a track ends and another of the same
class starts shortly after, nearby, they are the same object. Two passes, because dropouts
come in two flavours:

- *long gap, tight radius* (240 frames / 100 px) — vehicles stopped at the signal, where the
  object barely moves during a long dropout. A tight radius makes a long bridge safe.
- *short gap, loose radius* (45 frames / 220 px) — moving traffic, which covers ground quickly.

**Fix 2 — interpolate the holes (`interpolate_gaps.py`).** Stitching restores the identity but
leaves the gap empty, which is exactly what makes boxes blink. Since the object's position is
known either side of the hole, the box is linearly interpolated through it and flagged
`interpolated=True`.

### Result

| Metric | v1 baseline | Final |
|---|---|---|
| Track IDs (60 s) | 1,303 | **380** |
| Tracks lasting ≤3 frames | 39.8 % | **13.9 %** |
| Median track duration | 6 frames (0.2 s) | **166 frames (5.5 s)** |
| Track continuity¹ | 0.648 | **0.918** |

¹ frames where a track is present ÷ frames it spans. 0.648 means a third of each track was
holes — that was the flicker.

**28× longer median tracks and 3.4× fewer IDs.** Class counts also became plausible: 155
motorcycles per minute (from 447), which fits an Indian intersection where two-wheelers
genuinely dominate, rather than an artefact of fragmentation.

---

## 4. Results on the validation minute

380 tracks over 1,800 frames; 147,927 rows (34,413 interpolated).

| Class | Unique tracks | Mean confidence |
|---|---|---|
| motorcycle | 155 | 0.72 |
| car | 118 | 0.82 |
| pedestrian | 103 | 0.64 |
| LGV (van) | 75 | 0.75 |
| truck | 17 | 0.81 |
| bus | 6 | 0.73 |

The two-wheeler-dominant mix is characteristic of the location and is the main reason the
COCO-pretrained baseline was inadequate here: it found 9 motorcycles in the same minute,
which is not a plausible count for this scene.

---

## 5. Honest limitations

- **Detector dropout is mitigated, not solved.** Stitching and interpolation repair the
  trajectory after the fact; the detector still loses objects. The real fix is fine-tuning on
  annotated frames from this footage, which needs labels we do not have.
- **13.9 % of tracks still last ≤3 frames.** Some are genuine brief appearances at the frame
  edge; some are residual noise. Without ground truth these cannot be separated.
- **No MOTA/IDF1 numbers.** No annotations were provided, so the metrics here are internal
  consistency measures (fragmentation, continuity, plausibility of counts), not benchmark
  scores. They show relative improvement, not absolute accuracy.
- **Stitching can merge wrongly.** Two different vehicles stopping in the same spot within the
  gap window could be joined. Parameters were tuned to make this unlikely, not impossible.
- **Interpolated boxes are inferred, not observed.** They are flagged so downstream analysis
  can exclude them; treating them as detections would overstate recall.
- **HGV is not separated from truck.** VisDrone has a single `truck` class; splitting HGV from
  rigid trucks needs either a size heuristic in ground-plane metres (available once Level 4
  provides the homography) or a fine-tuned classifier.

---

## 6. What I would do next

1. **Fine-tune on this footage** — label a few hundred frames and fine-tune the VisDrone
   weights. Dropout is a domain-gap problem and this is the direct fix.
2. **SAHI tiled inference** — slicing 4K frames would materially improve small-object recall
   (motorcycles especially). Rejected here only because it multiplies inference cost beyond
   what a 4 GB GPU could complete in the time available.
3. **Motion-model-aware stitching** — extrapolate velocity across a gap rather than matching
   raw proximity, which would make long bridges safer for fast-moving traffic.
4. **HGV/LGV split in metric space** once the ground-plane projection from Level 4 exists.

---

## 7. Repository

| Path | Purpose |
|---|---|
| `src/track_visdrone.py` | Detection + tracking → trajectory CSV |
| `src/stitch_tracks.py` | Re-links fragmented tracks |
| `src/interpolate_gaps.py` | Fills dropouts, removes flicker |
| `src/fix_swaps.py` | ID-swap detector (diagnostic; found none) |
| `src/visualize.py` | Annotated video with class-coloured trails |
| `configs/botsort_visdrone.yaml` | Tuned tracker configuration |
| `requirements.txt` | Environment |

Reproduce:

```bash
python src/track_visdrone.py    --video data/raw_video/Intersection_Merged_convert_4k.mp4 \
                                --out data/output/tracks.csv --device 0 --imgsz 1280 --conf 0.55
python src/stitch_tracks.py     --in data/output/tracks.csv --out data/output/p1.csv --max-gap 240 --max-dist 100
python src/stitch_tracks.py     --in data/output/p1.csv     --out data/output/p2.csv --max-gap 45  --max-dist 220
python src/interpolate_gaps.py  --in data/output/p2.csv     --out data/output/tracks_smooth.csv
python src/visualize.py         --video data/raw_video/Intersection_Merged_convert_4k.mp4 \
                                --tracks data/output/tracks_smooth.csv --out data/output/annotated.mp4 --trail 200
```
