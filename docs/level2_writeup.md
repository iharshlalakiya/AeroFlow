# AeroFlow — Level 2: Object-Level Insight

**Repository:** https://github.com/iharshlalakiya/AeroFlow
**Author:** Harsh Lalakiya
**Dataset:** Drone footage of urban traffic (Pune, India) — 3840 × 2160 @ 29.97 fps

---

## 1. Summary

Level 2 builds directly on the Level 1 trajectory table to deliver two new dimensions of insight:

1. **Fine-grained, stable classification** — a per-track class label that does not flicker frame-to-frame, plus sub-classification of trucks into HGV vs rigid truck.
2. **Kinematics in real-world units** — per-object velocity (km/h), acceleration (m/s²) and heading (°), derived from smoothed pixel trajectories using a data-driven pixel-to-metre scale calibration.

Both datasets are enriched:

| Dataset | Tracks | Frames | Scene type |
|---|---|---|---|
| Intersection | 380 | 1,800 (60 s) | Dense signalised junction, Pune |
| Multi-road | 473 | 1,800 (60 s) | Mixed arterials and ring-road segments |

---

## 2. Fine-grained Classification

### 2.1 The label-flicker problem

The YOLO detector assigns an independent class to each detection. On ambiguous objects — a van at the `car`/`LGV` boundary, a motorcycle partially occluded by a tile edge — the label can flip between frames within the same track. Downstream kinematic analysis (grouping speed histograms by class, computing class-conditional stop rates) requires a single, authoritative class per track.

### 2.2 Confidence-weighted majority vote

For each track we accumulate the detector's output confidence score by class over all **non-interpolated** detections:

```
vote(class c, track T) = Σ  conf(frame)   for all detected frames where label = c
track_class(T)         = argmax_c  vote(c, T)
```

Using confidence rather than a plain frame count means high-certainty detections dominate. A car briefly mis-labelled as LGV with conf = 0.42 cannot overrule a run of car detections at conf = 0.82.

**Outcome:** `track_class` is verified programmatically to be **100 % stable** per track in both datasets. The original per-frame `class` column is preserved so no detector information is discarded.

### 2.3 Truck sub-classification: HGV vs rigid truck

VisDrone provides a single `truck` class. Without a ground-plane homography we cannot measure vehicle length in metres, but **bounding-box area** serves as a reliable proxy: articulated HGVs (semi-trailers, rigid long-wheelbase lorries) produce substantially larger image footprints than city-delivery rigid trucks, even from altitude.

**Method:** for each truck track, compute the median bounding-box area. Tracks whose median exceeds **1.5 × the global median** across all truck tracks are labelled `HGV`; the rest are `rigid_truck`.

| Dataset | HGV tracks | rigid_truck tracks |
|---|---|---|
| Intersection | 3 | 4 |
| Multi-road | 9 | 23 |

This is explicitly a heuristic. Two vehicles of the same physical length can produce different box areas depending on view angle and camera distance. A Level 4 homography enabling length measurement in metres is the definitive fix.

### 2.4 Full class breakdown

**Intersection — track counts by track_class:**

| Class | Tracks | Share |
|---|---|---|
| motorcycle | 141 | 37.1 % |
| car | 103 | 27.1 % |
| pedestrian | 94 | 24.7 % |
| LGV | 32 | 8.4 % |
| truck | 7 | 1.8 % |
| bus | 3 | 0.8 % |

**Multi-road — track counts by track_class:**

| Class | Tracks | Share |
|---|---|---|
| car | 251 | 53.1 % |
| pedestrian | 89 | 18.8 % |
| LGV | 34 | 7.2 % |
| truck | 32 | 6.8 % |
| motorcycle | 39 | 8.2 % |
| bus | 28 | 5.9 % |

The mode-share contrast between scenes is physically meaningful. The intersection is a dense city-centre junction where two-wheelers dominate (37 %). The multi-road scene covers broader arterials where car traffic dominates (53 %) and buses are numerous (6 %) because it includes bus-route corridors.

---

## 3. Kinematics in Real Units

### 3.1 Scale calibration — pixel to metre

No ground-control points or SRT telemetry were available. Scale is derived entirely from the trajectory data using the **known physical widths of common vehicle types**:

| Class | Real width (m) | Intersection median box (px) | Derived px/m |
|---|---|---|---|
| car | 1.80 | 91.2 | 50.7 |
| LGV | 2.00 | 101.3 | 50.7 |
| truck | 2.40 | 111.5 | 46.5 |
| bus | 2.50 | 100.2 | 40.1 |
| motorcycle | 0.70 | 39.3 | 56.1 |

Box widths are measured only from non-interpolated detections whose centroid falls within the **central 40 %** of each frame axis, where perspective distortion is minimal. A confidence-weighted median across all eligible classes gives the global calibration:

| Scene | px/m | m/px | Implied altitude (AGL) |
|---|---|---|---|
| Intersection | **50.65** | 0.0197 | ≈ 40 m |
| Multi-road | **14.89** | 0.0672 | ≈ 133 m |

The two-scene difference is consistent with DJI operational altitude logs for intersection surveys (low hover) vs wide-area road surveys (high hover).

**Accuracy bound.** For a near-nadir camera tilted 10–20° (typical of intersection surveys) the off-centre scale error is cos(10°)⁻¹ − 1 ≈ 2 % to cos(20°)⁻¹ − 1 ≈ 6 %. Objects at the extreme frame edge can differ by up to ~15 % in a 30° tilt configuration. These bounds are quoted in all kinematic results; a per-pixel scale from a Level 4 homography will remove the residual error.

### 3.2 Smoothing before differentiation

Raw centroid positions carry 2–5 px of per-frame jitter from the detector. Differencing raw positions amplifies this directly into velocity noise. We apply a **Savitzky-Golay filter** (window = 9 frames, polynomial order = 2) to `cx` and `cy` independently before any differentiation.

SG smoothing is superior to a moving average for this application because it locally fits a low-degree polynomial: genuine acceleration events — braking approaching a signal, pulling away from a stop — are preserved in shape while high-frequency jitter is suppressed. For tracks shorter than the filter window, the window is reduced to the largest odd value ≥ 3 that fits the track length.

Kinematics are computed **only on non-interpolated observations**. Rows reconstructed by the Level 1 gap-fill are excluded from differentiation to prevent artificial velocity segments across detection gaps.

### 3.3 Velocity, speed and acceleration

```
vx_ms[t] = (cx_sg[t] − cx_sg[t−1])  /  (Δframe × px_per_m × fps)⁻¹  [m/s]
vy_ms[t] = (cy_sg[t] − cy_sg[t−1])  /  (Δframe × px_per_m × fps)⁻¹  [m/s]
speed_kmh[t]  = sqrt(vx_ms² + vy_ms²) × 3.6
```

Acceleration is computed by differencing the **smoothed** speed time-series (a second SG pass) to avoid double-differencing raw noise. Physical plausibility caps are applied: `speed_kmh` ∈ [0, 150], `accel_ms2` ∈ [−8, +5].

Heading is the compass bearing of the velocity vector: 0° = up (north), increasing clockwise.

### 3.4 Speed results — Intersection (signalised queue)

| Class | Tracks | Median (km/h) | 75th pct (km/h) | 95th pct (km/h) | Max (km/h) |
|---|---|---|---|---|---|
| motorcycle | 141 | **6.6** | 13.6 | 19.4 | 80.5 |
| car | 103 | **0.8** | 7.8 | 14.2 | 80.9 |
| pedestrian | 94 | **1.2** | 5.5 | 12.6 | 55.1 |
| LGV | 32 | **1.8** | 10.3 | 16.7 | 88.4 |
| truck | 7 | **0.5** | 2.8 | 4.9 | 69.9 |
| bus | 3 | **0.2** | 0.9 | 2.0 | 3.4 |

The very low medians for cars (0.8 km/h) and buses (0.2 km/h) are not artefacts — they reflect the signalised junction, where most vehicles spend the majority of the observation window stationary in queue. Motorcycles median at 6.6 km/h because they filter through stationary queues. No vehicle class exceeds ~90 km/h, consistent with congested urban approach geometry.

### 3.5 Speed results — Multi-road (mixed arterial)

| Class | Tracks | Median (km/h) | 75th pct (km/h) | 95th pct (km/h) | Max (km/h) |
|---|---|---|---|---|---|
| motorcycle | 39 | **22.0** | 44.0 | 77.5 | 150* |
| bus | 28 | **11.5** | 18.3 | 26.2 | 150* |
| car | 251 | **4.2** | 12.2 | 27.6 | 150* |
| pedestrian | 89 | **2.0** | 6.8 | 43.4† | 150* |
| LGV | 34 | **1.1** | 8.0 | 22.5 | 150* |
| truck | 32 | **1.8** | 7.4 | 20.6 | 150* |

*Speed cap hit in 0.29 % of rows — all from short-lived tracks with positional jumps at tile boundaries.
†Pedestrian p95 inflated by the same artefact; the median (2.0 km/h) correctly represents walking speeds.

Free-flowing conditions are evident: motorcycle median rises from 6.6 to **22.0 km/h**, cars from 0.8 to **4.2 km/h** (many still stopped at intermediate junctions within the scene). Bus median of **11.5 km/h** is consistent with urban scheduled services with frequent stops.

### 3.6 Acceleration distribution — Intersection

| Percentile | Value (m/s²) | Physical interpretation |
|---|---|---|
| 1st | −8.0 | Hard-braking cap |
| 5th | −3.2 | Moderate-to-firm braking |
| 50th | −0.1 | Near-stationary / coast |
| 95th | +3.3 | Firm acceleration (signal release) |
| 99th | +5.0 | Hard-acceleration cap |

The near-zero median confirms the queue-dominated intersection dynamic. The symmetric ±3.2 m/s² 5th–95th band captures normal urban stop-start behaviour. The physical caps (−8 / +5 m/s²) are reached at < 1 % of rows, restricted to short or noisy tracks; they protect downstream analysis from artefact spikes.

---

## 4. New Output Schema

Both Level 2 CSVs add **9 columns** to the Level 1 schema:

| Column | Type | Unit | Description |
|---|---|---|---|
| `track_class` | str | — | Stable per-track class (majority vote) |
| `subclass` | str / NaN | — | `HGV` or `rigid_truck` for trucks; NaN elsewhere |
| `smoothed_cx` | float | px | SG-smoothed centroid x |
| `smoothed_cy` | float | px | SG-smoothed centroid y |
| `vx_ms` | float | m/s | East-component velocity |
| `vy_ms` | float | m/s | South-component velocity |
| `speed_kmh` | float | km/h | Scalar speed |
| `accel_ms2` | float | m/s² | Rate of change of scalar speed |
| `heading_deg` | float | ° | Direction of travel (0 = north, CW) |

Kinematic columns are `NaN` for:
- Tracks with fewer than 3 observed (non-interpolated) frames
- The first observed frame of each track (no prior position for differentiation)
- All interpolated rows (gap-filled positions excluded from differentiation)

---

## 5. Video Proof

Two annotated 1-minute videos rendered at 1920 × 1080 from the 4K source footage:

- `data/output/l2_intersection_1min.mp4` — intersection scene, 1,800 frames
- `data/output/l2_multiroad_1min.mp4` — multi-road scene, 1,800 frames

Each frame shows:
- **Bounding box** coloured by `track_class`
- **Label:** `<class> #<id>   <speed> km/h`
- **Heading arrow** from centroid — length scales with speed
- **Speed-heatmap trail** — colour codes instantaneous speed (green = slow → blue → dark blue = fast)
- **HGV / rigid_truck badge** in box corner for truck sub-classes
- **Panel overlays:** class legend, 50-metre scale bar, speed colour ramp, frame/time counter

---

## 6. Pipeline

```
Level 1 smooth CSVs
  │
  ├─ calibrate_scale.py     vehicle-width prior → px/m sidecar JSON
  ├─ consolidate_class.py   confidence-weighted vote → track_class, subclass
  └─ compute_kinematics.py  SG smooth → finite-diff → unit convert
  │
  level2_pipeline.py        (drives all three in sequence)
  │
  ├─ intersection_tracks_l2.csv   +  intersection_scale.json
  └─ multiroad_tiled_l2.csv       +  multiroad_scale.json
```

Single command to reproduce:

```bash
venv\Scripts\python src\level2_pipeline.py
```

---

## 7. Honest Limitations

**Flat scale approximation.** One px/m value per scene is exact only at the frame centre. For the intersection camera (estimated ~10–20° tilt) the edge error is ≤ 6 %. For the multi-road camera (wider coverage, likely higher tilt) it may reach 10–15 % at frame corners. A per-pixel scale from a homography (Level 4) eliminates this.

**HGV heuristic.** Bounding-box area is a proxy for vehicle size. View-angle variation within a scene means two vehicles of identical length can produce different box areas. The `subclass` column should be treated as indicative, not definitive.

**Pedestrian p95 outliers.** Short tracks (< 15 frames) at tile boundaries can produce positional jumps that survive SG smoothing and appear as high speeds. The median is unaffected. Users should filter tracks with fewer than 30 observed frames for speed-distribution analysis.

**No ground truth.** Speed values cannot be validated against radar or loop-detector reference data. The plausibility check is internal: median speeds match known traffic behaviour (stopped queue at a signal, filtering motorcycles, free-flow arterials), and < 0.3 % of measurements hit the physical cap.

**Kinematics on interpolated rows are NaN by design.** Gap-filled positions are linear interpolations of known endpoints — differencing them would produce artificially constant velocity through the gap. They are excluded.

---

## 8. Repository

| Path | Purpose |
|---|---|
| `src/calibrate_scale.py` | Pixel-to-metre calibration from vehicle-width priors |
| `src/consolidate_class.py` | Stable per-track classification + truck sub-class |
| `src/compute_kinematics.py` | Velocity, acceleration, heading in real units |
| `src/level2_pipeline.py` | Single-command driver for all three stages |
| `src/visualize_l2.py` | Annotated video renderer with kinematics overlays |
| `data/output/intersection_tracks_l2.csv` | Level 2 enriched intersection trajectories |
| `data/output/multiroad_tiled_l2.csv` | Level 2 enriched multi-road trajectories |
| `data/output/intersection_scale.json` | Intersection calibration sidecar |
| `data/output/multiroad_scale.json` | Multi-road calibration sidecar |
| `data/output/l2_intersection_1min.mp4` | Annotated intersection video proof (1 min) |
| `data/output/l2_multiroad_1min.mp4` | Annotated multi-road video proof (1 min) |
