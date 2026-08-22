# AeroFlow — Level 3: Aggregate Insight

**Repository:** https://github.com/iharshlalakiya/AeroFlow
**Dataset:** Intersection_Merged — 3840 × 2160 @ 29.97 fps, 1,800 frames (60 s)
**Scene:** Dense signalised junction, Pune, India

---

## 1. What Level 3 produces

Level 3 aggregates over individual trajectories to derive traffic-engineering quantities:

| Metric | Method | Output |
|---|---|---|
| Turning movement counts (TMC) | Heading-angle arm classification | Per class, per 15-sec interval |
| Queue length | Slow-vehicle count near stop lines | Per arm, every 5 frames |
| Speed profile | Grid median + CDF by class | 48×27 spatial grid + percentiles |
| Flow-density relationship | Vehicles/km vs vehicles/h in 5-s windows | Scatter pairs |
| Modal split | Class share by turn type | Count table |
| O-D summary | Entry arm → exit arm matrix | 4×4 demand table |

All outputs are derived purely from the Level 2 trajectory CSV (`intersection_tracks_l2.csv`).
No manual annotations of lane lines, stop lines or signal phases were provided.

---

## 2. Movement Classification

### 2.1 Method

The video frame is 3840 × 2160 with the intersection occupying the full scene.
We classify each track's movement by:

1. **Entry heading** — circular mean of `heading_deg` over the track's first 15 observed frames.
2. **Exit heading** — circular mean over the last 15 observed frames.
3. **Entry arm** — opposite of entry heading direction (e.g., heading NE → came from SW → entry arm = W).
4. **Exit arm** — direction the exit heading points toward.
5. **Turn type** — 90° rotation between entry arm and exit arm:
   - Same arm → U-turn | Opposite → Through | ±90° → Left/Right

**Intersection centre** — estimated as the median centroid of all observations with
`speed_kmh < 4`, the slowest vehicles which cluster at stop lines and mid-box positions.
Estimated centre: **(2029, 1118) px** ≈ the geometric centre of the signalised box.

302 of 380 tracks were classifiable (78 skipped — fewer than 4 heading observations,
typically very short edge-of-frame appearances).

### 2.2 Turn-type distribution

| Turn type | Count | Share |
|---|---|---|
| Through | 187 | 61.9 % |
| Right | 46 | 15.2 % |
| Left | 36 | 11.9 % |
| U-turn | 33 | 10.9 % |

The U-turn share of 10.9 % is notably high. This is consistent with Indian urban
intersections where median U-turns at signalised junctions are a regular and
permitted movement, unlike most Western contexts where they are rare or prohibited.

### 2.3 Origin–Destination matrix (all classes, 60 s)

Entry arm rows, exit arm columns. Values = track counts.

| Entry \ Exit | E | N | S | W | **Total** |
|---|---|---|---|---|---|
| **E** | 19 | 12 | 11 | 92 | **134** |
| **N** | 13 | 1 | 25 | 15 | **54** |
| **S** | 15 | 11 | 2 | 4 | **32** |
| **W** | 59 | 8 | 4 | 11 | **82** |
| **Total** | 106 | 32 | 42 | 122 | **302** |

**Key findings:**
- The **E→W through** movement (92 counts) and **W→E through** (59 counts) are the
  dominant flows, indicating a heavy east-west main corridor with a significant
  imbalance (E→W nearly 1.6× the return direction).
- **N→S through** (25) and **S→N through** (11) show the cross-street is less loaded,
  typical of a T-junction or secondary approach.
- Diagonal entries (E→E = 19, W→W = 11) are U-turns, confirming the U-turn observation.

### 2.4 Turning movement counts by 15-second interval

| Entry | Movement | 0–15 s | 15–30 s | 30–45 s | 45–60 s |
|---|---|---|---|---|---|
| E | Through | 22 | 24 | 28 | 18 |
| E | Right | 4 | 3 | 3 | 2 |
| E | Left | 2 | 4 | 3 | 2 |
| E | U-turn | 6 | 4 | 5 | 4 |
| W | Through | 14 | 18 | 15 | 12 |
| N | Through | 5 | 7 | 6 | 7 |
| S | Through | 3 | 3 | 2 | 3 |

The 15-second intervals do not show signal-phase cycling clearly in 60 seconds of footage
(a full signal cycle at this junction is likely 90–120 s). The counts are broadly stable
across intervals, confirming all approaches receive some green time within the window.

### 2.5 Modal split by turn type

| Class | Through | Right | Left | U-turn | Total | Share |
|---|---|---|---|---|---|---|
| motorcycle | 118 | 7 | 4 | 2 | **131** | 43.4 % |
| pedestrian | 49 | 17 | 9 | 4 | **79** | 26.2 % |
| car | 14 | 18 | 20 | 21 | **73** | 24.2 % |
| LGV | 4 | 3 | 3 | 4 | **14** | 4.6 % |
| truck | 2 | 1 | 0 | 0 | **3** | 1.0 % |
| bus | 0 | 0 | 0 | 2 | **2** | 0.7 % |

**Motorcycles dominate through movements** (90 % of their class goes straight), consistent
with two-wheeler lane discipline at Indian signals. **Cars and LGVs account for the bulk
of turning movements** — they generate disproportionate turning demand relative to their
mode share (cars are 24 % of tracks but make 59 % of left turns).

---

## 3. Queue Length

### 3.1 Method

At each 5-frame step (≈ 0.17 s), we count vehicles satisfying all three conditions:

- `speed_kmh < 4` (effectively stopped or near-stopped)
- Distance to intersection centre < 600 px (≈ 11.9 m — within the signal influence zone)
- Distance to intersection centre > 150 px (excludes vehicles actually mid-box)

Arm assignment is by position relative to centre (N = above, S = below, E = right, W = left).
Queue length in metres assumes a per-vehicle headway of 5 m (vehicle + gap), consistent
with mixed Indian urban traffic.

### 3.2 Results

| Metric | Value |
|---|---|
| Maximum queue (total) | **18 vehicles** |
| Maximum queue length | **90 m** |
| Mean queue (total) | **8.6 vehicles** |
| Mean queue length | **42.9 m** |
| Peak arm — N | 8 vehicles |
| Peak arm — W | 7 vehicles |
| Peak arm — E | 5 vehicles |
| Peak arm — S | 4 vehicles |

The northern arm carries the longest queues, which is consistent with a secondary
cross-street approach that gets proportionally less green time. The 42.9 m mean
queue implies near-continuous stop-and-go conditions, which matches the very low
median speed observed for cars (0.8 km/h) and buses (0.2 km/h).

---

## 4. Speed Profile

### 4.1 Spatial speed heatmap

The frame is divided into a **48 × 27 grid** (80 × 80 px per cell = **1.6 × 1.6 m**).
Median `speed_kmh` is computed per cell using all non-interpolated observations and
smoothed with a Gaussian filter (σ = 1.2 cells).

The heatmap shows a clear spatial structure:
- **Intersection box** (centre ~300 × 300 px zone): highest density of slow observations,
  median speeds near zero during red phases.
- **Mid-block on main corridor** (E–W axis): speeds increase to 10–20 km/h as vehicles
  accelerate after clearing the stop line.
- **Entry/approach zones**: mixed — some tracks at speed (late arrivals) and some already
  queued (early starters).

Speed range on grid: **0 – 28.1 km/h** (95th percentile, used as colour scale max).

### 4.2 Speed percentiles by class

| Class | p10 | p25 | p50 | p75 | p85 | p95 | Mean |
|---|---|---|---|---|---|---|---|
| motorcycle | 0.3 | 1.4 | **6.6** | 11.1 | 13.6 | 19.4 | 7.7 |
| car | 0.2 | 0.3 | **0.8** | 7.4 | 9.6 | 14.2 | 3.9 |
| pedestrian | 0.2 | 0.5 | **1.2** | 3.2 | 5.4 | 12.6 | 3.1 |
| LGV | 0.2 | 0.4 | **1.8** | 9.1 | 11.5 | 16.7 | 5.4 |
| truck | 0.1 | 0.3 | **0.5** | 0.9 | 1.4 | 4.9 | 1.2 |
| bus | 0.1 | 0.1 | **0.2** | 0.6 | 1.0 | 2.0 | 0.5 |

*(All km/h, from non-interpolated observations)*

The **85th percentile speed (V85)** is the standard traffic-engineering design speed:
- Motorcycle V85 = **13.6 km/h** — fast two-wheelers at this junction
- Car V85 = **9.6 km/h** — heavily queue-limited
- All modes: V85 < 20 km/h — confirms this is a severely congested approach

The near-zero median for buses (0.2 km/h) indicates buses spend almost all observed
time stationary — consistent with a bus stop on the approach or long red phases.

---

## 5. Flow-Density Relationship

### 5.1 Method

Using 5-second windows (≈ 150 frames):
- **Density** — unique vehicles present in frame / estimated zone length (km)
- **Flow** — vehicles whose first observation falls in window / window duration (veh/h)
- **Mean speed** — mean `speed_kmh` in window

Zone length estimated as 4 × queue radius × m/px ≈ 47 m → 0.047 km.

### 5.2 Results

| Window | Vehicles | Density (veh/km) | Flow (veh/h) | Mean speed (km/h) |
|---|---|---|---|---|
| 0–5 s | 65 | 1,393 | — | 4.2 |
| 5–10 s | 70 | 1,500 | 1,800 | 3.8 |
| 10–15 s | 78 | 1,670 | 2,160 | 4.1 |
| … | … | … | … | … |
| 55–60 s | 136 | 2,912 | — | 4.9 |

The density rises steadily from ~1,393 veh/km at the start to ~2,912 veh/km at the end
of the minute — the junction is progressively more congested as the observation window proceeds.
Mean speeds remain near 4–5 km/h throughout, consistent with the **congested branch of the
fundamental diagram** (high density, low speed, reduced flow). This confirms the junction
operates beyond its capacity during this period.

---

## 6. Honest Limitations

**Movement classification uses heading, not lane position.** For mixed-traffic Indian
intersections where vehicles do not keep strict lane discipline, heading is a more reliable
indicator than position-based lane assignment. However, heading-based classification cannot
distinguish a sharp lane-change from a genuine turning movement on very short tracks.

**No signal phase data.** Without loop detectors or signal controller logs, we cannot
separate per-phase volumes. The 15-second interval counts are therefore mixed-phase and
should not be used directly as saturation flow rates.

**Queue length uses a fixed 5 m headway.** Mixed Indian traffic (motorcycles, autos,
cyclists) can queue at shorter headways (~2–3 m). The 5 m figure overestimates physical
queue length by up to 40 %; the vehicle-count metric is unaffected.

**U-turn classification may include sharp right-turns.** The heading-based classifier
bins (entry_arm == exit_arm) → U-turn, but a vehicle that enters from the north, executes
a very sharp right, and exits north looks identical. In practice the 33 U-turns include
both genuine U-turns and tight right hooks.

**Flow-density values are scene-level, not lane-level.** A proper fundamental diagram
requires single-lane observation or lane assignment, which needs a homography.

---

## 7. Pipeline

```
intersection_tracks_l2.csv
  │
  src/level3_aggregate.py
  │  ├─ Movement classification (heading angle → arm → turn type)
  │  ├─ TMC by interval and class
  │  ├─ Queue length time-series
  │  ├─ Speed heatmap (48×27 grid)
  │  ├─ Flow-density (5-s windows)
  │  ├─ Modal split
  │  └─ Speed percentile profiles
  │
  data/output/l3_*.csv + l3_summary.json + l3_speed_heatmap.json
  │
  src/visualize_l3.py
  │  ├─ Speed heatmap background (semi-transparent)
  │  ├─ Live queue bars at each approach arm
  │  ├─ TMC flow arrows (refreshed every 15 s)
  │  ├─ Per-object class + speed + movement labels
  │  └─ Cumulative counts panel
  │
  data/output/l3_intersection_1min.mp4
```

Reproduce:
```bash
venv\Scripts\python src\level3_aggregate.py
venv\Scripts\python src\visualize_l3.py
```

---

## 8. Repository additions

| Path | Purpose |
|---|---|
| `src/level3_aggregate.py` | All Level 3 computations |
| `src/visualize_l3.py` | Annotated video with aggregate overlays |
| `data/output/l3_movements.csv` | Per-track movement classification |
| `data/output/l3_tmc.csv` | TMC by 15-s interval |
| `data/output/l3_tmc_total.csv` | TMC totals by class |
| `data/output/l3_queue.csv` | Queue length time-series |
| `data/output/l3_speed_heatmap.json` | Spatial speed grid |
| `data/output/l3_flow_density.csv` | Flow-density pairs |
| `data/output/l3_modal_split.csv` | Modal split by movement |
| `data/output/l3_speed_profile.csv` | Speed percentiles by class |
| `data/output/l3_summary.json` | All scalar metrics |
| `data/output/l3_intersection_1min.mp4` | 1-minute annotated video proof |
| `docs/level3_writeup.md` | This document |
