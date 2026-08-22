"""
Level 3 Aggregate Insight — Intersection Analysis.

Computes the following from intersection_tracks_l2.csv (1-minute clip):

  1. Movement classification   — approach arm + turn type per track (N/S/E/W, through/left/right)
  2. Turning movement counts   — TMC by class and 15-second interval
  3. Queue length              — per-arm vehicle count and metres at each 5-frame step
  4. Speed heatmap             — median speed_kmh on a spatial grid (for visualiser)
  5. Flow-density relationship — per 5-second windows
  6. Modal split               — class share by movement type
  7. Speed profile             — CDF and percentiles by class and movement
  8. Occupancy / density       — vehicles per 100 m of approach

All outputs are written to data/output/l3_*.csv|json.

Usage:
    python src/level3_aggregate.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

# ── constants ────────────────────────────────────────────────────────────────
FPS        = 29.97
PX_PER_M   = 50.65          # intersection scale (from level2 calibration)
M_PER_PX   = 1.0 / PX_PER_M
W, H       = 3840, 2160     # frame dimensions
INTERVAL_S = 15             # TMC aggregation interval (seconds)
INTERVAL_F = int(INTERVAL_S * FPS)
SLOW_KMH   = 4.0            # threshold for "queued" vehicle
GRID_COLS  = 48             # speed-heatmap grid columns
GRID_ROWS  = 27             # speed-heatmap grid rows  (48×27 = 16:9)

IN_CSV  = Path("data/output/intersection_tracks_l2.csv")
OUT_DIR = Path("data/output")

ARMS     = ["N", "E", "S", "W"]
ARM_DEG  = {       # heading IN this direction means you are GOING toward this arm
    "N": (315, 45),   # 315-360 or 0-45
    "E": (45, 135),
    "S": (135, 225),
    "W": (225, 315),
}

TURN_MAP = {        # (entry→exit delta in 90° steps, CCW positive) → turn type
    0: "u_turn",
    1: "left",      # 90° left
    2: "through",
    3: "right",     # 90° right = 270° CCW
}


# ── helpers ──────────────────────────────────────────────────────────────────

def heading_to_arm(hdg: float) -> str:
    """Map heading_deg (0=up, CW) to the arm the vehicle is HEADING TOWARD."""
    hdg = float(hdg) % 360
    lo, hi = ARM_DEG["N"]
    if hdg >= lo or hdg < hi:
        return "N"
    for arm in ["E", "S", "W"]:
        lo, hi = ARM_DEG[arm]
        if lo <= hdg < hi:
            return arm
    return "N"


def entry_arm_from_heading(hdg: float) -> str:
    """Entry arm = opposite of the direction the vehicle is heading toward."""
    opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
    return opp[heading_to_arm(hdg)]


def classify_turn(entry: str, exit_: str) -> str:
    ei = ARMS.index(entry)
    xi = ARMS.index(exit_)
    delta = (xi - ei) % 4
    return TURN_MAP[delta]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading L2 CSV …")
    df = pd.read_csv(IN_CSV, low_memory=False)
    df["interpolated"] = df["interpolated"].astype(bool)
    df["speed_kmh"]    = pd.to_numeric(df["speed_kmh"],    errors="coerce")
    df["heading_deg"]  = pd.to_numeric(df["heading_deg"],  errors="coerce")
    df["accel_ms2"]    = pd.to_numeric(df["accel_ms2"],    errors="coerce")
    df["vx_ms"]        = pd.to_numeric(df["vx_ms"],        errors="coerce")
    df["vy_ms"]        = pd.to_numeric(df["vy_ms"],        errors="coerce")

    obs = df[~df["interpolated"]].copy()

    # ── 1. Movement classification ─────────────────────────────────────────
    print("[1/7] Movement classification …")

    FIRST_N = 15   # frames used to determine entry heading
    LAST_N  = 15

    records = []
    for tid, grp in obs.groupby("track_id"):
        grp = grp.sort_values("frame")
        hdg_vals = grp["heading_deg"].dropna()
        if len(hdg_vals) < 4:
            continue

        # Entry heading = circular mean of first FIRST_N heading values
        first_hdg = hdg_vals.iloc[:FIRST_N]
        last_hdg  = hdg_vals.iloc[-LAST_N:]

        def circ_mean(angles):
            rad = np.deg2rad(angles)
            return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)

        entry_hdg = circ_mean(first_hdg.values)
        exit_hdg  = circ_mean(last_hdg.values)

        entry_arm = entry_arm_from_heading(entry_hdg)
        exit_arm  = heading_to_arm(exit_hdg)
        turn_type = classify_turn(entry_arm, exit_arm)

        # median speed and first/last frame
        med_spd = float(grp["speed_kmh"].median())
        records.append({
            "track_id":  tid,
            "track_class": grp["track_class"].iloc[0],
            "entry_arm": entry_arm,
            "exit_arm":  exit_arm,
            "turn_type": turn_type,
            "entry_hdg": round(entry_hdg, 1),
            "exit_hdg":  round(exit_hdg, 1),
            "first_frame": int(grp["frame"].min()),
            "last_frame":  int(grp["frame"].max()),
            "n_obs_frames": len(grp),
            "median_speed_kmh": round(med_spd, 2),
            "max_speed_kmh": round(float(grp["speed_kmh"].max()), 2) if grp["speed_kmh"].notna().any() else None,
        })

    mv = pd.DataFrame(records)
    mv.to_csv(OUT_DIR / "l3_movements.csv", index=False)
    print(f"  {len(mv)} classified tracks  (skipped {df.track_id.nunique() - len(mv)} short/no-heading)")

    print("  Movement matrix:")
    mat = pd.crosstab(mv.entry_arm, mv.exit_arm, margins=True)
    print(mat.to_string())

    print("  Turn-type distribution:")
    print(mv.turn_type.value_counts().to_string())

    # ── 2. Turning movement counts by interval ─────────────────────────────
    print("\n[2/7] Turning movement counts (TMC) …")
    max_frame = int(df.frame.max())
    intervals = range(0, max_frame, INTERVAL_F)

    tmc_rows = []
    for i_start in intervals:
        i_end  = i_start + INTERVAL_F
        t_start = i_start / FPS
        t_end   = i_end   / FPS
        # Tracks whose first frame falls in this interval
        active = mv[(mv.first_frame >= i_start) & (mv.first_frame < i_end)]
        for _, row in active.iterrows():
            tmc_rows.append({
                "interval_start_s": round(t_start, 2),
                "interval_end_s":   round(t_end,   2),
                "track_class":  row.track_class,
                "entry_arm":    row.entry_arm,
                "exit_arm":     row.exit_arm,
                "turn_type":    row.turn_type,
                "count":        1,
            })

    tmc = pd.DataFrame(tmc_rows) if tmc_rows else pd.DataFrame(
        columns=["interval_start_s","interval_end_s","track_class","entry_arm","exit_arm","turn_type","count"])
    tmc.to_csv(OUT_DIR / "l3_tmc.csv", index=False)

    # Aggregate TMC table (total counts)
    tmc_agg = mv.groupby(["entry_arm", "exit_arm", "turn_type", "track_class"]).size().reset_index(name="count")
    tmc_agg.to_csv(OUT_DIR / "l3_tmc_total.csv", index=False)

    print("  TMC total (all classes):")
    total_mv = mv.groupby(["entry_arm", "turn_type"]).size().reset_index(name="count")
    print(total_mv.to_string(index=False))

    # ── 3. Queue length ────────────────────────────────────────────────────
    print("\n[3/7] Queue lengths …")

    # Intersection centre = centroid of positions where speed < SLOW_KMH
    slow = obs[obs["speed_kmh"] < SLOW_KMH]
    cx_centre = float(slow["cx"].median()) if len(slow) > 10 else W / 2
    cy_centre = float(slow["cy"].median()) if len(slow) > 10 else H / 2
    print(f"  Intersection centre estimate: ({cx_centre:.0f}, {cy_centre:.0f}) px")

    # Stop-line distance: vehicles at queue are within QUEUE_RADIUS of centre
    # but NOT in the box (stopped at stop line, not mid-intersection)
    QUEUE_RADIUS_PX  = 600    # ~12 m each side
    CENTRE_EXCL_PX   = 150    # ignore vehicles actually in the box

    queue_rows = []
    step = 5   # every 5 frames
    for fid in range(0, max_frame + 1, step):
        frame_data = obs[obs["frame"] == fid]
        if frame_data.empty:
            continue
        frame_data = frame_data.copy()
        frame_data["dist_to_centre"] = np.sqrt(
            (frame_data["cx"] - cx_centre)**2 +
            (frame_data["cy"] - cy_centre)**2
        )
        # Queued: slow + within radius + not in box centre
        queued = frame_data[
            (frame_data["speed_kmh"] < SLOW_KMH) &
            (frame_data["dist_to_centre"] < QUEUE_RADIUS_PX) &
            (frame_data["dist_to_centre"] > CENTRE_EXCL_PX)
        ]

        total_queued = len(queued)
        # Per-arm queue (based on which side of centre the vehicle is on)
        arm_counts = {}
        for arm in ARMS:
            if arm == "N":
                arm_q = queued[queued["cy"] < cy_centre - CENTRE_EXCL_PX]
            elif arm == "S":
                arm_q = queued[queued["cy"] > cy_centre + CENTRE_EXCL_PX]
            elif arm == "W":
                arm_q = queued[queued["cx"] < cx_centre - CENTRE_EXCL_PX]
            else:  # E
                arm_q = queued[queued["cx"] > cx_centre + CENTRE_EXCL_PX]
            arm_counts[arm] = len(arm_q)

        # Queue length in metres: assume vehicles spaced ~5 m (vehicle + gap)
        SPACING_M = 5.0
        queue_rows.append({
            "frame": fid,
            "time_s": round(fid / FPS, 2),
            "queued_total": total_queued,
            "queue_length_m": round(total_queued * SPACING_M, 1),
            **{f"queue_{arm}": arm_counts[arm] for arm in ARMS},
            **{f"queue_{arm}_m": round(arm_counts[arm] * SPACING_M, 1) for arm in ARMS},
        })

    queue_df = pd.DataFrame(queue_rows)
    queue_df.to_csv(OUT_DIR / "l3_queue.csv", index=False)
    print(f"  Max total queue: {queue_df.queued_total.max()} vehicles  "
          f"({queue_df.queue_length_m.max():.0f} m)")
    print(f"  Mean queue: {queue_df.queued_total.mean():.1f} vehicles  "
          f"({queue_df.queue_length_m.mean():.1f} m)")

    # ── 4. Speed heatmap grid ─────────────────────────────────────────────
    print("\n[4/7] Speed heatmap …")
    cell_w = W / GRID_COLS
    cell_h = H / GRID_ROWS
    speed_grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    count_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

    speed_obs = obs[obs["speed_kmh"].notna()]
    for _, row in speed_obs.iterrows():
        c = min(int(row["cx"] / cell_w), GRID_COLS - 1)
        r = min(int(row["cy"] / cell_h), GRID_ROWS - 1)
        if np.isnan(speed_grid[r, c]):
            speed_grid[r, c] = row["speed_kmh"]
        else:
            speed_grid[r, c] = (speed_grid[r, c] + row["speed_kmh"]) / 2
        count_grid[r, c] += 1

    # Replace NaN with 0 for smoothing, then Gaussian blur
    filled = np.where(np.isnan(speed_grid), 0, speed_grid)
    smoothed = gaussian_filter(filled, sigma=1.2)
    mask = count_grid > 0

    # Save as JSON for visualiser
    speed_heatmap = {
        "grid_cols": GRID_COLS,
        "grid_rows": GRID_ROWS,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "values": smoothed.tolist(),
        "counts": count_grid.tolist(),
        "vmin": 0,
        "vmax": float(np.nanpercentile(speed_grid[mask], 95)) if mask.any() else 30,
    }
    with open(OUT_DIR / "l3_speed_heatmap.json", "w") as f:
        json.dump(speed_heatmap, f)

    print(f"  Grid: {GRID_COLS}×{GRID_ROWS}  "
          f"({cell_w:.0f}×{cell_h:.0f} px per cell = "
          f"{cell_w*M_PER_PX:.1f}×{cell_h*M_PER_PX:.1f} m)")
    print(f"  Speed range on grid: 0 – {speed_heatmap['vmax']:.1f} km/h")

    # ── 5. Flow-density relationship ──────────────────────────────────────
    print("\n[5/7] Flow-density …")
    # Reference zone: entire frame. Density = vehicles in zone / zone area (km)
    # Flow = vehicles newly appearing per second
    ZONE_LEN_M = (QUEUE_RADIUS_PX * 4) * M_PER_PX   # approx intersection length

    fd_rows = []
    WINDOW_F = int(5 * FPS)   # 5-second windows
    for w_start in range(0, max_frame, WINDOW_F):
        w_end = w_start + WINDOW_F
        w_frames = obs[(obs["frame"] >= w_start) & (obs["frame"] < w_end)]
        if w_frames.empty:
            continue

        # Density: unique vehicles present / zone length (veh/km)
        n_vehs = w_frames["track_id"].nunique()
        density = n_vehs / max(ZONE_LEN_M / 1000, 0.001)

        # Flow: vehicles whose first frame is in this window / window duration (veh/h)
        entering = mv[(mv["first_frame"] >= w_start) & (mv["first_frame"] < w_end)]
        flow = len(entering) / (WINDOW_F / FPS / 3600)

        # Mean speed in window
        mean_spd = float(w_frames["speed_kmh"].mean())

        fd_rows.append({
            "window_start_s": round(w_start / FPS, 2),
            "n_vehicles":     n_vehs,
            "density_veh_km": round(density, 2),
            "flow_veh_h":     round(flow, 1),
            "mean_speed_kmh": round(mean_spd, 2),
        })

    fd_df = pd.DataFrame(fd_rows)
    fd_df.to_csv(OUT_DIR / "l3_flow_density.csv", index=False)
    print(f"  Windows: {len(fd_df)}  |  "
          f"flow range {fd_df.flow_veh_h.min():.0f}–{fd_df.flow_veh_h.max():.0f} veh/h  |  "
          f"density {fd_df.density_veh_km.min():.1f}–{fd_df.density_veh_km.max():.1f} veh/km")

    # ── 6. Modal split by movement ────────────────────────────────────────
    print("\n[6/7] Modal split …")
    modal = mv.groupby(["turn_type", "track_class"]).size().reset_index(name="count")
    modal_pct = mv.groupby(["track_class"]).size().reset_index(name="total")
    modal.to_csv(OUT_DIR / "l3_modal_split.csv", index=False)

    total_by_class = mv.track_class.value_counts()
    total_n = len(mv)
    print("  Modal split (overall):")
    for cls, cnt in total_by_class.items():
        print(f"    {cls:12s}  {cnt:3d}  ({cnt/total_n*100:.1f}%)")

    print("  Turn type distribution:")
    for t, cnt in mv.turn_type.value_counts().items():
        print(f"    {t:10s}  {cnt:3d}  ({cnt/total_n*100:.1f}%)")

    # ── 7. Speed profile (CDF by class) ──────────────────────────────────
    print("\n[7/7] Speed profiles …")
    pct = [10, 25, 50, 75, 85, 95]
    spd_profile_rows = []
    for cls, grp in obs.groupby("track_class"):
        vals = grp["speed_kmh"].dropna()
        if len(vals) < 10:
            continue
        row = {"track_class": cls, "n": len(vals)}
        for p in pct:
            row[f"p{p}_kmh"] = round(float(np.percentile(vals, p)), 2)
        row["mean_kmh"] = round(float(vals.mean()), 2)
        spd_profile_rows.append(row)
    spd_profile = pd.DataFrame(spd_profile_rows)
    spd_profile.to_csv(OUT_DIR / "l3_speed_profile.csv", index=False)
    print(spd_profile.to_string(index=False))

    # ── Summary JSON (for write-up and visualiser) ────────────────────────
    print("\nGenerating summary …")
    summary = {
        "fps": FPS,
        "px_per_m": PX_PER_M,
        "duration_s": round(max_frame / FPS, 1),
        "total_tracks": int(df.track_id.nunique()),
        "classified_tracks": len(mv),
        "intersection_centre_px": [round(cx_centre, 1), round(cy_centre, 1)],
        "interval_s": INTERVAL_S,
        "turn_type_counts": mv.turn_type.value_counts().to_dict(),
        "entry_arm_counts": mv.entry_arm.value_counts().to_dict(),
        "class_counts": mv.track_class.value_counts().to_dict(),
        "queue_stats": {
            "max_total_veh": int(queue_df.queued_total.max()),
            "max_length_m": float(queue_df.queue_length_m.max()),
            "mean_total_veh": round(float(queue_df.queued_total.mean()), 1),
            "mean_length_m": round(float(queue_df.queue_length_m.mean()), 1),
            "per_arm_max": {
                arm: int(queue_df[f"queue_{arm}"].max()) for arm in ARMS
            },
        },
        "flow_density": {
            "max_flow_veh_h": float(fd_df.flow_veh_h.max()),
            "mean_flow_veh_h": round(float(fd_df.flow_veh_h.mean()), 1),
            "max_density_veh_km": float(fd_df.density_veh_km.max()),
            "mean_density_veh_km": round(float(fd_df.density_veh_km.mean()), 1),
        },
        "speed_by_class": {
            row["track_class"]: {
                "p50_kmh": row["p50_kmh"],
                "p85_kmh": row["p85_kmh"],
                "p95_kmh": row["p95_kmh"],
                "mean_kmh": row["mean_kmh"],
            }
            for _, row in spd_profile.iterrows()
        },
        "tmc_total": {
            f"{k[0]}->{k[1]}": int(v)
            for k, v in tmc_agg.groupby(["entry_arm","turn_type"])["count"].sum().items()
        },
    }
    with open(OUT_DIR / "l3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nLevel 3 aggregate complete. Outputs in {OUT_DIR}/")
    print("  l3_movements.csv       — per-track movement classification")
    print("  l3_tmc.csv             — TMC by interval")
    print("  l3_tmc_total.csv       — TMC totals by class")
    print("  l3_queue.csv           — queue length time-series")
    print("  l3_speed_heatmap.json  — spatial speed grid")
    print("  l3_flow_density.csv    — flow-density pairs")
    print("  l3_modal_split.csv     — modal split by movement")
    print("  l3_speed_profile.csv   — speed percentiles by class")
    print("  l3_summary.json        — all scalars for write-up")


if __name__ == "__main__":
    main()
