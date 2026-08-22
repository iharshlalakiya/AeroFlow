"""
Kinematics engine — per-object velocity, acceleration and heading in real units.

Pipeline per track
------------------
1. Sort by frame.
2. Apply Savitzky-Golay smoothing (window=9, poly=2) to cx and cy separately.
   This suppresses detector jitter before differentiation.  Tracks shorter
   than the window are smoothed with a proportionally smaller odd window.
3. Finite-difference the smoothed positions → vx_px, vy_px  (px / frame).
4. Convert to m/s using the calibrated px_per_m and fps.
5. Compute scalar speed in km/h.
6. Finite-difference speed → accel_ms2.  Apply SG smoothing again.
7. Compute heading_deg (0 = up / north, clockwise, range 0–360).
8. Clip to physical limits and mark NaN for very short tracks (< 3 frames).

Scale approximation note
------------------------
We use a flat (single) px/m factor for the whole frame, which is exact only
at the frame centre.  Near the edges the true scale can differ by up to ~15 %
for a 45° camera tilt.  This is acceptable for Level 2; a proper perspective
homography (Level 4) will remove the remaining error.

Usage
-----
    python src/compute_kinematics.py \\
        --tracks data/output/intersection_tracks_classed.csv \\
        --scale  data/output/intersection_scale.json \\
        --fps    29.97 \\
        --out    data/output/intersection_tracks_l2.csv
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Physical plausibility caps
MAX_SPEED_KMH  =  150.0   # urban drone footage; nothing should exceed motorway
MIN_SPEED_KMH  =    0.0
MAX_ACCEL_MS2  =    5.0   # hard acceleration (≈ 0–100 in 5.5 s)
MIN_ACCEL_MS2  =   -8.0   # emergency braking

MIN_SG_WINDOW  = 3         # smallest odd SG window we allow
DEFAULT_SG_WIN = 9
SG_POLY        = 2
MIN_FRAMES_FOR_KIN = 3     # tracks shorter than this get NaN kinematics


def _odd(n: int) -> int:
    """Return n if odd, else n-1 (must be ≥ 3)."""
    n = max(n, MIN_SG_WINDOW)
    return n if n % 2 == 1 else n - 1


def _sg_smooth(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Apply Savitzky-Golay; fall back to no-op for very short arrays."""
    w = _odd(min(window, len(arr)))
    if w < poly + 2 or len(arr) < w:
        return arr.astype(float)
    return savgol_filter(arr.astype(float), w, poly)


def compute_kinematics(df: pd.DataFrame, px_per_m: float, fps: float) -> pd.DataFrame:
    """
    Returns df with new kinematic columns appended.
    Operates only on observed (non-interpolated) detections for differentiation;
    kinematics values are then merged back to the full dataframe.
    """
    df = df.sort_values(["track_id", "frame"]).copy()

    # Columns to populate
    for col in ["smoothed_cx", "smoothed_cy",
                "vx_ms", "vy_ms", "speed_kmh",
                "accel_ms2", "heading_deg"]:
        df[col] = np.nan

    m_per_px = 1.0 / px_per_m

    results = []  # list of partial DataFrames

    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame").copy()

        if len(grp) < MIN_FRAMES_FOR_KIN:
            results.append(grp)
            continue

        # Use only observed rows for differentiation to avoid SG distortion
        # across interpolated gaps; we'll fill back via frame index later.
        obs_mask = ~grp["interpolated"].astype(bool)

        if obs_mask.sum() < MIN_FRAMES_FOR_KIN:
            results.append(grp)
            continue

        # Smooth full track (including interpolated) for display positions
        grp["smoothed_cx"] = _sg_smooth(grp["cx"].values, DEFAULT_SG_WIN, SG_POLY)
        grp["smoothed_cy"] = _sg_smooth(grp["cy"].values, DEFAULT_SG_WIN, SG_POLY)

        # Work on observed rows for velocity
        obs = grp[obs_mask].copy()
        obs["s_cx"] = _sg_smooth(obs["cx"].values, DEFAULT_SG_WIN, SG_POLY)
        obs["s_cy"] = _sg_smooth(obs["cy"].values, DEFAULT_SG_WIN, SG_POLY)

        frames = obs["frame"].values.astype(float)
        s_cx   = obs["s_cx"].values
        s_cy   = obs["s_cy"].values

        # px / frame  →  m / s
        dt = np.diff(frames)                       # frame gaps (usually 1, >1 after gaps)
        dvx_px = np.diff(s_cx) / dt               # px/frame (corrected for frame gaps)
        dvy_px = np.diff(s_cy) / dt

        vx_ms = dvx_px * m_per_px * fps
        vy_ms = dvy_px * m_per_px * fps
        speed = np.sqrt(vx_ms**2 + vy_ms**2)

        # Smooth speed before differentiating for acceleration
        speed_smooth = _sg_smooth(speed, DEFAULT_SG_WIN, SG_POLY)
        dt2 = 0.5 * (dt[:-1] + dt[1:]) / fps     # centred frame intervals in seconds
        accel = np.diff(speed_smooth) / dt2 if len(speed_smooth) > 1 else np.array([])
        accel = _sg_smooth(accel, DEFAULT_SG_WIN, SG_POLY) if len(accel) > 0 else accel

        # Heading (atan2 convention: 0=up/north, clockwise)
        heading = (np.degrees(np.arctan2(dvx_px, -dvy_px)) % 360)

        # Convert to km/h and clip
        speed_kmh = np.clip(speed * 3.6, MIN_SPEED_KMH, MAX_SPEED_KMH)
        accel_clip = np.clip(accel, MIN_ACCEL_MS2, MAX_ACCEL_MS2) if len(accel) else accel

        # Assign back to obs rows.  Velocity is computed at the midpoint between
        # consecutive frames; we assign it to the *later* frame (forward difference).
        obs_idx = obs.index.tolist()

        for i, idx in enumerate(obs_idx[1:]):
            grp.at[idx, "vx_ms"]       = round(float(vx_ms[i]),        4)
            grp.at[idx, "vy_ms"]       = round(float(vy_ms[i]),        4)
            grp.at[idx, "speed_kmh"]   = round(float(speed_kmh[i]),    2)
            grp.at[idx, "heading_deg"] = round(float(heading[i]),       2)

        for i, idx in enumerate(obs_idx[2:]):
            if i < len(accel_clip):
                grp.at[idx, "accel_ms2"] = round(float(accel_clip[i]), 4)

        results.append(grp)

    return pd.concat(results, ignore_index=True).sort_values(["track_id", "frame"])


def main():
    p = argparse.ArgumentParser(description="Compute per-object kinematics in real units.")
    p.add_argument("--tracks", required=True, help="Classed trajectory CSV")
    p.add_argument("--scale",  required=True, help="Scale JSON (from calibrate_scale.py)")
    p.add_argument("--fps",    type=float, default=29.97, help="Video frame rate")
    p.add_argument("--out",    required=True, help="Output L2 CSV path")
    args = p.parse_args()

    with open(args.scale) as f:
        scale = json.load(f)
    px_per_m = scale["px_per_m"]

    print(f"Using px_per_m = {px_per_m:.4f}  "
          f"({scale['m_per_px']:.4f} m/px)  [method: {scale['method']}]")

    df = pd.read_csv(args.tracks)
    df = compute_kinematics(df, px_per_m=px_per_m, fps=args.fps)
    df.to_csv(args.out, index=False, float_format="%.4f")

    # Summary statistics
    obs = df[~df["interpolated"].astype(bool) & df["speed_kmh"].notna()]
    print("\nSpeed summary (km/h) — observed detections only:")
    print(obs.groupby("track_class")["speed_kmh"].describe(percentiles=[.25, .5, .75, .95]).round(2).to_string())
    print("\nAcceleration summary (m/s²) — observed detections only:")
    print(obs.groupby("track_class")["accel_ms2"].describe(percentiles=[.05, .50, .95]).round(3).to_string())
    print(f"\nWritten to: {args.out}")


if __name__ == "__main__":
    main()
