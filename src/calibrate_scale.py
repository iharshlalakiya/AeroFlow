"""
Scale calibration: derive a pixel-to-metre conversion factor from the data itself.

Method
------
Real-world vehicles have known typical widths:
  car        ≈ 1.80 m  (compact–mid-size sedan)
  LGV        ≈ 2.00 m  (light goods / van)
  truck      ≈ 2.40 m  (rigid truck)
  bus        ≈ 2.50 m  (standard urban bus)
  motorcycle ≈ 0.70 m  (handlebars to handlebars)

We compare these against the *observed median bounding-box widths* (in px) for
non-interpolated detections whose box centre falls within the central 40 % of
the frame (less perspective distortion at the edges).

px_per_m = median_px_width / real_world_m_width

A weighted median across all eligible classes gives one global scale factor.
This is a flat-plane approximation; objects near the frame edges will have a
true scale up to ~20 % different depending on camera tilt, but for aerial nadir
or near-nadir drone footage the error is small at mid-frame.

Usage
-----
    python src/calibrate_scale.py \\
        --tracks data/output/intersection_tracks_smooth.csv \\
        --out    data/output/intersection_scale.json
"""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

# Known typical widths in metres (conservative mid-range values)
REAL_WIDTH_M = {
    "car":        1.80,
    "LGV":        2.00,
    "truck":      2.40,
    "bus":        2.50,
    "motorcycle": 0.70,
    # pedestrians excluded — shoulder width is too variable and posture changes box width
}

# Weight by confidence in the physical prior (trucks/buses more distinctive)
CLASS_WEIGHT = {
    "car":        1.0,
    "LGV":        0.9,
    "truck":      1.0,
    "bus":        1.0,
    "motorcycle": 0.8,
}


def calibrate(df: pd.DataFrame, central_fraction: float = 0.40) -> dict:
    """
    Returns a dict with px_per_m, contributing class estimates, and metadata.
    """
    # Frame dimensions from coordinate ranges
    frame_w = df["cx"].max()
    frame_h = df["cy"].max()

    # Central zone filter
    cx_lo = frame_w * (0.5 - central_fraction / 2)
    cx_hi = frame_w * (0.5 + central_fraction / 2)
    cy_lo = frame_h * (0.5 - central_fraction / 2)
    cy_hi = frame_h * (0.5 + central_fraction / 2)

    df = df[~df["interpolated"]].copy()
    df["box_w"] = df["x2"] - df["x1"]
    # Filter to central zone and sane box widths (at least 5 px)
    central = df[
        (df["cx"] >= cx_lo) & (df["cx"] <= cx_hi) &
        (df["cy"] >= cy_lo) & (df["cy"] <= cy_hi) &
        (df["box_w"] >= 5)
    ]

    class_col = "class" if "class" in central.columns else "taxonomy_class"

    estimates = {}
    for cls, real_m in REAL_WIDTH_M.items():
        sub = central[central[class_col] == cls]["box_w"]
        if len(sub) < 20:
            continue
        median_px = float(np.median(sub))
        px_per_m = median_px / real_m
        estimates[cls] = {
            "px_per_m":    round(px_per_m, 4),
            "median_px_w": round(median_px, 2),
            "real_m_w":    real_m,
            "n_samples":   int(len(sub)),
            "weight":      CLASS_WEIGHT[cls],
        }

    if not estimates:
        print("WARNING: not enough central detections for any class — "
              "falling back to full-frame detections.", file=sys.stderr)
        df2 = df.copy()
        for cls, real_m in REAL_WIDTH_M.items():
            sub = df2[df2[class_col] == cls]["box_w"]
            if len(sub) < 5:
                continue
            median_px = float(np.median(sub))
            estimates[cls] = {
                "px_per_m":    round(median_px / real_m, 4),
                "median_px_w": round(median_px, 2),
                "real_m_w":    real_m,
                "n_samples":   int(len(sub)),
                "weight":      CLASS_WEIGHT[cls],
            }

    # Weighted median of per-class px_per_m values
    vals = np.array([v["px_per_m"] for v in estimates.values()])
    wts  = np.array([v["weight"]   for v in estimates.values()])
    # Weighted median: sort by value, find where cumulative weight crosses 0.5
    order = np.argsort(vals)
    vals_s, wts_s = vals[order], wts[order]
    cum_w = np.cumsum(wts_s)
    half_w = cum_w[-1] / 2.0
    idx = int(np.searchsorted(cum_w, half_w))
    px_per_m = float(vals_s[idx])

    return {
        "px_per_m":          round(px_per_m, 4),
        "m_per_px":          round(1.0 / px_per_m, 6),
        "per_class":         estimates,
        "central_fraction":  central_fraction,
        "frame_w_px":        round(float(frame_w), 1),
        "frame_h_px":        round(float(frame_h), 1),
        "method":            "vehicle_width_prior",
    }


def main():
    p = argparse.ArgumentParser(description="Estimate px/m scale from vehicle box widths.")
    p.add_argument("--tracks", required=True, help="Smooth trajectory CSV (Level 1 output)")
    p.add_argument("--out",    required=True, help="Output JSON path for scale sidecar")
    p.add_argument("--central-fraction", type=float, default=0.40,
                   help="Fraction of frame centre to use (default 0.40)")
    args = p.parse_args()

    df = pd.read_csv(args.tracks)
    result = calibrate(df, central_fraction=args.central_fraction)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scale calibration complete.")
    print(f"  px_per_m : {result['px_per_m']:.4f}")
    print(f"  m_per_px : {result['m_per_px']:.6f}")
    print(f"  Per-class contributions:")
    for cls, v in result["per_class"].items():
        print(f"    {cls:12s}  median_box={v['median_px_w']:.1f}px  "
              f"real={v['real_m_w']:.2f}m  "
              f"→ {v['px_per_m']:.2f} px/m  (n={v['n_samples']})")
    print(f"  Written to: {args.out}")


if __name__ == "__main__":
    main()
