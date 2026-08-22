"""
Level 2 pipeline driver.

Runs the three Level 2 stages in order for one or both input datasets:

  1. calibrate_scale     — derive px/m from vehicle-width priors
  2. consolidate_class   — stable per-track label + truck sub-class
  3. compute_kinematics  — velocity, acceleration, heading in real units

Usage (single dataset)
----------------------
    python src/level2_pipeline.py \\
        --intersection \\
        --multiroad

Usage (custom paths)
--------------------
    python src/level2_pipeline.py \\
        --tracks data/output/intersection_tracks_smooth.csv \\
        --out    data/output/intersection_tracks_l2.csv \\
        --fps    29.97

Outputs (defaults)
------------------
  data/output/intersection_scale.json
  data/output/intersection_tracks_l2.csv
  data/output/multiroad_scale.json
  data/output/multiroad_tiled_l2.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src/ to path so we can import siblings directly
sys.path.insert(0, str(Path(__file__).parent))

from calibrate_scale   import calibrate
from consolidate_class import consolidate
from compute_kinematics import compute_kinematics

FPS = 29.97

# ---------------------------------------------------------------------------
# Default dataset definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "intersection": {
        "tracks": "data/output/intersection_tracks_smooth.csv",
        "scale":  "data/output/intersection_scale.json",
        "out":    "data/output/intersection_tracks_l2.csv",
    },
    "multiroad": {
        "tracks": "data/output/multiroad_tiled_smooth.csv",
        "scale":  "data/output/multiroad_scale.json",
        "out":    "data/output/multiroad_tiled_l2.csv",
    },
}


def run_dataset(tracks_path: str, scale_path: str, out_path: str, fps: float):
    print(f"\n{'='*60}")
    print(f"Dataset : {tracks_path}")
    print(f"Output  : {out_path}")
    print(f"{'='*60}")

    # ── 1. Scale calibration ────────────────────────────────────────────────
    print("\n[1/3] Scale calibration …")
    df = pd.read_csv(tracks_path)
    scale = calibrate(df)
    with open(scale_path, "w") as f:
        json.dump(scale, f, indent=2)
    print(f"  px_per_m = {scale['px_per_m']:.4f}  ({scale['m_per_px']:.5f} m/px)")
    for cls, v in scale["per_class"].items():
        print(f"    {cls:12s}  {v['median_px_w']:.1f} px  /  {v['real_m_w']:.2f} m  "
              f"-> {v['px_per_m']:.2f} px/m  (n={v['n_samples']})")

    # ── 2. Class consolidation ──────────────────────────────────────────────
    print("\n[2/3] Class consolidation …")
    df = consolidate(df)
    print(f"  Track-class distribution:")
    for cls, cnt in df.drop_duplicates("track_id")["track_class"].value_counts().items():
        print(f"    {cls:14s}  {cnt}")
    sub = df[df["subclass"].notna()]
    if not sub.empty:
        print(f"  Truck sub-classes:")
        for sc, cnt in sub.drop_duplicates("track_id")["subclass"].value_counts().items():
            print(f"    {sc:14s}  {cnt}")

    # ── 3. Kinematics ───────────────────────────────────────────────────────
    print("\n[3/3] Kinematics …")
    df = compute_kinematics(df, px_per_m=scale["px_per_m"], fps=fps)

    # Write output
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  Output written to: {out_path}")

    # Summary
    obs = df[~df["interpolated"].astype(bool) & df["speed_kmh"].notna()]
    print("\n  Speed summary by track_class (km/h):")
    spd = obs.groupby("track_class")["speed_kmh"].agg(["mean", "median", "max", "count"])
    spd.columns = ["mean", "median", "max", "n"]
    print(spd.round(1).to_string())

    return df


def main():
    p = argparse.ArgumentParser(description="AeroFlow Level 2 pipeline.")
    p.add_argument("--intersection", action="store_true",
                   help="Process the intersection dataset (default paths)")
    p.add_argument("--multiroad",    action="store_true",
                   help="Process the multi-road dataset (default paths)")
    p.add_argument("--tracks", help="Custom CSV path (overrides --intersection/--multiroad)")
    p.add_argument("--scale",  help="Custom scale JSON output path")
    p.add_argument("--out",    help="Custom L2 CSV output path")
    p.add_argument("--fps",    type=float, default=FPS)
    args = p.parse_args()

    if args.tracks:
        # Custom single-dataset mode
        scale_path = args.scale or args.out.replace(".csv", "_scale.json")
        run_dataset(args.tracks, scale_path, args.out, args.fps)
    else:
        if not args.intersection and not args.multiroad:
            # Default: run both
            args.intersection = True
            args.multiroad    = True

        if args.intersection:
            d = DATASETS["intersection"]
            run_dataset(d["tracks"], d["scale"], d["out"], args.fps)

        if args.multiroad:
            d = DATASETS["multiroad"]
            run_dataset(d["tracks"], d["scale"], d["out"], args.fps)

    print("\nLevel 2 pipeline complete.")


if __name__ == "__main__":
    main()
