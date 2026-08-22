"""
Per-track classification consolidation.

The YOLO detector can emit different class labels for the same object in
different frames (especially when partially occluded or at the edge of a tile).
For Level 2 the label must be *stable across the track lifetime*.

Method
------
For each track we take a confidence-weighted majority vote over all
non-interpolated detections and record the winning class as `track_class`.

Sub-classification
------------------
`truck` bounding boxes span a wide range of physical sizes.  Without a
homography we cannot measure ground-plane length precisely, but a box-area
proxy separates heavy goods vehicles (HGV, long-wheelbase) from rigid trucks:

  - Median box area for truck tracks → baseline
  - Tracks whose median box area > 1.5 × baseline → `HGV`
  - Otherwise → `rigid_truck`

This is explicitly flagged as a heuristic in the write-up.

Usage
-----
    python src/consolidate_class.py \\
        --tracks data/output/intersection_tracks_smooth.csv \\
        --out    data/output/intersection_tracks_classed.csv
"""

import argparse

import numpy as np
import pandas as pd

# HGV threshold multiplier (median track area × multiplier)
HGV_AREA_MULTIPLIER = 1.5


def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df with two new columns added:
      track_class  — stable per-track class label
      subclass     — sub-classification within 'truck' (HGV / rigid_truck) or NaN
    """
    class_col = "class" if "class" in df.columns else "taxonomy_class"

    track_class_map: dict[int, str] = {}
    subclass_map:    dict[int, str] = {}

    detected = df[~df["interpolated"]].copy()

    # Pre-compute track median box area for HGV split
    detected["box_area"] = (detected["x2"] - detected["x1"]) * (detected["y2"] - detected["y1"])

    truck_areas = (
        detected[detected[class_col] == "truck"]
        .groupby("track_id")["box_area"]
        .median()
    )
    global_truck_median = float(truck_areas.median()) if len(truck_areas) else None

    for tid, grp in detected.groupby("track_id"):
        # Confidence-weighted vote
        votes: dict[str, float] = {}
        for _, row in grp.iterrows():
            cls = row[class_col]
            conf = float(row["conf"])
            votes[cls] = votes.get(cls, 0.0) + conf

        winner = max(votes, key=votes.__getitem__)
        track_class_map[int(tid)] = winner

        # Sub-classify trucks
        if winner == "truck" and global_truck_median is not None:
            track_median_area = float(grp["box_area"].median())
            if track_median_area > global_truck_median * HGV_AREA_MULTIPLIER:
                subclass_map[int(tid)] = "HGV"
            else:
                subclass_map[int(tid)] = "rigid_truck"
        else:
            subclass_map[int(tid)] = np.nan  # type: ignore[assignment]

    df = df.copy()
    df["track_class"] = df["track_id"].map(track_class_map)
    df["subclass"]    = df["track_id"].map(subclass_map)

    return df


def main():
    p = argparse.ArgumentParser(description="Stable per-track class consolidation.")
    p.add_argument("--tracks", required=True, help="Smooth trajectory CSV")
    p.add_argument("--out",    required=True, help="Output CSV path")
    args = p.parse_args()

    df = pd.read_csv(args.tracks)
    df = consolidate(df)
    df.to_csv(args.out, index=False)

    class_col = "class" if "class" in df.columns else "taxonomy_class"
    print("Track-class consolidation done.")
    print(f"  Rows:   {len(df):,}")
    print(f"  Tracks: {df.track_id.nunique():,}")
    print("  track_class distribution:")
    for cls, cnt in df.drop_duplicates("track_id")["track_class"].value_counts().items():
        print(f"    {cls:12s}  {cnt}")
    if df["subclass"].notna().any():
        print("  Truck sub-classes:")
        for sc, cnt in df[df["subclass"].notna()].drop_duplicates("track_id")["subclass"].value_counts().items():
            print(f"    {sc:15s}  {cnt}")
    print(f"  Written to: {args.out}")


if __name__ == "__main__":
    main()
