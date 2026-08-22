"""
Post-process a tracking CSV to merge fragmented tracks that are almost certainly
the same real object: same class, one track ends shortly before another begins,
close in space. Produces longer, more continuous tracks and trail lines without
re-running detection/tracking.

Usage:
    python src/stitch_tracks.py --in data/output/intersection_tracks_visdrone_v3.csv --out data/output/intersection_tracks_visdrone_v3_stitched.csv
"""
import argparse

import numpy as np
import pandas as pd


def stitch(df: pd.DataFrame, max_gap: int, max_dist: float, class_col: str) -> pd.DataFrame:
    tracks = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        tracks.append({
            "track_id": tid,
            "cls": g.iloc[0][class_col],
            "start_frame": g.iloc[0].frame,
            "end_frame": g.iloc[-1].frame,
            "start_xy": (g.iloc[0].cx, g.iloc[0].cy),
            "end_xy": (g.iloc[-1].cx, g.iloc[-1].cy),
        })

    # candidate (a ends, b starts shortly after, same class, close in space)
    candidates = []
    for a in tracks:
        for b in tracks:
            if a["track_id"] == b["track_id"] or a["cls"] != b["cls"]:
                continue
            gap = b["start_frame"] - a["end_frame"]
            if gap <= 0 or gap > max_gap:
                continue
            dist = np.hypot(b["start_xy"][0] - a["end_xy"][0], b["start_xy"][1] - a["end_xy"][1])
            if dist <= max_dist:
                candidates.append((dist, a["track_id"], b["track_id"]))

    candidates.sort(key=lambda c: c[0])

    parent = {t["track_id"]: t["track_id"] for t in tracks}
    used_as_successor = set()
    used_as_predecessor = set()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dist, a_id, b_id in candidates:
        if a_id in used_as_predecessor or b_id in used_as_successor:
            continue
        ra, rb = find(a_id), find(b_id)
        if ra == rb:
            continue
        parent[rb] = ra
        used_as_predecessor.add(a_id)
        used_as_successor.add(b_id)

    df = df.copy()
    df["track_id"] = df["track_id"].apply(lambda t: find(t))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-gap", type=int, default=20, help="Max frame gap to bridge (default ~0.7s at 30fps)")
    p.add_argument("--max-dist", type=float, default=80.0, help="Max pixel distance between end/start points")
    p.add_argument("--class-col", default="taxonomy_class")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    before = df.track_id.nunique()
    out = stitch(df, args.max_gap, args.max_dist, args.class_col)
    after = out.track_id.nunique()
    print(f"tracks before: {before}, after stitching: {after} ({before - after} merged)")

    dur_before = df.groupby("track_id").frame.count()
    dur_after = out.groupby("track_id").frame.count()
    print(f"tracks <=3 frames before: {(dur_before <= 3).mean()*100:.1f}%")
    print(f"tracks <=3 frames after:  {(dur_after <= 3).mean()*100:.1f}%")
    print(f"median duration before: {dur_before.median()}, after: {dur_after.median()}")

    out.to_csv(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
