"""
Fill detection dropouts inside each track by linear interpolation.

After stitching, a track can still be missing frames in the middle (the detector lost the
object for a moment). Those holes are what make the boxes blink. Since we know where the
object was before and after, we can interpolate the box through the gap.

Usage:
    python src/interpolate_gaps.py --in data/output/intersection_tracks_final.csv --out data/output/intersection_tracks_smooth.csv
"""
import argparse

import numpy as np
import pandas as pd

BOX_COLS = ["x1", "y1", "x2", "y2", "cx", "cy"]


def interpolate_track(g: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    g = g.sort_values("frame")
    frames = g.frame.to_numpy()
    rows = [g]

    for i in range(len(frames) - 1):
        gap = frames[i + 1] - frames[i]
        if gap <= 1 or gap - 1 > max_gap:
            continue
        a = g.iloc[i]
        b = g.iloc[i + 1]
        missing = np.arange(frames[i] + 1, frames[i + 1])
        w = (missing - frames[i]) / gap  # 0..1 across the gap

        filled = pd.DataFrame({"frame": missing})
        for col in BOX_COLS:
            filled[col] = a[col] + (b[col] - a[col]) * w
        filled["track_id"] = a.track_id
        for col in g.columns:
            if col not in filled.columns:
                filled[col] = a[col]
        filled["interpolated"] = True
        rows.append(filled)

    out = pd.concat(rows, ignore_index=True).sort_values("frame")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-gap", type=int, default=90,
                   help="Largest hole (frames) to fill; longer holes are left alone")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    df["interpolated"] = False

    pieces = [interpolate_track(g, args.max_gap) for _, g in df.groupby("track_id")]
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["frame", "track_id"]).reset_index(drop=True)

    added = int(out.interpolated.sum())
    print(f"rows: {len(df)} -> {len(out)} ({added} interpolated)")

    # how continuous are tracks now?
    def continuity(d):
        spans = d.groupby("track_id").frame.agg(["min", "max", "count"])
        expected = spans["max"] - spans["min"] + 1
        return (spans["count"] / expected).mean()

    print(f"mean track continuity (frames present / frames spanned): "
          f"{continuity(df):.3f} -> {continuity(out):.3f}")

    out.to_csv(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
