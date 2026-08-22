"""
Detect and repair ID swaps in a tracking CSV.

A real road user moves smoothly frame to frame. When a tracker swaps two identities
during an overlap, both tracks show a sudden position jump in the SAME frame, and each
lands near where the other one just was. This script finds those mutual jumps and swaps
the identities back from that frame onward.

Usage:
    python src/fix_swaps.py --in data/output/intersection_tracks_v4_stitched.csv --out data/output/intersection_tracks_v4_fixed.csv
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def find_swaps(df, jump_thresh, pair_dist_thresh):
    """Return list of (frame, track_a, track_b) where a and b appear to have swapped."""
    df = df.sort_values(["frame", "track_id"])
    # position of each track per frame
    pos = {}
    for row in df.itertuples(index=False):
        pos[(row.frame, row.track_id)] = (row.cx, row.cy)

    frames = sorted(df.frame.unique())
    tracks_by_frame = df.groupby("frame").track_id.apply(list).to_dict()

    swaps = []
    for prev_f, cur_f in zip(frames, frames[1:]):
        if cur_f - prev_f > 2:
            continue
        common = set(tracks_by_frame.get(prev_f, [])) & set(tracks_by_frame.get(cur_f, []))
        # tracks that jumped a lot this frame
        jumpers = []
        for t in common:
            px, py = pos[(prev_f, t)]
            cx, cy = pos[(cur_f, t)]
            d = np.hypot(cx - px, cy - py)
            if d > jump_thresh:
                jumpers.append((t, (px, py), (cx, cy)))

        # look for mutual jumps: a landed near b's old spot and vice versa
        for i in range(len(jumpers)):
            for j in range(i + 1, len(jumpers)):
                ta, a_prev, a_cur = jumpers[i]
                tb, b_prev, b_cur = jumpers[j]
                d_ab = np.hypot(a_cur[0] - b_prev[0], a_cur[1] - b_prev[1])
                d_ba = np.hypot(b_cur[0] - a_prev[0], b_cur[1] - a_prev[1])
                if d_ab < pair_dist_thresh and d_ba < pair_dist_thresh:
                    swaps.append((cur_f, ta, tb))
    return swaps


def apply_swaps(df, swaps):
    """Swap identities from each detected swap frame onward."""
    df = df.copy()
    # process chronologically so later swaps see earlier corrections
    for frame, ta, tb in sorted(swaps):
        mask_a = (df.track_id == ta) & (df.frame >= frame)
        mask_b = (df.track_id == tb) & (df.frame >= frame)
        df.loc[mask_a, "track_id"] = -ta  # temp marker to avoid clobber
        df.loc[mask_b, "track_id"] = ta
        df.loc[df.track_id == -ta, "track_id"] = tb
    return df


def motion_smoothness(df):
    """Mean per-frame displacement across all tracks — lower is smoother."""
    jumps = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        d = np.hypot(g.cx.diff(), g.cy.diff()).dropna()
        jumps.extend(d.tolist())
    arr = np.array(jumps)
    return arr.mean(), np.percentile(arr, 99)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jump-thresh", type=float, default=60.0,
                   help="Pixel displacement in one frame that counts as a suspicious jump")
    p.add_argument("--pair-dist", type=float, default=50.0,
                   help="How close a jumper must land to the other's previous position")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    mean_b, p99_b = motion_smoothness(df)
    swaps = find_swaps(df, args.jump_thresh, args.pair_dist)
    print(f"detected {len(swaps)} probable ID swaps")

    out = apply_swaps(df, swaps)
    mean_a, p99_a = motion_smoothness(out)
    print(f"mean per-frame displacement: {mean_b:.2f}px -> {mean_a:.2f}px")
    print(f"99th pct displacement:       {p99_b:.2f}px -> {p99_a:.2f}px")

    out.to_csv(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
