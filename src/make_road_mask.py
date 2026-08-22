"""
Derive a road mask from existing tracking output.

Road users only travel on the road, so the union of every detection box across a clip is a
good empirical map of the drivable area. Dilating and closing that union gives a mask we can
use to (a) restrict where we run inference and (b) reject detections that land on rooftops.

Usage:
    python src/make_road_mask.py --tracks data/output/multiroad_tracks_smooth.csv \
        --video data/raw_video/Multi_Road_Merged_convert_4k.mp4 \
        --out data/output/multiroad_mask.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def build_mask(df, w, h, dilate_px, min_conf):
    mask = np.zeros((h, w), np.uint8)

    if "conf" in df.columns:
        df = df[df.conf >= min_conf]
    if "interpolated" in df.columns:
        df = df[~df.interpolated.astype(bool)]

    for r in df.itertuples(index=False):
        x1, y1 = max(0, int(r.x1)), max(0, int(r.y1))
        x2, y2 = min(w - 1, int(r.x2)), min(h - 1, int(r.y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    # close small holes, then dilate so we keep a margin around the travelled area
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask = cv2.dilate(mask, k_dil)

    # drop specks that are not part of the corridor
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        thresh = max(areas.max() * 0.02, 5000)
        for i, a in enumerate(areas, start=1):
            if a >= thresh:
                keep[labels == i] = 255
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dilate", type=int, default=61, help="Dilation kernel size in px")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--preview", help="Optional path to save a mask-overlay preview image")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()

    df = pd.read_csv(args.tracks)
    mask = build_mask(df, w, h, args.dilate, args.min_conf)

    coverage = (mask > 0).mean()
    ys, xs = np.where(mask > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    print(f"mask covers {coverage*100:.1f}% of the frame")
    print(f"road bbox: {bbox}")
    if bbox:
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"road bbox size: {bw}x{bh} px  ({bw*bh/(w*h)*100:.1f}% of frame area)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, mask)
    print(f"Written {args.out}")

    if args.preview and ok:
        overlay = frame.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] +
                             0.45 * np.array([0, 200, 0])).astype(np.uint8)
        cv2.imwrite(args.preview, overlay)
        print(f"Written {args.preview}")


if __name__ == "__main__":
    main()
