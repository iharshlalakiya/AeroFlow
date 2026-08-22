"""
Overlays tracked trajectories from a CSV (produced by track.py) onto the source video,
so you can visually verify identity persistence through occlusion/crossing.

Usage:
    python src/visualize.py --video data/raw_video/Intersection_Merged.MP4 \
        --tracks data/output/intersection_tracks.csv \
        --out data/output/intersection_annotated.mp4
"""
import argparse
import csv
from collections import defaultdict

import cv2

COLORS = {
    "car": (0, 200, 0),
    "truck": (0, 128, 255),
    "bus": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "pedestrian": (255, 0, 255),
    "unknown": (200, 200, 200),
}


def load_tracks(csv_path):
    by_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_frame[int(row["frame"])].append(row)
    return by_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for a quick preview clip")
    args = p.parse_args()

    tracks = load_tracks(args.tracks)

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    trail = defaultdict(list)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        for row in tracks.get(frame_idx, []):
            tid = row["track_id"]
            cls = row["class"]
            color = COLORS.get(cls, COLORS["unknown"])
            x1, y1, x2, y2 = map(float, (row["x1"], row["y1"], row["x2"], row["y2"]))
            cx, cy = float(row["cx"]), float(row["cy"])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls}#{tid}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            trail[tid].append((int(cx), int(cy)))
            trail[tid] = trail[tid][-30:]

        for tid, pts in trail.items():
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"rendered frame {frame_idx}")

    cap.release()
    out.release()
    print(f"Annotated video written to {args.out}")


if __name__ == "__main__":
    main()
