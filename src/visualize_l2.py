"""
Level 2 annotated video — overlays kinematics on the source footage.

What is drawn per object (every frame it is present):
  - Bounding box coloured by track_class
  - Label:  <class> #<id>   <speed> km/h
  - Heading arrow from centroid (length proportional to speed)
  - Trail coloured by speed:  green (slow) → yellow → red (fast)
  - HGV/rigid_truck badge shown in top-left of box

Overlay panels:
  - Class-colour legend (bottom-left)
  - Scale bar (bottom-right)
  - Frame / time counter (top-left)
  - Speed colour scale (bottom-centre)

Output is rendered at 1920×1080 (half of 4K) for manageable file size.

Usage
-----
    python src/visualize_l2.py \\
        --video  data/raw_video/Intersection_Merged_convert_4k.mp4 \\
        --tracks data/output/intersection_tracks_l2.csv \\
        --scale  data/output/intersection_scale.json \\
        --out    data/output/l2_intersection_1min.mp4 \\
        --max-frames 1800

    python src/visualize_l2.py \\
        --video  data/raw_video/Multi_Road_Merged_convert_4k.mp4 \\
        --tracks data/output/multiroad_tiled_l2.csv \\
        --scale  data/output/multiroad_scale.json \\
        --out    data/output/l2_multiroad_1min.mp4 \\
        --max-frames 1800
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import cv2
import numpy as np

# ── colours ─────────────────────────────────────────────────────────────────
CLASS_COLOR = {
    "car":        (50,  205,  50),   # lime green
    "LGV":        (0,   165, 255),   # orange
    "truck":      (0,   100, 200),   # steel blue
    "bus":        (180,   0, 180),   # magenta
    "motorcycle": (0,   220, 220),   # cyan
    "pedestrian": (220,  80, 220),   # violet
    "HGV":        (0,    60, 180),   # dark blue badge
}
UNKNOWN_COLOR = (160, 160, 160)

# Speed colour ramp: 0 km/h → green, 30 km/h → yellow, 60+ km/h → red
SPEED_RAMP = [
    (0,   (50,  220,  50)),
    (15,  (50,  220, 140)),
    (30,  (30,  200, 220)),
    (50,  (20,  120, 255)),
    (70,  (0,    40, 255)),
    (100, (0,     0, 200)),
]


def speed_color(kmh: float) -> tuple:
    """Interpolate speed → BGR colour."""
    if kmh is None or math.isnan(kmh):
        return (120, 120, 120)
    ramp = SPEED_RAMP
    for i in range(1, len(ramp)):
        s0, c0 = ramp[i - 1]
        s1, c1 = ramp[i]
        if kmh <= s1:
            t = (kmh - s0) / (s1 - s0)
            return tuple(int(c0[j] + t * (c1[j] - c0[j])) for j in range(3))
    return ramp[-1][1]


def load_tracks(csv_path: str) -> dict:
    """Load L2 CSV → {frame_idx: [row_dict, ...]}"""
    by_frame = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_frame[int(row["frame"])].append(row)
    return by_frame


def draw_heading_arrow(frame, cx, cy, heading_deg, speed_kmh, scale=1.0):
    """Draw a small directional arrow from centroid."""
    if heading_deg is None or math.isnan(float(heading_deg)):
        return
    spd = float(speed_kmh) if speed_kmh and not math.isnan(float(speed_kmh)) else 0
    length = max(8, min(60, spd * 1.2)) * scale
    rad = math.radians(float(heading_deg))
    ex = int(cx + length * math.sin(rad))
    ey = int(cy - length * math.cos(rad))
    col = speed_color(spd)
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), col,
                    max(1, int(2 * scale)), tipLength=0.4, line_type=cv2.LINE_AA)


def draw_legend(frame, h, w, px_per_m, m_per_px):
    """Draw class legend, scale bar, and speed ramp."""
    pad = 12
    lh  = 24

    # ── class legend (bottom-left) ──
    classes = [("car", "car"), ("motorcycle", "motorcycle"), ("pedestrian", "pedestrian"),
               ("LGV", "LGV/van"), ("truck", "truck"), ("bus", "bus")]
    ly = h - pad - len(classes) * lh - 4
    cv2.rectangle(frame, (pad - 4, ly - 4),
                  (pad + 130, h - pad + 4), (20, 20, 20), -1)
    for cls, label in classes:
        col = CLASS_COLOR.get(cls, UNKNOWN_COLOR)
        cv2.rectangle(frame, (pad, ly), (pad + 16, ly + lh - 4), col, -1)
        cv2.putText(frame, label, (pad + 22, ly + lh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        ly += lh

    # ── scale bar (bottom-right) ──
    bar_m = 50                          # represent 50 metres
    bar_px = int(bar_m * px_per_m)
    bx2 = w - pad
    bx1 = bx2 - bar_px
    by  = h - pad
    cv2.rectangle(frame, (bx1 - 4, by - 22), (bx2 + 4, by + 8), (20, 20, 20), -1)
    cv2.line(frame, (bx1, by), (bx2, by), (230, 230, 230), 3, cv2.LINE_AA)
    cv2.line(frame, (bx1, by - 6), (bx1, by + 6), (230, 230, 230), 2)
    cv2.line(frame, (bx2, by - 6), (bx2, by + 6), (230, 230, 230), 2)
    cv2.putText(frame, f"50 m ({bar_px} px)", (bx1, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # ── speed colour ramp (bottom-centre) ──
    rw, rh2 = 200, 14
    rx = w // 2 - rw // 2
    ry = h - pad - 4
    for i in range(rw):
        kmh = i / rw * 70
        col = speed_color(kmh)
        cv2.line(frame, (rx + i, ry), (rx + i, ry - rh2), col, 1)
    cv2.rectangle(frame, (rx - 2, ry - rh2 - 2), (rx + rw + 2, ry + 2), (50, 50, 50), 1)
    cv2.putText(frame, "0", (rx - 8, ry + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(frame, "70+ km/h", (rx + rw - 40, ry + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(frame, "Speed", (rx + rw // 2 - 14, ry - rh2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def main():
    p = argparse.ArgumentParser(description="Level 2 kinematics video renderer.")
    p.add_argument("--video",       required=True)
    p.add_argument("--tracks",      required=True, help="L2 CSV")
    p.add_argument("--scale",       required=True, help="Scale JSON")
    p.add_argument("--out",         required=True)
    p.add_argument("--max-frames",  type=int, default=1800)
    p.add_argument("--trail",       type=int, default=90,
                   help="Trail length in frames")
    p.add_argument("--output-size", nargs=2, type=int, default=[1920, 1080],
                   metavar=("W", "H"), help="Output resolution (default 1920 1080)")
    args = p.parse_args()

    with open(args.scale) as f:
        scale = json.load(f)
    px_per_m = scale["px_per_m"]
    m_per_px = scale["m_per_px"]

    print(f"Loading tracks …")
    by_frame = load_tracks(args.tracks)
    print(f"  {sum(len(v) for v in by_frame.values()):,} rows across "
          f"{len(by_frame)} frames")

    cap = cv2.VideoCapture(args.video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 29.97

    out_w, out_h = args.output_size
    scale_x = out_w / src_w
    scale_y = out_h / src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))

    # Per-track trail buffer: deque of (pt, speed_kmh)
    trail: dict[str, list] = defaultdict(list)

    limit = min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"Rendering {limit} frames at {out_w}×{out_h} …")

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok:
            break

        # Downscale source frame
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        rows = by_frame.get(fidx, [])

        # ── update trails ────────────────────────────────────────────────────
        for row in rows:
            tid = row["track_id"]
            cx  = float(row["cx"]) * scale_x
            cy  = float(row["cy"]) * scale_y
            spd_raw = row.get("speed_kmh", "")
            spd = float(spd_raw) if spd_raw and spd_raw not in ("", "nan") else float("nan")
            trail[tid].append(((int(cx), int(cy)), spd))
            if len(trail[tid]) > args.trail:
                trail[tid] = trail[tid][-args.trail:]

        # ── draw trails ──────────────────────────────────────────────────────
        for tid, pts in trail.items():
            for i in range(1, len(pts)):
                col = speed_color(pts[i][1])
                alpha_factor = i / len(pts)          # fade older segments
                faded = tuple(int(c * alpha_factor) for c in col)
                cv2.line(frame, pts[i-1][0], pts[i][0], faded, 2, cv2.LINE_AA)

        # ── draw boxes + labels + arrows ─────────────────────────────────────
        for row in rows:
            cls   = row.get("track_class") or row.get("class", "unknown")
            tid   = row["track_id"]
            sub   = row.get("subclass", "")
            col   = CLASS_COLOR.get(cls, UNKNOWN_COLOR)

            x1 = float(row["x1"]) * scale_x
            y1 = float(row["y1"]) * scale_y
            x2 = float(row["x2"]) * scale_x
            y2 = float(row["y2"]) * scale_y
            cx = float(row["cx"]) * scale_x
            cy = float(row["cy"]) * scale_y

            # semi-transparent box fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)

            # speed value
            spd_raw = row.get("speed_kmh", "")
            if spd_raw and spd_raw not in ("", "nan"):
                spd = float(spd_raw)
                spd_str = f"{spd:.1f} km/h"
                spd_col = speed_color(spd)
            else:
                spd_str = ""
                spd_col = UNKNOWN_COLOR

            # class + id label
            label_top = f"{cls} #{tid}"
            lx, ly = int(x1), max(int(y1) - 18, 14)
            (tw, th), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (lx - 1, ly - th - 3), (lx + tw + 2, ly + 3),
                          (20, 20, 20), -1)
            cv2.putText(frame, label_top, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

            # speed label below class label
            if spd_str:
                sy = ly + th + 14
                (sw, sh), _ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(frame, (lx - 1, sy - sh - 2), (lx + sw + 2, sy + 2),
                              (20, 20, 20), -1)
                cv2.putText(frame, spd_str, (lx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, spd_col, 1, cv2.LINE_AA)

            # HGV / subclass badge
            if sub and sub not in ("nan", ""):
                badge_col = CLASS_COLOR.get("HGV", (0, 60, 180))
                bx, by2 = int(x1), int(y2) + 14
                cv2.rectangle(frame, (bx - 1, by2 - 12), (bx + 38, by2 + 2), badge_col, -1)
                cv2.putText(frame, sub[:3].upper(), (bx + 1, by2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # heading arrow
            hdg_raw = row.get("heading_deg", "")
            if hdg_raw and hdg_raw not in ("", "nan"):
                draw_heading_arrow(frame, cx, cy, float(hdg_raw),
                                   float(spd_raw) if spd_raw and spd_raw != "nan" else 0,
                                   scale=max(scale_x, scale_y))

        # ── overlays ─────────────────────────────────────────────────────────
        draw_legend(frame, out_h, out_w, px_per_m * scale_x, m_per_px / scale_x)

        # frame/time counter
        t_sec = fidx / fps
        ts = f"t = {int(t_sec // 60):02d}:{t_sec % 60:05.2f}  frame {fidx}"
        cv2.rectangle(frame, (4, 4), (320, 26), (20, 20, 20), -1)
        cv2.putText(frame, ts, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 200), 1, cv2.LINE_AA)

        # active object count
        n_obj = len(rows)
        cv2.putText(frame, f"Objects: {n_obj}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)

        writer.write(frame)

        if fidx % 150 == 0:
            print(f"  frame {fidx}/{limit}  ({fidx/fps:.1f}s)")

    cap.release()
    writer.release()
    print(f"\nDone. Written to: {args.out}")


if __name__ == "__main__":
    main()
