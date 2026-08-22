"""
Level 3 annotated video — aggregate insight overlays on intersection footage.

Overlays (on top of per-object kinematics from L2):
  1. Speed heatmap   — semi-transparent coloured grid behind the scene
  2. Queue bars      — live bar-chart at each approach arm (updates every frame)
  3. TMC arrows      — arrow at each arm showing cumulative movement counts
                       (refreshed each 15-second interval)
  4. Counts panel    — running totals by class + turn type in corner
  5. Time bar        — progress bar + interval marker

Usage:
    python src/visualize_l3.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
VIDEO_IN  = Path("data/raw_video/Intersection_Merged_convert_4k.mp4")
TRACKS_L2 = Path("data/output/intersection_tracks_l2.csv")
MOVEMENT  = Path("data/output/l3_movements.csv")
QUEUE_CSV = Path("data/output/l3_queue.csv")
TMC_CSV   = Path("data/output/l3_tmc.csv")
HEATMAP   = Path("data/output/l3_speed_heatmap.json")
SUMMARY   = Path("data/output/l3_summary.json")
VIDEO_OUT = Path("data/output/l3_intersection_1min.mp4")

FPS       = 29.97
MAX_FRAMES = 1800
OUT_W, OUT_H = 1920, 1080
INTERVAL_F = int(15 * FPS)    # TMC refresh interval

# ── colours ───────────────────────────────────────────────────────────────────
CLASS_COLOR = {
    "car":        (50,  205,  50),
    "LGV":        (0,   165, 255),
    "truck":      (0,   100, 200),
    "bus":        (180,   0, 180),
    "motorcycle": (0,   220, 220),
    "pedestrian": (220,  80, 220),
}
UNKNOWN_COLOR = (160, 160, 160)

ARM_POS = {}   # will be computed from intersection centre
ARM_COLOR = {"N": (255,200,50), "S": (50,200,255), "E": (100,255,100), "W": (255,100,100)}

# Speed colour: low=cool blue, medium=green, high=red (reversed for heatmap)
def speed_to_bgr(v, vmin=0, vmax=30):
    t = np.clip((v - vmin) / max(vmax - vmin, 1), 0, 1)
    # Blue(slow) → Cyan → Green → Yellow → Red(fast)
    if t < 0.25:
        r, g, b = 0, int(t*4*255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((1-(t-0.25)*4)*255)
    elif t < 0.75:
        r, g, b = int((t-0.5)*4*255), 255, 0
    else:
        r, g, b = 255, int((1-(t-0.75)*4)*255), 0
    return (b, g, r)   # BGR


def draw_text_bg(img, text, pos, font_scale=0.5, color=(230,230,230),
                 thickness=1, bg=(20,20,20), pad=3):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def build_heatmap_overlay(hm_data, out_w, out_h, alpha=0.38):
    """Pre-render the speed heatmap as a full-frame BGRA overlay."""
    rows, cols = hm_data["grid_rows"], hm_data["grid_cols"]
    cw = out_w / cols
    ch = out_h / rows
    vmin, vmax = hm_data["vmin"], hm_data["vmax"]
    values = np.array(hm_data["values"])
    counts = np.array(hm_data["counts"])

    overlay = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if counts[r, c] == 0:
                continue
            v = float(values[r, c])
            bgr = speed_to_bgr(v, vmin, vmax)
            x1, y1 = int(c * cw), int(r * ch)
            x2, y2 = min(int((c+1)*cw), out_w-1), min(int((r+1)*ch), out_h-1)
            overlay[y1:y2, x1:x2, 0] = bgr[0]
            overlay[y1:y2, x1:x2, 1] = bgr[1]
            overlay[y1:y2, x1:x2, 2] = bgr[2]
            overlay[y1:y2, x1:x2, 3] = int(alpha * 255)

    # Convert to BGR for blending
    bgr_overlay = overlay[:, :, :3].astype(np.uint8)
    alpha_mask  = overlay[:, :, 3:4].astype(float) / 255.0
    return bgr_overlay, alpha_mask


def draw_queue_bars(frame, queue_row, cx, cy, scale_x, scale_y):
    """Draw a small bar at each arm showing queue length."""
    if queue_row is None:
        return
    BAR_MAX = 12          # vehicles → full bar
    BAR_LEN = 80          # px in output
    cx_s = int(cx * scale_x)
    cy_s = int(cy * scale_y)
    offsets = {"N": (0, -120), "S": (0, 120), "E": (120, 0), "W": (-120, 0)}
    for arm, (dx, dy) in offsets.items():
        n = int(queue_row.get(f"queue_{arm}", 0))
        col = ARM_COLOR.get(arm, (200,200,200))
        px, py = cx_s + dx, cy_s + dy
        # background
        cv2.rectangle(frame, (px-5, py-5), (px+BAR_LEN+5, py+16), (20,20,20), -1)
        # bar
        bar_w = int(np.clip(n / BAR_MAX, 0, 1) * BAR_LEN)
        if bar_w > 0:
            cv2.rectangle(frame, (px, py), (px+bar_w, py+10), col, -1)
        cv2.putText(frame, f"{arm}: {n}veh", (px, py+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def draw_tmc_arrows(frame, tmc_interval, mv_df, cx, cy, scale_x, scale_y):
    """Draw arrows from intersection centre toward each exit arm, thickness ∝ volume."""
    cx_s, cy_s = int(cx * scale_x), int(cy * scale_y)
    ARM_DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    ARM_LABEL_OFF = {"N": (-20, -10), "S": (-20, 20), "E": (10, 0), "W": (-60, 0)}

    interval_mv = tmc_interval if tmc_interval is not None else mv_df
    arm_counts = interval_mv.groupby("exit_arm").size().to_dict() if not interval_mv.empty else {}

    total = max(sum(arm_counts.values()), 1)
    for arm, (dx, dy) in ARM_DIRS.items():
        n = arm_counts.get(arm, 0)
        if n == 0:
            continue
        col = ARM_COLOR.get(arm, (200,200,200))
        thickness = max(2, min(12, int(n / total * 40)))
        ex = cx_s + dx * 130
        ey = cy_s + dy * 130
        cv2.arrowedLine(frame, (cx_s, cy_s), (int(ex), int(ey)),
                        col, thickness, tipLength=0.25, line_type=cv2.LINE_AA)
        lox, loy = ARM_LABEL_OFF[arm]
        draw_text_bg(frame, f"{arm} {n}", (int(ex)+lox, int(ey)+loy),
                     font_scale=0.55, color=col)


def draw_counts_panel(frame, cumulative, out_w, out_h, turn_counts, frame_n_obj):
    """Bottom-right info panel."""
    pw, ph = 220, 200
    px, py = out_w - pw - 8, out_h - ph - 8
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (15,15,15), -1)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (60,60,60), 1)

    y = py + 18
    cv2.putText(frame, "CUMULATIVE COUNTS", (px+8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,255), 1, cv2.LINE_AA)
    y += 16
    for cls, cnt in sorted(cumulative.items()):
        col = CLASS_COLOR.get(cls, UNKNOWN_COLOR)
        cv2.putText(frame, f"{cls}: {cnt}", (px+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        y += 15

    y += 4
    cv2.putText(frame, "TURN TYPES", (px+8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,255,180), 1, cv2.LINE_AA)
    y += 16
    for tt, cnt in sorted(turn_counts.items()):
        cv2.putText(frame, f"{tt}: {cnt}", (px+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210,210,210), 1, cv2.LINE_AA)
        y += 14

    y += 4
    cv2.putText(frame, f"On screen: {frame_n_obj}", (px+10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,220,100), 1, cv2.LINE_AA)


def draw_time_bar(frame, frame_idx, max_frames, out_w, out_h, interval_f):
    """Progress bar + interval marker at top."""
    bx1, bx2 = 0, out_w
    by = 8
    bh = 6
    # Background
    cv2.rectangle(frame, (bx1, by), (bx2, by+bh), (40,40,40), -1)
    # Progress
    prog = int(frame_idx / max_frames * out_w)
    cv2.rectangle(frame, (bx1, by), (prog, by+bh), (80,200,120), -1)
    # Interval ticks
    for i in range(0, max_frames, interval_f):
        tx = int(i / max_frames * out_w)
        cv2.line(frame, (tx, by-2), (tx, by+bh+2), (200,200,100), 1)
    # Time label
    t = frame_idx / FPS
    cv2.putText(frame, f"{int(t//60):02d}:{t%60:05.2f}", (bx1+4, by+bh+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1, cv2.LINE_AA)


def draw_heatmap_legend(frame, vmin, vmax, out_w, out_h):
    """Speed heatmap colour legend."""
    lw, lh = 150, 12
    lx = out_w // 2 - lw // 2
    ly = out_h - 30
    for i in range(lw):
        v = vmin + i / lw * (vmax - vmin)
        col = speed_to_bgr(v, vmin, vmax)
        cv2.line(frame, (lx+i, ly), (lx+i, ly+lh), col, 1)
    cv2.rectangle(frame, (lx-1, ly-1), (lx+lw+1, ly+lh+1), (80,80,80), 1)
    draw_text_bg(frame, f"{vmin:.0f}", (lx-20, ly+lh), font_scale=0.35, color=(200,200,200))
    draw_text_bg(frame, f"{vmax:.0f} km/h", (lx+lw+2, ly+lh), font_scale=0.35, color=(200,200,200))
    draw_text_bg(frame, "Speed heatmap", (lx+40, ly-4), font_scale=0.35, color=(200,200,200))


def main():
    print("Loading data …")
    tracks   = pd.read_csv(TRACKS_L2, low_memory=False)
    mv_df    = pd.read_csv(MOVEMENT)
    queue_df = pd.read_csv(QUEUE_CSV)
    tmc_df   = pd.read_csv(TMC_CSV) if TMC_CSV.exists() else pd.DataFrame()
    summary  = json.load(open(SUMMARY))

    cx_centre = summary["intersection_centre_px"][0]
    cy_centre = summary["intersection_centre_px"][1]

    with open(HEATMAP) as f:
        hm_data = json.load(f)

    # Group track data by frame
    by_frame = defaultdict(list)
    for row in tracks.itertuples():
        by_frame[row.frame].append(row)

    # Group queue data by frame (nearest)
    queue_by_frame = {int(r.frame): r._asdict() for r in queue_df.itertuples()}

    # Group movements by track_id for quick lookup
    mv_by_tid = mv_df.set_index("track_id").to_dict("index")

    # Scale factors
    src_w, src_h = 3840, 2160
    sx = OUT_W / src_w
    sy = OUT_H / src_h

    print("Building speed heatmap overlay …")
    hm_bgr, hm_alpha = build_heatmap_overlay(hm_data, OUT_W, OUT_H, alpha=0.32)

    cap = cv2.VideoCapture(str(VIDEO_IN))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, FPS, (OUT_W, OUT_H))

    trail = defaultdict(list)
    cumulative = defaultdict(int)   # cumulative track-class counts
    turn_counts = defaultdict(int)  # cumulative turn-type counts

    # Pre-compute TMC per interval
    tmc_intervals = {}
    if not tmc_df.empty:
        for t_start in tmc_df["interval_start_s"].unique():
            tmc_intervals[t_start] = tmc_df[tmc_df["interval_start_s"] == t_start]

    seen_tracks = set()

    print(f"Rendering {MAX_FRAMES} frames …")
    for fidx in range(MAX_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        # ── Apply speed heatmap background ───────────────────────────────────
        frame = (frame * (1 - hm_alpha) + hm_bgr * hm_alpha).astype(np.uint8)

        rows = by_frame.get(fidx, [])

        # Update trails
        for r in rows:
            tid   = str(r.track_id)
            cx_s  = int(float(r.cx) * sx)
            cy_s  = int(float(r.cy) * sy)
            spd   = float(r.speed_kmh) if str(r.speed_kmh) not in ("nan","") else float("nan")
            trail[tid].append(((cx_s, cy_s), spd))
            trail[tid] = trail[tid][-60:]

            # Update cumulative on first appearance
            if tid not in seen_tracks:
                seen_tracks.add(tid)
                cls = str(getattr(r, "track_class", r.track_class if hasattr(r,"track_class") else ""))
                cumulative[cls] += 1
                mv = mv_by_tid.get(r.track_id, {})
                if mv:
                    turn_counts[mv.get("turn_type","?")] += 1

        # Draw trails
        for tid, pts in trail.items():
            for i in range(1, len(pts)):
                col = (80,80,80) if math.isnan(pts[i][1]) else \
                      speed_to_bgr(pts[i][1], 0, 30)
                alpha_t = i / len(pts)
                faded = tuple(int(c * alpha_t) for c in col)
                cv2.line(frame, pts[i-1][0], pts[i][0], faded, 2, cv2.LINE_AA)

        # Draw boxes + speed labels
        for r in rows:
            cls = str(getattr(r,"track_class",""))
            col = CLASS_COLOR.get(cls, UNKNOWN_COLOR)
            x1 = int(float(r.x1)*sx); y1 = int(float(r.y1)*sy)
            x2 = int(float(r.x2)*sx); y2 = int(float(r.y2)*sy)
            cx_s = int(float(r.cx)*sx); cy_s = int(float(r.cy)*sy)

            # box
            overlay = frame.copy()
            cv2.rectangle(overlay,(x1,y1),(x2,y2),col,-1)
            cv2.addWeighted(overlay,0.12,frame,0.88,0,frame)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)

            # Movement label from mv_by_tid
            mv_info = mv_by_tid.get(r.track_id, {})
            turn_label = mv_info.get("turn_type","?")[:2].upper() if mv_info else ""
            entry_arm  = mv_info.get("entry_arm","") if mv_info else ""

            spd_raw = str(r.speed_kmh)
            if spd_raw not in ("nan",""):
                spd_val = float(spd_raw)
                spd_str = f"{spd_val:.1f}"
                spd_col = speed_to_bgr(spd_val, 0, 30)
            else:
                spd_str, spd_col = "", UNKNOWN_COLOR

            label = f"{cls[:3]} #{r.track_id} {entry_arm}{turn_label}"
            draw_text_bg(frame, label, (x1, max(y1-2, 12)), font_scale=0.44, color=col)
            if spd_str:
                draw_text_bg(frame, f"{spd_str}km/h", (x1, max(y1+12, 24)),
                             font_scale=0.40, color=spd_col)

        # ── Queue bars (nearest 5-frame step) ─────────────────────────────
        q_frame = (fidx // 5) * 5
        queue_row = queue_by_frame.get(q_frame, None)
        draw_queue_bars(frame, queue_row, cx_centre, cy_centre, sx, sy)

        # ── TMC arrows (refresh each interval) ─────────────────────────────
        interval_s = (fidx // INTERVAL_F) * 15
        interval_mv = tmc_intervals.get(float(interval_s), mv_df) if tmc_intervals else mv_df
        draw_tmc_arrows(frame, interval_mv, mv_df, cx_centre, cy_centre, sx, sy)

        # ── Panels ─────────────────────────────────────────────────────────
        draw_counts_panel(frame, cumulative, OUT_W, OUT_H, turn_counts, len(rows))
        draw_time_bar(frame, fidx, MAX_FRAMES, OUT_W, OUT_H, INTERVAL_F)
        draw_heatmap_legend(frame, hm_data["vmin"], hm_data["vmax"], OUT_W, OUT_H)

        # Intersection centre dot
        cx_s = int(cx_centre * sx); cy_s = int(cy_centre * sy)
        cv2.circle(frame, (cx_s, cy_s), 6, (0,0,255), -1)
        cv2.circle(frame, (cx_s, cy_s), 8, (255,255,255), 1)

        writer.write(frame)
        if fidx % 150 == 0:
            print(f"  frame {fidx}/{MAX_FRAMES}")

    cap.release()
    writer.release()
    print(f"\nDone. Written to: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
