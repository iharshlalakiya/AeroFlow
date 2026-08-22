"""
Level 1: Detection & Tracking
Runs YOLO detection + ByteTrack/BoT-SORT tracking on drone footage,
maps classes to the required taxonomy, and writes per-frame trajectories.

Usage:
    python src/track.py --video data/raw_video/Intersection_Merged.MP4 --out data/output/intersection_tracks.csv
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

# COCO class id -> our required taxonomy
COCO_TO_TAXONOMY = {
    0: "pedestrian",
    1: "motorcycle",
    2: "car",
    3: "motorcycle",   # motorbike
    5: "bus",
    7: "truck",        # further split car/LGV/HGV/truck needs a fine-tuned model
}

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out", required=True, help="Path to output CSV of trajectories")
    p.add_argument("--model", default="yolov8s.pt", help="YOLO checkpoint")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (bytetrack.yaml or botsort.yaml)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--device", default=0, help="CUDA device id or 'cpu'")
    p.add_argument("--save-video", action="store_true", help="Save annotated output video")
    return p


def main():
    args = build_arg_parser().parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy"])

        results = model.track(
            source=str(video_path),
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            classes=list(COCO_TO_TAXONOMY.keys()),
            persist=True,
            stream=True,
            save=args.save_video,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                label = COCO_TO_TAXONOMY.get(cls, "unknown")
                writer.writerow([frame_idx, tid, label, round(float(conf), 4),
                                  round(float(x1), 1), round(float(y1), 1),
                                  round(float(x2), 1), round(float(y2), 1),
                                  round(float(cx), 1), round(float(cy), 1)])

            if frame_idx % 100 == 0:
                print(f"frame {frame_idx} processed")

    print(f"Done. Trajectories written to {out_path}")


if __name__ == "__main__":
    main()
