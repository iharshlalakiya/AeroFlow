"""
Level 1: Detection & Tracking using a VisDrone-pretrained YOLOv8s model,
better suited to aerial/top-down imagery than stock COCO weights.

Usage:
    python src/track_visdrone.py --video data/raw_video/Intersection_Merged.MP4 --out data/output/intersection_tracks_visdrone.csv
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

# VisDrone class id -> our required taxonomy (used as a secondary "taxonomy" column;
# nothing is filtered out at detection time, every VisDrone class is tracked and kept)
VISDRONE_TO_TAXONOMY = {
    0: "pedestrian",        # pedestrian
    1: "pedestrian",        # people (crowd/group)
    2: "motorcycle",        # bicycle (closest two-wheeler bucket)
    3: "car",                # car
    4: "LGV",                # van -> light goods vehicle
    5: "truck",              # truck
    6: "motorcycle",        # tricycle (closest two-wheeler bucket)
    7: "motorcycle",        # awning-tricycle
    8: "bus",                # bus
    9: "motorcycle",        # motor
}


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--tracker", default="configs/botsort_aerial.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default=0)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main():
    args = build_arg_parser().parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "raw_class", "taxonomy_class", "conf",
                          "x1", "y1", "x2", "y2", "cx", "cy"])

        results = model.track(
            source=str(video_path),
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            persist=True,
            stream=True,
            save=args.save_video,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                raw_label = model.names[int(cls)]
                taxonomy_label = VISDRONE_TO_TAXONOMY.get(int(cls), "unknown")
                writer.writerow([frame_idx, tid, raw_label, taxonomy_label, round(float(conf), 4),
                                  round(float(x1), 1), round(float(y1), 1),
                                  round(float(x2), 1), round(float(y2), 1),
                                  round(float(cx), 1), round(float(cy), 1)])

            if frame_idx % 100 == 0:
                print(f"frame {frame_idx} processed")

    print(f"Done. Trajectories written to {out_path}")


if __name__ == "__main__":
    main()
