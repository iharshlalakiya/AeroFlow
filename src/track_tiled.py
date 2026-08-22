"""
Tiled (SAHI-style) detection + tracking, restricted to a road mask.

Two problems this solves for oblique wide-angle footage:

* Downscaling a 4K frame to 1280 px destroys small distant vehicles. Running the detector on
  native-resolution tiles instead preserves them.
* Tiling a whole 4K frame is far too expensive on a small GPU. But road users only occupy the
  road, so we tile ONLY the tiles that overlap the road mask - typically a small fraction of
  the frame. That is what makes tiled inference affordable here.

Detections are merged across tile seams with NMS, filtered to the mask, and fed to BoT-SORT.

Usage:
    python src/track_tiled.py --video data/raw_video/Multi_Road_Merged_convert_4k.mp4 \
        --mask data/output/multiroad_mask.png \
        --out data/output/multiroad_tiled.csv --max-frames 1800
"""
import argparse
import csv
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from torchvision.ops import batched_nms
from ultralytics.trackers.bot_sort import BOTSORT

VISDRONE_TO_TAXONOMY = {
    0: "pedestrian", 3: "car", 4: "LGV", 5: "truck", 8: "bus", 9: "motorcycle",
}


def plan_tiles(mask, tile, overlap):
    """Grid the mask bounding box into tiles, keeping only those containing road."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise SystemExit("Mask is empty.")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    step = int(tile * (1 - overlap))
    h, w = mask.shape

    tiles = []
    for ty in range(y0, y1, step):
        for tx in range(x0, x1, step):
            tx2, ty2 = min(tx + tile, w), min(ty + tile, h)
            tx1, ty1 = max(0, tx2 - tile), max(0, ty2 - tile)
            if mask[ty1:ty2, tx1:tx2].any():
                tiles.append((tx1, ty1, tx2, ty2))
    # de-duplicate tiles clamped onto the same region
    return sorted(set(tiles))


def load_tracker_cfg(path, frame_rate, device):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("gmc_method", "none")
    cfg.setdefault("device", device)
    cfg.setdefault("frame_rate", frame_rate)
    return BOTSORT(SimpleNamespace(**cfg))


def detect_frame(model, frame, tiles, conf, iou, device, batch):
    """Run the detector over the road tiles and return full-frame [x1,y1,x2,y2,conf,cls]."""
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        crops = [frame[ty1:ty2, tx1:tx2] for tx1, ty1, tx2, ty2 in chunk]
        results = model.predict(crops, conf=conf, device=device, verbose=False,
                                classes=list(VISDRONE_TO_TAXONOMY.keys()))
        for (tx1, ty1, _, _), r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            b[:, [0, 2]] += tx1
            b[:, [1, 3]] += ty1
            dets.append(np.concatenate(
                [b, r.boxes.conf.cpu().numpy()[:, None], r.boxes.cls.cpu().numpy()[:, None]], 1))

    if not dets:
        return np.zeros((0, 6), np.float32)
    dets = np.concatenate(dets, 0)

    # merge duplicates from overlapping tiles (class-aware NMS)
    t = torch.from_numpy(dets).float()
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou)
    return t[keep].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--tracker", default="configs/botsort_visdrone.yaml")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default=0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"Could not read mask {args.mask}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    tiles = plan_tiles(mask, args.tile, args.overlap)
    full_grid = int(np.ceil(w / (args.tile * (1 - args.overlap)))) * \
                int(np.ceil(h / (args.tile * (1 - args.overlap))))
    print(f"{len(tiles)} road tiles of {args.tile}px "
          f"(a full-frame grid would be ~{full_grid} - {len(tiles)/full_grid*100:.0f}% of the cost)")

    model = YOLO(args.model)
    names = model.names
    tracker = load_tracker_cfg(args.tracker, fps, args.device)

    limit = min(args.max_frames or total, total)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "raw_class", "taxonomy_class", "conf",
                          "x1", "y1", "x2", "y2", "cx", "cy"])

        for frame_idx in range(limit):
            ok, frame = cap.read()
            if not ok:
                break

            dets = detect_frame(model, frame, tiles, args.conf, args.iou, args.device, args.batch)

            # keep only detections whose centre sits on the road
            if len(dets):
                cx = np.clip(((dets[:, 0] + dets[:, 2]) / 2).astype(int), 0, w - 1)
                cy = np.clip(((dets[:, 1] + dets[:, 3]) / 2).astype(int), 0, h - 1)
                dets = dets[mask[cy, cx] > 0]

            data = torch.from_numpy(np.concatenate(
                [dets[:, :4], dets[:, 4:5], dets[:, 5:6]], 1).astype(np.float32)) \
                if len(dets) else torch.zeros((0, 6))
            tracks = tracker.update(Boxes(data, (h, w)), frame)

            for t in tracks:
                x1, y1, x2, y2, tid, conf, cls = t[0], t[1], t[2], t[3], int(t[4]), t[5], int(t[6])
                writer.writerow([frame_idx, tid, names[cls],
                                  VISDRONE_TO_TAXONOMY.get(cls, "unknown"), round(float(conf), 4),
                                  round(float(x1), 1), round(float(y1), 1),
                                  round(float(x2), 1), round(float(y2), 1),
                                  round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1)])

            if frame_idx % 100 == 0:
                print(f"frame {frame_idx}/{limit}")

    cap.release()
    print(f"Done. Trajectories written to {args.out}")


if __name__ == "__main__":
    main()
