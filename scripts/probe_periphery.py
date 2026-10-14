"""Measured probe: can any detector see the missed objects at the image periphery?

The three ObChange ground-truth misses (bowl, mug, skillet) sit at the extreme border of the
captured sweep: the bowl's spot projects 1-6 px inside the image border in only 4 of the 60
frames, the mug's in 9. This script answers the only question that matters for recall: do the
pixels there support detection at all?

It runs the pipeline's detector (YOLO-World over ROOM_VOCAB) at low confidence on
  1. the full frames, and
  2. 2x-upscaled border strips (outer 120 px of every side),
and reports every hit whose box touches the outer 80 px of the frame, with its class,
confidence and border distance. If nothing bowl/mug-class fires even there, the misses are not
recoverable by detection and the recall ceiling is a property of the capture, not of the rules.

  python scripts/probe_periphery.py data/datasets/obchange/frames/before \
      data/datasets/obchange/frames/after --out outputs/probe_periphery.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from detect_objects import ROOM_VOCAB, WORLD_MODEL, frame_paths  # noqa: E402

FOCUS = {"bowl", "mug", "cup", "bottle", "plate", "pan"}
BORDER_PX = 80          # a hit is interesting if its box touches this close to the image border
STRIP_PX = 120          # crop width of each border strip
UPSCALE = 2


def border_dist(x1, y1, x2, y2, W, H) -> int:
    """Distance in pixels from the box to the nearest image border (0 = touches)."""
    return int(round(min(x1, y1, W - x2, H - y2)))


def run_side(model, device, files: list[Path]) -> list[dict]:
    import cv2
    hits = []
    for f in files:
        res = model.predict(str(f), verbose=False, device=device)[0]
        W, H = res.orig_shape[1], res.orig_shape[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            d = border_dist(x1, y1, x2, y2, W, H)
            cls = model.names[int(b.cls)]
            if d <= BORDER_PX or cls in FOCUS:
                hits.append({"frame": f.name, "cls": cls, "conf": round(float(b.conf), 3),
                             "border_px": d, "box": [round(v) for v in (x1, y1, x2, y2)], "view": "full"})
        # border strips, upscaled: the detector sees a periphery object at 2x resolution
        for side, (sx1, sy1, sx2, sy2) in {
            "left": (0, 0, STRIP_PX, H), "right": (W - STRIP_PX, 0, W, H),
            "top": (0, 0, W, STRIP_PX), "bottom": (0, H - STRIP_PX, W, H)}.items():
            crop = res.orig_img[sy1:sy2, sx1:sx2]
            big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            r2 = model.predict(big, verbose=False, device=device)[0]
            for b in r2.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1, x2 = x1 / UPSCALE + sx1, x2 / UPSCALE + sx1
                y1, y2 = y1 / UPSCALE + sy1, y2 / UPSCALE + sy1
                d = border_dist(x1, y1, x2, y2, W, H)
                cls = model.names[int(b.cls)]
                if d <= BORDER_PX or cls in FOCUS:
                    hits.append({"frame": f.name, "cls": cls, "conf": round(float(b.conf), 3),
                                 "border_px": d, "box": [round(v) for v in (x1, y1, x2, y2)],
                                 "view": f"strip-{side}-{UPSCALE}x"})
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, nargs="+", help="frame directories (e.g. before and after)")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--min-conf", type=float, default=0.10, help="low cut: this is a visibility probe")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLOWorld
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(str(WORLD_MODEL))
    model.set_classes(ROOM_VOCAB)

    report = {}
    for d in args.frames:
        files = frame_paths(d, args.max_frames)
        hits = [h for h in run_side(model, device, files) if h["conf"] >= args.min_conf]
        report[d.name] = hits
        per_cls: dict[str, int] = {}
        for h in hits:
            per_cls[h["cls"]] = per_cls.get(h["cls"], 0) + 1
        print(f"{d.name}: {len(files)} frames, {len(hits)} border/focus hits at conf >= {args.min_conf}")
        print(f"  by class: {dict(sorted(per_cls.items(), key=lambda kv: -kv[1]))}")
        for cls in sorted(FOCUS):
            n = [h for h in hits if h["cls"] == cls]
            if n:
                print(f"  {cls}: {len(n)} hits, frames {sorted({h['frame'] for h in n})[:12]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
