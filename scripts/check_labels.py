"""Consistency check for the OWLv2 labeller on real frames (no ground truth needed).

YOLO-World labels each confident box with a class from ROOM_VOCAB. The same crop is then labelled by OWLv2 over
its own vocabulary (semantics/describe.py). Agreement is counted only for crops whose YOLO-World class is also an
OWLv2 class, so the comparison is fair. This measures agreement between two detectors, not accuracy against truth.

  python scripts/check_labels.py data/before/frames --out outputs/label_check --max-frames 40 --min-conf 0.5
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from detect_objects import ROOM_VOCAB, WORLD_MODEL, frame_paths  # noqa: E402
from semantics.describe import VOCAB, OwlLabeler  # noqa: E402

OWL_NAMES = {v.removeprefix("a ").removeprefix("an ") for v in VOCAB}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "label_check")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLOWorld
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLOWorld(str(WORLD_MODEL))
    yolo.set_classes(ROOM_VOCAB)

    crops = []  # (frame, yolo class, confidence, crop array)
    for f in frame_paths(args.frames, args.max_frames):
        img = Image.open(f).convert("RGB")
        arr = np.asarray(img)
        for b in yolo.predict(str(f), verbose=False, device=device)[0].boxes:
            c = float(b.conf)
            if c < args.min_conf:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy[0].tolist())
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crops.append((f.name, yolo.names[int(b.cls)], c, arr[max(0, y0):y1, max(0, x0):x1]))
    print(f"crops: {len(crops)} from {len(frame_paths(args.frames, args.max_frames))} frames")

    owl = OwlLabeler(device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []
    for frame, yc, conf, crop in crops:
        label, score = owl(np.ascontiguousarray(crop))
        rows.append({"frame": frame, "yolo_world": yc, "yolo_conf": round(conf, 3),
                     "owlv2": label, "owlv2_score": round(score, 3)})

    comparable = [r for r in rows if r["yolo_world"] in OWL_NAMES]
    agree = [r for r in comparable if r["owlv2"] == r["yolo_world"]]
    per_class = Counter(r["yolo_world"] for r in comparable)
    per_class_ok = Counter(r["yolo_world"] for r in agree)
    summary = {
        "crops": len(rows),
        "comparable_crops": len(comparable),
        "agree": len(agree),
        "agreement": round(len(agree) / len(comparable), 3) if comparable else None,
        "per_class": {k: {"n": per_class[k], "agree": per_class_ok[k]} for k in sorted(per_class)},
        "note": "agreement between YOLO-World and OWLv2 on real crops; not accuracy against ground truth",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'labels.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
