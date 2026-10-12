"""Consistency check for the OWLv2 labeller on real frames, and accuracy against hand labels when they exist.

YOLO-World labels each confident box with a class from ROOM_VOCAB. The same crop is then labelled by OWLv2 over
its own vocabulary (semantics/describe.py). Agreement is counted only for crops whose YOLO-World class is also an
OWLv2 class, so the comparison is fair. Without `--truth` this measures agreement between two detectors, not
accuracy. With `--truth` (a crop-id to label map written while reviewing `--save-crops` output) it also reports
accuracy of both detectors against those labels (a label is a string or {"dominant": str, "acceptable": [str]}).

  python scripts/check_labels.py data/before/frames --out outputs/label_check --max-frames 40 --min-conf 0.5
  python scripts/check_labels.py data/before/frames --out outputs/label_check --save-crops
  python scripts/check_labels.py data/before/frames --out outputs/label_check --truth outputs/label_check/hand_labels.json
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
# the two vocabularies name the same thing differently; score on normalised labels
SYNONYMS = {"cardboard box": "box", "cup": "mug", "water bottle": "bottle", "potted plant": "plant",
            "television": "tv", "armchair": "chair", "couch": "sofa"}


def norm(label: str) -> str:
    label = label.strip().lower()
    return SYNONYMS.get(label, label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "label_check")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-crops", action="store_true", help="write each crop to <out>/crops/<id>.png for hand labelling")
    ap.add_argument("--truth", type=Path, default=None,
                    help="JSON map of crop id -> hand label; adds accuracy of both detectors against it")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLOWorld
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLOWorld(str(WORLD_MODEL))
    yolo.set_classes(ROOM_VOCAB)

    crops = []  # (crop id, frame, box, yolo class, confidence, crop array)
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
            crops.append((f"{f.stem}_{len(crops):03d}", f.name, [x0, y0, x1, y1], yolo.names[int(b.cls)], c,
                          arr[max(0, y0):y1, max(0, x0):x1]))
    print(f"crops: {len(crops)} from {len(frame_paths(args.frames, args.max_frames))} frames")

    if args.save_crops:
        crop_dir = args.out / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for cid, _, _, _, _, crop in crops:
            Image.fromarray(crop).save(crop_dir / f"{cid}.png")
        print(f"wrote {len(crops)} crops to {crop_dir}")

    owl = OwlLabeler(device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []
    for cid, frame, box, yc, conf, crop in crops:
        label, score = owl(np.ascontiguousarray(crop))
        rows.append({"id": cid, "frame": frame, "bbox": box, "yolo_world": yc, "yolo_conf": round(conf, 3),
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

    if args.truth:
        truth = json.loads(args.truth.read_text())
        # each entry is a label string or {"dominant": str, "acceptable": [str]}; "acceptable" names
        # every object clearly visible in the crop (multi-object crops), "dominant" the main one.
        entry = lambda t: {"dominant": t, "acceptable": [t]} if isinstance(t, str) else t  # noqa: E731
        scored = [r for r in rows if r["id"] in truth]
        for r in rows:
            t = entry(truth[r["id"]]) if r["id"] in truth else None
            r["hand_label"] = t
            if t:
                for key, field in (("yolo_world", "yolo"), ("owlv2", "owl")):
                    lab = norm(r[key])
                    r[f"{field}_ok"] = lab in {norm(a) for a in t["acceptable"]}
                    r[f"{field}_strict_ok"] = lab == norm(t["dominant"])
        yolo_ok = [r for r in scored if r["yolo_ok"]]
        owl_ok = [r for r in scored if r["owl_ok"]]
        yolo_strict = [r for r in scored if r["yolo_strict_ok"]]
        owl_strict = [r for r in scored if r["owl_strict_ok"]]
        summary["hand_labels"] = {
            "labelled": len(scored),
            "unlabelled": len(rows) - len(scored),
            "yolo_world_correct": len(yolo_ok),
            "owlv2_correct": len(owl_ok),
            "yolo_world_accuracy": round(len(yolo_ok) / len(scored), 3) if scored else None,
            "owlv2_accuracy": round(len(owl_ok) / len(scored), 3) if scored else None,
            "yolo_world_strict_accuracy": round(len(yolo_strict) / len(scored), 3) if scored else None,
            "owlv2_strict_accuracy": round(len(owl_strict) / len(scored), 3) if scored else None,
            "note": ("accuracy against hand labels written while reviewing crops; correct = label names any "
                     "object clearly visible in the crop (acceptable set), strict = label names the dominant one. "
                     "Synonyms across the two vocabularies are normalised (cardboard box=box, cup=mug). "
                     "The crops are YOLO-World proposals, so this scores label accuracy on proposed boxes, "
                     "not detection recall. The labeller is the agent, so this is not independent ground truth"),
        }
        summary["note"] = "includes accuracy against hand labels (see hand_labels)"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'labels.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
