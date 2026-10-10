"""Object-level change detection: YOLO boxes lifted to 3D with the MASt3R pointmaps.

For every reconstructed frame, YOLOv8 finds objects (COCO classes). The box's central region is looked up in
that frame's pointmap, giving a 3D position in the capture's frame (metres). Observations of the same class
that lie within --merge m of each other are merged into one object per capture, and objects seen in fewer
than --min-frames frames are dropped. Objects of the same class are then matched across captures (Hungarian
assignment on 3D distance):

  stayed   matched, displacement < --same m
  moved    matched, --same m <= displacement <= --max-move m, displacement reported in metres
  removed  only in the before capture
  added    only in the after capture

  python scripts/detect_objects.py \
      --before-recon outputs/reconstructions/before.npz --before-frames data/before/frames \
      --after-recon outputs/reconstructions/after_aligned.npz --after-frames data/after/frames \
      --out outputs/objects

Null check (run this as well): pass the control capture as the 'after' side. Every object should come out as
'stayed' there; anything else is a false alarm of the object method.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8n.pt"


def frame_paths(frames_dir: Path, max_frames: int) -> list[Path]:
    """Same frame list and subsampling as reconstruct.py, so frame i matches pointmap i."""
    manifest = frames_dir / "manifest.json"
    if manifest.exists():
        files = [frames_dir / f["file"] for f in json.loads(manifest.read_text())["frames"]]
    else:
        files = sorted(frames_dir.glob("frame_*.png"))
    if len(files) > max_frames:
        idx = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        files = [files[i] for i in idx]
    return files


def observations(recon: dict, files: list[Path], model, device, min_conf: float,
                 inner: float = 0.4, conf_q: float = 0.5) -> list[dict]:
    """Lift each confident YOLO box to a 3D point (median of its inner region's confident pointmap points)."""
    H, W = recon["conf"].shape[1:]
    out = []
    for i, f in enumerate(files):
        img_w, img_h = Image.open(f).size
        sx, sy = W / img_w, H / img_h
        conf_map = recon["conf"][i]
        pos = conf_map[conf_map > 0]
        thr = np.quantile(pos, conf_q) if pos.size else np.inf
        res = model.predict(str(f), verbose=False, device=device)[0]
        for b in res.boxes:
            c = float(b.conf)
            if c < min_conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
            hw, hh = (x2 - x1) * sx * inner / 2, (y2 - y1) * sy * inner / 2
            x0, x1i = max(0, int(cx - hw)), min(W, int(cx + hw) + 1)
            y0, y1i = max(0, int(cy - hh)), min(H, int(cy + hh) + 1)
            P = recon["points"][i][y0:y1i, x0:x1i].reshape(-1, 3)
            K = conf_map[y0:y1i, x0:x1i].reshape(-1)
            ok = np.isfinite(P).all(1) & (K >= thr)
            if ok.sum() < 20:
                continue
            out.append({"frame": i, "cls": model.names[int(b.cls)], "conf": c,
                        "xyz": np.median(P[ok], 0), "n": int(ok.sum())})
    return out


def cluster(obs: list[dict], merge_m: float, min_frames: int) -> list[dict]:
    """One object per class per capture: greedy merge of observations within merge_m metres."""
    clusters: list[dict] = []
    for o in sorted(obs, key=lambda o: -o["conf"]):
        best = None
        for c in clusters:
            if c["cls"] != o["cls"]:
                continue
            d = float(np.linalg.norm(o["xyz"] - c["xyz"]))
            if d < merge_m and (best is None or d < best[0]):
                best = (d, c)
        if best is None:
            clusters.append({"cls": o["cls"], "members": [o], "xyz": o["xyz"].copy()})
        else:
            c = best[1]
            c["members"].append(o)
            c["xyz"] = np.median([m["xyz"] for m in c["members"]], axis=0)
    out = []
    for c in clusters:
        frames = {m["frame"] for m in c["members"]}
        if len(frames) >= min_frames:
            out.append({"cls": c["cls"], "xyz": c["xyz"], "frames": len(frames),
                        "conf": float(np.median([m["conf"] for m in c["members"]]))})
    return out


def match(before: list[dict], after: list[dict], same_m: float, max_move_m: float) -> list[dict]:
    res = []
    for cls in sorted({o["cls"] for o in before + after}):
        B = [o for o in before if o["cls"] == cls]
        A = [o for o in after if o["cls"] == cls]
        used_b, used_a = set(), set()
        if B and A:
            D = np.array([[np.linalg.norm(b["xyz"] - a["xyz"]) for a in A] for b in B])
            rows, cols = linear_sum_assignment(D)
            for i, j in zip(rows, cols):
                d = float(D[i, j])
                if d > max_move_m:
                    continue
                used_b.add(i)
                used_a.add(j)
                res.append({"cls": cls, "kind": "stayed" if d < same_m else "moved",
                            "displacement_m": round(d, 3),
                            "before_xyz": np.round(B[i]["xyz"], 3).tolist(),
                            "after_xyz": np.round(A[j]["xyz"], 3).tolist(),
                            "frames_before": B[i]["frames"], "frames_after": A[j]["frames"]})
        for i, b in enumerate(B):
            if i not in used_b:
                res.append({"cls": cls, "kind": "removed", "displacement_m": None,
                            "before_xyz": np.round(b["xyz"], 3).tolist(), "after_xyz": None,
                            "frames_before": b["frames"], "frames_after": 0})
        for j, a in enumerate(A):
            if j not in used_a:
                res.append({"cls": cls, "kind": "added", "displacement_m": None,
                            "before_xyz": None, "after_xyz": np.round(a["xyz"], 3).tolist(),
                            "frames_before": 0, "frames_after": a["frames"]})
    return res


def run_side(recon_path: Path, frames_dir: Path, max_frames: int, model, device, args) -> tuple[list[dict], int]:
    recon = dict(np.load(recon_path))
    files = frame_paths(frames_dir, max_frames)
    if len(files) != len(recon["points"]):
        raise SystemExit(f"{recon_path.name}: {len(recon['points'])} pointmaps but {len(files)} frames; "
                         f"pass the frames used for this reconstruction")
    obs = observations(recon, files, model, device, args.min_conf)
    return cluster(obs, args.merge, args.min_frames), len(obs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-recon", type=Path, required=True)
    ap.add_argument("--before-frames", type=Path, required=True)
    ap.add_argument("--after-recon", type=Path, required=True)
    ap.add_argument("--after-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--min-conf", type=float, default=0.3, help="YOLO confidence cut-off")
    ap.add_argument("--merge", type=float, default=0.3, help="metres: observations closer than this are one object")
    ap.add_argument("--min-frames", type=int, default=3, help="an object must be seen in at least this many frames")
    ap.add_argument("--same", type=float, default=0.3, help="metres: matched objects closer than this have stayed")
    ap.add_argument("--max-move", type=float, default=4.0, help="metres: matches farther than this are not the same object")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))

    before, n_b = run_side(args.before_recon, args.before_frames, args.max_frames, model, device, args)
    after, n_a = run_side(args.after_recon, args.after_frames, args.max_frames, model, device, args)
    changes = match(before, after, args.same, args.max_move)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "objects.json").write_text(json.dumps(changes, indent=2))
    counts = {k: sum(c["kind"] == k for c in changes) for k in ("stayed", "moved", "added", "removed")}
    print(f"objects: before {len(before)} (from {n_b} YOLO hits), after {len(after)} (from {n_a}) -> {counts}")
    for c in changes:
        if c["kind"] == "moved":
            print(f"  moved {c['cls']}: {c['displacement_m']:.2f} m  {c['before_xyz']} -> {c['after_xyz']}")
    print(f"wrote {args.out / 'objects.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
