"""Object-level change detection: YOLO boxes lifted to 3D with the MASt3R pointmaps, checked for visibility.

For every reconstructed frame, YOLOv8 finds objects (COCO classes). The box's central region is looked up in
that frame's pointmap, giving a 3D position in the shared floor frame (metres). Observations of the same class
that lie within --merge m of each other are merged into one object per capture, and objects seen in fewer than
--min-frames frames are dropped. Same-class objects are then matched across captures (Hungarian assignment on
3D distance).

Each unmatched or moved object is then checked against the other capture. Its 3D spot is projected into every
frame of the other capture and compared with that frame's depth at the same pixel:

  empty     the ray reaches the spot and the surface is behind it: nothing is there now
  occupied  a surface sits at or just in front of the spot: something is there now
  unknown   the spot is not in view, or the surface in front hides it

An object counts as removed (or added) only when the spot is 'empty' in at least --min-votes frames and
'occupied' in none. A matched pair counts as moved only when both its old and its new spot are 'empty'.
Everything else is reported as 'unconfirmed' rather than as a change.

  stayed       matched, displacement < --same m
  moved        matched, --same m <= displacement <= --max-move m, both spots confirmed empty
  removed      only in the before capture, spot confirmed empty in the after capture
  added        only in the after capture, spot confirmed empty in the before capture
  unconfirmed  anything the visibility check could not confirm (kept for review, not counted as a change)

  python scripts/detect_objects.py \\
      --before-recon outputs/reconstructions/before.npz --before-frames data/before/frames \\
      --after-recon outputs/reconstructions/after_aligned.npz --after-frames data/after/frames \\
      --out outputs/objects

Null check (run this as well): pass the control capture as the 'after' side. Every object should come out as
'stayed' or 'unconfirmed' there; any 'moved', 'removed' or 'added' is a false alarm of the object method.
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


def depth_maps(recon: dict, conf_q: float = 0.5) -> np.ndarray:
    """Per frame: depth z in that camera of every confident pointmap point (NaN where not confident)."""
    P, C, c2w = recon["points"], recon["conf"], recon["c2w"]
    Z = np.full(C.shape, np.nan, np.float32)
    for i in range(C.shape[0]):
        pos = C[i][C[i] > 0]
        thr = np.quantile(pos, conf_q) if pos.size else np.inf
        ok = np.isfinite(P[i]).all(-1) & (C[i] >= thr)
        w2c = np.linalg.inv(c2w[i])
        Pc = P[i] @ w2c[:3, :3].T + w2c[:3, 3]
        Z[i] = np.where(ok, Pc[..., 2], np.nan)
    return Z


def project(X: np.ndarray, w2c: np.ndarray, focal: float, H: int, W: int):
    """Pixel (u, v) and depth z of world point X in a camera; None if behind the camera or off the image."""
    Xc = w2c[:3, :3] @ X + w2c[:3, 3]
    z = Xc[2]
    if z < 0.1:
        return None
    u = focal * Xc[0] / z + W / 2
    v = focal * Xc[1] / z + H / 2
    if not (0 <= u < W and 0 <= v < H):
        return None
    return int(round(u)), int(round(v)), float(z)


def spot_status(X: np.ndarray, other: dict, Z: np.ndarray, margin: float, radius: float,
                min_votes: int) -> dict:
    """Look at world point X from every frame of the other capture and vote empty / occupied / hidden.

    The ray through X reaches the surface behind it (depth > z + margin): X is empty in that frame.
    A surface within [z - radius, z + margin]: something occupies X. A surface in front of X by more than
    radius: X is hidden in that frame and gives no vote.
    """
    H, W = Z.shape[1:]
    empty = occupied = seen = 0
    for i in range(Z.shape[0]):
        p = project(X, np.linalg.inv(other["c2w"][i]), float(other["focals"][i]), H, W)
        if p is None:
            continue
        u, v, zx = p
        win = Z[i, max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        zs = win[np.isfinite(win)]
        if zs.size == 0:
            continue
        seen += 1
        zo = float(np.median(zs))
        if zo > zx + margin:
            empty += 1
        elif zo >= zx - radius:
            occupied += 1
    if occupied == 0 and empty >= min_votes:
        status = "empty"
    elif occupied > 0:
        status = "occupied"
    else:
        status = "unknown"
    return {"status": status, "empty": empty, "occupied": occupied, "seen": seen}


def pair_objects(before: list[dict], after: list[dict], max_move_m: float):
    """Hungarian match of same-class objects. Returns (pairs, unmatched_before, unmatched_after)."""
    pairs, unmatched_b, unmatched_a = [], [], []
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
                pairs.append((B[i], A[j], d))
        unmatched_b += [b for i, b in enumerate(B) if i not in used_b]
        unmatched_a += [a for j, a in enumerate(A) if j not in used_a]
    return pairs, unmatched_b, unmatched_a


def classify(before: list[dict], after: list[dict], rec_b: dict, rec_a: dict,
             Zb: np.ndarray, Za: np.ndarray, args) -> list[dict]:
    """Turn matched and unmatched objects into stayed / moved / removed / added / unconfirmed records."""
    def check(X, other, Z):
        return spot_status(X, other, Z, args.margin, args.radius, args.min_votes)

    out = []
    pairs, ub, ua = pair_objects(before, after, args.max_move)
    for b, a, d in pairs:
        rec = {"cls": b["cls"], "displacement_m": round(d, 3),
               "vertical_m": round(float(a["xyz"][2] - b["xyz"][2]), 3),
               "before_xyz": np.round(b["xyz"], 3).tolist(), "after_xyz": np.round(a["xyz"], 3).tolist(),
               "frames_before": b["frames"], "frames_after": a["frames"]}
        if d < args.same:
            out.append({**rec, "kind": "stayed"})
            continue
        old_spot = check(b["xyz"], rec_a, Za)   # the old spot should now be empty
        new_spot = check(a["xyz"], rec_b, Zb)   # the new spot should have been empty before
        ok = old_spot["status"] == "empty" and new_spot["status"] == "empty"
        out.append({**rec, "kind": "moved" if ok else "unconfirmed", "side": "matched pair",
                    "old_spot": old_spot, "new_spot": new_spot})
    for b in ub:
        s = check(b["xyz"], rec_a, Za)
        kind = "removed" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": b["cls"], "kind": kind, "side": "before only", "displacement_m": None,
                    "before_xyz": np.round(b["xyz"], 3).tolist(), "after_xyz": None,
                    "frames_before": b["frames"], "frames_after": 0, "spot_in_after": s})
    for a in ua:
        s = check(a["xyz"], rec_b, Zb)
        kind = "added" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": a["cls"], "kind": kind, "side": "after only", "displacement_m": None,
                    "before_xyz": None, "after_xyz": np.round(a["xyz"], 3).tolist(),
                    "frames_before": 0, "frames_after": a["frames"], "spot_in_before": s})
    return out


def side_objects(recon: dict, files: list[Path], model, device, args) -> list[dict]:
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
    ap.add_argument("--margin", type=float, default=0.15, help="metres: depth gap that counts as empty space")
    ap.add_argument("--radius", type=float, default=0.4, help="metres: a surface this far in front of a spot hides it")
    ap.add_argument("--min-votes", type=int, default=2, help="frames that must see a spot as empty")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))

    rec_b = dict(np.load(args.before_recon))
    rec_a = dict(np.load(args.after_recon))
    before, n_b = side_objects(rec_b, frame_paths(args.before_frames, args.max_frames), model, device, args)
    after, n_a = side_objects(rec_a, frame_paths(args.after_frames, args.max_frames), model, device, args)
    Zb, Za = depth_maps(rec_b), depth_maps(rec_a)
    changes = classify(before, after, rec_b, rec_a, Zb, Za, args)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "objects.json").write_text(json.dumps(changes, indent=2))
    kinds = ("stayed", "moved", "removed", "added", "unconfirmed")
    counts = {k: sum(c["kind"] == k for c in changes) for k in kinds}
    print(f"objects: before {len(before)} (from {n_b} YOLO hits), after {len(after)} (from {n_a}) -> {counts}")
    for c in changes:
        if c["kind"] in ("moved", "removed", "added"):
            print(f"  {c['kind']:<8} {c['cls']}: {c['displacement_m'] if c['displacement_m'] is not None else '-'} m")
    print(f"wrote {args.out / 'objects.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
