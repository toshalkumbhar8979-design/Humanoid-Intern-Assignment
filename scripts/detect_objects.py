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
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "models" / "yolov8n.pt"
WORLD_MODEL = ROOT / "models" / "yolov8s-worldv2.pt"
# Open-vocabulary prompts for YOLO-World: room objects that COCO does not name (desk, mug, pillow, book, ...).
# Chosen before looking at the results, from typical desk and living-room objects.
ROOM_VOCAB = ["chair", "table", "desk", "backpack", "bag", "suitcase", "box", "cup", "mug", "bottle",
              "laptop", "monitor", "keyboard", "mouse", "book", "phone", "plant", "lamp", "pillow", "shoe",
              "bowl", "toy", "remote", "speaker", "printer", "person", "bed", "sofa", "tv", "refrigerator"]


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
                 inner: float = 0.4, conf_q: float = 0.5, lift: str = "base") -> list[dict]:
    """Lift each confident YOLO box to a 3D point (median of confident pointmap points in a window).

    lift="base" samples the bottom band of the box (the object's support contact), which stays at the
    same height when the viewpoint changes; lift="inner" samples the box centre, which drifts between
    views on large objects (a bed's centre is mattress in one view and headboard in another)."""
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
            if lift == "base":
                y0, y1i = max(0, int(y2 * sy - 2 * hh)), min(H, int(y2 * sy) + 1)
            else:
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


def vote_status(empty: int, occupied: int, seen: int, min_votes: int, min_seen: int = 2) -> str:
    """empty / occupied / unknown from per-frame votes.

    A status needs `min_seen` frames that saw the spot and `min_votes` of them agreeing, so no single
    frame decides: one occupied vote among empties is 'unknown', not 'occupied'. Mixed evidence is
    'unknown' too. Hidden frames give no vote and are not counted in `seen`.
    """
    if seen < min_seen:
        return "unknown"
    if occupied == 0 and empty >= min_votes:
        return "empty"
    if empty == 0 and occupied >= min_votes:
        return "occupied"
    return "unknown"


def spot_status(X: np.ndarray, other: dict, Z: np.ndarray, margin: float, radius: float,
                min_votes: int, min_seen: int = 2) -> dict:
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
    status = vote_status(empty, occupied, seen, min_votes, min_seen)
    return {"status": status, "empty": empty, "occupied": occupied, "seen": seen}


def footprint_status(X: np.ndarray, other: dict, Z: np.ndarray, args) -> dict:
    """Vote on the object's footprint: its centre and 8 points on a ring at the same height.

    A spot is 'empty' when at least two thirds of the sample points that get a vote are empty, and 'occupied'
    when at least one third are occupied. This is the footprint test that replaces the single-point test: a
    centre point that lands on a support or a wall no longer decides the answer on its own.
    """
    samples = [X] + [X + np.array([args.footprint * np.cos(t), args.footprint * np.sin(t), 0.0])
                     for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    votes = [spot_status(s, other, Z, args.margin, args.radius, args.min_votes, args.min_seen)["status"]
             for s in samples]
    n_empty, n_occ = votes.count("empty"), votes.count("occupied")
    voting = n_empty + n_occ
    if voting < 3:
        status = "unknown"
    elif n_occ * 3 >= voting:
        status = "occupied"
    elif n_empty * 3 >= 2 * voting:
        status = "empty"
    else:
        status = "unknown"
    return {"status": status, "empty_samples": n_empty, "occupied_samples": n_occ,
            "unknown_samples": len(samples) - voting, "samples": len(samples)}


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


def vertical_reason(vertical_m: float, max_vertical: float) -> str | None:
    """Reason a candidate move is rejected for a vertical jump, or None if the jump is allowed.

    The floor frame has z up and objects in these rooms stand on the floor, so a real move on the floor
    changes x and y, not height. A jump above --max-vertical is a mismatch of the object's 3D centre.
    """
    if abs(vertical_m) <= max_vertical:
        return None
    return f"vertical jump {vertical_m:+.2f} m (floor objects do not rise; likely a centre mismatch)"


def to_floor_xyz(xyz, R: np.ndarray, shift: float) -> np.ndarray:
    """Map a recon-frame point into the floor frame (fitted floor at z = 0, z up).

    MASt3R's frame is arbitrary (in the real capture 'up' is mostly along raw -z), so height
    comparisons and reported positions must use this frame, not the recon frame.
    """
    q = R @ np.asarray(xyz, dtype=float)
    q[2] += shift
    return q


def classify(before: list[dict], after: list[dict], rec_b: dict, rec_a: dict,
             Zb: np.ndarray, Za: np.ndarray, args, to_floor=None) -> list[dict]:
    """Turn matched and unmatched objects into stayed / moved / removed / added / unconfirmed records.

    Positions and vertical_m are reported in the floor frame (pass `to_floor`); the footprint checks
    stay in the recon frame, where the camera poses live.
    """
    def check(X, other, Z):
        return footprint_status(X, other, Z, args)

    def floor(xyz):
        return np.asarray(xyz) if to_floor is None else to_floor(xyz)

    out = []
    pairs, ub, ua = pair_objects(before, after, args.max_move)
    for b, a, d in pairs:
        fb, fa = floor(b["xyz"]), floor(a["xyz"])
        rec = {"cls": b["cls"], "displacement_m": round(d, 3),
               "vertical_m": round(float(fa[2] - fb[2]), 3),
               "before_xyz": np.round(fb, 3).tolist(), "after_xyz": np.round(fa, 3).tolist(),
               "frames_before": b["frames"], "frames_after": a["frames"]}
        if d < args.same:
            out.append({**rec, "kind": "stayed"})
            continue
        old_spot = check(b["xyz"], rec_a, Za)   # the old spot should now be empty
        new_spot = check(a["xyz"], rec_b, Zb)   # the new spot should have been empty before
        ok = old_spot["status"] == "empty" and new_spot["status"] == "empty"
        reason = vertical_reason(rec["vertical_m"], args.max_vertical)
        confirmed = ok and reason is None
        out.append({**rec, "kind": "moved" if confirmed else "unconfirmed", "side": "matched pair",
                    "old_spot": old_spot, "new_spot": new_spot, "reason": reason})
    for b in ub:
        s = check(b["xyz"], rec_a, Za)
        kind = "removed" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": b["cls"], "kind": kind, "side": "before only", "displacement_m": None,
                    "before_xyz": np.round(floor(b["xyz"]), 3).tolist(), "after_xyz": None,
                    "frames_before": b["frames"], "frames_after": 0, "spot_in_after": s})
    for a in ua:
        s = check(a["xyz"], rec_b, Zb)
        kind = "added" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": a["cls"], "kind": kind, "side": "after only", "displacement_m": None,
                    "before_xyz": None, "after_xyz": np.round(floor(a["xyz"]), 3).tolist(),
                    "frames_before": 0, "frames_after": a["frames"], "spot_in_before": s})
    return out


def describe_objects(changes: list[dict]) -> str:
    """Plain-language summary of objects.json, with positions in the floor frame (metres)."""
    lines = ["# Object changes", "",
             "Positions are metres in the floor frame (x, y, z up). Labels are YOLO class names; "
             "no colour or appearance is used.", ""]
    for kind in ("moved", "removed", "added"):
        for c in (c for c in changes if c["kind"] == kind):
            if kind == "moved":
                lines.append(f"- moved {c['cls']}: {c['displacement_m']:.2f} m "
                             f"(vertical {c['vertical_m']:+.2f} m), from {c['before_xyz']} to {c['after_xyz']}.")
            elif kind == "removed":
                lines.append(f"- removed {c['cls']}: last seen at {c['before_xyz']}; the spot is empty in the after capture.")
            else:
                lines.append(f"- added {c['cls']}: seen at {c['after_xyz']}; the spot is empty in the before capture.")
    n_stay = sum(c["kind"] == "stayed" for c in changes)
    lines += ["", f"Stayed (matched, under the stay threshold): {n_stay}.", "", "## Unconfirmed", "",
              "These could not be confirmed as changes. They are listed for review only.", ""]
    for c in (c for c in changes if c["kind"] == "unconfirmed"):
        if c["side"] == "matched pair":
            why = f"candidate move, old spot {c['old_spot']['status']}, new spot {c['new_spot']['status']}"
            if c.get("reason"):
                why += f"; {c['reason']}"
        else:
            key = "spot_in_after" if c["side"] == "before only" else "spot_in_before"
            why = f"spot {c[key]['status']} in the other capture"
        lines.append(f"- {c['cls']} ({c['side']}): {why}.")
    return "\n".join(lines) + "\n"


def side_objects(recon: dict, files: list[Path], model, device, args) -> list[dict]:
    obs = observations(recon, files, model, device, args.min_conf, lift=args.lift)
    return cluster(obs, args.merge, args.min_frames), len(obs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-recon", type=Path, required=True)
    ap.add_argument("--before-frames", type=Path, required=True)
    ap.add_argument("--after-recon", type=Path, required=True)
    ap.add_argument("--after-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", choices=["yolo", "yolo-world"], default="yolo",
                    help="yolo: COCO classes (80). yolo-world: open vocabulary, prompts in ROOM_VOCAB")
    ap.add_argument("--model", type=Path, default=None,
                    help="weights; default models/yolov8n.pt, or models/yolov8s-worldv2.pt for yolo-world")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--min-conf", type=float, default=0.3, help="YOLO confidence cut-off")
    ap.add_argument("--merge", type=float, default=0.3, help="metres: observations closer than this are one object")
    ap.add_argument("--min-frames", type=int, default=3, help="an object must be seen in at least this many frames")
    ap.add_argument("--same", type=float, default=0.3, help="metres: matched objects closer than this have stayed")
    ap.add_argument("--max-move", type=float, default=4.0, help="metres: matches farther than this are not the same object")
    ap.add_argument("--margin", type=float, default=0.15, help="metres: depth gap that counts as empty space")
    ap.add_argument("--radius", type=float, default=0.4, help="metres: a surface this far in front of a spot hides it")
    ap.add_argument("--min-votes", type=int, default=2, help="frames that must agree on a spot's status")
    ap.add_argument("--min-seen", type=int, default=2, help="frames that must see a spot before its status counts")
    ap.add_argument("--lift", choices=["base", "inner"], default="base",
                    help="3D point for a box: base = bottom band (view-stable height), inner = box centre")
    ap.add_argument("--footprint", type=float, default=0.15, help="metres: radius of the footprint ring around each object centre")
    ap.add_argument("--max-vertical", type=float, default=0.15,
                    help="metres: a candidate move with a larger vertical jump is unconfirmed (floor objects do not rise)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO, YOLOWorld
    device = 0 if torch.cuda.is_available() else "cpu"
    if args.detector == "yolo-world":
        model = YOLOWorld(str(args.model or WORLD_MODEL))
        model.set_classes(ROOM_VOCAB)
    else:
        model = YOLO(str(args.model or MODEL))

    rec_b = dict(np.load(args.before_recon))
    rec_a = dict(np.load(args.after_recon))
    before, n_b = side_objects(rec_b, frame_paths(args.before_frames, args.max_frames), model, device, args)
    after, n_a = side_objects(rec_a, frame_paths(args.after_frames, args.max_frames), model, device, args)
    Zb, Za = depth_maps(rec_b), depth_maps(rec_a)
    # report in the fitted floor frame: MASt3R's frame has an arbitrary up, so height rules and
    # published positions must be gravity-aligned (same frame as bev.png and evaluate_changes --frame floor)
    from registration.align import surface_points
    from visualization.plots import floor_transform
    pts_b, _ = surface_points(rec_b)
    R, shift = floor_transform(pts_b, rec_b["c2w"][0])
    changes = classify(before, after, rec_b, rec_a, Zb, Za, args,
                       to_floor=lambda xyz: to_floor_xyz(xyz, R, shift))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "objects.json").write_text(json.dumps(changes, indent=2))
    (args.out / "objects.md").write_text(describe_objects(changes))
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
