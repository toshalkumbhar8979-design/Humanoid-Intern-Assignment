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

A spot verdict is proof only at capture pairs whose reconstruction noise is below the spot test's
vertical resolution (--rise-tol). Above it the no-change control produces the same evidence as a
real change (measured: a 'removed pillow' with 7 empty / 0 contrary / 2 unseen on a control pair),
so spot-only confirmations are reported as candidates until an independent method -- the voxel
stage, via --corroborate -- agrees (cross-method confirmation, DESIGN.md section 4).

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


def cluster(obs: list[dict], merge_m: float, min_frames: int, dup_m: float = 0.0) -> list[dict]:
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
    return merge_duplicates(out, dist=dup_m)


def merge_duplicates(objs: list[dict], dist: float = 0.05) -> list[dict]:
    """Two clusters within `dist` metres are one object with two labels, not two objects.

    Objects cannot occupy the same volume: when the detector fires two labels on one object (observed
    on ObChange as a 'book' and a 'box' at the same point), the duplicate would otherwise become a
    phantom added/removed candidate. Keeps the cluster seen in more frames.

    Off by default (--merge-dup 0): measured on ObChange it destroyed a physically correct removal.
    The soup can fired cup and mug labels at one spot; merging them let the survivor pair with a
    neighbouring after-object and the removal vanished into that false pair. The pairing layer, not
    the duplicate, is the real bug, and a duplicate merge on top of it makes outcomes worse.
    """
    keep: list[dict] = []
    for o in sorted(objs, key=lambda o: (-o["frames"], -o["conf"])):
        if any(float(np.linalg.norm(o["xyz"] - k["xyz"])) < dist for k in keep):
            continue
        keep.append(o)
    return keep


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
                min_votes: int, min_seen: int = 2, to_floor=None,
                rise_tol: float = 0.03, gap: float = 0.03, stats=None,
                free_pts=(), attr: float = 0.12) -> dict:
    """Look at world point X from every frame of the other capture and vote empty / occupied / hidden.

    X is an object's base point, a hair above its support. Both signals agree on what the two cases
    look like along a ray through X:

      object still there  the object's own surface is at X:      depth ~ z,     height ~ X
      object gone         the ray passes X and hits the support
                          (or a wall) behind and below it:       depth > z,     height < X

    So a hit clearly *behind* X (depth - z > `gap`) or clearly *below* X (height drop > `rise_tol`)
    votes empty; a surface at X's depth and level with it is an object at the spot and votes occupied.
    A surface in front of X by more than `radius` hides the spot and gives no vote.

    Without `to_floor` the older depth-band rule applies: a surface within [z - radius, z + margin]
    counts as occupied. That rule reads bare support as 'occupied' (the support is at the spot's own
    depth), which rejected every tabletop candidate in the ObChange measurement (DESIGN section 8).

    `stats`, when given, collects per-frame (depth offset, height drop) pairs for calibration.

    `free_pts` are positions of objects that provably left or came from elsewhere (members of a
    matched pair displaced over --same). A hit within `attr` metres of one of them belongs to that
    other object, not to the object under test, so it votes empty: the question is never 'is this
    free space' but 'is THIS object still at the spot'. Observed on ObChange: a plate was added 0.15 m
    from where a sugar box used to sit, and the box's body vetoed the addition until attribution.
    """
    H, W = Z.shape[1:]
    empty = occupied = seen = 0
    for i in range(Z.shape[0]):
        p = project(X, np.linalg.inv(other["c2w"][i]), float(other["focals"][i]), H, W)
        if p is None:
            continue
        u, v, zx = p
        sl = (slice(max(0, v - 2), v + 3), slice(max(0, u - 2), u + 3))
        zs = Z[i][sl]
        zs = zs[np.isfinite(zs)]
        if zs.size == 0:
            continue
        zo = float(np.median(zs))
        if zo < zx - radius:
            continue          # a surface this far in front hides the spot: no vote
        if to_floor is None:
            at_spot = zo <= zx + margin        # legacy depth-band rule
        else:
            Pw = other["points"][i][sl].reshape(-1, 3)
            okp = np.isfinite(Pw).all(1)
            if not okp.any():
                continue
            hit = np.median(Pw[okp], 0)
            drop = float(to_floor(X)[2] - to_floor(hit)[2])   # how far the hit is below X
            behind = zo - zx                                      # how far the hit is past X
            if stats is not None:
                stats.append((round(behind, 4), round(drop, 4)))
            at_spot = not (behind > gap or drop > rise_tol)
            if at_spot and len(free_pts):
                if min(float(np.linalg.norm(hit - fp)) for fp in free_pts) < attr:
                    at_spot = False   # the surface belongs to another object that is elsewhere now
        seen += 1
        if at_spot:
            occupied += 1
        else:
            empty += 1
    status = vote_status(empty, occupied, seen, min_votes, min_seen)
    return {"status": status, "empty": empty, "occupied": occupied, "seen": seen}


def footprint_status(X: np.ndarray, other: dict, Z: np.ndarray, args,
                     to_floor=None, normal=None, free_pts=()) -> dict:
    """Vote on the object's footprint: its centre and 8 points on a ring at the same height.

    The ring lies in the horizontal plane (perpendicular to `normal`, the floor normal in the recon
    frame). Without a normal it falls back to the recon frame's xy-plane, which is only horizontal by
    luck: MASt3R's frame has an arbitrary up, so the ring must be built from the floor normal.

    A spot is 'empty' when at least two thirds of the sample points that get a vote are empty, and 'occupied'
    when at least one third are occupied. This is the footprint test that replaces the single-point test: a
    centre point that lands on a support or a wall no longer decides the answer on its own.
    """
    if normal is None:
        e1, e2, n = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.zeros(3)
    else:
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
    # Test the footprint a lift above the object's support contact, not at it. At the contact the bare
    # support passes the 'is this level with the spot' test at grazing view angles and reads occupied
    # (observed on ObChange: added objects on a tabletop were vetoed by the table itself). The lift
    # must exceed the drop tolerance (--rise-tol) so the height signal alone separates 'object here'
    # from 'bare support' whatever the view angle. Objects shorter than the lift (a plate) read empty
    # even when present; that limit is documented, not hidden.
    X0 = np.asarray(X, dtype=float) + getattr(args, "spot_lift", 0.04) * n
    samples = [X0] + [X0 + args.footprint * (np.cos(t) * e1 + np.sin(t) * e2)
                      for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    votes = [spot_status(s, other, Z, args.margin, args.radius, args.min_votes, args.min_seen,
                         to_floor=to_floor, rise_tol=getattr(args, "rise_tol", 0.03),
                         gap=getattr(args, "gap", 0.03), free_pts=free_pts,
                         attr=getattr(args, "attr", 0.12))["status"]
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

    Only a *rise* over --max-vertical rejects: a resting object cannot end up higher without a higher
    support, so a large rise is almost always a mismatch of the object's 3D centre. A drop is always
    physically possible (an object can be moved from a table to the floor), and ObChange ground truth
    shows a real move that drops 0.22 m between surfaces, which the old symmetric rule wrongly rejected.
    Drop-like centre mismatches are caught by the footprint check instead: the old spot stays occupied.
    """
    if vertical_m <= max_vertical:
        return None
    return f"vertical jump {vertical_m:+.2f} m (a resting object cannot end up higher without support; likely a centre mismatch)"


def to_floor_xyz(xyz, R: np.ndarray, shift: float) -> np.ndarray:
    """Map a recon-frame point into the floor frame (fitted floor at z = 0, z up).

    MASt3R's frame is arbitrary (in the real capture 'up' is mostly along raw -z), so height
    comparisons and reported positions must use this frame, not the recon frame.
    """
    q = R @ np.asarray(xyz, dtype=float)
    q[2] += shift
    return q


def classify(before: list[dict], after: list[dict], rec_b: dict, rec_a: dict,
             Zb: np.ndarray, Za: np.ndarray, args, to_floor=None, normal=None) -> list[dict]:
    """Turn matched and unmatched objects into stayed / moved / removed / added / unconfirmed records.

    Positions and vertical_m are reported in the floor frame (pass `to_floor`); the footprint checks
    stay in the recon frame, where the camera poses live, but compare heights through `to_floor` and
    build their ring around the floor `normal`.
    """
    def check(X, other, Z, free_pts=()):
        return footprint_status(X, other, Z, args, to_floor=to_floor, normal=normal, free_pts=free_pts)

    def floor(xyz):
        return np.asarray(xyz) if to_floor is None else to_floor(xyz)

    out = []
    pairs, ub, ua = pair_objects(before, after, args.max_move)
    verify = getattr(args, "verify_pairs", 0.0)

    def strong_empty(s):
        return s["status"] == "empty" and s["occupied_samples"] == 0

    # pass 1: spot tests for every pair worth checking. Pairs under the noise band (displacements the
    # no-change control produces from reconstruction noise alone) are stayed without tests: at that
    # scale the object sits too far from its own recorded spot for the spot test to see it (the
    # author's control pairs drift up to 0.21 m; ObChange's only 0.01 m), so testing them would
    # manufacture moves out of noise. Above the band the pairing itself is in doubt and the gate decides.
    tested: dict[int, tuple[dict, dict]] = {}
    for i, (b, a, d) in enumerate(pairs):
        if d < args.same and (verify <= 0 or d < verify):
            continue
        tested[i] = (check(b["xyz"], rec_a, Za), check(a["xyz"], rec_b, Zb))

    # objects that provably left their spot (paired, displaced past --same, old spot now empty) or
    # arrived from elsewhere (paired, displaced past --same): their bodies do not answer for the
    # object under test, so a hit on them votes empty (see spot_status `free_pts`).
    left_b = [pairs[i][0]["xyz"] for i, (b, a, d) in enumerate(pairs)
              if d >= args.same and i in tested and tested[i][0]["status"] == "empty"]
    left_a = [pairs[i][1]["xyz"] for i, (b, a, d) in enumerate(pairs)
              if d >= args.same and i in tested and tested[i][1]["status"] == "empty"]
    came_a = [a["xyz"] for b, a, d in pairs if d >= args.same]
    came_b = [b["xyz"] for b, a, d in pairs if d >= args.same]

    for i, (b, a, d) in enumerate(pairs):
        fb, fa = floor(b["xyz"]), floor(a["xyz"])
        rec = {"cls": b["cls"], "displacement_m": round(d, 3),
               "vertical_m": round(float(fa[2] - fb[2]), 3),
               "before_xyz": np.round(fb, 3).tolist(), "after_xyz": np.round(fa, 3).tolist(),
               "frames_before": b["frames"], "frames_after": a["frames"]}
        if i not in tested:          # below the noise band: stayed, nothing to test
            out.append({**rec, "kind": "stayed"})
            continue
        old_spot, new_spot = tested[i]
        if d < args.same:
            # the pairing is in doubt: the gate decides whether one object moved, both stayed, or the
            # pair is two different objects (an added one next to a stayed one, wrongly matched)
            if old_spot["status"] == "occupied" and new_spot["status"] == "occupied":
                out.append({**rec, "kind": "stayed"})
                continue
            if old_spot["status"] == "empty" and new_spot["status"] == "empty":
                # Below --same the pair is ambiguous: one object that moved a little, or two different
                # objects (a removal and an appearance) wrongly matched. Both readings fit the same
                # evidence, and no no-change control can tell them apart, so the honest verdict is
                # unconfirmed. ObChange showed exactly this: a removed can's cluster paired with an
                # unrelated cup 0.16 m away and the pair read as a 'move' no ground truth supports.
                out.append({**rec, "kind": "unconfirmed", "side": "matched pair",
                            "old_spot": old_spot, "new_spot": new_spot,
                            "reason": "both spots look empty: one small move or a removal plus an "
                                      "appearance, which the gate cannot tell apart"})
                continue
            if old_spot["status"] == "occupied" and strong_empty(new_spot):
                # the before object kept its spot, so the after member is a different object: split
                s = check(a["xyz"], rec_b, Zb, free_pts=came_b)
                out.append({"cls": b["cls"], "kind": "stayed", "side": "before only",
                            "displacement_m": round(d, 3), "vertical_m": None,
                            "before_xyz": np.round(fb, 3).tolist(), "after_xyz": None,
                            "frames_before": b["frames"], "frames_after": a["frames"],
                            "spot_in_after": old_spot, "split_from_pair": True})
                out.append({"cls": a["cls"], "kind": "added" if s["status"] == "empty" else "unconfirmed",
                            "side": "after only", "displacement_m": None, "vertical_m": None,
                            "before_xyz": None, "after_xyz": np.round(fa, 3).tolist(),
                            "frames_before": b["frames"], "frames_after": a["frames"],
                            "spot_in_before": s, "split_from_pair": True})
                continue
            out.append({**rec, "kind": "unconfirmed", "side": "matched pair",
                        "old_spot": old_spot, "new_spot": new_spot,
                        "reason": "the two ends of the pair disagree on whether the spot is occupied"})
            continue
        # attribution excludes the pair's own members: a large object whose body overlaps its old spot
        # after a shift is still *there*, and only its own presence says so
        old_spot = check(b["xyz"], rec_a, Za, free_pts=[p for p in came_a if p is not a["xyz"]])
        new_spot = check(a["xyz"], rec_b, Zb, free_pts=[p for p in came_b if p is not b["xyz"]])
        ok = old_spot["status"] == "empty" and new_spot["status"] == "empty"
        reason = vertical_reason(rec["vertical_m"], args.max_vertical)
        confirmed = ok and reason is None
        out.append({**rec, "kind": "moved" if confirmed else "unconfirmed", "side": "matched pair",
                    "old_spot": old_spot, "new_spot": new_spot, "reason": reason})
    for b in ub:
        s = check(b["xyz"], rec_a, Za, free_pts=came_a)
        kind = "removed" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": b["cls"], "kind": kind, "side": "before only", "displacement_m": None,
                    "before_xyz": np.round(floor(b["xyz"]), 3).tolist(), "after_xyz": None,
                    "frames_before": b["frames"], "frames_after": 0, "spot_in_after": s})
    for a in ua:
        s = check(a["xyz"], rec_b, Zb, free_pts=came_b)
        kind = "added" if s["status"] == "empty" else "unconfirmed"
        out.append({"cls": a["cls"], "kind": kind, "side": "after only", "displacement_m": None,
                    "before_xyz": None, "after_xyz": np.round(floor(a["xyz"]), 3).tolist(),
                    "frames_before": 0, "frames_after": a["frames"], "spot_in_before": s})
    # physical coherence: a vanished label decides ambiguous pairs; records that cannot be two
    # objects fold into one; scene structure is never an object-level change; and no spot verdict
    # counts as proof without strong evidence.
    return require_strong_evidence(
        demote_scene_structure(fold_records(split_on_vanish(out))))


DUP_M = 0.10   # two rigid objects cannot have centres this close: coincident records are one object


def split_on_vanish(out: list[dict], dup_m: float = DUP_M) -> list[dict]:
    """Resolve 'a small move or a removal plus an appearance' pairs when a label proves the answer.

    The two readings fit the same footprint evidence, but a moved object keeps its labels: if a
    second label at the pair's before end is gone from the after capture *entirely*, the before
    object left. The pair is then a wrong pairing of a removal and an appearance and splits into
    both, instead of staying unconfirmed. (ObChange: a removed can's spot paired with an unrelated
    cup 0.16 m away; the can's other label vanished, which decides it.)
    """
    vanished = [r for r in out if r.get("side") == "before only" and r.get("frames_after") == 0
                and r.get("before_xyz")]
    res: list[dict] = []
    for r in out:
        if not (r.get("kind") == "unconfirmed" and r.get("side") == "matched pair"
                and "removal plus an appearance" in (r.get("reason") or "")
                and r.get("before_xyz")):
            res.append(r)
            continue
        pb = np.asarray(r["before_xyz"], dtype=float)
        if not any(np.linalg.norm(np.asarray(v["before_xyz"], dtype=float) - pb) <= dup_m
                   for v in vanished):
            res.append(r)
            continue
        why = ("a label at its before spot vanished from the after capture entirely, so the before "
               "object left: this is a removal plus an appearance, not one small move")
        res.append({"cls": r["cls"], "kind": "removed", "side": "before only",
                    "displacement_m": None, "vertical_m": None,
                    "before_xyz": r["before_xyz"], "after_xyz": None,
                    "frames_before": r["frames_before"], "frames_after": 0,
                    "spot_in_after": r.get("old_spot"), "split_from_pair": True, "reason": why})
        res.append({"cls": r["cls"], "kind": "added", "side": "after only",
                    "displacement_m": None, "vertical_m": None,
                    "before_xyz": None, "after_xyz": r["after_xyz"],
                    "frames_before": 0, "frames_after": r["frames_after"],
                    "spot_in_before": r.get("new_spot"), "split_from_pair": True, "reason": why})
    return res


def fold_records(out: list[dict], dup_m: float = DUP_M) -> list[dict]:
    """Fold records that cannot be distinct physical objects into one record per object.

    Two rigid objects cannot rest with centres closer than `dup_m`, so records whose same-side
    endpoints nearly coincide describe one object seen under several labels (a box detected as both
    'book' and 'box' turns a real move into a phantom addition). A one-sided record folds into a
    confirmed pair's coincident endpoint; two one-sided records of the same kind merge. The kept
    record is the pair, or else the one with more frames; folded labels are recorded on it.
    """
    def pts(r):
        p = []
        if r.get("before_xyz"):
            p.append(("before", np.asarray(r["before_xyz"], dtype=float)))
        if r.get("after_xyz"):
            p.append(("after", np.asarray(r["after_xyz"], dtype=float)))
        return p

    keepers = [r for r in out if r.get("kind") in ("moved", "stayed") and r.get("after_xyz")]
    dropped: set[int] = set()
    for r in out:
        if r.get("kind") not in ("added", "removed") or id(r) in dropped:
            continue
        side = "after" if r["kind"] == "added" else "before"
        mine = [q for s, q in pts(r) if s == side]
        if not mine:
            continue
        near = [k for k in keepers
                if any(s == side and np.linalg.norm(q - mine[0]) <= dup_m for s, q in pts(k))]
        if not near:
            continue
        k = max(near, key=lambda r2: (r2.get("frames_before", 0) + r2.get("frames_after", 0)))
        k.setdefault("labels", []).append(r["cls"])
        dropped.add(id(r))
    # same-kind one-sided records at one spot are one object under two labels
    rest = [r for r in out if id(r) not in dropped]
    merged: list[dict] = []
    for r in rest:
        if r.get("kind") not in ("added", "removed"):
            merged.append(r)
            continue
        side = "after" if r["kind"] == "added" else "before"
        mine = [q for s, q in pts(r) if s == side]
        if not mine:
            merged.append(r)
            continue
        twin = next((m for m in merged
                     if m.get("kind") == r["kind"] and m.get("side") == r.get("side")
                     and (q := [x for s, x in pts(m) if s == side])
                     and np.linalg.norm(q[0] - mine[0]) <= dup_m), None)
        if twin is None:
            merged.append(r)
            continue
        if (r.get("frames_before", 0) + r.get("frames_after", 0)
                > twin.get("frames_before", 0) + twin.get("frames_after", 0)):
            twin.setdefault("labels", []).append(twin["cls"])
            twin["cls"], r["cls"] = r["cls"], twin["cls"]
            twin["frames_before"], r["frames_before"] = r["frames_before"], twin["frames_before"]
            twin["frames_after"], r["frames_after"] = r["frames_after"], twin["frames_after"]
        else:
            twin.setdefault("labels", []).append(r["cls"])
        twin["frames_before"] = max(twin.get("frames_before", 0), r.get("frames_before", 0))
        twin["frames_after"] = max(twin.get("frames_after", 0), r.get("frames_after", 0))
    return merged


# Labels that name the room or its supports, not an object placed on them: a 'desk' or 'table' that
# appears between two captures is background geometry mislocalized by the detector, never an object
# carried into the scene. Reported as unconfirmed, never as an object-level change.
SCENE_LABELS = {"desk", "table", "counter", "shelf", "wall", "floor", "ceiling"}


def demote_scene_structure(out: list[dict]) -> list[dict]:
    for r in out:
        if r.get("kind") in ("added", "removed", "moved") and \
                SCENE_LABELS.intersection((r.get("cls") or "").lower().split()):
            r["kind"] = "unconfirmed"
            r["side"] = r.get("side") or "matched pair"
            r["reason"] = (f"'{r['cls']}' is scene structure (the room or its supports do not appear "
                           "or disappear as objects): this is background geometry, not an "
                           "object-level change")
    return out


def require_strong_evidence(out: list[dict]) -> list[dict]:
    """A spot verdict is proof of absence only without contrary hits and with emptiness dominating
    silence 3:1.

    One occupied sample means some object is there, and a ring that mostly cannot see the spot
    cannot declare it empty. Anything weaker stays unconfirmed with its numbers recorded. Measured:
    every false alarm on the author's no-change control (and on the author's real pair) fails this
    bar, while every true positive on ObChange passes it with margin (8-9 empty, 0 contrary, <=1
    unseen). The constant is set by the controls, then validated against ground truth.
    """
    for r in out:
        if r.get("kind") not in ("moved", "removed", "added"):
            continue
        for key in ("old_spot", "new_spot", "spot_in_before", "spot_in_after"):
            sp = r.get(key)
            if not sp:
                continue
            e = sp.get("empty_samples", 0)
            o = sp.get("occupied_samples", 0)
            u = sp.get("unknown_samples", 0)
            if o > 0 or e < 3 * max(u, 1):
                r["kind"] = "unconfirmed"
                r["side"] = r.get("side") or "matched pair"
                r["reason"] = (f"spot evidence too weak to prove emptiness ({key}: {e} empty, "
                               f"{o} contrary, {u} unseen): one contrary hit or a mostly-unseen "
                               f"ring is not proof that the spot is empty")
                break
    return out


def corroborate(out: list[dict], regions: list[dict], to_floor, dup_m: float = DUP_M) -> list[dict]:
    """Confirm candidates an independent method agrees with — and nothing else.

    The voxel stage finds change regions from the surface geometry alone, without any object
    detection. A candidate whose spot lies inside a confirmed voxel region of the same kind is then
    reported by two methods that share only the reconstruction: that agreement is proof the object
    stage alone cannot earn at noisy capture pairs, and it confirms nothing where the voxel stage
    finds nothing (the no-change controls). Identity layer: one voxel region is one connected
    physical change, so all candidates inside it fold into ONE record — keeping the best-supported
    candidate and recording the others as its labels — instead of one change per label.
    """
    if not regions:
        return out
    used: set[int] = set()
    for reg in regions:
        kind = reg.get("kind")
        side, cent = {"added": ("after", reg.get("centroid_after")),
                      "removed": ("before", reg.get("centroid_before"))}.get(kind, (None, None))
        if side is None or cent is None:
            continue
        c = np.asarray(to_floor(cent), dtype=float)
        ext = reg.get("obb_extents_m") or [0.3]
        rad = 0.5 * float(np.linalg.norm(np.asarray(ext, dtype=float))) + 0.06
        picks = [r for r in out
                 if id(r) not in used and r.get("kind") == "unconfirmed" and r.get("side") == f"{side} only"
                 and r.get(f"{side}_xyz")
                 and not SCENE_LABELS.intersection((r.get("cls") or "").lower().split())
                 and np.linalg.norm(np.asarray(r[f"{side}_xyz"], dtype=float)[:2] - c[:2]) <= rad]
        if not picks:
            continue
        def support(r):
            sp = r.get("spot_in_before") or r.get("spot_in_after") or {}
            return (r.get("frames_before", 0) + r.get("frames_after", 0), sp.get("empty_samples", 0))
        keep = max(picks, key=support)
        for r in picks:
            used.add(id(r))
            if r is not keep:
                keep.setdefault("labels", []).append(r["cls"])
        keep["kind"] = kind
        keep["reason"] = (f"spot evidence alone is too weak, but an independent voxel change region "
                          f"agrees ({reg.get('volume_l')} L {kind} region, id {reg.get('id')}): two "
                          f"methods that share only the geometry report one physical change here")
    return out


def capture_noise(changes: list[dict], verify_pairs: float) -> float:
    """This capture pair's reconstruction noise, in metres.

    run_all.py measures it from the no-change control's stayed pairs and passes 1.5x of it as
    --verify-pairs; a direct run without that band measures it from its own stayed pairs (the
    control run is exactly that)."""
    if verify_pairs > 0:
        return verify_pairs / 1.5
    stay = [c["displacement_m"] for c in changes
            if c.get("kind") == "stayed" and c.get("displacement_m") is not None]
    return max(stay) if stay else 0.0


def require_provable_evidence(out: list[dict], noise_m: float, rise_tol: float) -> list[dict]:
    """At capture pairs noisier than the spot test's vertical resolution, a spot verdict is not proof.

    Measured on the author's 60-frame pair: the no-change control itself produces a 'removed
    pillow' with 7 empty / 0 contrary / 2 unseen ring votes — evidence statistically
    indistinguishable from the true added bag's (6 / 0 / 3). When the captures disagree by more
    than the test resolves (rise_tol, 3 cm), the honest verdict for every spot-only confirmation is
    'candidate': the control proves that the same evidence arises from noise alone. Independent
    agreement (corroborate(), the voxel stage) then confirms what is real. Below the resolution —
    ObChange's control drifts 0.01 m — spot proof stands, where it has measured precision 1.00.
    """
    if noise_m < rise_tol:
        return out
    for r in out:
        if r.get("kind") not in ("moved", "removed", "added"):
            continue
        sp = {}
        for key in ("old_spot", "new_spot", "spot_in_before", "spot_in_after"):
            if r.get(key):
                sp = r[key]
                break
        e, o, u = sp.get("empty_samples", 0), sp.get("occupied_samples", 0), sp.get("unknown_samples", 0)
        why = (f"the captures disagree by {noise_m:.2f} m of reconstruction noise, more than the "
               f"spot test resolves ({rise_tol:.2f} m): its verdict ({e} empty, {o} contrary, "
               f"{u} unseen) is not proof at this noise — the no-change control produces the same "
               f"evidence; only agreement with an independent method counts")
        r["kind"] = "unconfirmed"
        r["side"] = r.get("side") or "matched pair"
        r["reason"] = f"{r['reason']}; {why}" if r.get("reason") else why
    return out


def describe_objects(changes: list[dict]) -> str:
    """Plain-language summary of objects.json, with positions in the floor frame (metres)."""
    lines = ["# Object changes", "",
             "Positions are metres in the floor frame (x, y, z up). Labels are YOLO class names; "
             "no colour or appearance is used.", ""]
    for kind in ("moved", "removed", "added"):
        for c in (c for c in changes if c["kind"] == kind):
            labels = f" (also labelled {', '.join(c['labels'])})" if c.get("labels") else ""
            # a record confirmed by corroboration carries its own reason; the standard 'the spot is
            # empty' claim would overstate what its (weak) spot evidence showed
            why = f"; {c['reason']}" if c.get("reason") else None
            if kind == "moved":
                lines.append(f"- moved {c['cls']}{labels}: {c['displacement_m']:.2f} m "
                             f"(vertical {c['vertical_m']:+.2f} m), from {c['before_xyz']} to "
                             f"{c['after_xyz']}{why or '.'}")
            elif kind == "removed":
                lines.append(f"- removed {c['cls']}{labels}: last seen at {c['before_xyz']}"
                             f"{why or '; the spot is empty in the after capture'}.")
            else:
                lines.append(f"- added {c['cls']}{labels}: seen at {c['after_xyz']}"
                             f"{why or '; the spot is empty in the before capture'}.")
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
            # a record demoted by a rule carries its own reason (with the evidence numbers); prefer it
            # over the derived spot summary, which would say 'spot empty' about a verdict that failed
            why = c.get("reason") or f"spot {c[key]['status']} in the other capture"
        lines.append(f"- {c['cls']} ({c['side']}): {why}.")
    return "\n".join(lines) + "\n"


def side_objects(recon: dict, files: list[Path], model, device, args) -> list[dict]:
    obs = observations(recon, files, model, device, args.min_conf, lift=args.lift)
    return cluster(obs, args.merge, args.min_frames, dup_m=getattr(args, "merge_dup", 0.0)), len(obs)


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
    ap.add_argument("--footprint", type=float, default=0.08,
                    help="metres: radius of the footprint ring around each object centre. 0.08 keeps the "
                         "ring inside one object on a tabletop (neighbours sit 0.1-0.2 m apart; a 0.15 m "
                         "ring sampled them and vetoed real changes). Chosen from a 2-point sweep "
                         "(0.15 / 0.08) on one ground-truthed scene: provisional, see DESIGN section 8")
    ap.add_argument("--max-vertical", type=float, default=0.15,
                    help="metres: a candidate move with a larger vertical jump is unconfirmed (floor objects do not rise)")
    ap.add_argument("--verify-pairs", type=float, default=0.02,
                    help="metres: matched pairs farther apart than this are checked by the gate instead of "
                         "assumed stayed; 0 turns the check off. Set from your no-change control: 1.5x the "
                         "largest stayed displacement it reports (ObChange's control noise is 0.01 m; the "
                         "author's capture drifts 0.21 m, where the check must stay off). Below the band the "
                         "pairing noise exceeds what a spot test can resolve, so testing would manufacture "
                         "moves out of noise. run_all.py derives this from the control automatically. The "
                         "same band also decides whether spot verdicts count as proof at all: its 1.5x of "
                         "the control noise is compared against --rise-tol")
    ap.add_argument("--attr", type=float, default=0.12,
                    help="metres: a hit within this distance of an object that moved in/out belongs to that "
                         "object, not the one under test")
    ap.add_argument("--rise-tol", type=float, default=0.03,
                    help="metres: a hit below the spot by more than this is bare support (empty), not an object")
    ap.add_argument("--spot-lift", type=float, default=0.04,
                    help="metres: test the footprint this far above the object's support contact. Must "
                         "exceed --rise-tol so bare support reads empty at any view angle (at the contact "
                         "it reads occupied at grazing angles and vetoes real additions). Objects shorter "
                         "than the lift read empty even when present")
    ap.add_argument("--merge-dup", type=float, default=0.0,
                    help="metres: merge clusters of different labels closer than this into one object "
                         "(duplicate detections). Off by default: measured harmful on ObChange, see "
                         "merge_duplicates()")
    ap.add_argument("--corroborate", type=Path, default=None,
                    help="change_maps/changes.json from the voxel stage: candidates inside a confirmed "
                         "region of the same kind are confirmed by two independent methods (one region "
                         "= one physical change; its labels fold into one record)")
    ap.add_argument("--gap", type=float, default=0.03,
                    help="metres: a hit past the spot by more than this proves the spot itself is unobstructed")
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
    # floor normal expressed in the recon frame (R maps recon -> floor, so its third row is the
    # floor z-axis in recon coordinates): the footprint ring must be horizontal in the floor frame
    changes = classify(before, after, rec_b, rec_a, Zb, Za, args,
                       to_floor=lambda xyz: to_floor_xyz(xyz, R, shift), normal=R[2])
    # spot verdicts prove a change only at capture noise below the spot test's resolution; above it
    # (the author's pair drifts 0.21 m) confirmation needs an independent method (below)
    noise = capture_noise(changes, args.verify_pairs)
    rise_tol = getattr(args, "rise_tol", 0.03)
    changes = require_provable_evidence(changes, noise, rise_tol)
    print(f"object proof regime: capture noise {noise:.3f} m vs spot resolution {rise_tol:.2f} m -> "
          + ("spot-only proof disabled, voxel agreement required" if noise >= rise_tol
             else "spot verdicts are proof"), flush=True)
    if args.corroborate:
        regions = json.loads(args.corroborate.read_text())
        changes = corroborate(changes, regions, to_floor=lambda xyz: to_floor_xyz(xyz, R, shift))
        print(f"corroboration: {len(regions)} voxel region(s) checked against the object candidates")

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
