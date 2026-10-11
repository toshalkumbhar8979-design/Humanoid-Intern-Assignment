"""Score detected changes against a ground-truth list (precision / recall).

Ground truth is a JSON file in the format of data/synthetic/ground_truth.json, or a copy of
data/real_ground_truth.template.json filled in by hand. Each entry has a type (added | removed | moved).
Positions are optional. When given (metres, x and y only):
  removed -> before_xy    added -> after_xy    moved -> before_xy and after_xy
a detection matches only if its centroid lies within --tol of every given position.
Without positions, an entry matches any unused detection of the same type.

  python scripts/evaluate_changes.py outputs/change_maps/changes.json data/real_ground_truth.json
  python scripts/evaluate_changes.py outputs/change_maps/changes.json gt.json --frame floor \
         --before outputs/reconstructions/before.npz

--frame raw    detection centroids as stored (MASt3R before-capture frame; use for synthetic GT)
--frame floor  centroids rotated so the floor is z = 0 (same frame as bev.png); needs --before
--truth-frame raw    (default) ground truth is in the raw frame; with --frame floor it is rotated too,
                     which needs before_z / after_z (height in the raw frame) on each entry that has a position
--truth-frame floor ground truth is already in the floor frame (no conversion)
Exit code 0 when the files are read and scored; 1 on malformed input. A low score is not an error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KINDS = ("added", "removed", "moved")


def _xy(p) -> np.ndarray:
    return np.asarray(p, dtype=float)[:2]


def distance(det: dict, gt: dict, tol: float) -> float | None:
    """Largest horizontal distance between a detection and a GT entry, or None if they cannot match.

    Returns 0.0 when the GT entry has no positions (type-only match).
    """
    if det["kind"] != gt["type"]:
        return None
    dists = []
    for gt_key, det_key in (("before_xy", "centroid_before"), ("after_xy", "centroid_after")):
        if gt.get(gt_key) is None:
            continue
        if det.get(det_key) is None:
            return None
        dists.append(float(np.linalg.norm(_xy(det[det_key]) - _xy(gt[gt_key]))))
    if not dists:
        return 0.0
    d = max(dists)
    return d if d <= tol else None


def score(detections: list[dict], truth: list[dict], tol: float = 0.5) -> dict:
    """Greedy one-to-one matching in ground-truth order (each detection is used at most once)."""
    used: set[int] = set()
    matches = []  # (gt index, detection index, distance)
    for gi, g in enumerate(truth):
        best = None
        for di, d in enumerate(detections):
            if di in used:
                continue
            dist = distance(d, g, tol)
            if dist is not None and (best is None or dist < best[1]):
                best = (di, dist)
        if best is not None:
            used.add(best[0])
            matches.append((gi, best[0], best[1]))
    tp = len(matches)
    fn = len(truth) - tp
    fp = len(detections) - tp
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "matches": matches,
        "missed_gt": [gi for gi in range(len(truth)) if gi not in {m[0] for m in matches}],
        "unmatched_det": [di for di in range(len(detections)) if di not in used],
    }


def objects_to_detections(objects: list[dict]) -> list[dict]:
    """Map detect_objects.py records to the scorer's schema. Only confirmed moved / added / removed count.

    stayed and unconfirmed records are not detections: an unconfirmed object is not reported as a change.
    """
    out = []
    for i, o in enumerate(objects):
        if o.get("kind") not in KINDS:
            continue
        out.append({"id": i, "kind": o["kind"], "cls": o.get("cls"),
                    "centroid_before": o.get("before_xyz"), "centroid_after": o.get("after_xyz")})
    return out


def floor_frame(before_npz: Path):
    """(R, shift) of the bird's-eye floor frame for a before reconstruction (the frame bev.png uses)."""
    from registration.align import surface_points
    from visualization.plots import floor_transform

    recon = dict(np.load(before_npz))
    pts, _ = surface_points(recon)
    return floor_transform(pts, recon["c2w"][0])


def _to_floor_point(p, R: np.ndarray, shift: float) -> list[float]:
    q = R @ np.asarray(p, dtype=float)
    q[2] += shift
    return q.tolist()


def to_floor_frame(detections: list[dict], R: np.ndarray, shift: float) -> list[dict]:
    """Rotate detection centroids into the floor frame (floor normal -> z, floor at z = 0)."""
    out = []
    for d in detections:
        e = dict(d)
        for key in ("centroid_before", "centroid_after"):
            if d.get(key) is not None:
                e[key] = _to_floor_point(d[key], R, shift)
        out.append(e)
    return out


def truth_to_floor_frame(truth: list[dict], R: np.ndarray, shift: float) -> list[dict]:
    """Rotate raw-frame ground truth into the floor frame.

    The rotation mixes height into x and y, so each position needs its height (before_z / after_z, metres,
    raw frame). Without it the conversion is undefined and this raises ValueError.
    """
    out = []
    for g in truth:
        e = dict(g)
        for xy_key in ("before_xy", "after_xy"):
            if g.get(xy_key) is None:
                continue
            z_key = xy_key.replace("_xy", "_z")
            if g.get(z_key) is None:
                raise ValueError(f"ground-truth entry {g.get('name', '?')!r}: {xy_key} needs {z_key} "
                                 "(height in the raw frame) to convert to the floor frame")
            e[xy_key] = _to_floor_point([*g[xy_key], g[z_key]], R, shift)[:2]
        out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections", type=Path, help="changes.json written by detect_changes.py")
    ap.add_argument("truth", type=Path, help="ground-truth JSON")
    ap.add_argument("--tol", type=float, default=0.5, help="position tolerance in metres (default 0.5)")
    ap.add_argument("--frame", choices=("raw", "floor"), default="raw")
    ap.add_argument("--truth-frame", choices=("raw", "floor"), default="raw",
                    help="frame of the ground truth when --frame floor (default raw)")
    ap.add_argument("--before", type=Path, default=None, help="before reconstruction .npz (needed for --frame floor)")
    ap.add_argument("--objects", action="store_true",
                    help="detections are objects.json from detect_objects.py (only confirmed moved/added/removed count)")
    args = ap.parse_args()

    try:
        detections = json.loads(args.detections.read_text())
        truth = json.loads(args.truth.read_text())["changes"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read inputs: {e}", file=sys.stderr)
        return 1
    if not isinstance(detections, list) or not isinstance(truth, list):
        print("error: detections must be a list and truth must have a 'changes' list", file=sys.stderr)
        return 1
    for g in truth:
        if g.get("type") not in KINDS:
            print(f"error: ground-truth entry {g.get('name')!r} has type {g.get('type')!r}, expected one of {KINDS}",
                  file=sys.stderr)
            return 1
    if args.objects:
        detections = objects_to_detections(detections)
    if args.truth_frame == "floor" and args.frame != "floor":
        print("error: --truth-frame floor only applies with --frame floor", file=sys.stderr)
        return 1
    if args.frame == "floor":
        if args.before is None:
            print("error: --frame floor needs --before", file=sys.stderr)
            return 1
        R, shift = floor_frame(args.before)
        detections = to_floor_frame(detections, R, shift)
        if args.truth_frame == "raw":
            try:
                truth = truth_to_floor_frame(truth, R, shift)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

    res = score(detections, truth, args.tol)
    print(f"ground truth: {len(truth)} changes, detections: {len(detections)}, tol {args.tol} m, frame {args.frame}")
    for gi, di, dist in res["matches"]:
        g, d = truth[gi], detections[di]
        print(f"  MATCH  {g.get('name', gi)!s:16s} {g['type']:8s} -> detection #{d.get('id', di)} "
              f"({d['kind']}), distance {dist:.2f} m")
    for gi in res["missed_gt"]:
        g = truth[gi]
        print(f"  MISSED {g.get('name', gi)!s:16s} {g['type']:8s}")
    for di in res["unmatched_det"]:
        d = detections[di]
        print(f"  FALSE  detection #{d.get('id', di)} ({d['kind']})")
    p = "n/a" if res["precision"] is None else f"{res['precision']:.2f}"
    r = "n/a" if res["recall"] is None else f"{res['recall']:.2f}"
    print(f"TP {res['tp']}  FP {res['fp']}  FN {res['fn']}  precision {p}  recall {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
