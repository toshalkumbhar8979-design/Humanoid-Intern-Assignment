"""Score a pipeline run against ObChange ground truth (type level, plus a move analysis).

The dataset scenes share one coordinate frame but the pipeline reports in its own floor frame, so
positions cannot be matched directly; entries match by change type (evaluate_changes.py treats a
position-less ground truth that way). The move analysis below is frame-independent where it can be:
vertical components are comparable (both frames are z up), and displacement vectors are compared up
to one global yaw between the two frames.

  python scripts/score_obchange.py outputs/obchange_2v3 data/datasets/obchange/gt_kitchen_2v3.json

Sections:
  1. object changes (objects/objects.json, confirmed only) vs the full ground truth, scored under
     three vertical rules: as run, the old symmetric rule, and the rise-only variant (a drop is
     physically possible, so only vertical_m > --max-vertical rejects a move).
  2. voxel regions (change_maps/changes.json) vs the added/removed subset: one region cannot
     represent a move, so `moved` entries are out of the region detector's reach by construction.
  3. candidate coverage: what the footprint gate rejects (all candidates of any kind, no gate).
  4. per GT move: matched candidate, measured vertical_m (what the rule acts on), and how well one
     global yaw explains the measured displacement vectors against the ground truth's.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_changes as ev  # noqa: E402

VERTICAL_MARK = "vertical jump"


def move_decision(o: dict, rule: str, max_vertical: float) -> str | None:
    """The kind a matched pair gets under a vertical rule, or None to keep the record as recorded.

    'as-run' keeps the run's own decision; 'symmetric' rejects |vertical| over the limit (the rule the
    run had before the ObChange measurement); 'rise-only' rejects only rises, keeping drops.
    Non-vertical unconfirmed records stay unconfirmed under every rule.
    """
    v = o.get("vertical_m")
    if rule == "as-run":
        return None
    if o.get("kind") == "moved":
        if rule == "symmetric" and v is not None and abs(v) > max_vertical:
            return "unconfirmed"
        return None
    if o.get("kind") == "unconfirmed" and VERTICAL_MARK in (o.get("reason") or ""):
        if rule == "rise-only" and v is not None and v <= max_vertical:
            return "moved"
    return None


def objects_as_detections(records: list[dict], rule: str, max_vertical: float) -> list[dict]:
    """Confirmed object changes as scorer detections under the given vertical rule."""
    dets = []
    for i, o in enumerate(records):
        kind = move_decision(o, rule, max_vertical) or o.get("kind")
        if kind not in ev.KINDS:
            continue
        dets.append({"id": i, "kind": kind, "cls": o.get("cls"),
                     "centroid_before": o.get("before_xyz"), "centroid_after": o.get("after_xyz")})
    return dets


def objects_as_candidates(records: list[dict], same: float = 0.3) -> list[dict]:
    """What the run would report if the footprint gate rejected nothing (measurement only).

    A before-only record is a removed candidate, an after-only record an added candidate, and a
    matched pair that moved at least `same` metres a moved candidate. The gate exists to keep false
    alarms out (the no-change controls stay clean), so this coverage number is the recall the gate
    has to work with, not a replacement for it.
    """
    dets = []
    for i, o in enumerate(records):
        side = o.get("side")
        if o.get("kind") == "stayed":
            continue   # a split pair's kept half: it stayed at its spot, it is not a removal
        if side == "before only":
            kind = "removed"
        elif side == "after only":
            kind = "added"
        elif side == "matched pair" and (o.get("displacement_m") or 0) >= same:
            kind = "moved"
        else:
            continue
        dets.append({"id": i, "kind": kind, "cls": o.get("cls"),
                     "centroid_before": o.get("before_xyz"), "centroid_after": o.get("after_xyz")})
    return dets


def type_only(gt: list[dict]) -> list[dict]:
    """Ground truth with positions dropped, so entries match by change type alone.

    The dataset's positions are in its own frame while the pipeline reports in the floor frame, so a
    distance between the two is meaningless. evaluate_changes.py matches position-less entries by
    type. Index order (and names) are preserved.
    """
    return [{"name": g.get("name"), "type": g["type"]} for g in gt]


def se2_fit(gt_pts: list[np.ndarray], meas_pts: list[np.ndarray]) -> tuple[float, np.ndarray, list[float]]:
    """Best yaw + translation mapping ground-truth positions to measured ones (both z up).

    The dataset frame and the floor frame are both gravity-aligned, so they differ by one yaw about z
    plus a translation. Returns (degrees, xy translation, per-point residuals). The fit is only
    trustworthy with 3+ points; with fewer the residuals are near-zero by construction and say
    nothing.
    """
    g = np.array([p[:2] for p in gt_pts], dtype=float)
    m = np.array([p[:2] for p in meas_pts], dtype=float)
    gc, mc = g.mean(0), m.mean(0)
    g0, m0 = g - gc, m - mc
    num = sum(float(a[0] * b[1] - a[1] * b[0]) for a, b in zip(g0, m0))
    den = sum(float(np.dot(a, b)) for a, b in zip(g0, m0))
    yaw = math.atan2(num, den)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    t = mc - R @ gc
    resid = [float(np.linalg.norm(R @ g[i] + t - m[i])) for i in range(len(g))]
    return math.degrees(yaw), t, resid


def yaw_fit(gt_vecs: list[np.ndarray], meas_vecs: list[np.ndarray]) -> tuple[float, list[float]]:
    """Best yaw mapping ground-truth displacements to measured ones; returns (degrees, per-vector residuals)."""
    num = sum(float(g[0] * m[1] - g[1] * m[0]) for g, m in zip(gt_vecs, meas_vecs))
    den = sum(float(np.dot(g[:2], m[:2])) for g, m in zip(gt_vecs, meas_vecs))
    yaw = math.atan2(num, den)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    resid = [float(np.linalg.norm(R @ g[:2] - m[:2])) for g, m in zip(gt_vecs, meas_vecs)]
    return math.degrees(yaw), resid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run output directory (contains objects/ and change_maps/)")
    ap.add_argument("gt", type=Path, help="ground truth JSON from obchange_gt.py (with positions)")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--max-vertical", type=float, default=0.15)
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())["changes"]
    records = json.loads((args.run / "objects" / "objects.json").read_text())
    regions = json.loads((args.run / "change_maps" / "changes.json").read_text())

    print(f"ground truth: {len(gt)} changes "
          f"({sum(g['type'] == 'added' for g in gt)} added, "
          f"{sum(g['type'] == 'removed' for g in gt)} removed, "
          f"{sum(g['type'] == 'moved' for g in gt)} moved)\n")

    print("== 1. object changes (type level; dataset frame differs from the floor frame) ==")
    notes = {
        "as-run": "decision as the run recorded it",
        "symmetric": f"old rule: |vertical| over {args.max_vertical} m rejects",
        "rise-only": f"new rule: only rises over {args.max_vertical} m reject; drops are real moves",
    }
    for rule in ("as-run", "symmetric", "rise-only"):
        dets = objects_as_detections(records, rule, args.max_vertical)
        res = ev.score(dets, type_only(gt), tol=args.tol)
        p = "n/a" if res["precision"] is None else f"{res['precision']:.2f}"
        r = "n/a" if res["recall"] is None else f"{res['recall']:.2f}"
        print(f"  rule {rule:<10}: TP {res['tp']}  FP {res['fp']}  FN {res['fn']}  "
              f"precision {p}  recall {r}  ({notes[rule]})")
        for gi in res["missed_gt"]:
            print(f"    MISSED {gt[gi].get('name', '?')} ({gt[gi]['type']})")

    print("\n== 2. voxel regions (added/removed only: a region cannot represent a move) ==")
    subset = [g for g in gt if g["type"] in ("added", "removed")]
    res = ev.score(regions, type_only(subset), tol=args.tol)
    p = "n/a" if res["precision"] is None else f"{res['precision']:.2f}"
    r = "n/a" if res["recall"] is None else f"{res['recall']:.2f}"
    print(f"  TP {res['tp']}  FP {res['fp']}  FN {res['fn']}  precision {p}  recall {r}  "
          f"({len(regions)} regions vs {len(subset)} type-compatible entries)")
    for gi in res["missed_gt"]:
        print(f"    MISSED {subset[gi].get('name', '?')} ({subset[gi]['type']})")

    print("\n== 3. candidate coverage (what the footprint gate rejects) ==")
    cands = objects_as_candidates(records)
    res = ev.score(cands, type_only(gt), tol=args.tol)
    p = "n/a" if res["precision"] is None else f"{res['precision']:.2f}"
    r = "n/a" if res["recall"] is None else f"{res['recall']:.2f}"
    print(f"  TP {res['tp']}  FP {res['fp']}  FN {res['fn']}  precision {p}  recall {r}  "
          f"({len(cands)} candidates of any kind vs {len(gt)} entries, gate removed)")
    for gi in res["missed_gt"]:
        print(f"    MISSED {gt[gi].get('name', '?')} ({gt[gi]['type']})")
    for di in res["unmatched_det"]:
        d = cands[di]
        print(f"    EXTRA  {d['cls']} ({d['kind']})")
    gt_pos, meas_pos = [], []
    for gi, di, _ in res["matches"]:
        g, d = gt[gi], cands[di]
        for gk, dk in (("before_xy", "centroid_before"), ("after_xy", "centroid_after")):
            if g.get(gk) is not None and d.get(dk) is not None:
                gt_pos.append(np.array([*g[gk], 0.0]))
                meas_pos.append(np.array(d[dk], dtype=float))
    if len(gt_pos) >= 6:
        yaw, t, resid = se2_fit(gt_pos, meas_pos)
        print(f"  position consistency: one SE(2) (yaw {yaw:+.1f} deg, translation "
              f"[{t[0]:+.2f}, {t[1]:+.2f}] m) maps the dataset frame to the floor frame; residuals "
              f"{' '.join(f'{r:.2f} m' for r in resid)} over {len(resid)} positions")
        print("  (small residuals: the type-level matches are the right physical objects under one "
              "rigid transform, not just the right types)")
    else:
        print(f"  position consistency: only {len(gt_pos)} matched positions, need 6 for an SE(2) fit")

    print("\n== 4. moved objects against the ground truth ==")
    pairs = [o for o in records if o.get("side") == "matched pair"]
    gt_moves = [g for g in gt if g["type"] == "moved"]
    used = set()
    gt_vecs, meas_vecs, shown = [], [], []
    for g in gt_moves:
        target = np.array([*g["after_xy"], g.get("after_z", 0.0)]) - \
                 np.array([*g["before_xy"], g.get("before_z", 0.0)])
        best, best_d = None, None
        for i, o in enumerate(pairs):
            if i in used or o.get("before_xyz") is None or o.get("after_xyz") is None:
                continue
            d = float(np.linalg.norm(np.array(o["after_xyz"]) - np.array(o["before_xyz"])
                                     - np.array([target[0], target[1], 0.0])))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is None:
            shown.append((g, None, None))
            continue
        used.add(best)
        o = pairs[best]
        shown.append((g, o, best_d))
        gt_vecs.append(target)
        meas_vecs.append(np.array(o["after_xyz"]) - np.array(o["before_xyz"]))
    for g, o, d in shown:
        if o is None:
            print(f"  {g['name']:<20} no matched pair at all")
            continue
        v = o.get("vertical_m")
        state = "confirmed" if o.get("kind") == "moved" else f"unconfirmed ({o.get('reason', '?')})"
        print(f"  {g['name']:<20} gt dz {g.get('after_z', 0) - g.get('before_z', 0):+.2f} m | "
              f"measured vertical {('n/a' if v is None else f'{v:+.2f} m')} | {state}")
    if len(gt_vecs) >= 2:
        yaw, resid = yaw_fit(gt_vecs, meas_vecs)
        print(f"  displacement consistency: one yaw of {yaw:+.1f} deg maps gt to measured, "
              f"residuals {' '.join(f'{r:.2f} m' for r in resid)} over {len(resid)} moves")
        print("  (with 2 moves this fits exactly by construction; residuals are only informative for 3+)")
    else:
        print("  displacement consistency: fewer than 2 matched moves, no yaw check possible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
