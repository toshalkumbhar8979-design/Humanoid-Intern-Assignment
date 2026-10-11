"""Unit tests for scripts/evaluate_changes.py (matching rules only, no model weights needed).

Run:  python tests/test_evaluate_changes.py      (or: pytest tests/test_evaluate_changes.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_changes as ev  # noqa: E402


def det(kind, before=None, after=None, i=0):
    return {"id": i, "kind": kind, "centroid_before": None if before is None else [*before, 0.0],
            "centroid_after": None if after is None else [*after, 0.0]}


def test_perfect_match_scores_full():
    truth = [{"type": "moved", "before_xy": [0.5, 0.05], "after_xy": [-0.7, 0.05]},
             {"type": "added", "after_xy": [-0.35, 0.98]},
             {"type": "removed", "before_xy": [-1.1, 1.15]}]
    dets = [det("moved", (0.52, 0.0), (-0.66, 0.1), 1), det("added", None, (-0.3, 1.0), 2),
            det("removed", (-1.0, 1.1), None, 3)]
    r = ev.score(dets, truth, tol=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 0, 0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0


def test_wrong_place_is_miss_and_false_alarm():
    truth = [{"type": "moved", "before_xy": [0.5, 0.05], "after_xy": [-0.7, 0.05]}]
    dets = [det("moved", (0.5, 0.0), (1.8, 0.0), 1)]  # right start, wrong end (2.5 m off)
    r = ev.score(dets, truth, tol=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_type_only_entry_matches_any_detection_of_that_type():
    truth = [{"type": "added"}]
    r = ev.score([det("added", None, (3.0, 3.0), 1)], truth)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_wrong_kind_never_matches():
    truth = [{"type": "removed", "before_xy": [0.0, 0.0]}]
    r = ev.score([det("added", None, (0.0, 0.0), 1)], truth)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_each_detection_matches_at_most_once():
    truth = [{"type": "added"}, {"type": "added"}]
    r = ev.score([det("added", None, (1.0, 1.0), 1)], truth)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)


def test_no_detections_gives_zero_recall_and_undefined_precision():
    r = ev.score([], [{"type": "moved", "before_xy": [0, 0], "after_xy": [1, 0]}])
    assert r["recall"] == 0.0 and r["precision"] is None and r["fn"] == 1


def test_moved_needs_both_positions_within_tolerance():
    truth = [{"type": "moved", "before_xy": [0.0, 0.0], "after_xy": [1.0, 0.0]}]
    ok = det("moved", (0.1, 0.0), (1.1, 0.0), 1)
    bad_after = det("moved", (0.1, 0.0), (2.0, 0.0), 2)
    assert ev.distance(ok, truth[0], 0.5) is not None
    assert ev.distance(bad_after, truth[0], 0.5) is None


def test_object_adapter_keeps_only_confirmed_changes():
    objs = [
        {"kind": "stayed", "cls": "chair", "before_xyz": [0, 0, 0], "after_xyz": [0, 0, 0]},
        {"kind": "unconfirmed", "cls": "cup", "before_xyz": None, "after_xyz": [1, 1, 0]},
        {"kind": "removed", "cls": "backpack", "before_xyz": [1.0, 2.0, 0.4], "after_xyz": None},
        {"kind": "moved", "cls": "bed", "before_xyz": [0, 0, 0], "after_xyz": [2.0, 0.0, 0]},
    ]
    dets = ev.objects_to_detections(objs)
    assert [d["kind"] for d in dets] == ["removed", "moved"]
    assert dets[0]["centroid_before"] == [1.0, 2.0, 0.4] and dets[0]["centroid_after"] is None
    truth = [{"type": "removed", "before_xy": [1.0, 2.0]}, {"type": "moved", "before_xy": [0, 0], "after_xy": [2, 0]}]
    r = ev.score(dets, truth, tol=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)


def test_synthetic_ground_truth_loads_and_is_well_formed():
    import json
    gt = json.loads((ROOT / "data" / "synthetic" / "ground_truth.json").read_text())["changes"]
    assert all(g["type"] in ev.KINDS for g in gt)


def _tilted_floor_npz() -> Path:
    """A before reconstruction that is one plane: a floor tilted about 8 degrees, seen by a camera at the origin."""
    H, W, F, s = 60, 80, 60.0, 0.14
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    denom = (vv - H / 2) / F - s                  # floor: y = 1.2 + s * z (camera y points down)
    z = np.where(denom > 0.05, 1.2 / np.maximum(denom, 1e-6), 0.0)
    ok = (z > 0.3) & (z < 6.0)
    pts = np.stack([(uu - W / 2) / F * z, (vv - H / 2) / F * z, z], -1)
    path = Path(tempfile.mkdtemp()) / "before.npz"
    np.savez(path, points=pts[None], conf=ok[None].astype(np.float32),
             colors=np.zeros((1, H, W, 3), np.float32), c2w=np.eye(4)[None], focals=np.array([F]))
    return path


def test_floor_frame_matches_truth_only_when_truth_is_converted():
    npz = _tilted_floor_npz()
    R, shift = ev.floor_frame(npz)
    # a centroid far from the camera axis: the tilt moves its x and y by a clear margin
    P = np.array([1.2, 0.3, 3.0])
    det = [{"id": 1, "kind": "removed", "centroid_before": P.tolist(), "centroid_after": None}]
    truth = [{"type": "removed", "name": "box", "before_xy": P[:2].tolist(), "before_z": float(P[2])}]
    f_det = ev.to_floor_frame(det, R, shift)
    assert ev.score(f_det, ev.truth_to_floor_frame(truth, R, shift), tol=0.05)["tp"] == 1
    # the old behaviour: rotating only the detections puts the same object far from its truth
    assert ev.score(f_det, truth, tol=0.05)["tp"] == 0
    try:
        ev.truth_to_floor_frame([{"type": "removed", "name": "x", "before_xy": [1.0, 0.3]}], R, shift)
    except ValueError:
        pass
    else:
        raise AssertionError("truth without before_z must not be converted silently")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"PASS {len(tests)} tests")
