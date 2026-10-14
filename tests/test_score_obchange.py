"""Unit tests for scripts/score_obchange.py (rule reclassification and yaw consistency fit).

Run:  pytest tests/test_score_obchange.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_obchange as so  # noqa: E402


def rec(kind="unconfirmed", vertical=0.2, reason=None):
    return {"kind": kind, "side": "matched pair", "vertical_m": vertical,
            "reason": reason if reason is not None else
            (f"vertical jump {vertical:+.2f} m (floor objects do not rise; likely a centre mismatch)"
             if kind == "unconfirmed" else None),
            "before_xyz": [0, 0, 0], "after_xyz": [0.5, 0, vertical]}


def test_move_decision_rules():
    drop, rise = rec(vertical=-0.22), rec(vertical=+0.40)
    assert so.move_decision(drop, "rise-only", 0.15) == "moved"   # real move down: keep
    assert so.move_decision(rise, "rise-only", 0.15) is None      # centre-mismatch rise: reject
    assert so.move_decision(drop, "symmetric", 0.15) is None      # old rule rejects the real drop
    assert so.move_decision(rec(kind="moved", vertical=-0.22, reason=None), "symmetric", 0.15) == "unconfirmed"
    assert so.move_decision(drop, "as-run", 0.15) is None         # as-run keeps the record's kind


def test_objects_as_detections_reclassifies_per_rule():
    records = [rec(vertical=-0.22), rec(vertical=+0.40), rec(kind="stayed", vertical=0.0, reason=None)]
    as_run = so.objects_as_detections(records, "as-run", 0.15)
    rise = so.objects_as_detections(records, "rise-only", 0.15)
    assert [d["kind"] for d in as_run] == []                     # both unconfirmed as recorded
    assert [d["kind"] for d in rise] == ["moved"]                # only the drop is recovered
    # a move the run confirmed with a big rise is rejected again under the symmetric rule
    recs = [rec(kind="moved", vertical=+0.40, reason=None)]
    assert [d["kind"] for d in so.objects_as_detections(recs, "symmetric", 0.15)] == []


def test_type_only_drops_cross_frame_positions():
    # dataset-frame positions must not be compared against floor-frame detections: the scorer
    # matches by change type alone (positions would be meaningless distances)
    gt = [{"name": "004_sugar_box", "type": "moved", "before_xy": [1.95, -0.60], "after_xy": [1.43, -0.23],
           "before_z": 0.93, "after_z": 0.93}]
    stripped = so.type_only(gt)
    assert stripped == [{"name": "004_sugar_box", "type": "moved"}]
    dets = [{"id": 0, "kind": "moved", "cls": "book",
             "centroid_before": [0.35, 0.40, 0.02], "centroid_after": [-0.31, 0.44, 0.01]}]
    res = so.ev.score(dets, stripped, tol=0.5)
    assert res["tp"] == 1 and res["recall"] == 1.0       # type-level match
    wrong = so.ev.score(dets, gt, tol=0.5)
    assert wrong["tp"] == 0                              # cross-frame positions never match


def test_objects_as_candidates_counts_what_the_gate_rejects():
    records = [
        {"cls": "book", "kind": "unconfirmed", "side": "matched pair", "displacement_m": 0.665,
         "before_xyz": [0.4, 0.4, 0.0], "after_xyz": [-0.3, 0.4, 0.0]},
        {"cls": "mug", "kind": "unconfirmed", "side": "before only", "displacement_m": None,
         "before_xyz": [0.5, 0.4, 0.0], "after_xyz": None},
        {"cls": "box", "kind": "unconfirmed", "side": "after only", "displacement_m": None,
         "before_xyz": None, "after_xyz": [-0.3, 0.4, 0.0]},
        {"cls": "bowl", "kind": "stayed", "side": None, "displacement_m": 0.05,
         "before_xyz": [0, 0, 0], "after_xyz": [0.05, 0, 0]},
        {"cls": "cup", "kind": "unconfirmed", "side": "matched pair", "displacement_m": 0.12,
         "before_xyz": [1, 0, 0], "after_xyz": [1.1, 0, 0]},
    ]
    dets = so.objects_as_candidates(records)
    assert [(d["cls"], d["kind"]) for d in dets] == [("book", "moved"), ("mug", "removed"), ("box", "added")]


def test_se2_fit_recovers_a_known_rigid_transform():
    rng = np.random.default_rng(11)
    theta = math.radians(-20.0)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    t = np.array([1.2, -0.7])
    gt_pts, meas_pts = [], []
    for _ in range(5):
        g = rng.normal(size=3)
        m = np.array([*(R @ g[:2] + t), 0.0])
        gt_pts.append(g)
        meas_pts.append(m)
    yaw, tt, resid = so.se2_fit(gt_pts, meas_pts)
    assert abs(yaw - (-20.0)) < 1e-6
    assert np.allclose(tt, t, atol=1e-9)
    assert max(resid) < 1e-9


def test_yaw_fit_recovers_a_known_rotation():
    rng = np.random.default_rng(7)
    theta = math.radians(30.0)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    gt, meas = [], []
    for _ in range(4):
        g = rng.normal(size=3)
        m = R @ g[:2]
        gt.append(g)
        meas.append(np.array([m[0], m[1], 0.0]))
    yaw, resid = so.yaw_fit(gt, meas)
    assert abs(yaw - 30.0) < 1e-6
    assert max(resid) < 1e-9


def test_yaw_fit_residuals_catch_inconsistent_displacements():
    # two vectors the same and one that contradicts them: the fit cannot explain all three
    gt = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    meas = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    _, resid = so.yaw_fit(gt, meas)
    assert max(resid) > 0.5
