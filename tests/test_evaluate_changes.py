"""Unit tests for scripts/evaluate_changes.py (matching rules only, no model weights needed).

Run:  python tests/test_evaluate_changes.py      (or: pytest tests/test_evaluate_changes.py)
"""
import sys
from pathlib import Path

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


def test_synthetic_ground_truth_loads_and_is_well_formed():
    import json
    gt = json.loads((ROOT / "data" / "synthetic" / "ground_truth.json").read_text())["changes"]
    assert all(g["type"] in ev.KINDS for g in gt)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"PASS {len(tests)} tests")
