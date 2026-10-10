"""Unit tests for scripts/detect_objects.py (rules and text only, no model weights needed).

Run:  python tests/test_detect_objects.py      (or: pytest tests/test_detect_objects.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detect_objects as do  # noqa: E402


def test_small_vertical_change_is_allowed():
    assert do.vertical_reason(0.10, 0.15) is None
    assert do.vertical_reason(-0.15, 0.15) is None


def test_large_vertical_jump_is_rejected_with_reason():
    reason = do.vertical_reason(0.36, 0.15)
    assert reason is not None and "vertical jump +0.36 m" in reason


def test_objects_md_states_the_vertical_reason():
    changes = [{
        "cls": "bed", "kind": "unconfirmed", "side": "matched pair", "displacement_m": 0.365,
        "vertical_m": 0.36, "before_xyz": [0, 0, 0], "after_xyz": [0.3, 0, 0.36],
        "frames_before": 3, "frames_after": 3,
        "old_spot": {"status": "occupied"}, "new_spot": {"status": "occupied"},
        "reason": do.vertical_reason(0.36, 0.15),
    }]
    text = do.describe_objects(changes)
    assert "bed (matched pair): candidate move, old spot occupied, new spot occupied; vertical jump" in text


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"PASS {len(tests)} tests")
