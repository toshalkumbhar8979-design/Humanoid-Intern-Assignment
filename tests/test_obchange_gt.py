"""Unit tests for scripts/obchange_gt.py (annotation parsing + change-list derivation).

Run:  python tests/test_obchange_gt.py      (or: pytest tests/test_obchange_gt.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import obchange_gt as og  # noqa: E402

# 4x4 row-major pose lines as written by the dataset's .anno files
CANON = "005_tomato_soup_can (1): -1 0 0 2.13602 0 -1 0 -0.603173 0 0 1 0.895 0 0 0 1 "


def write_scene(tmp: Path, name: str, lines: list[str]) -> Path:
    d = tmp / name / "planes" / "0"
    d.mkdir(parents=True)
    (d / "merged_plane_clouds_ds002.anno").write_text("\n".join(lines) + "\n")
    return tmp / name


def test_scene_objects_parses_poses_and_skips_gt_files(tmp_path):
    write_scene(tmp_path, "sceneA", [CANON, "025_mug (1): 1 0 0 2.73 0 -1 0 -1.09 0 0 -1 0.97 0 0 0 1 "])
    # a _GT.anno (point-index labels) must not be read as poses
    gt_file = tmp_path / "sceneA" / "planes" / "0" / "merged_plane_clouds_ds002_GT.anno"
    gt_file.write_text("005_tomato_soup_can 1 2 3\n")
    objs = og.scene_objects(tmp_path / "sceneA")
    assert set(objs) == {"005_tomato_soup_can", "025_mug"}
    assert og.centre(objs["005_tomato_soup_can"]) == [2.13602, -0.603173, 0.895]


def test_derive_classifies_added_removed_moved():
    ref = {"a": [1, 0, 0, 0.0, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1],
           "b": [1, 0, 0, 1.0, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    test = {"b": [1, 0, 0, 1.05, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1],   # 5 cm = within tol
            "c": [1, 0, 0, 2.0, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    changes = og.derive(ref, test, move_tol=0.15)
    kinds = {c["name"]: c["type"] for c in changes}
    assert kinds == {"a": "removed", "c": "added"}   # b stayed: 5 cm < 15 cm


def test_derive_reports_moves_with_both_positions():
    ref = {"mug": [1, 0, 0, 0.0, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    test = {"mug": [1, 0, 0, 0.48, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    (c,) = og.derive(ref, test, move_tol=0.15)
    assert c["type"] == "moved"
    assert c["before_xy"] == [0.0, 0.0] and c["after_xy"] == [0.48, 0.0]
    assert c["before_z"] == 0.9 and c["after_z"] == 0.9
    assert abs(c["displacement_m"] - 0.48) < 1e-9


def test_move_tolerance_is_respected():
    ref = {"mug": [1, 0, 0, 0.0, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    test = {"mug": [1, 0, 0, 0.15, 0, 1, 0, 0.0, 0, 0, 1, 0.9, 0, 0, 0, 1]}
    assert og.derive(ref, test, move_tol=0.15) == []          # exactly at tol: not a move
    assert len(og.derive(ref, test, move_tol=0.14)) == 1


def test_cli_no_positions_strips_coordinates(tmp_path, monkeypatch, capsys):
    write_scene(tmp_path, "sceneA", [CANON])
    write_scene(tmp_path, "sceneB", ["025_mug (1): 1 0 0 2.73 0 -1 0 -1.09 0 0 -1 0.97 0 0 0 1 "])
    out = tmp_path / "gt.json"
    monkeypatch.setattr(sys, "argv", ["obchange_gt.py", str(tmp_path / "sceneA"), str(tmp_path / "sceneB"),
                                      "--out", str(out), "--no-positions"])
    assert og.main() == 0
    doc = json.loads(out.read_text())
    assert doc["reference_objects"] == 1 and doc["test_objects"] == 1
    assert {c["type"] for c in doc["changes"]} == {"removed", "added"}
    for c in doc["changes"]:
        assert "before_xy" not in c and "after_xy" not in c
        assert "before_z" not in c and "after_z" not in c
