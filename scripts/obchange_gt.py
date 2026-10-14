"""Derive a change list from ObChange annotations, in the format of evaluate_changes.py.

ObChange (TU Wien, CC BY 4.0, https://researchdata.tuwien.at/records/y3ggy-hxp10) annotates every
scene with one line per object: a YCB name and its 4x4 pose in the scene reconstruction frame

    005_tomato_soup_can (1): r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz 0 0 0 1

Comparing a reference scene's objects with a test scene's gives the change list: an object only in
the test scene is `added`, one only in the reference is `removed`, and one whose centre moved more
than --move-tol is `moved` (with both positions). Positions are metres in the dataset's scene frame
(z up; tabletop heights around 0.9 m confirm it). The scene frame is not the pipeline's floor frame,
so score with --frame raw / type-only unless the clouds are registered first (see DESIGN.md).

  python scripts/obchange_gt.py data/datasets/obchange/extracted/Kitchen/scene1 \\
      data/datasets/obchange/extracted/Kitchen/scene2 --out data/datasets/obchange/kitchen_1v2.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANNO = re.compile(r"^(?P<name>\S+)\s*(\(\d+\))?:\s*(?P<vals>[-0-9.eE+ ]+)$")


def scene_objects(scene_dir: Path) -> dict[str, list[float]]:
    """Object name -> 4x4 pose (row-major) for every object annotated in the scene's planes."""
    objs: dict[str, list[float]] = {}
    for anno in sorted(scene_dir.glob("planes/*/*.anno")):
        if anno.name.endswith("_GT.anno"):
            continue
        for line in anno.read_text().splitlines():
            m = ANNO.match(line.strip())
            if not m:
                continue
            vals = [float(v) for v in m.group("vals").split()]
            if len(vals) != 16:
                continue
            objs.setdefault(m.group("name"), vals)
    return objs


def centre(pose: list[float]) -> list[float]:
    return [pose[3], pose[7], pose[11]]


def derive(ref: dict, test: dict, move_tol: float) -> list[dict]:
    changes = []
    for name in sorted(set(ref) | set(test)):
        in_ref, in_test = name in ref, name in test
        if in_ref and not in_test:
            changes.append({"name": name, "type": "removed",
                            "before_xy": centre(ref[name])[:2], "before_z": centre(ref[name])[2]})
        elif in_test and not in_ref:
            changes.append({"name": name, "type": "added",
                            "after_xy": centre(test[name])[:2], "after_z": centre(test[name])[2]})
        else:
            b, a = centre(ref[name]), centre(test[name])
            if math.dist(b, a) > move_tol:
                changes.append({"name": name, "type": "moved",
                                "before_xy": b[:2], "after_xy": a[:2],
                                "before_z": b[2], "after_z": a[2],
                                "displacement_m": round(math.dist(b, a), 3)})
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help="reference scene directory (planes/*/...anno)")
    ap.add_argument("test", type=Path, help="test scene directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--move-tol", type=float, default=0.15,
                    help="metres: centre displacement that counts as a move (default 0.15)")
    ap.add_argument("--no-positions", action="store_true",
                    help="write type-only entries (for scoring before cross-frame registration)")
    args = ap.parse_args()

    ref, test = scene_objects(args.reference), scene_objects(args.test)
    changes = derive(ref, test, args.move_tol)
    if args.no_positions:
        for c in changes:
            for key in ("before_xy", "after_xy", "before_z", "after_z", "displacement_m"):
                c.pop(key, None)
    doc = {
        "scene": f"ObChange {args.reference.name} vs {args.test.name}",
        "frame_note": "positions are metres in the dataset scene frame (z up), not the pipeline floor frame",
        "reference_objects": len(ref),
        "test_objects": len(test),
        "changes": changes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    kinds = {"added": 0, "removed": 0, "moved": 0}
    for c in changes:
        kinds[c["type"]] += 1
    print(f"{doc['scene']}: ref {len(ref)} objects, test {len(test)} -> "
          f"{kinds['added']} added, {kinds['removed']} removed, {kinds['moved']} moved")
    for c in changes:
        print(f"  {c['type']:<8} {c['name']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
