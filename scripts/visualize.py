"""Phase 6 - visualise the changes in 3D and in a floor-aligned bird's-eye view.

    python scripts/visualize.py outputs/reconstructions/before.npz \
        outputs/reconstructions/after_aligned.npz outputs/change_maps/changes.pkl \
        --descriptions outputs/descriptions/changes.json

Writes outputs/visualizations/bev.png (static figure) and outputs/visualizations/viewer.html
(interactive; open in any browser, works offline).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registration.align import surface_points  # noqa: E402
from visualization.plots import bev_png, floor_transform, transform_changes, viewer_html  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("changes", type=Path)
    ap.add_argument("--descriptions", type=Path, default=None, help="changes.json from describe_changes.py")
    ap.add_argument("--objects", type=Path, default=None,
                    help="objects.json from detect_objects.py; confirmed object changes are drawn as diamonds "
                         "(already in the floor frame)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "visualizations")
    args = ap.parse_args()

    before = dict(np.load(args.before))
    after = dict(np.load(args.after))
    with open(args.changes, "rb") as f:
        changes = pickle.load(f)

    labels = {}
    if args.descriptions and args.descriptions.exists():
        for r in json.loads(args.descriptions.read_text()):
            colour = f"{r['colour']} " if r.get("colour") else ""
            labels[r["id"]] = f"{colour}{r['label']}"

    b_pts, b_cols = surface_points(before)
    a_pts, a_cols = surface_points(after)
    R, shift = floor_transform(b_pts, before["c2w"][0])

    def to_floor(p: np.ndarray) -> np.ndarray:
        q = p @ R.T
        q[:, 2] += shift
        return q

    b_pts_f, a_pts_f = to_floor(b_pts), to_floor(a_pts)
    changes_f = transform_changes(changes, R, shift)

    objs = []
    if args.objects and args.objects.exists():
        objs = [o for o in json.loads(args.objects.read_text())
                if o.get("kind") in ("moved", "added", "removed")]

    args.out.mkdir(parents=True, exist_ok=True)
    bev_png(b_pts_f, a_pts_f, changes_f, labels, args.out / "bev.png", objects=objs)
    viewer_html(b_pts_f, b_cols, a_pts_f, a_cols, changes_f, labels, args.out / "viewer.html", objects=objs)
    print(f"wrote {args.out / 'bev.png'} and {args.out / 'viewer.html'}")


if __name__ == "__main__":
    main()
