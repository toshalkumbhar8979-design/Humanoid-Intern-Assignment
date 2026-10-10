"""Phase 4 - geometric change detection.

    python scripts/detect_changes.py outputs/reconstructions/before.npz \
        outputs/reconstructions/after_aligned.npz

Writes outputs/change_maps/changes.pkl (full change records for later steps) and
outputs/change_maps/changes.json (human-readable summary).
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

from change_detection.voxel_change import ChangeConfig, detect_changes  # noqa: E402
from reconstruction.mast3r_backend import to_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "change_maps")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--min-voxels", type=int, default=50)
    ap.add_argument("--null", type=Path, default=None,
                    help="aligned reconstruction of an UNCHANGED room (control). Changes must be larger than the "
                         "biggest artifact the detector finds on it.")
    args = ap.parse_args()

    before_seq = to_sequence("before", dict(np.load(args.before)))
    after_seq = to_sequence("after", dict(np.load(args.after)))
    cfg = ChangeConfig(voxel=args.voxel, min_voxels=args.min_voxels)
    noise_max = 0
    if args.null is not None:
        # measure the largest artifact on a scene that did not change, with no size cutoff
        null_seq = to_sequence("null", dict(np.load(args.null)))
        null_changes = detect_changes(before_seq, null_seq, ChangeConfig(voxel=args.voxel, min_voxels=1))
        noise_max = max((c.voxels for c in null_changes), default=0)
        cfg = ChangeConfig(voxel=args.voxel, min_voxels=max(args.min_voxels, noise_max + 1))
        print(f"noise floor from control: {len(null_changes)} artifacts, largest {noise_max} voxels")
    changes = detect_changes(before_seq, after_seq, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "changes.pkl", "wb") as f:
        pickle.dump(changes, f)
    summary = [{
        "id": c.id, "kind": c.kind, "volume_l": round(c.volume_m3 * 1e3, 2),
        "obb_extents_m": [round(float(x), 3) for x in c.obb_extents],
        "displacement_m": None if c.displacement_m is None else round(float(c.displacement_m), 3),
        "confidence": round(c.confidence, 3),
        "centroid_before": None if c.centroid_before is None else np.round(c.centroid_before, 3).tolist(),
        "centroid_after": None if c.centroid_after is None else np.round(c.centroid_after, 3).tolist(),
    } for c in changes]
    (args.out / "changes.json").write_text(json.dumps(summary, indent=2))
    counts = {k: sum(c.kind == k for c in changes) for k in ("added", "removed", "moved")}
    print(f"detected {len(changes)} changes {counts} -> {args.out}")


if __name__ == "__main__":
    main()
