"""Phase 3 - align the 'after' reconstruction into the 'before' frame.

    python scripts/align.py outputs/reconstructions/before.npz outputs/reconstructions/after.npz

Writes outputs/reconstructions/after_aligned.npz (points and camera poses transformed by T)
and outputs/alignment/alignment.json with the transform and quality metrics.
Check the overlay (outputs/visualizations/bev.png) before trusting the changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registration.align import align, estimate_up, surface_points, transform_recon  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    before = dict(np.load(args.before))
    after = dict(np.load(args.after))
    pb, pa = surface_points(before)[0], surface_points(after)[0]
    up_b = estimate_up(pb, before["c2w"][0])[0]
    up_a = estimate_up(pa, after["c2w"][0])[0]
    res = align(pb, pa, up_before=up_b, up_after=up_a)
    print("alignment:", res.summary())
    if res.fitness < 0.3:
        print("WARNING: low overlap - the two captures may not share enough structure; check the overlay.")

    aligned = transform_recon(after, res.T)
    (args.out / "reconstructions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "reconstructions" / "after_aligned.npz", **aligned)
    (args.out / "alignment").mkdir(parents=True, exist_ok=True)
    (args.out / "alignment" / "alignment.json").write_text(
        json.dumps({"T_after_to_before": res.T.tolist(), **res.summary()}, indent=2))
    print("wrote", args.out / "reconstructions" / "after_aligned.npz")


if __name__ == "__main__":
    main()
