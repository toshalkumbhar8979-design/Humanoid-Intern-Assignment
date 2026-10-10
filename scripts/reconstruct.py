"""Phase 2 - reconstruct one sequence with MASt3R.

    python scripts/reconstruct.py data/before/frames --name before --out outputs/reconstructions

Writes outputs/reconstructions/<name>.npz (points, conf, colours, poses, focals) and
<name>.ply (confidence-filtered colour point cloud in the sequence's own frame).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconstruction import mast3r_backend  # noqa: E402


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="directory with frame_XXXX.png and manifest.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "reconstructions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scene-graph", default="swin-2",
                    help="pair graph; pair count grows with frames x window, so keep it small")
    ap.add_argument("--max-frames", type=int, default=40,
                    help="subsample uniformly to at most this many frames (host-RAM limit)")
    ap.add_argument("--niter", type=int, default=300)
    args = ap.parse_args()

    manifest = json.loads((args.frames / "manifest.json").read_text())
    paths = [args.frames / f["file"] for f in manifest["frames"]]
    if len(paths) > args.max_frames:
        idx = np.linspace(0, len(paths) - 1, args.max_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    if len(paths) < 3:
        raise SystemExit(f"need at least 3 frames, got {len(paths)}")

    rec = mast3r_backend.reconstruct(paths, device=args.device, size=args.size,
                                     scene_graph=args.scene_graph, niter=args.niter)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.name}.npz", **rec)

    keep = rec["conf"] >= np.quantile(rec["conf"], 0.3)
    stride = 4
    pts = rec["points"][:, ::stride, ::stride][keep[:, ::stride, ::stride]]
    cols = rec["colors"][:, ::stride, ::stride][keep[:, ::stride, ::stride]]
    write_ply(args.out / f"{args.name}.ply", pts, cols)
    print(f"{args.name}: {len(paths)} frames -> {args.out / (args.name + '.npz')} ({len(pts)} preview points)")


if __name__ == "__main__":
    main()
