"""Graphs of the pipeline's own results. They work without ground truth.

Writes three PNGs to <out_dir>:
  top_down_difference.png  top-down view in the floor frame: cells occupied only in the before capture
                           (red), only in the after capture (blue), or in both (grey)
  noise_floor.png          size of every candidate change on the real pair, against the same measure on the
                           no-change control, with the threshold the pipeline applies
  floor_heights.png        height of surface points above the fitted floor, per capture

  python scripts/make_graphs.py <before.npz> <after_aligned.npz> <control_aligned.npz> <out_dir>

All three captures are expressed in the before capture's frame, and the floor is the one fitted to the
before capture. Occupancy depends on what the camera saw, so differences partly reflect coverage.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from change_detection.voxel_change import ChangeConfig, detect_changes  # noqa: E402
from reconstruction.mast3r_backend import to_sequence  # noqa: E402
from registration.align import surface_points  # noqa: E402
from visualization.plots import floor_transform  # noqa: E402

CELL = 0.05          # metres per grid cell (same as the voxel size)
MIN_VOXELS = 50      # the pipeline's default size cutoff (scripts/detect_changes.py --min-voxels)


def to_floor(pts: np.ndarray, R: np.ndarray, shift: float) -> np.ndarray:
    q = pts @ R.T
    q[:, 2] += shift
    return q


def top_down(pts: np.ndarray, lo: np.ndarray, shape: tuple) -> np.ndarray:
    """Boolean occupancy grid of points that lie above the floor and below 2 m."""
    grid = np.zeros(shape, dtype=bool)
    p = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 2.0)]
    ix = np.clip(((p[:, 0] - lo[0]) / CELL).astype(int), 0, shape[1] - 1)
    iy = np.clip(((p[:, 1] - lo[1]) / CELL).astype(int), 0, shape[0] - 1)
    grid[iy, ix] = True
    return grid


def plot_top_down(b, a, lo, shape, out: Path):
    gb, ga = top_down(b, lo, shape), top_down(a, lo, shape)
    rgb = np.full((*shape, 3), 0.96)
    rgb[gb & ga] = (0.55, 0.55, 0.55)
    rgb[gb & ~ga] = (0.90, 0.20, 0.20)
    rgb[ga & ~gb] = (0.20, 0.45, 1.00)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="lower", extent=[lo[0], lo[0] + shape[1] * CELL, lo[1], lo[1] + shape[0] * CELL])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down, floor frame\nred = occupied only before, blue = only after, grey = both")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return int((gb & ~ga).sum()), int((ga & ~gb).sum())


def plot_noise_floor(seq_b, seq_a, seq_c, out: Path):
    cfg = ChangeConfig(voxel=CELL, min_voxels=1)
    real = [c.voxels for c in detect_changes(seq_b, seq_a, cfg)]
    null = [c.voxels for c in detect_changes(seq_b, seq_c, cfg)]
    noise_max = max(null, default=0)
    threshold = max(MIN_VOXELS, noise_max + 1)
    hi = max(real + null + [threshold]) + 1
    bins = np.logspace(0, np.log10(hi), 30)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(null, bins=bins, alpha=0.6, color="grey", label=f"no-change control ({len(null)} regions)")
    ax.hist(real, bins=bins, alpha=0.6, color="tab:blue", label=f"real pair ({len(real)} regions)")
    ax.axvline(threshold, color="tab:red", ls="--", label=f"threshold {threshold} voxels")
    ax.set_xscale("log")
    ax.set_xlabel("region size (voxels, log scale)")
    ax.set_ylabel("count")
    ax.set_title("Candidate regions vs the noise floor (5 cm voxels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return len(real), len(null), noise_max, threshold, sum(v >= threshold for v in real)


def plot_heights(hs: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-1.0, 1.0001, 0.02)
    colours = {"before": "tab:blue", "after": "tab:orange", "control": "grey"}
    for name, z in hs.items():
        ax.hist(z, bins=bins, histtype="step", lw=1.8, color=colours[name],
                label=f"{name} ({(z < -0.05).mean() * 100:.0f}% below -5 cm)")
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("height above fitted floor (m)")
    ax.set_ylabel("surface points")
    ax.set_title("Surface heights above the floor (points below 0 should be rare)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recon = {k: dict(np.load(p)) for k, p in (("before", args.before), ("after", args.after),
                                             ("control", args.control))}
    pts = {k: surface_points(r)[0] for k, r in recon.items()}
    R, shift = floor_transform(pts["before"], recon["before"]["c2w"][0])
    f = {k: to_floor(p, R, shift) for k, p in pts.items()}

    everything = np.concatenate([f["before"], f["after"]])[:, :2]
    lo = np.percentile(everything, 0.5, axis=0)
    hi = np.percentile(everything, 99.5, axis=0)
    shape = (int(np.ceil((hi[1] - lo[1]) / CELL)), int(np.ceil((hi[0] - lo[0]) / CELL)))
    only_b, only_a = plot_top_down(f["before"], f["after"], lo, shape, args.out / "top_down_difference.png")
    print(f"top_down_difference.png: cells only before {only_b}, only after {only_a}")

    seqs = {"before": to_sequence("before", recon["before"]),
            "after": to_sequence("after", recon["after"]),
            "control": to_sequence("control", recon["control"])}
    n_real, n_null, noise_max, thr, n_over = plot_noise_floor(
        seqs["before"], seqs["after"], seqs["control"], args.out / "noise_floor.png")
    print(f"noise_floor.png: real {n_real} regions, control {n_null} (largest {noise_max} voxels), "
          f"threshold {thr}, real regions over threshold {n_over}")

    plot_heights({k: f[k][:, 2] for k in f}, args.out / "floor_heights.png")
    print("floor_heights.png written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
