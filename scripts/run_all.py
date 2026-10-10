"""One-command reproduction of the full pipeline.

    python scripts/run_all.py                         # uses data/before/before.mp4 and data/after/after.mp4
    python scripts/run_all.py --no-labeler            # skip OWLv2 labels (geometry and colour still reported)
    python scripts/run_all.py --before A.mp4 --after B.mp4

Each stage is a standalone script in scripts/ and can be re-run on its own.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
O = ROOT / "outputs"


def run(*parts) -> None:
    cmd = [sys.executable, *map(str, parts)]
    print("\n$ " + " ".join(Path(c).name if i == 1 else str(c) for i, c in enumerate(cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, default=ROOT / "data" / "before" / "before.mp4")
    ap.add_argument("--after", type=Path, default=ROOT / "data" / "after" / "after.mp4")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--no-labeler", action="store_true")
    ap.add_argument("--max-frames", type=int, default=40, help="frame cap per reconstruction (host-RAM limit)")
    args = ap.parse_args()

    frames = {"before": ROOT / "data" / "before" / "frames", "after": ROOT / "data" / "after" / "frames"}
    recon = O / "reconstructions"
    for name, video in (("before", args.before), ("after", args.after)):
        run(S / "extract_frames.py", video, frames[name], "--fps", args.fps)
    for name in ("before", "after"):
        run(S / "reconstruct.py", frames[name], "--name", name, "--out", recon, "--max-frames", args.max_frames)
    run(S / "align.py", recon / "before.npz", recon / "after.npz", "--out", O)
    # no-change control: the same room reconstructed again from every other frame, then aligned
    ctrl = O / "control"
    run(S / "make_control.py", frames["before"], ctrl / "frames")
    run(S / "reconstruct.py", ctrl / "frames", "--name", "control", "--out", ctrl / "reconstructions",
        "--max-frames", args.max_frames)
    run(S / "align.py", recon / "before.npz", ctrl / "reconstructions" / "control.npz", "--out", ctrl / "align")
    run(S / "detect_changes.py", recon / "before.npz", recon / "after_aligned.npz", "--out", O / "change_maps",
        "--null", ctrl / "align" / "reconstructions" / "after_aligned.npz")
    desc = [S / "describe_changes.py", recon / "before.npz", recon / "after_aligned.npz",
            O / "change_maps" / "changes.pkl", "--out", O / "descriptions"]
    run(*(desc + (["--no-labeler"] if args.no_labeler else [])))
    run(S / "visualize.py", recon / "before.npz", recon / "after_aligned.npz", O / "change_maps" / "changes.pkl",
        "--descriptions", O / "descriptions" / "changes.json", "--out", O / "visualizations")
    run(S / "make_graphs.py", recon / "before.npz", recon / "after_aligned.npz",
        ctrl / "align" / "reconstructions" / "after_aligned.npz", O / "graphs")
    # object level: YOLO boxes lifted to 3D, matched across captures; the control is the null check
    run(S / "detect_objects.py", "--before-recon", recon / "before.npz", "--before-frames", frames["before"],
        "--after-recon", recon / "after_aligned.npz", "--after-frames", frames["after"],
        "--max-frames", args.max_frames, "--out", O / "objects")
    run(S / "detect_objects.py", "--before-recon", recon / "before.npz", "--before-frames", frames["before"],
        "--after-recon", ctrl / "align" / "reconstructions" / "after_aligned.npz", "--after-frames", ctrl / "frames",
        "--max-frames", args.max_frames, "--out", O / "objects_null")
    print(f"\nDone. See {O / 'descriptions' / 'changes.md'}, {O / 'visualizations'} and {O / 'graphs'}")


if __name__ == "__main__":
    main()
