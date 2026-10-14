"""One-command reproduction of the full pipeline.

    python scripts/run_all.py                         # uses data/before/before.mp4 and data/after/after.mp4
    python scripts/run_all.py --no-labeler            # skip OWLv2 labels (geometry and colour still reported)
    python scripts/run_all.py --before A.mp4 --after B.mp4

Each stage is a standalone script in scripts/ and can be re-run on its own.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--max-frames", type=int, default=60,
                    help="frame cap per reconstruction (host-RAM limit; 60 measured to fit this 8 GB GPU, "
                         "and the densest sampling the documentation's numbers are measured at)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs", help="output directory (default outputs/)")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="directory for the extracted frames (default data/<before|after>/frames beside the videos)")
    args = ap.parse_args()

    O = args.out
    if args.frames_dir:
        frames = {"before": args.frames_dir / "before", "after": args.frames_dir / "after"}
    else:
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
    # Object level: YOLO boxes lifted to 3D, matched across captures; the control is the null check.
    # The control runs FIRST and with pair verification off: its stayed-pair displacements measure
    # this capture pair's reconstruction noise, and the pair run then gets --verify-pairs just above
    # it (1.5x, floor 0.02 m). Only pairings farther apart than that are checked by the gate; closer
    # ones are stayed outright, because at that scale the spot test resolves nothing (it would
    # manufacture moves out of noise on a noisy capture). This runs before visualize.py so the
    # confirmed object changes can be drawn in the 3D views.
    run(S / "detect_objects.py", "--before-recon", recon / "before.npz", "--before-frames", frames["before"],
        "--after-recon", ctrl / "align" / "reconstructions" / "after_aligned.npz", "--after-frames", ctrl / "frames",
        "--max-frames", args.max_frames, "--detector", "yolo-world", "--out", O / "objects_null",
        "--verify-pairs", 0)
    recs = json.loads((O / "objects_null" / "objects.json").read_text())
    stay = [r["displacement_m"] for r in recs
            if r.get("kind") == "stayed" and r.get("displacement_m") is not None]
    noise = max(stay) if stay else 0.0
    verify = max(0.02, 1.5 * noise)
    print(f"object stage: control noise {noise:.3f} m -> pair runs use --verify-pairs {verify:.3f}", flush=True)
    run(S / "detect_objects.py", "--before-recon", recon / "before.npz", "--before-frames", frames["before"],
        "--after-recon", recon / "after_aligned.npz", "--after-frames", frames["after"],
        "--max-frames", args.max_frames, "--detector", "yolo-world", "--out", O / "objects",
        "--verify-pairs", verify, "--corroborate", O / "change_maps" / "changes.json")
    run(S / "visualize.py", recon / "before.npz", recon / "after_aligned.npz", O / "change_maps" / "changes.pkl",
        "--descriptions", O / "descriptions" / "changes.json", "--objects", O / "objects" / "objects.json",
        "--out", O / "visualizations")
    run(S / "make_graphs.py", recon / "before.npz", recon / "after_aligned.npz",
        ctrl / "align" / "reconstructions" / "after_aligned.npz", O / "graphs")
    print(f"\nDone. See {O / 'descriptions' / 'changes.md'}, {O / 'visualizations'} and {O / 'graphs'}")


if __name__ == "__main__":
    main()
