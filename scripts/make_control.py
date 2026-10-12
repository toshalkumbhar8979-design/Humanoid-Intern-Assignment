"""Build a no-change control from one capture: every other frame, reconstructed independently.

    python scripts/make_control.py data/before/frames outputs/control/frames

The control shows the same unchanged room, so any change detected between it and the original
is reconstruction noise. The pipeline uses it to set the smallest region that may count as a change.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="frames directory with manifest.json")
    ap.add_argument("out", type=Path, help="output frames directory")
    ap.add_argument("--parity", choices=["odd", "even"], default="odd",
                    help="which frames to keep: odd (default) or even indices; two parities give two independent controls")
    args = ap.parse_args()

    man = json.loads((args.frames / "manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    kept = []
    for i, f in enumerate(man["frames"]):
        if i % 2 == (1 if args.parity == "odd" else 0):
            shutil.copy(args.frames / f["file"], args.out / f["file"])
            kept.append(f)
    (args.out / "manifest.json").write_text(json.dumps({"frames": kept}, indent=2))
    print(f"control ({args.parity}): {len(kept)} frames -> {args.out}")


if __name__ == "__main__":
    main()
