"""Phase 5 - semantic labels and natural-language descriptions.

    python scripts/describe_changes.py outputs/reconstructions/before.npz \
        outputs/reconstructions/after_aligned.npz outputs/change_maps/changes.pkl

Writes outputs/descriptions/changes.json and outputs/descriptions/changes.md.
Use --no-labeler to skip OWLv2 (labels become 'object'; geometry and colour are still reported).
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

from semantics.describe import OwlLabeler, describe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("changes", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "descriptions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-labeler", action="store_true")
    args = ap.parse_args()

    recons = {"before": dict(np.load(args.before)), "after": dict(np.load(args.after))}
    with open(args.changes, "rb") as f:
        changes = pickle.load(f)
    labeler = None if args.no_labeler else OwlLabeler(device=args.device)
    records = describe(changes, recons, labeler)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "changes.json").write_text(json.dumps(records, indent=2))
    lines = ["# What changed in the room", ""]
    lines += [f"- **{r['kind']}** ({r['confidence']:.2f} conf.) - {r['text']}" for r in records]
    note = "Labels from OWLv2 open-vocabulary detection" if labeler else "Labels skipped (--no-labeler)"
    lines += ["", f"_{note}; distances are metres from MASt3R's metric scale._"]
    (args.out / "changes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
