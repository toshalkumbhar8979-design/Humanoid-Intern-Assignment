"""Phase 1 - extract frames from a capture and drop low-quality ones.

    python scripts/extract_frames.py data/before/before.mp4 data/before/frames --fps 3

Quality filter (plan Sec. 4.1):
  * reject frames with a black band / near-black frame (encoder or renderer glitches),
  * reject blurry frames: variance of the Laplacian below a fraction of the sequence median.
Kept frames are written as frame_XXXX.png and a manifest.json lists what was kept and why.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def has_black_band(bgr: np.ndarray) -> bool:
    # any row that is (almost) entirely black means the frame is broken
    row_max = bgr.max(axis=(1, 2))
    return bool((row_max < 30).any())


def clear_frames(out_dir: Path) -> None:
    """Remove a previous extraction's frames and manifest from `out_dir`.

    Without this, frames left over from an earlier run (different --fps or blur settings) survive in
    the directory and pollute the downstream 40-frame sample: two identical runs of the pipeline
    could then reconstruct different scenes and report different changes. Measured: the stale files
    flipped the reported change on the same pair of videos. This run's output replaces them.
    """
    for stale in sorted(out_dir.glob("frame_*.png")):
        stale.unlink()
    (out_dir / "manifest.json").unlink(missing_ok=True)


def extract(video: Path, out_dir: Path, fps: float, blur_frac: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_frames(out_dir)
    tmp = Path(tempfile.mkdtemp(prefix="frames_"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps}",
         str(tmp / "raw_%05d.png")],
        check=True,
    )
    raw = sorted(tmp.glob("raw_*.png"))
    scores, imgs = [], []
    for p in raw:
        img = cv2.imread(str(p))
        imgs.append((p, img))
        scores.append(sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    median = float(np.median(scores)) if scores else 0.0

    kept, rejected = [], []
    for idx, ((p, img), s) in enumerate(zip(imgs, scores)):
        reason = None
        if img.mean() < 20 or has_black_band(img):
            reason = "black_or_banded"
        elif s < blur_frac * median:
            reason = "blurry"
        if reason:
            rejected.append({"source": p.name, "reason": reason, "sharpness": round(s, 2)})
            continue
        name = f"frame_{len(kept):04d}.png"
        shutil.copy(p, out_dir / name)
        kept.append({"file": name, "source_index": idx, "sharpness": round(s, 2)})

    manifest = {
        "video": str(video), "fps": fps, "extracted": len(raw), "kept": len(kept),
        "rejected": rejected, "median_sharpness": round(median, 2), "frames": kept,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=float, default=3.0, help="sampling rate (plan: 2-5 fps)")
    ap.add_argument("--blur-frac", type=float, default=0.5,
                    help="reject frames with Laplacian variance below this fraction of the median")
    args = ap.parse_args()
    m = extract(args.video, args.out_dir, args.fps, args.blur_frac)
    print(f"{args.video.name}: extracted {m['extracted']}, kept {m['kept']}, "
          f"rejected {len(m['rejected'])} -> {args.out_dir}")


if __name__ == "__main__":
    main()
