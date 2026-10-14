"""Convert an ObChange rosbag (HSR robot RGB stream) to an mp4 the pipeline can ingest.

The room ZIPs of ObChange (TU Wien, CC BY 4.0) hold one rosbag per recording with the tf tree,
the RGB and depth streams and the camera intrinsics. This reads the RGB stream, subsamples it to
about --fps frames per second of recording, and writes an mp4 that `run_all.py` accepts as
`--before` / `--after` like any phone video.

  python scripts/obchange_to_frames.py data/datasets/obchange/Kitchen/scene2.bag \\
      data/datasets/obchange/videos/kitchen_scene2.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def rgb_topic(reader) -> str:
    """First image topic that is not depth; prefer names containing 'rgb'."""
    topics = [t for t, info in reader.topics.items()
              if info.msgtype and "image" in info.msgtype.lower() and "depth" not in t.lower()]
    if not topics:
        listed = ", ".join(f"{t} ({info.msgtype})" for t, info in sorted(reader.topics.items()))
        raise SystemExit(f"no RGB image topic in bag; topics: {listed}")
    topics.sort(key=lambda t: ("rgb" not in t.lower(), t))
    return topics[0]


def decode(msg, msgtype: str) -> np.ndarray | None:
    """RGB uint8 image from a sensor_msgs/Image or CompressedImage message."""
    if "CompressedImage" in msgtype:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding not in ("rgb8", "bgr8"):
        return None
    img = arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path, help="output .mp4")
    ap.add_argument("--fps", type=float, default=3.0, help="frames per second of recording to keep")
    ap.add_argument("--max-frames", type=int, default=120, help="cap on extracted frames")
    args = ap.parse_args()

    import cv2
    from rosbags.highlevel import AnyReader

    frames: list[np.ndarray] = []
    stamps: list[float] = []
    with AnyReader([args.bag]) as reader:
        topic = rgb_topic(reader)
        conns = [c for c in reader.connections if c.topic == topic]
        print(f"{args.bag.name}: topic {topic} ({conns[0].msgtype})")
        last_kept = None
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if last_kept is not None and t - last_kept < 1.0 / args.fps:
                continue
            img = decode(msg, conn.msgtype)
            if img is None:
                continue
            frames.append(img)
            stamps.append(t)
            last_kept = t
            if len(frames) >= args.max_frames:
                break
    if len(frames) < 3:
        raise SystemExit(f"only {len(frames)} frames decoded from {args.bag}")
    dur = max(stamps[-1] - stamps[0], 1e-6)
    h, w = frames[0].shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         float(max(1.0, round(len(frames) / dur))), (w, h))
    if not vw.isOpened():
        raise SystemExit("cannot open mp4 writer")
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"{len(frames)} frames over {dur:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
