"""Generate a synthetic 'before / after' pair of orbit videos with ground-truth changes.

Used as a reproducible stand-in for phone captures (see README "Capture protocol").
The scene is a small desk corner. Between 'before' and 'after' it contains:

  * chair        MOVED  ~1.2 m to the left
  * cardboard box MOVED  ~0.8 m
  * backpack     ADDED  on the desk
  * plant        REMOVED from the windowsill corner

Writes:
  data/before/before.mp4, data/after/after.mp4   (1920x1080 is slow; 960x540 @ 30 fps)
  data/synthetic/ground_truth.json               (used only for evaluation)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image

W, H, FPS, SECONDS = 960, 540, 30, 24
FOCAL = 720.0
BACKGROUND_RGB = np.array([237, 235, 230])  # matches set_background in render_video
CENTER = np.array([0.0, 0.35, 0.6])


# ----------------------------------------------------------------------------- scene
def _paint(mesh, color, rng, noise=0.04):
    mesh.compute_vertex_normals()
    cols = np.clip(np.asarray(color) + rng.normal(0, noise, (len(mesh.vertices), 3)), 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    return mesh


def _box(size, center, yaw, color, rng):
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(-np.array(size) / 2)
    m.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]), center=(0, 0, 0))
    m.translate(center)
    return _paint(m, color, rng)


def _cyl(radius, height, center, color, rng):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=24)
    m.translate(center)
    return _paint(m, color, rng)


def _sphere(radius, center, color, rng):
    m = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=20)
    m.translate(center)
    return _paint(m, color, rng)


def _floor(rng):
    # Larger floors made the offscreen renderer return background-only frames (see DESIGN.md).
    xs = np.linspace(-1.6, 1.6, 65)
    ys = np.linspace(-1.4, 1.4, 57)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx, gy, np.zeros_like(gx)], -1).reshape(-1, 3)
    tris = []
    nx, ny = len(xs), len(ys)
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = (i + 1) * ny + j + 1
            d = i * ny + j + 1
            tris += [[a, b, c], [a, c, d]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(np.array(tris)))
    checker = ((np.floor(verts[:, 0] / 0.2) + np.floor(verts[:, 1] / 0.2)) % 2)[:, None]
    base = 0.55 + 0.12 * checker
    cols = np.clip(base + rng.normal(0, 0.03, base.shape), 0, 1) * np.array([1.0, 0.95, 0.88])
    m.vertex_colors = o3d.utility.Vector3dVector(cols)
    m.compute_vertex_normals()
    return m


def build_scene(state: str, rng: np.random.Generator):
    """state in {'before','after'}; returns list of (name, mesh)."""
    # No wall meshes: large boxes in the view made the Open3D renderer output black frames.
    parts = [("floor", _floor(rng))]

    # desk (against back wall)
    wood = (0.62, 0.42, 0.24)
    parts.append(("desk_top", _box((1.2, 0.6, 0.04), (0.0, 1.0, 0.75), 0, wood, rng)))
    for k, (dx, dy) in enumerate([(-0.55, 0.72), (0.55, 0.72), (-0.55, 1.28), (0.55, 1.28)]):
        parts.append((f"desk_leg{k}", _box((0.04, 0.04, 0.73), (dx, dy, 0.365), 0, wood, rng)))
    parts.append(("bottle", _cyl(0.035, 0.22, (0.35, 0.92, 0.77 + 0.11), (0.1, 0.35, 0.9), rng)))

    # windowsill-style shelf + plant (before only)
    if state == "before":
        parts.append(("plant_pot", _cyl(0.1, 0.16, (-1.1, 1.15, 0.08), (0.7, 0.35, 0.2), rng)))
        parts.append(("plant_leaves", _sphere(0.15, (-1.1, 1.15, 0.42), (0.15, 0.55, 0.2), rng)))

    # chair (black office chair, moved)
    chair_xy = (0.5, 0.05) if state == "before" else (-0.7, 0.05)
    cx, cy = chair_xy
    black = (0.08, 0.08, 0.09)
    parts.append(("chair_seat", _box((0.46, 0.46, 0.06), (cx, cy, 0.46), 0.3, black, rng)))
    parts.append(("chair_back", _box((0.46, 0.06, 0.5), (cx, cy - 0.23, 0.74), 0.3, black, rng)))
    parts.append(("chair_post", _cyl(0.03, 0.4, (cx, cy, 0.23), (0.3, 0.3, 0.3), rng)))
    for k, a in enumerate(np.linspace(0, 2 * np.pi, 5, endpoint=False)):
        px, py = cx + 0.2 * np.cos(a), cy + 0.2 * np.sin(a)
        parts.append((f"chair_leg{k}", _box((0.05, 0.05, 0.03), (px, py, 0.015), 0, (0.3, 0.3, 0.3), rng)))

    # cardboard box on floor (moved)
    bx, by = ((-0.5, -0.45) if state == "before" else (0.3, -0.55))
    parts.append(("cardboard_box", _box((0.4, 0.3, 0.3), (bx, by, 0.15), 0.1, (0.72, 0.56, 0.36), rng)))

    # backpack (added on the desk, after only)
    if state == "after":
        parts.append(("backpack", _box((0.3, 0.16, 0.38), (-0.35, 0.98, 0.79 + 0.19), 0.0, (0.75, 0.12, 0.12), rng)))
    return parts


def ground_truth():
    return {
        "scene": "synthetic desk corner (2 orbit videos, 4 changes)",
        "changes": [
            {"name": "chair", "type": "moved", "label": "black office chair",
             "before_xy": [0.5, 0.05], "after_xy": [-0.7, 0.05], "displacement_m": 1.2},
            {"name": "cardboard_box", "type": "moved", "label": "cardboard box",
             "before_xy": [-0.5, -0.45], "after_xy": [0.3, -0.55], "displacement_m": 0.8},
            {"name": "backpack", "type": "added", "label": "red backpack",
             "after_xy": [-0.35, 0.98]},
            {"name": "plant", "type": "removed", "label": "potted plant",
             "before_xy": [-1.1, 1.15]},
        ],
    }


# ----------------------------------------------------------------------------- camera
def look_at_extrinsic(eye, target, up=(0, 0, 1.0)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ eye
    return E


def orbit_pose(t: float, start_deg: float, end_deg: float, rng_jitter: np.ndarray):
    """t in [0,1]; eye on an arc in front of the desk; smooth height variation."""
    phi = np.deg2rad(start_deg + (end_deg - start_deg) * t)
    R = 2.7
    eye = CENTER + np.array([R * np.cos(phi), R * np.sin(phi), 0.0])
    eye[2] = 1.25 + 0.25 * np.sin(2 * np.pi * t * 1.5) + rng_jitter[0]
    eye[0] += rng_jitter[1]
    return eye


# ----------------------------------------------------------------------------- render
def render_video(state: str, out_mp4: Path, start_deg: float, end_deg: float, seed: int, tmp: Path):
    rng = np.random.default_rng(seed)
    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background(np.array([0.93, 0.92, 0.9, 1.0]))
    r.scene.scene.set_sun_light([0.4, -0.5, -0.8], [1.0, 0.98, 0.94], 70000)
    r.scene.scene.enable_sun_light(True)
    r.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, np.array([0.5, -0.4, -0.7]))
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = 0.8
    for name, mesh in build_scene(state, np.random.default_rng(0)):
        r.scene.add_geometry(name, mesh, mat)

    K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
    n = FPS * SECONDS
    frames_dir = tmp / state
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = i / (n - 1)
        jitter = rng.normal(0, 0.01, 2)
        eye = orbit_pose(t, start_deg, end_deg, jitter)
        r.setup_camera(K, look_at_extrinsic(eye, CENTER), W, H)
        # Open3D's offscreen renderer intermittently returns black or background-only buffers,
        # so redraw until the frame actually contains the scene (bounded retries).
        for _ in range(12):
            img = np.asarray(r.render_to_image())
            # valid = not black, and a real share of pixels differ from the flat background colour
            differs = np.abs(img.astype(int) - BACKGROUND_RGB).sum(-1) > 60
            no_black_rows = (img.max(axis=(1, 2)) > 30).all()  # renderer sometimes leaves a black band
            if img.mean() > 20 and differs.mean() > 0.05 and no_black_rows:
                break
        else:
            # Keep the last render; the Phase-1 quality filter (extract_frames.py) rejects it.
            print(f"warning: {state} frame {i} still has render artefacts after retries")
        Image.fromarray(img).save(frames_dir / f"f{i:05d}.png")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "f%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_mp4)],
        check=True,
    )
    shutil.rmtree(frames_dir)
    print(f"wrote {out_mp4} ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    root = Path(args.root)
    tmp = Path(tempfile.mkdtemp(prefix="synth_"))
    # Different viewpoints for the two captures (plan: "capture from different viewpoints")
    # Arcs stay in front of the open side of the room: the left wall would occlude a camera further left.
    render_video("before", root / "before" / "before.mp4", 240, 300, seed=1, tmp=tmp)
    render_video("after", root / "after" / "after.mp4", 250, 310, seed=2, tmp=tmp)
    gt_path = root / "synthetic" / "ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(ground_truth(), indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    print("ground truth ->", gt_path)


if __name__ == "__main__":
    main()
