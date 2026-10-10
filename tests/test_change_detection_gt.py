"""Check the geometric change detector against the synthetic ground truth.

Depth and colour come from ray-casting the GT meshes from the known orbit cameras
(so occlusion is real). This mimics what a dense reconstruction would provide.
Run:  python tests/test_change_detection_gt.py
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_synthetic_data as synth  # noqa: E402
from change_detection.voxel_change import ChangeConfig, Sequence, detect_changes  # noqa: E402

W, H = 128, 72
F = synth.FOCAL * W / synth.W
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def scene_mesh(state: str) -> o3d.t.geometry.TriangleMesh:
    rng = np.random.default_rng(0)
    legacy = o3d.geometry.TriangleMesh()
    for _, m in synth.build_scene(state, rng):
        legacy += m
    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def render_sequence(name: str, state: str, start: float, end: float, n_cams: int = 24) -> Sequence:
    tmesh = scene_mesh(state)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(tmesh)
    tri = tmesh.triangle.indices.numpy()
    vcol = tmesh.vertex.colors.numpy()
    c2w_list, pts_list, col_list, far_list = [], [], [], []
    for t in np.linspace(0, 1, n_cams):
        eye = synth.orbit_pose(t, start, end, np.zeros(2))
        E = synth.look_at_extrinsic(eye, synth.CENTER)
        rays_t = o3d.core.Tensor(
            o3d.t.geometry.RaycastingScene.create_rays_pinhole(K, E, W, H).numpy(), dtype=o3d.core.Dtype.Float32)
        ans = rc.cast_rays(rays_t)
        t_hit = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy()
        ok = np.isfinite(t_hit)
        r = rays_t.numpy()
        pts = r[ok, :3] + t_hit[ok, None] * r[ok, 3:6]
        cols = vcol[tri[prim[ok], 0]]
        # rays that hit nothing: MASt3R depth covers the whole frame, so emulate that with far points
        # that only carve free space (they never become surfaces)
        far = r[~ok, :3] + 5.0 * r[~ok, 3:6]
        c2w_list.append(np.linalg.inv(E))
        pts_list.append(pts)
        col_list.append(cols)
        far_list.append(far)
    return Sequence(name=name, cam_c2w=np.stack(c2w_list), cam_points=pts_list,
                    cam_colors=col_list, cam_free_points=far_list)


def main():
    before = render_sequence("before", "before", 240, 300)
    after = render_sequence("after", "after", 250, 310)
    cfg = ChangeConfig(voxel=0.05)
    changes = detect_changes(before, after, cfg)
    print(f"detected {len(changes)} change regions")
    for c in changes:
        cb = None if c.centroid_before is None else np.round(c.centroid_before, 2).tolist()
        ca = None if c.centroid_after is None else np.round(c.centroid_after, 2).tolist()
        disp = "" if c.displacement_m is None else f" disp={c.displacement_m:.2f}m"
        col = "" if c.colour is None else f" rgb={np.round(c.colour, 2).tolist()}"
        print(f"  #{c.id} {c.kind:8s} vol={c.volume_m3 * 1e3:5.1f}L conf={c.confidence:.2f}"
              f" ext={np.round(c.obb_extents, 2).tolist()}{col}"
              f" before={cb} after={ca}{disp} view={c.best_view}")
    print("ground truth:", [(g["name"], g["type"]) for g in synth.ground_truth()["changes"]])
    return changes


if __name__ == "__main__":
    main()
