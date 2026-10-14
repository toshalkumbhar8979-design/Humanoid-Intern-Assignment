"""Phase 3 - align the 'after' reconstruction into the 'before' frame.

Two captures from different viewpoints have arbitrary relative poses, and FPFH+RANSAC alone can
lock onto a symmetric wrong solution (for example a 180 degree flip) in a room with a large floor.
We therefore combine:

1. Gravity prior: the floor normal of each sequence is estimated (RANSAC plane fit, signed with
   camera 0). That fixes two rotation degrees of freedom, leaving only yaw.
2. Candidate initialisations: a 30-degree yaw sweep around the floor normal, plus FPFH+RANSAC.
3. Point-to-plane ICP from every candidate; keep the best by fitness, then inlier RMSE.
4. Trimmed refinement: drop the worst (1 - trim_quantile) residuals so that moved objects do not
   pull the static background off-register.

The result T maps 'after' coordinates to 'before' coordinates (homogeneous 4x4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


@dataclass
class AlignResult:
    T: np.ndarray            # (4, 4) after -> before
    fitness: float           # fraction of 'after' points with a neighbour in 'before' (final ICP)
    inlier_rmse: float       # metres
    coarse_fitness: float    # fitness of the FPFH+RANSAC initialisation
    trimmed_fraction: float  # fraction of residuals discarded in the refinement

    def summary(self) -> dict:
        return {"fitness": round(self.fitness, 4), "inlier_rmse_m": round(self.inlier_rmse, 4),
                "coarse_fitness": round(self.coarse_fitness, 4),
                "trimmed_fraction": round(self.trimmed_fraction, 3)}


def _refine_plane(seed_pts: np.ndarray, all_pts: np.ndarray, n0: np.ndarray, thr: float = 0.03,
                  iters: int = 4):
    """Least-squares refit of a plane seeded by RANSAC inliers.

    RANSAC on a noisy cloud lands on a slightly different plane each run (4-16 deg spread on the same
    points), so each refinement step re-collects every point within `thr` of the current plane and refits
    its normal by SVD. Returns (unit normal, refined inlier points).
    """
    n = n0 / np.linalg.norm(n0)
    c = seed_pts.mean(0)
    inl = seed_pts
    for _ in range(iters):
        near = all_pts[np.abs((all_pts - c) @ n) < thr]
        if len(near) < 3:
            break
        inl = near
        c = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[-1]
    return n, inl


def _estimate_up_once(points: np.ndarray, cam0_c2w: np.ndarray, n_planes: int = 6,
                      min_frac: float = 0.02, horizontal_cos: float = 0.6):
    """One RANSAC pass: returns (up, floor_offset, floor_inlier_points).

    The floor is not always the largest plane: walls and desk faces can have more points. So we fit the
    top few planes and keep the horizontal ones (normal within ~53 deg of camera 0's y axis, which points
    down in a level phone capture). Among those, the floor is the lowest. 'up' opposes y_cam.
    """
    pts = points.astype(np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.random_down_sample(min(1.0, 60000 / max(len(points), 1)))
    y_cam = cam0_c2w[:3, :3] @ np.array([0.0, 1.0, 0.0])
    min_inl = max(100, int(min_frac * len(pcd.points)))

    best = None  # (height of floor inliers along the candidate up, up, offset)
    rest = pcd
    for _ in range(n_planes):
        if len(rest.points) < min_inl:
            break
        plane, inliers = rest.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=2000)
        if len(inliers) < min_inl:
            break
        n = np.asarray(plane[:3], dtype=np.float64)
        n /= np.linalg.norm(n)
        if abs(n @ y_cam) >= horizontal_cos:
            seed_pts = np.asarray(rest.select_by_index(inliers).points)
            n, inl = _refine_plane(seed_pts, np.asarray(rest.points), n)
            up = -n if n @ y_cam > 0 else n
            h = float(np.median(inl @ up))
            if best is None or h < best[0]:
                best = (h, up, h, inl)
        rest = rest.select_by_index(inliers, invert=True)

    if best is None:  # no horizontal plane found: fall back to the largest plane
        plane, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=2000)
        n = np.asarray(plane[:3], dtype=np.float64)
        n /= np.linalg.norm(n)
        up = -n if n @ y_cam > 0 else n
        inl = np.asarray(pcd.select_by_index(inliers).points)
        return up, float(np.median(inl @ up)), inl
    return best[1], best[2], best[3]


def estimate_up(points: np.ndarray, cam0_c2w: np.ndarray, runs: int = 15, **kwargs):
    """Unit 'up' vector and floor offset for one reconstruction, robust to RANSAC randomness.

    RANSAC picks a slightly different floor on each run (4-16 deg spread on the same points), so we take
    the coordinate-wise median of `runs` passes (after sign alignment) and recompute the offset from the
    pooled floor inliers along that 'up'. The RNG is seeded, so the fit is reproducible: identical inputs
    give an identical floor frame (measured by running the object stage twice and diffing its output).
    """
    o3d.utility.random.seed(0)
    results = [_estimate_up_once(points, cam0_c2w, **kwargs) for _ in range(runs)]
    ref = results[0][0]
    ups = np.array([u if u @ ref >= 0 else -u for u, _, _ in results])
    up = np.median(ups, axis=0)
    up /= np.linalg.norm(up)
    pooled = np.concatenate([inl for _, _, inl in results])
    return up, float(np.median(pooled @ up))


def _to_pcd(pts: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    p = p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return p


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _icp(src, tgt, dist: float, T0: np.ndarray, iters: int):
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return o3d.pipelines.registration.registration_icp(
        src, tgt, dist, T0, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))


def _gravity_inits(points_before: np.ndarray, points_after: np.ndarray,
                   up_before: np.ndarray, up_after: np.ndarray) -> list[np.ndarray]:
    """Rotations that map the after floor normal onto the before one, with a yaw sweep."""
    up_b = up_before / np.linalg.norm(up_before)
    up_a = up_after / np.linalg.norm(up_after)
    R0 = Rotation.align_vectors([up_b], [up_a])[0].as_matrix()   # R0 @ up_a == up_b
    c_a, c_b = points_after.mean(0), points_before.mean(0)
    inits = []
    for yaw in np.arange(0, 360, 30):
        R = Rotation.from_rotvec(up_b * np.deg2rad(yaw)).as_matrix() @ R0
        inits.append(_pose(R, c_b - R @ c_a))
    return inits


def _colored_refine(pts_a: np.ndarray, col_a: np.ndarray, pts_b: np.ndarray, col_b: np.ndarray,
                    T: np.ndarray, voxel: float) -> np.ndarray:
    """Colour-aware ICP refinement (Park, Zhou, Koltun, ICCV 2017), multi-scale as in the Open3D tutorial.

    The colour term locks the tangential direction that point-to-plane ICP can slide along on a flat floor.
    Colours must be in [0, 1].
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_a))
    src.colors = o3d.utility.Vector3dVector(np.clip(col_a, 0, 1).astype(np.float64))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_b))
    tgt.colors = o3d.utility.Vector3dVector(np.clip(col_b, 0, 1).astype(np.float64))
    for radius, iters in ((voxel, 50), (voxel / 2, 30), (voxel / 4, 14)):
        s = src.voxel_down_sample(radius)
        t = tgt.voxel_down_sample(radius)
        s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        r = o3d.pipelines.registration.registration_colored_icp(
            s, t, radius, T, o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6,
                                                              max_iteration=iters))
        T = r.transformation
    return T


def align(points_before: np.ndarray, points_after: np.ndarray, voxel: float = 0.04,
          trim_quantile: float = 0.85, up_before: np.ndarray | None = None,
          up_after: np.ndarray | None = None, colors_before: np.ndarray | None = None,
          colors_after: np.ndarray | None = None, use_color: bool = False) -> AlignResult:
    src, tgt = _to_pcd(points_after, voxel), _to_pcd(points_before, voxel)
    o3d.utility.random.seed(0)   # deterministic RANSAC: same inputs, same alignment

    # FPFH + RANSAC (global, no prior)
    fpfh_r = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    src_f = o3d.pipelines.registration.compute_fpfh_feature(src, fpfh_r)
    tgt_f = o3d.pipelines.registration.compute_fpfh_feature(tgt, fpfh_r)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200_000, 0.999))
    inits = [coarse.transformation]

    # gravity prior + yaw sweep
    if up_before is not None and up_after is not None:
        inits += _gravity_inits(points_before, points_after, up_before, up_after)

    # ICP from every candidate; keep the best fit
    best_T, best_key = None, None
    for T0 in inits:
        r = _icp(src, tgt, voxel * 3, T0, 50)          # wide correspondence gate for the sweep
        r = _icp(src, tgt, voxel * 1.0, r.transformation, 50)
        key = (r.fitness, -r.inlier_rmse)
        if best_key is None or key > best_key:
            best_key, best_T = key, r.transformation

    # trimmed refinement: discard the worst residuals (moved objects) and re-fit
    full = points_after @ best_T[:3, :3].T + best_T[:3, 3]
    d = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full)).compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_before))))
    keep = d <= np.quantile(d, trim_quantile)
    trimmed = _to_pcd(points_after[keep], voxel)
    final = _icp(trimmed, tgt, voxel * 0.8, best_T, 60).transformation

    # optional colour refinement on the full clouds (off by default; see DESIGN.md for the measurement)
    if use_color and colors_before is not None and colors_after is not None:
        final = _colored_refine(points_after, colors_after, points_before, colors_before, final, voxel)

    ev = o3d.pipelines.registration.evaluate_registration(src, tgt, voxel * 1.0, final)
    return AlignResult(T=final, fitness=float(ev.fitness), inlier_rmse=float(ev.inlier_rmse),
                       coarse_fitness=float(coarse.fitness), trimmed_fraction=float(1 - keep.mean()))


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def surface_mask(conf: np.ndarray, conf_quantile: float = 0.3) -> np.ndarray:
    """Pixels that are real surface observations: positive confidence, top (1 - q) of those."""
    pos = conf > 0
    if not pos.any():
        return pos
    return pos & (conf >= np.quantile(conf[pos], conf_quantile))


def surface_points(recon: dict, stride: int = 4, conf_quantile: float = 0.3):
    """Confident surface points and their colours from a reconstruction (subsampled by stride)."""
    m = surface_mask(recon["conf"], conf_quantile)[:, ::stride, ::stride]
    pts = recon["points"][:, ::stride, ::stride][m].reshape(-1, 3)
    cols = recon["colors"][:, ::stride, ::stride][m].reshape(-1, 3)
    return pts, cols


def transform_recon(recon: dict, T: np.ndarray) -> dict:
    """Apply a rigid transform to a reconstruction's pointmaps and camera poses."""
    out = dict(recon)
    out["points"] = apply(T, recon["points"].reshape(-1, 3)).reshape(recon["points"].shape)
    out["c2w"] = np.einsum("ij,njk->nik", T, recon["c2w"])
    return out
